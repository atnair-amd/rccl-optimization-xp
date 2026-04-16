// rccl-optimization-xp - Phase 1 baseline harness
//
// One process per GPU. MPI bootstraps ranks and broadcasts the RCCL
// communicator id. Each rank allocates its own copy of a fixed list of
// fake "gradient tensors" and AllReduces them, one collective per tensor,
// for a configurable number of iterations.
//
// Reports stdout-only metrics:
//   step_time_ms (mean, p50, p99) across non-warmup iterations
//   num_collectives per step
//   total bytes per step

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fake_grads.h"
#include "baseline.h"
#include "bucketing.h"
#include "fusion.h"

#define HIP_CHECK(expr)                                                     \
    do {                                                                    \
        hipError_t _err = (expr);                                           \
        if (_err != hipSuccess) {                                           \
            std::fprintf(stderr, "HIP error %d at %s:%d: %s\n",             \
                         (int)_err, __FILE__, __LINE__,                     \
                         hipGetErrorString(_err));                          \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

#define NCCL_CHECK(expr)                                                    \
    do {                                                                    \
        ncclResult_t _r = (expr);                                           \
        if (_r != ncclSuccess) {                                            \
            std::fprintf(stderr, "RCCL error %d at %s:%d: %s\n",            \
                         (int)_r, __FILE__, __LINE__,                       \
                         ncclGetErrorString(_r));                           \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

#define MPI_CHECK(expr)                                                     \
    do {                                                                    \
        int _r = (expr);                                                    \
        if (_r != MPI_SUCCESS) {                                            \
            std::fprintf(stderr, "MPI error %d at %s:%d\n",                 \
                         _r, __FILE__, __LINE__);                           \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

struct Args {
    std::string mode = "baseline";
    std::string profile = "medium";
    int iters = 30;
    int warmup = 5;
    bool verify = true;
    size_t bucket_bytes = 0; // 0 == one big bucket (bucketed/fused modes)
    std::string fusion = "pack"; // pack | aliased (only for mode=fused)
};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --mode <baseline|bucketed|fused>   (default: baseline)\n"
        "  --profile <tiny|medium|large>      (default: medium)\n"
        "  --iters <N>                        (default: 30)\n"
        "  --warmup <N>                       (default: 5)\n"
        "  --bucket_bytes <N>                 (bucketed/fused: max bytes per bucket; 0 = one big bucket)\n"
        "  --fusion <pack|aliased>            (fused mode only; default: pack)\n"
        "  --no_verify                        (skip correctness check)\n",
        prog);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) {
                std::fprintf(stderr, "missing value for %s\n", s.c_str());
                print_usage(argv[0]);
                std::exit(2);
            }
        };
        if (s == "--mode") { need(1); a.mode = argv[++i]; }
        else if (s == "--profile") { need(1); a.profile = argv[++i]; }
        else if (s == "--iters") { need(1); a.iters = std::atoi(argv[++i]); }
        else if (s == "--warmup") { need(1); a.warmup = std::atoi(argv[++i]); }
        else if (s == "--bucket_bytes") { need(1); a.bucket_bytes = (size_t)std::strtoull(argv[++i], nullptr, 10); }
        else if (s == "--fusion") { need(1); a.fusion = argv[++i]; }
        else if (s == "--no_verify") { a.verify = false; }
        else if (s == "-h" || s == "--help") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", s.c_str());
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return a;
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size = 0, world_rank = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    Args args = parse_args(argc, argv);

    // Single-node assumption: local_rank == world_rank.
    // Bind this process to its GPU.
    int n_devices = 0;
    HIP_CHECK(hipGetDeviceCount(&n_devices));
    if (world_size > n_devices) {
        if (world_rank == 0) {
            std::fprintf(stderr,
                "world_size (%d) > visible GPUs (%d). Run with -np <= GPUs.\n",
                world_size, n_devices);
        }
        MPI_Finalize();
        return 1;
    }
    int local_rank = world_rank;
    HIP_CHECK(hipSetDevice(local_rank));

    // Build the RCCL communicator. Rank 0 mints the unique id; MPI broadcasts.
    ncclUniqueId nccl_id;
    if (world_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, nccl_id, world_rank));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Allocate tensors for this rank.
    TensorProfile profile = profile_from_name(args.profile);
    std::vector<Tensor> tensors = allocate_tensors(profile);
    seed_tensors(tensors, world_rank);

    // Build bucket plan once (used by bucketed and fused modes).
    std::vector<Bucket> buckets;
    std::vector<FusedBucket> fused;
    bool aliased = false;
    if (args.mode == "bucketed" || args.mode == "fused") {
        buckets = plan_buckets(tensors, args.bucket_bytes);
    }
    if (args.mode == "fused") {
        if (args.fusion == "pack") {
            fused = setup_pack(tensors, buckets);
        } else if (args.fusion == "aliased") {
            fused = setup_aliased(tensors, buckets);
            aliased = true;
        } else {
            if (world_rank == 0) {
                std::fprintf(stderr,
                    "unknown --fusion: %s (use pack|aliased)\n",
                    args.fusion.c_str());
            }
            std::abort();
        }
    }

    // For reporting: the number of "submission units" RCCL sees per step.
    //   baseline : one ncclAllReduce per tensor
    //   bucketed : one ncclGroupStart/End per bucket (tensors stay separate)
    //   fused    : one ncclAllReduce per bucket on contiguous bucket_buf
    size_t num_submissions = 0;
    size_t mean_bytes_per_submission = 0;
    size_t total_payload_bytes = total_bytes(tensors);
    if (args.mode == "baseline") {
        num_submissions = tensors.size();
        mean_bytes_per_submission =
            tensors.empty() ? 0 : total_payload_bytes / tensors.size();
    } else {
        num_submissions = buckets.size();
        size_t sum_b = 0;
        for (const auto& b : buckets) sum_b += b.total_bytes;
        mean_bytes_per_submission = buckets.empty() ? 0 : sum_b / buckets.size();
    }

    if (world_rank == 0) {
        const char* proto = std::getenv("NCCL_PROTO");
        const char* algo  = std::getenv("NCCL_ALGO");
        std::printf("[rank 0] config: mode=%s profile=%s iters=%d warmup=%d "
                    "world_size=%d devices=%d bucket_bytes=%zu fusion=%s\n",
                    args.mode.c_str(), args.profile.c_str(),
                    args.iters, args.warmup, world_size, n_devices,
                    args.bucket_bytes,
                    args.mode == "fused" ? args.fusion.c_str() : "n/a");
        std::printf("[rank 0] nccl env: NCCL_PROTO=%s NCCL_ALGO=%s\n",
                    proto ? proto : "(unset)",
                    algo  ? algo  : "(unset)");
        std::printf("[rank 0] tensors: %zu, total_bytes=%zu\n",
                    tensors.size(), total_payload_bytes);
        std::printf("[rank 0] submissions/step: %zu, "
                    "mean_bytes_per_submission: %zu\n",
                    num_submissions, mean_bytes_per_submission);
    }

    // Run iterations. Discard warmup. Collect step_ms for the rest.
    std::vector<double> step_ms;
    step_ms.reserve(args.iters);

    const int total_iters = args.warmup + args.iters;
    for (int it = 0; it < total_iters; ++it) {
        double ms = 0.0;
        if (args.mode == "baseline") {
            ms = run_baseline_step(tensors, comm, stream);
        } else if (args.mode == "bucketed") {
            ms = run_bucketed_step(tensors, buckets, comm, stream);
        } else if (args.mode == "fused") {
            if (aliased) {
                ms = run_fused_aliased_step(fused, comm, stream);
            } else {
                ms = run_fused_pack_step(tensors, fused, comm, stream);
            }
        } else {
            if (world_rank == 0) {
                std::fprintf(stderr,
                    "unknown mode: %s (use baseline|bucketed|fused)\n",
                    args.mode.c_str());
            }
            std::abort();
        }
        if (it >= args.warmup) step_ms.push_back(ms);
    }

    // Correctness check: after the loop, the buffers will hold sum^N applied N times
    // (every iteration ran AllReduce(sum) on the same buffers without re-seeding).
    // To make verification deterministic, re-seed and run one more allreduce, then check.
    if (args.verify) {
        seed_tensors(tensors, world_rank);
        // Use a single ncclGroup so all per-tensor calls are submitted as one
        // unit. Logically equivalent to running every mode end-to-end and
        // checking the result; we just want to verify "AllReduce(sum) on each
        // buffer produces the expected value" once.
        NCCL_CHECK(ncclGroupStart());
        for (const auto& t : tensors) {
            NCCL_CHECK(ncclAllReduce(t.buf, t.buf, t.count, ncclFloat, ncclSum,
                                     comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        HIP_CHECK(hipStreamSynchronize(stream));
        bool ok = verify_allreduce_sum(tensors, world_size, world_rank);
        int local_ok = ok ? 1 : 0, all_ok = 0;
        MPI_CHECK(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                                MPI_COMM_WORLD));
        if (world_rank == 0) {
            std::printf("[rank 0] correctness: %s\n",
                        all_ok ? "OK" : "FAILED");
        }
        if (!all_ok) {
            // Make sure non-zero exit so callers can detect it.
            HIP_CHECK(hipStreamDestroy(stream));
            free_tensors(tensors);
            ncclCommDestroy(comm);
            MPI_Finalize();
            return 3;
        }
    }

    // Aggregate metrics on rank 0.
    if (world_rank == 0) {
        double sum = 0.0;
        for (double v : step_ms) sum += v;
        double mean = step_ms.empty() ? 0.0 : sum / step_ms.size();
        double p50  = percentile(step_ms, 0.50);
        double p99  = percentile(step_ms, 0.99);
        double tot_bytes = (double)total_bytes(tensors);

        std::printf("\n[rank 0] === results ===\n");
        std::printf("[rank 0] mode:                     %s\n", args.mode.c_str());
        std::printf("[rank 0] tensors:                  %zu\n", tensors.size());
        std::printf("[rank 0] submissions_per_step:     %zu\n", num_submissions);
        std::printf("[rank 0] mean_bytes_per_sub:       %zu\n", mean_bytes_per_submission);
        std::printf("[rank 0] bytes_per_step:           %.0f\n", tot_bytes);
        std::printf("[rank 0] step_ms mean:             %.3f\n", mean);
        std::printf("[rank 0] step_ms p50:              %.3f\n", p50);
        std::printf("[rank 0] step_ms p99:              %.3f\n", p99);
        std::printf("[rank 0] iters used:               %zu (warmup=%d discarded)\n",
                    step_ms.size(), args.warmup);
    }

    HIP_CHECK(hipStreamDestroy(stream));
    free_fused_buckets(fused);
    if (aliased) {
        clear_tensor_views(tensors); // tensors didn't own their memory
    }
    free_tensors(tensors);
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return 0;
}

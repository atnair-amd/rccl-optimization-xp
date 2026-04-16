#include "bucketing.h"

#include <cstdio>
#include <cstdlib>

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

std::vector<Bucket> plan_buckets(const std::vector<Tensor>& tensors,
                                 size_t bucket_bytes) {
    std::vector<Bucket> out;
    if (tensors.empty()) return out;

    Bucket cur;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const size_t t_bytes = tensors[i].bytes();
        const bool overflow =
            (bucket_bytes > 0) &&
            (!cur.tensor_indices.empty()) &&
            (cur.total_bytes + t_bytes > bucket_bytes);
        if (overflow) {
            out.push_back(std::move(cur));
            cur = Bucket{};
        }
        cur.tensor_indices.push_back(i);
        cur.total_bytes += t_bytes;
    }
    if (!cur.tensor_indices.empty()) out.push_back(std::move(cur));
    return out;
}

double run_bucketed_step(const std::vector<Tensor>& tensors,
                         const std::vector<Bucket>& buckets,
                         ncclComm_t comm,
                         hipStream_t stream) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, stream));
    for (const auto& b : buckets) {
        NCCL_CHECK(ncclGroupStart());
        for (size_t idx : b.tensor_indices) {
            const auto& t = tensors[idx];
            NCCL_CHECK(ncclAllReduce(t.buf, t.buf, t.count,
                                     ncclFloat, ncclSum,
                                     comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return (double)ms;
}

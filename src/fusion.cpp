#include "fusion.h"

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

// Build per-bucket layout (offsets, counts, total_count) from the logical plan.
static FusedBucket build_layout(const std::vector<Tensor>& tensors,
                                const Bucket& b) {
    FusedBucket fb;
    fb.tensor_indices = b.tensor_indices;
    fb.offsets.reserve(b.tensor_indices.size());
    fb.counts.reserve(b.tensor_indices.size());
    size_t cur_off = 0;
    for (size_t idx : b.tensor_indices) {
        fb.offsets.push_back(cur_off);
        fb.counts.push_back(tensors[idx].count);
        cur_off += tensors[idx].count;
    }
    fb.total_count = cur_off;
    return fb;
}

std::vector<FusedBucket> setup_pack(const std::vector<Tensor>& tensors,
                                    const std::vector<Bucket>& plan) {
    std::vector<FusedBucket> out;
    out.reserve(plan.size());
    for (const auto& b : plan) {
        FusedBucket fb = build_layout(tensors, b);
        HIP_CHECK(hipMalloc(&fb.bucket_buf, fb.total_count * sizeof(float)));
        out.push_back(std::move(fb));
    }
    return out;
}

std::vector<FusedBucket> setup_aliased(std::vector<Tensor>& tensors,
                                       const std::vector<Bucket>& plan) {
    // First free the original per-tensor allocations: their data will live
    // inside the bucket buffers after this.
    for (auto& t : tensors) {
        if (t.buf) HIP_CHECK(hipFree(t.buf));
        t.buf = nullptr;
    }

    std::vector<FusedBucket> out;
    out.reserve(plan.size());
    for (const auto& b : plan) {
        FusedBucket fb = build_layout(tensors, b);
        HIP_CHECK(hipMalloc(&fb.bucket_buf, fb.total_count * sizeof(float)));
        for (size_t i = 0; i < fb.tensor_indices.size(); ++i) {
            const size_t idx = fb.tensor_indices[i];
            tensors[idx].buf = fb.bucket_buf + fb.offsets[i];
        }
        out.push_back(std::move(fb));
    }
    return out;
}

void free_fused_buckets(std::vector<FusedBucket>& fbs) {
    for (auto& fb : fbs) {
        if (fb.bucket_buf) HIP_CHECK(hipFree(fb.bucket_buf));
        fb.bucket_buf = nullptr;
    }
    fbs.clear();
}

void clear_tensor_views(std::vector<Tensor>& tensors) {
    for (auto& t : tensors) {
        t.buf = nullptr;
    }
}

double run_fused_pack_step(const std::vector<Tensor>& tensors,
                           const std::vector<FusedBucket>& fbs,
                           ncclComm_t comm,
                           hipStream_t stream) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, stream));
    for (const auto& fb : fbs) {
        // Pack: memcpy each tensor's data into the bucket at its offset.
        for (size_t i = 0; i < fb.tensor_indices.size(); ++i) {
            const auto& t = tensors[fb.tensor_indices[i]];
            HIP_CHECK(hipMemcpyAsync(fb.bucket_buf + fb.offsets[i],
                                     t.buf,
                                     t.bytes(),
                                     hipMemcpyDeviceToDevice,
                                     stream));
        }
        // One AllReduce on the contiguous bucket.
        NCCL_CHECK(ncclAllReduce(fb.bucket_buf, fb.bucket_buf, fb.total_count,
                                 ncclFloat, ncclSum, comm, stream));
        // Unpack: memcpy reduced values back into each tensor.
        for (size_t i = 0; i < fb.tensor_indices.size(); ++i) {
            const auto& t = tensors[fb.tensor_indices[i]];
            HIP_CHECK(hipMemcpyAsync(t.buf,
                                     fb.bucket_buf + fb.offsets[i],
                                     t.bytes(),
                                     hipMemcpyDeviceToDevice,
                                     stream));
        }
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return (double)ms;
}

double run_fused_aliased_step(const std::vector<FusedBucket>& fbs,
                              ncclComm_t comm,
                              hipStream_t stream) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, stream));
    for (const auto& fb : fbs) {
        NCCL_CHECK(ncclAllReduce(fb.bucket_buf, fb.bucket_buf, fb.total_count,
                                 ncclFloat, ncclSum, comm, stream));
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return (double)ms;
}

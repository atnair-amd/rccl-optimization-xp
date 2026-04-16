#include "baseline.h"

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

double run_baseline_step(const std::vector<Tensor>& tensors,
                         ncclComm_t comm,
                         hipStream_t stream) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, stream));
    for (const auto& t : tensors) {
        NCCL_CHECK(ncclAllReduce(t.buf, t.buf, t.count,
                                 ncclFloat, ncclSum,
                                 comm, stream));
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return (double)ms;
}

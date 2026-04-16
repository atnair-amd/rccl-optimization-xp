#include "fake_grads.h"

#include <hip/hip_runtime.h>
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

TensorProfile profile_from_name(const std::string& name) {
    TensorProfile p;
    p.name = name;
    if (name == "tiny") {
        // 100 tensors x 64 KiB = 6.4 MiB total
        const size_t per = 64 * 1024;
        for (int i = 0; i < 100; ++i) p.sizes_bytes.push_back(per);
    } else if (name == "medium") {
        // 50 tensors x 1 MiB = 50 MiB total
        const size_t per = 1024 * 1024;
        for (int i = 0; i < 50; ++i) p.sizes_bytes.push_back(per);
    } else if (name == "large") {
        // 10 tensors x 64 MiB = 640 MiB total
        const size_t per = 64ull * 1024 * 1024;
        for (int i = 0; i < 10; ++i) p.sizes_bytes.push_back(per);
    } else {
        std::fprintf(stderr, "unknown profile: %s (use tiny|medium|large)\n",
                     name.c_str());
        std::abort();
    }
    return p;
}

TensorProfile profile_from_uniform(int n, size_t bytes_each) {
    TensorProfile p;
    p.name = "uniform";
    for (int i = 0; i < n; ++i) p.sizes_bytes.push_back(bytes_each);
    return p;
}

std::vector<Tensor> allocate_tensors(const TensorProfile& p) {
    std::vector<Tensor> ts;
    ts.reserve(p.sizes_bytes.size());
    for (size_t bytes : p.sizes_bytes) {
        Tensor t;
        t.count = bytes / sizeof(float);
        HIP_CHECK(hipMalloc(&t.buf, t.count * sizeof(float)));
        ts.push_back(t);
    }
    return ts;
}

void free_tensors(std::vector<Tensor>& tensors) {
    for (auto& t : tensors) {
        if (t.buf) HIP_CHECK(hipFree(t.buf));
        t.buf = nullptr;
        t.count = 0;
    }
    tensors.clear();
}

// Tiny kernel: fill a buffer with a constant value.
__global__ void fill_kernel(float* buf, float val, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

void seed_tensors(std::vector<Tensor>& tensors, int rank) {
    const float val = static_cast<float>(rank + 1);
    for (auto& t : tensors) {
        const size_t n = t.count;
        const int block = 256;
        const size_t grid = (n + block - 1) / block;
        hipLaunchKernelGGL(fill_kernel,
                           dim3((unsigned)grid), dim3((unsigned)block), 0, 0,
                           t.buf, val, n);
    }
    HIP_CHECK(hipDeviceSynchronize());
}

bool verify_allreduce_sum(const std::vector<Tensor>& tensors,
                          int world_size,
                          int rank) {
    // expected = sum_{r=0..world_size-1} (r + 1)
    const float expected = (float)world_size * ((float)world_size + 1.0f) / 2.0f;
    for (size_t i = 0; i < tensors.size(); ++i) {
        float head = 0.0f, tail = 0.0f;
        const auto& t = tensors[i];
        HIP_CHECK(hipMemcpy(&head, t.buf, sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&tail, t.buf + (t.count - 1), sizeof(float),
                            hipMemcpyDeviceToHost));
        if (head != expected || tail != expected) {
            std::fprintf(stderr,
                "[rank %d] correctness FAIL on tensor %zu: "
                "head=%f tail=%f expected=%f (world_size=%d)\n",
                rank, i, head, tail, expected, world_size);
            return false;
        }
    }
    return true;
}

size_t total_bytes(const std::vector<Tensor>& tensors) {
    size_t s = 0;
    for (const auto& t : tensors) s += t.bytes();
    return s;
}

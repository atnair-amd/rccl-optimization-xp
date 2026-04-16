#pragma once

#include <cstddef>
#include <string>
#include <vector>

// A fake "gradient tensor" living on a single GPU.
// Each rank owns its own copy of every tensor (different data per rank).
struct Tensor {
    float* buf = nullptr;     // device pointer
    size_t count = 0;         // number of float elements

    size_t bytes() const { return count * sizeof(float); }
};

// A profile is just a list of per-tensor sizes in bytes.
// Each rank allocates the same shapes (so AllReduce makes sense).
struct TensorProfile {
    std::string name;
    std::vector<size_t> sizes_bytes;
};

// Built-in profiles for quick experimentation.
//   tiny   : many small tensors  (latency regime)
//   medium : a moderate mix      (LL128 regime)
//   large  : few large tensors   (bandwidth regime)
TensorProfile profile_from_name(const std::string& name);

// A uniform profile: N tensors of equal size.
TensorProfile profile_from_uniform(int n, size_t bytes_each);

// Allocate device buffers for every tensor in the profile.
// Caller owns the returned vector and must call free_tensors().
std::vector<Tensor> allocate_tensors(const TensorProfile& p);

// Free device buffers and reset the vector.
void free_tensors(std::vector<Tensor>& tensors);

// Fill every tensor with the constant value (rank + 1) on the device.
// After AllReduce(sum) across world_size ranks, every element should
// equal sum_{r=0..world_size-1} (r + 1) = world_size * (world_size + 1) / 2.
void seed_tensors(std::vector<Tensor>& tensors, int rank);

// Verify the AllReduce(sum) result by sampling the first element of each
// tensor. Returns true on success. Prints a clear error on failure.
// Only meaningful after seed_tensors() + AllReduce(sum) on every tensor.
bool verify_allreduce_sum(const std::vector<Tensor>& tensors,
                          int world_size,
                          int rank);

// Total bytes across all tensors in the profile (host-side info).
size_t total_bytes(const std::vector<Tensor>& tensors);

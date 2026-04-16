#pragma once

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <vector>

#include "fake_grads.h"
#include "bucketing.h"

// A bucket whose backing storage is a single contiguous device buffer.
// Used for both pack and aliased modes; the difference is who owns the
// per-tensor storage:
//   pack    : tensors keep their own allocations; bucket_buf is a scratch
//             buffer used to copy data in/out around each AllReduce.
//   aliased : tensors do NOT have their own allocations; tensor.buf points
//             inside bucket_buf. No copies needed.
struct FusedBucket {
    float* bucket_buf = nullptr;            // device pointer, owned
    size_t total_count = 0;                 // float elements in bucket
    std::vector<size_t> tensor_indices;     // tensors that live in this bucket
    std::vector<size_t> offsets;            // float offset into bucket_buf
    std::vector<size_t> counts;             // float count per tensor
};

// PACK mode setup:
//   - tensors keep their own device allocations
//   - allocate one extra contiguous "bucket_buf" per bucket
//   - run_fused_pack_step copies tensors -> bucket -> allreduce -> tensors
std::vector<FusedBucket> setup_pack(const std::vector<Tensor>& tensors,
                                    const std::vector<Bucket>& plan);

// ALIASED mode setup:
//   - free each tensor's original device allocation
//   - allocate one contiguous bucket_buf per bucket
//   - rewrite tensors[i].buf to point inside the right bucket_buf
// After this, callers must NOT call free_tensors() on the same tensors vector;
// call free_fused_buckets() and clear_tensor_views() instead.
std::vector<FusedBucket> setup_aliased(std::vector<Tensor>& tensors,
                                       const std::vector<Bucket>& plan);

// Free bucket_buf for every FusedBucket. Safe for both pack and aliased
// configurations. Resets the vector.
void free_fused_buckets(std::vector<FusedBucket>& fbs);

// Zero out tensor.buf pointers without calling hipFree (used after aliased
// teardown so a later free_tensors() is a no-op).
void clear_tensor_views(std::vector<Tensor>& tensors);

// Pack-mode step:
//   for each bucket: gather (memcpy) tensors -> bucket_buf
//                    one ncclAllReduce on bucket_buf
//                    scatter (memcpy) bucket_buf -> tensors
double run_fused_pack_step(const std::vector<Tensor>& tensors,
                           const std::vector<FusedBucket>& fbs,
                           ncclComm_t comm,
                           hipStream_t stream);

// Aliased-mode step:
//   for each bucket: one ncclAllReduce on bucket_buf
// (tensors share storage with the bucket, so nothing to pack)
double run_fused_aliased_step(const std::vector<FusedBucket>& fbs,
                              ncclComm_t comm,
                              hipStream_t stream);

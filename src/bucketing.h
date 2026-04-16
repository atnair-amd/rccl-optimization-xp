#pragma once

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <vector>

#include "fake_grads.h"

// A bucket is a logical group of consecutive tensors. Tensors keep their
// original allocations; this struct just records "these N tensors should
// be communicated together as one ncclGroup".
struct Bucket {
    std::vector<size_t> tensor_indices;
    size_t total_bytes = 0;
};

// Greedy partition: walk tensors in order, start a new bucket whenever
// adding the next tensor would exceed `bucket_bytes`. If `bucket_bytes == 0`,
// every tensor becomes its own bucket (so this degrades to baseline shape).
std::vector<Bucket> plan_buckets(const std::vector<Tensor>& tensors,
                                 size_t bucket_bytes);

// Run one step in "bucketed" mode: for each bucket, wrap its AllReduce
// calls in ncclGroupStart/End so RCCL submits them as one group.
//
// IMPORTANT: this does NOT pack tensors into a contiguous buffer. Tensors
// remain in their own allocations. Bucketing here only changes how many
// "submission events" RCCL sees per step. Phase 3 (fusion) is the next step
// where memory layout actually changes.
//
// Returns elapsed milliseconds for the whole step (HIP event timed).
double run_bucketed_step(const std::vector<Tensor>& tensors,
                         const std::vector<Bucket>& buckets,
                         ncclComm_t comm,
                         hipStream_t stream);

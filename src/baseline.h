#pragma once

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <vector>

#include "fake_grads.h"

// Run one "step" of the baseline mode: one ncclAllReduce per tensor,
// back to back, on a single stream. Returns elapsed milliseconds for
// the whole step (host wall time bounded by hipEvent timings).
double run_baseline_step(const std::vector<Tensor>& tensors,
                         ncclComm_t comm,
                         hipStream_t stream);

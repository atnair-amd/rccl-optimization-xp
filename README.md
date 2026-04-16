# rccl-optimization-xp

Minimal C++/HIP/RCCL learning harness for studying bucketing, fusion, and
RCCL protocol/algorithm tradeoffs on a single 8x MI300X node.

One process per GPU. MPI bootstraps the RCCL communicator. Fake gradient
tensors so we can sweep sizes/counts independently of any model.

## Build

Requires ROCm (with RCCL) and OpenMPI installed system-wide.

```bash
make
```

Defaults assume `ROCM=/opt/rocm` and standard OpenMPI paths on Ubuntu 22.04.
Override on the command line if needed:

```bash
make ROCM=/opt/rocm MPI_INC=/path/to/mpi/include MPI_LIB=/path/to/mpi/lib
```

## Run

```bash
mpirun -np 8 ./bench --mode <baseline|bucketed|fused> [other flags]
```

CLI:

| flag | meaning | default |
|---|---|---|
| `--mode <baseline\|bucketed\|fused>` | which mode to run | `baseline` |
| `--profile <tiny\|medium\|large>` | tensor size profile | `medium` |
| `--iters <N>` | timed iterations | `30` |
| `--warmup <N>` | warmup iterations to discard | `5` |
| `--bucket_bytes <N>` | max bytes per bucket (0 = one big bucket); used by `bucketed` and `fused` | `0` |
| `--fusion <pack\|aliased>` | only for `fused`; how the bucket buffer is laid out | `pack` |
| `--no_verify` | skip the post-loop correctness check | off |

Profiles:

- `tiny`   - 100 tensors x 64 KiB     (latency regime)
- `medium` - 50 tensors x 1 MiB       (LL128 regime)
- `large`  - 10 tensors x 64 MiB      (bandwidth regime)

## Modes

### `baseline`
One `ncclAllReduce` per tensor, back to back, on a single stream. Reference
point.

### `bucketed`
Group consecutive tensors into buckets up to `--bucket_bytes`. For each
bucket, wrap its per-tensor `ncclAllReduce` calls in `ncclGroupStart/End`
so RCCL submits them as one group. Tensors keep their own allocations -
no memory layout change.

### `fused`
Group consecutive tensors into buckets, then either:
- `--fusion pack`    - tensors keep their own allocations; per step,
  `hipMemcpyAsync` them into a contiguous bucket buffer, do one `ncclAllReduce`,
  and copy back.
- `--fusion aliased` - allocate one flat buffer per bucket up front; tensor
  pointers are offsets into it. One `ncclAllReduce` per bucket. No copies.

## Output

Stdout-only. Rank 0 prints config, NCCL env vars in effect, the bucket
plan, and per-step timing aggregates (mean / p50 / p99) over `--iters`
non-warmup iterations.

## Comparison scripts

Each script runs the bench in a few configurations and prints rank-0 lines.

```bash
bash compare_phase2.sh tiny     # baseline vs bucketed
bash compare_phase3.sh tiny     # bucketed vs fused/pack vs fused/aliased
bash compare_phase4.sh          # NCCL_PROTO x NCCL_ALGO sweep across profiles
```

## What the phases teach

| Phase | Question it answers |
|---|---|
| 1: baseline | What does naive one-allreduce-per-tensor cost? |
| 2: bucketed | How much does grouping submissions help (no memory change)? |
| 3: fused/pack | What does `pack-and-copy` add on top of grouping? |
| 3: fused/aliased | What does `contiguous memory from the start` actually buy? |
| 4: protocol/algo | Do `NCCL_PROTO` / `NCCL_ALGO` matter on top of app-level choices? |

## Known good results (8x MI300X, banff-cyxtera-s70-2, tiny profile)

| mode | submissions/step | step_ms mean |
|---|---|---|
| baseline | 100 | 2.50 |
| bucketed (1 group) | 1 | 0.21 |
| fused/pack (1 bucket) | 1 | 0.99 |
| fused/aliased (1 bucket) | 1 | 0.08 |

Aliased fusion is roughly 30x baseline because we eliminate both the
per-tensor launch overhead and the per-step packing memcopies.

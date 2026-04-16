#!/usr/bin/env bash
# Phase 4: sweep NCCL_PROTO and NCCL_ALGO at the fused/aliased setting
# (one big bucket, no copies). This isolates the RCCL-level knob effect
# from the app-level fusion choice.
#
# Usage: ./compare_phase4.sh [profile]   (default: all three)

set -e

iters=30
warmup=5

cd "$(dirname "$0")"

profiles="${1:-tiny medium large}"

run_one() {
    local profile="$1"
    local proto="$2"
    local algo="$3"
    echo "------ profile=${profile} NCCL_PROTO=${proto} NCCL_ALGO=${algo} ------"
    NCCL_PROTO="${proto}" NCCL_ALGO="${algo}" \
        mpirun -np 8 \
            -x NCCL_PROTO -x NCCL_ALGO \
            ./bench --mode fused --fusion aliased --profile "${profile}" \
            --iters "${iters}" --warmup "${warmup}" --bucket_bytes 0 \
            --no_verify \
            2>&1 | grep -E '^\[rank 0\] (config:|nccl env:|step_ms (mean|p99)|submissions/step)'
    echo ""
}

for profile in $profiles; do
    echo "============================================================"
    echo "  PROFILE = ${profile}"
    echo "============================================================"
    for proto in Simple LL LL128; do
        for algo in Ring Tree; do
            run_one "${profile}" "${proto}" "${algo}"
        done
    done
done

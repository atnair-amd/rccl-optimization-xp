#!/usr/bin/env bash
# Phase 3 comparison: bucketed (no fusion) vs fused/pack vs fused/aliased.
# Usage: ./compare_phase3.sh [profile]   (default: tiny)

set -e
profile="${1:-tiny}"
iters=30
warmup=5

cd "$(dirname "$0")"

run_one() {
    local mode="$1"
    local bb="$2"
    local fusion="${3:-pack}"
    echo "====== mode=${mode} profile=${profile} bucket_bytes=${bb} fusion=${fusion} ======"
    mpirun -np 8 ./bench --mode "${mode}" --profile "${profile}" \
        --iters "${iters}" --warmup "${warmup}" \
        --bucket_bytes "${bb}" --fusion "${fusion}" \
        2>&1 | grep -E '^\[rank 0\]'
    echo ""
}

# baseline reference
run_one baseline 0
# bucketed (logical grouping only, no packing)
run_one bucketed 0
# fused with pack (one big bucket, copy in/out)
run_one fused 0 pack
# fused with aliased (one big bucket, no copies)
run_one fused 0 aliased
# fused with smaller buckets to see the granularity tradeoff
run_one fused 1048576 pack
run_one fused 1048576 aliased

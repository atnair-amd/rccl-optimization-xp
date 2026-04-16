#!/usr/bin/env bash
# Compare baseline vs bucketed at a few bucket sizes for one profile.
# Usage: ./compare_phase2.sh [profile]   (default: tiny)

set -e
profile="${1:-tiny}"
iters=30
warmup=5

cd "$(dirname "$0")"

run_one() {
    local mode="$1"
    local bb="$2"
    echo "====== mode=${mode} profile=${profile} bucket_bytes=${bb} ======"
    mpirun -np 8 ./bench --mode "${mode}" --profile "${profile}" \
        --iters "${iters}" --warmup "${warmup}" --bucket_bytes "${bb}" \
        2>&1 | grep -E '^\[rank 0\]'
    echo ""
}

run_one baseline 0
run_one bucketed 0
run_one bucketed 262144
run_one bucketed 1048576
run_one bucketed 8388608
run_one bucketed 134217728

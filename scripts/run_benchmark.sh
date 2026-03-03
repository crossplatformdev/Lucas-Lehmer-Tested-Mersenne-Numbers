#!/usr/bin/env bash
# scripts/run_benchmark.sh
#
# Run a bounded benchmark and write machine-readable CSV output.
#
# Usage:
#   scripts/run_benchmark.sh [OUTPUT_CSV] [START_INDEX] [MAX_INDEX] [THREADS]
#
# Defaults:
#   OUTPUT_CSV   = bin/benchmark.csv
#   START_INDEX  = 14   (p=9689)
#   MAX_INDEX    = 27   (exponents[0..26], p ≤ 44497)
#   THREADS      = 1
#
# The script runs the benchmark twice: once at THREADS=1 and once at THREADS=max,
# appending a 'threads' column to distinguish the runs.
#
# Example:
#   scripts/run_benchmark.sh bin/ci_bench.csv 14 27 0

set -euo pipefail

BIN="${BIGNUM_BIN:-./bin/bignum}"
OUTPUT="${1:-bin/benchmark.csv}"
START="${2:-14}"
MAX_IDX="${3:-27}"
THREADS_ARG="${4:-1}"

mkdir -p "$(dirname "$OUTPUT")"

run_once() {
    local t="$1"
    local tmp
    tmp="$(mktemp /tmp/bench_XXXXXX.csv)"
    LL_STOP_AFTER_ONE=0 \
    LL_MAX_EXPONENT_INDEX="$MAX_IDX" \
    LL_BENCH_OUTPUT="$tmp" \
        "$BIN" "$START" "$t" >/dev/null
    # Prepend thread count column
    awk -F, -v threads="$t" '
        NR==1 { print "threads," $0 }
        NR>1  { print threads "," $0 }
    ' "$tmp"
    rm -f "$tmp"
}

echo "=== Benchmark: start_index=$START  max_index=$MAX_IDX  binary=$BIN ==="

# Write header once
echo "threads,p,is_prime,time_sec" > "$OUTPUT"

# 1-thread run
echo "--- 1 thread ---"
run_once 1 | tail -n +2 >> "$OUTPUT"

# Multi-thread run (use requested threads or max)
if [ "$THREADS_ARG" -ne 1 ] 2>/dev/null; then
    echo "--- $THREADS_ARG threads ---"
    run_once "$THREADS_ARG" | tail -n +2 >> "$OUTPUT"
fi

echo "=== Results written to $OUTPUT ==="
echo "Rows: $(( $(wc -l < "$OUTPUT") - 1 ))"
cat "$OUTPUT"

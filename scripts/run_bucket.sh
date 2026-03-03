#!/usr/bin/env bash
# run_bucket.sh – Run the Lucas-Lehmer prime sweep for one power-of-two bucket.
#
# Usage:
#   scripts/run_bucket.sh <bucket_n> [threads] [dry_run]
#
# Environment overrides (all optional):
#   LL_BUCKET_N                     – bucket number (1-64); overrides arg
#   LL_THREADS                      – worker thread count (0 = all cores)
#   LL_DRY_RUN                      – 1 = print plan only, 0 = run tests
#   LL_REVERSE_ORDER                – 1 = largest exponent first
#   LL_MAX_EXPONENTS_PER_JOB        – safety cap on exponents per bucket
#   LL_RESUME_FROM_EXPONENT         – skip exponents below this value
#   LL_STOP_AFTER_FIRST_PRIME_RESULT– stop after first new discovery
#   LL_OUTPUT_DIR                   – directory for output files
#   LL_BENCHMARK_MODE               – 1 = skip is_prime_exponent() check
#   LL_PROGRESS                     – 1 = show per-iteration progress
#
# Examples:
#   scripts/run_bucket.sh 5                    # run bucket 5, all cores
#   scripts/run_bucket.sh 5 4                  # run bucket 5, 4 threads
#   scripts/run_bucket.sh 5 1 1                # dry-run, 1 thread
#   LL_DRY_RUN=1 scripts/run_bucket.sh 5       # dry-run via env var

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${REPO_ROOT}/bin/bignum"

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not found: ${BIN}" >&2
    echo "Run 'make all' first." >&2
    exit 1
fi

# Parse positional args (env vars take precedence if set).
BUCKET_N="${LL_BUCKET_N:-${1:-}}"
THREADS_ARG="${2:-0}"
DRY_RUN_ARG="${3:-0}"

if [[ -z "${BUCKET_N}" ]]; then
    echo "Usage: $0 <bucket_n> [threads] [dry_run]" >&2
    exit 1
fi

if (( BUCKET_N < 1 || BUCKET_N > 64 )); then
    echo "Error: bucket_n must be in [1, 64], got ${BUCKET_N}" >&2
    exit 1
fi

OUTPUT_DIR="${LL_OUTPUT_DIR:-bucket_out/bucket_${BUCKET_N}/}"
mkdir -p "${OUTPUT_DIR}"

echo "=== run_bucket.sh: bucket ${BUCKET_N} ==="
echo "  binary      : ${BIN}"
echo "  output_dir  : ${OUTPUT_DIR}"
echo ""

exec env \
    LL_SWEEP_MODE=power_bucket_primes \
    LL_BUCKET_N="${BUCKET_N}" \
    LL_THREADS="${LL_THREADS:-${THREADS_ARG}}" \
    LL_DRY_RUN="${LL_DRY_RUN:-${DRY_RUN_ARG}}" \
    LL_REVERSE_ORDER="${LL_REVERSE_ORDER:-0}" \
    LL_MAX_EXPONENTS_PER_JOB="${LL_MAX_EXPONENTS_PER_JOB:-0}" \
    LL_RESUME_FROM_EXPONENT="${LL_RESUME_FROM_EXPONENT:-0}" \
    LL_STOP_AFTER_FIRST_PRIME_RESULT="${LL_STOP_AFTER_FIRST_PRIME_RESULT:-0}" \
    LL_OUTPUT_DIR="${OUTPUT_DIR}" \
    LL_BENCHMARK_MODE="${LL_BENCHMARK_MODE:-0}" \
    LL_PROGRESS="${LL_PROGRESS:-0}" \
    "${BIN}" 0 "${LL_THREADS:-${THREADS_ARG}}"

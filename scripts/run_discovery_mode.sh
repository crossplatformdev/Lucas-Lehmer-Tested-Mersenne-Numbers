#!/usr/bin/env bash
# scripts/run_discovery_mode.sh
# Run the discover mode locally, mirroring the GitHub Actions workflow.
#
# Usage:
#   scripts/run_discovery_mode.sh [threads]
#
# Key environment variables (all optional):
#   LL_SINGLE_EXPONENT   Explicit first exponent (default: 0)
#   LL_MIN_EXPONENT      Lower bound exclusive    (default: 136279841)
#   LL_MAX_EXPONENT      Upper bound inclusive    (default: 200000000)
#   LL_SHARD_COUNT       Number of shards         (default: 1)
#   LL_SHARD_INDEX       This shard (0-based)     (default: 0)
#   LL_REVERSE_ORDER     1 to run largest first   (default: 0)
#   LL_STOP_AFTER_N_CASES  Safety limit           (default: 0 = no limit)
#   LL_DRY_RUN           1 to print plan only     (default: 0)
#   LL_PROGRESS          1 for LL progress output (default: 0)
#   LL_OUTPUT_DIR        Where to write results   (default: discover_out/)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/bin/bignum"

if [ ! -x "${BIN}" ]; then
    echo "Binary not found: ${BIN}"
    echo "Run 'make all' first."
    exit 1
fi

THREADS="${1:-0}"

export LL_SWEEP_MODE=discover
export LL_SINGLE_EXPONENT="${LL_SINGLE_EXPONENT:-0}"
export LL_MIN_EXPONENT="${LL_MIN_EXPONENT:-136279841}"
export LL_MAX_EXPONENT="${LL_MAX_EXPONENT:-200000000}"
export LL_SHARD_COUNT="${LL_SHARD_COUNT:-1}"
export LL_SHARD_INDEX="${LL_SHARD_INDEX:-0}"
export LL_REVERSE_ORDER="${LL_REVERSE_ORDER:-0}"
export LL_STOP_AFTER_N_CASES="${LL_STOP_AFTER_N_CASES:-0}"
export LL_DRY_RUN="${LL_DRY_RUN:-0}"
export LL_PROGRESS="${LL_PROGRESS:-0}"
export LL_OUTPUT_DIR="${LL_OUTPUT_DIR:-${REPO_ROOT}/discover_out/}"

mkdir -p "${LL_OUTPUT_DIR}"

echo "=== run_discovery_mode.sh ==="
echo "  binary        : ${BIN}"
echo "  threads       : ${THREADS} (0=all cores)"
echo "  output_dir    : ${LL_OUTPUT_DIR}"
echo "  single_exp    : ${LL_SINGLE_EXPONENT}"
echo "  min_excl      : ${LL_MIN_EXPONENT}"
echo "  max_incl      : ${LL_MAX_EXPONENT}"
echo "  shard         : ${LL_SHARD_INDEX}/${LL_SHARD_COUNT}"
echo "  reverse       : ${LL_REVERSE_ORDER}"
echo "  stop_after_n  : ${LL_STOP_AFTER_N_CASES}"
echo "  dry_run       : ${LL_DRY_RUN}"
echo ""

"${BIN}" 0 "${THREADS}"

if [ -f "${LL_OUTPUT_DIR}/discovery_event.json" ]; then
    echo ""
    echo "*** NEW MERSENNE PRIME FOUND — see ${LL_OUTPUT_DIR}/discovery_event.json ***"
    cat "${LL_OUTPUT_DIR}/discovery_event.json"
fi

#!/usr/bin/env bash
# scripts/run_expr_mod_gmp.sh
#
# Bash runner for bin/ll_fft – the Mersenne pipeline binary.
#
# Computes s_{p-2} mod (2^p - 1) for:
#   1. Every known Mersenne-prime exponent (in batches).
#   2. A forward search from SEARCH_START for the first prime p
#      whose result is 0 (i.e. a new Mersenne-prime candidate).
#
# Usage:
#   scripts/run_expr_mod_gmp.sh [BATCH_SIZE [SEARCH_START [MAX_PRIME_CHECKS [KNOWN_LIMIT]]]]
#
# Defaults:
#   BATCH_SIZE       = 24
#   SEARCH_START     = 136279841
#   MAX_PRIME_CHECKS = 1000000
#   KNOWN_LIMIT      = 0   (0 = use all known exponents)
#
# Environment overrides (alternative to positional args):
#   EXPR_BATCH_SIZE       EXPR_SEARCH_START   EXPR_MAX_PRIME_CHECKS
#   EXPR_KNOWN_LIMIT      EXPR_BIN            EXPR_SRC
#
# Outputs a CSV file to bin/expr_batch_results_<timestamp>.csv.
#
# Notes on integer-overflow safety:
#   The Miller-Rabin primality test uses bash 64-bit signed arithmetic.
#   Intermediate products are at most (n-1)^2 where n is the candidate.
#   For candidates up to ~3 billion, (n-1)^2 < 9e18 < 2^63, which is safe.
#   With bases {2,3,5,7,11,13,17} the test is deterministic for all n < 3.3e24.

set -euo pipefail

# ── Locate repository root ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Parameters ───────────────────────────────────────────────────────────────
BATCH_SIZE="${EXPR_BATCH_SIZE:-${1:-24}}"
SEARCH_START="${EXPR_SEARCH_START:-${2:-136279841}}"
MAX_PRIME_CHECKS="${EXPR_MAX_PRIME_CHECKS:-${3:-1000000}}"
KNOWN_LIMIT="${EXPR_KNOWN_LIMIT:-${4:-0}}"

EXE="${EXPR_BIN:-${REPO_ROOT}/bin/ll_fft}"
SRC="${EXPR_SRC:-${REPO_ROOT}/src/ll_fft.cpp}"

# ── Build if necessary ───────────────────────────────────────────────────────
if [[ ! -x "${EXE}" ]]; then
    echo "[$(date)] Building ll_fft..."
    g++ -O3 -std=c++17 -march=native -mtune=native -Wall -Wextra -Wpedantic \
        -o "${EXE}" "${SRC}"
fi

# ── Known Mersenne-prime exponents ───────────────────────────────────────────
KNOWN_EXPONENTS=(
    2 3 5 7 13 17 19 31 61 89 107 127
    521 607 1279 2203 2281 3217 4253 4423 9689 9941 11213 19937
    21701 23209 44497 86243 110503 132049 216091 756839 859433 1257787 1398269
    2976221 3021377 6972593 13466917 20996011 24036583 25964951 30402457
    32582657 37156667 42643801 43112609 57885161 74207281 77232917 82589933
    136279841
)

# Apply KNOWN_LIMIT
if [[ "${KNOWN_LIMIT}" -gt 0 ]]; then
    total=${#KNOWN_EXPONENTS[@]}
    take=$(( KNOWN_LIMIT < total ? KNOWN_LIMIT : total ))
    KNOWN_EXPONENTS=("${KNOWN_EXPONENTS[@]:0:${take}}")
fi

# ── CSV output ───────────────────────────────────────────────────────────────
mkdir -p "${REPO_ROOT}/bin"
STAMP="$(date '+%Y%m%d_%H%M%S')"
CSV_PATH="${REPO_ROOT}/bin/expr_batch_results_${STAMP}.csv"
echo "kind,index,batch,exp,result,elapsed_ms" > "${CSV_PATH}"

# ── Portable nanosecond timestamp ────────────────────────────────────────────
# date +%s%N works on Linux (GNU coreutils).  On macOS/BSD use python3 as a
# fallback so the script remains portable outside the Linux CI runner.
_now_ns() {
    if date +%s%N 2>/dev/null | grep -qv '^[0-9]*N$'; then
        date +%s%N
    else
        python3 -c "import time; print(int(time.monotonic_ns()))"
    fi
}

# ── invoke_expr_run ──────────────────────────────────────────────────────────
# Args: <exponent>
# Sets globals: INVOKE_RESULT  INVOKE_ELAPSED_MS  INVOKE_EXIT_CODE
invoke_expr_run() {
    local exp="$1"
    local start_ns end_ns output exit_code

    start_ns=$(_now_ns)
    output=$("${EXE}" "${exp}" 2>/dev/null) && exit_code=0 || exit_code=$?
    end_ns=$(_now_ns)

    INVOKE_ELAPSED_MS=$(( (end_ns - start_ns) / 1000000 ))

    if [[ "${exit_code}" -ne 0 ]]; then
        INVOKE_RESULT="ERROR:exit${exit_code}"
    else
        # Last non-empty line of stdout (mirrors PS Select-Object -Last 1)
        INVOKE_RESULT="$(echo "${output}" | grep -v '^[[:space:]]*$' | tail -1)"
    fi

    INVOKE_EXIT_CODE="${exit_code}"
}

# ── miller_rabin_modpow ──────────────────────────────────────────────────────
# Computes (base ^ exp) mod m using 64-bit bash arithmetic.
miller_rabin_modpow() {
    local base="$1" exp="$2" m="$3"
    local result=1
    base=$(( base % m ))
    while (( exp > 0 )); do
        if (( exp & 1 )); then
            result=$(( result * base % m ))
        fi
        exp=$(( exp >> 1 ))
        base=$(( base * base % m ))
    done
    echo "${result}"
}

# ── is_prime ─────────────────────────────────────────────────────────────────
# Deterministic Miller-Rabin primality test with bases {2,3,5,7,11,13,17}.
# Returns 0 (success/true) if n is prime, 1 (failure/false) if composite.
is_prime() {
    local n="$1"

    (( n < 2 ))               && return 1
    (( n == 2 || n == 3 ))    && return 0
    (( n % 2 == 0 ))          && return 1

    # Factor out powers of 2: write n-1 = d * 2^s
    local d=$(( n - 1 ))
    local s=0
    while (( d % 2 == 0 )); do
        d=$(( d / 2 ))
        s=$(( s + 1 ))
    done

    local bases=(2 3 5 7 11 13 17)
    local a x r witness_passed

    for a in "${bases[@]}"; do
        (( a >= n )) && continue

        x=$(miller_rabin_modpow "${a}" "${d}" "${n}")

        if (( x == 1 || x == n - 1 )); then
            continue
        fi

        witness_passed=0
        for (( r = 1; r < s; r++ )); do
            x=$(( x * x % n ))
            if (( x == n - 1 )); then
                witness_passed=1
                break
            fi
        done

        [[ "${witness_passed}" -eq 0 ]] && return 1
    done

    return 0
}

# ── Run known exponents in batches ───────────────────────────────────────────
echo "=== Running known exponents in batches of ${BATCH_SIZE} ==="

total_known=${#KNOWN_EXPONENTS[@]}
idx=0
batch=0
start=0

while (( start < total_known )); do
    batch=$(( batch + 1 ))
    end=$(( start + BATCH_SIZE - 1 ))
    (( end >= total_known )) && end=$(( total_known - 1 ))

    echo ""
    echo "[Batch ${batch}] indices $(( start + 1 ))..$(( end + 1 ))"

    for (( i = start; i <= end; i++ )); do
        idx=$(( idx + 1 ))
        exp="${KNOWN_EXPONENTS[$i]}"

        invoke_expr_run "${exp}"

        echo "KNOWN,${idx},${batch},${exp},${INVOKE_ELAPSED_MS}"
        printf "KNOWN,%d,%d,%s,,%d\n" "${idx}" "${batch}" "${exp}" "${INVOKE_ELAPSED_MS}" \
            >> "${CSV_PATH}"
    done

    start=$(( start + BATCH_SIZE ))
done

# ── Search for first prime exponent with result 0 ────────────────────────────
echo ""
echo "=== Searching first prime exponent >= ${SEARCH_START} with result 0 ==="

candidate="${SEARCH_START}"
(( candidate < 2 )) && candidate=2
(( candidate > 2 && candidate % 2 == 0 )) && candidate=$(( candidate + 1 ))

prime_checks=0
found=0

while (( prime_checks < MAX_PRIME_CHECKS )); do
    if is_prime "${candidate}"; then
        prime_checks=$(( prime_checks + 1 ))

        invoke_expr_run "${candidate}"

        echo "SEARCH,${prime_checks},-,${candidate},${INVOKE_RESULT},${INVOKE_ELAPSED_MS}"
        printf "SEARCH,%d,-,%s,%s,%d\n" \
            "${prime_checks}" "${candidate}" "${INVOKE_RESULT}" "${INVOKE_ELAPSED_MS}" \
            >> "${CSV_PATH}"

        if [[ "${INVOKE_RESULT}" == "Residuo = 0" ]]; then
            printf "FOUND,,,%s,0,\n" "${candidate}" >> "${CSV_PATH}"
            echo ""
            echo "First prime exponent with result 0: ${candidate}"
            found=1
            break
        fi
    fi

    if (( candidate == 2 )); then
        candidate=3
    else
        candidate=$(( candidate + 2 ))
    fi
done

if [[ "${found}" -eq 0 ]]; then
    echo ""
    echo "Search stopped after ${prime_checks} prime checks (MaxPrimeChecks=${MAX_PRIME_CHECKS})."
fi

echo ""
echo "Finished. Results saved in ${CSV_PATH}"

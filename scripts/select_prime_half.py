#!/usr/bin/env python3
"""
scripts/select_prime_half.py
-----------------------------
Filter a batch matrix JSON (as produced by split_bucket_batches) to include
only the lower or upper half of the prime exponents, ordered ascending by value.

Usage:
    echo '<batch-matrix-json>' | python3 scripts/select_prime_half.py <lower_half|upper_half>

Output:
    Filtered batch matrix JSON (suitable for use as a GitHub Actions matrix).

Half definition:
    total = sum of batch_size across all batches (total remaining prime count)
    mid   = total // 2
    lower = first 'mid' primes  (indices 0 .. mid-1, ascending by value)
    upper = last  'total - mid' primes (indices mid .. total-1)

When 'total' is odd the upper half receives one more prime than the lower half.
When 'total' == 1 the lower half is empty and the upper half contains the single
prime; this is an expected edge case that callers should guard against.

Partial-batch handling:
  lower – the last batch may be trimmed: batch_size is reduced so that the
          worker tests exactly the right number of primes.  batch_min_exponent
          is unchanged; the binary uses LL_MAX_EXPONENTS_PER_JOB (= batch_size)
          as the count limit.
  upper – the first batch may be trimmed: batch_min_exponent is advanced to
          the correct prime value and batch_size is reduced accordingly.  The
          binary uses LL_RESUME_FROM_EXPONENT (= batch_min_exponent) to start
          at the right prime (>= that value, inclusive).
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from generate_bucket_primes import enumerate_bucket_primes  # noqa: E402


def select_half(matrix: list, half: str) -> list:
    """Filter the batch matrix to the lower or upper half of prime exponents.

    Args:
        matrix: List of batch descriptor dicts, sorted ascending by exponent,
                as produced by split_bucket_batches.
        half:   'lower_half' or 'upper_half'.

    Returns:
        Filtered list of batch descriptors covering the requested half.
        May be empty if the total prime count is 0 or the split produces an
        empty side (e.g. single prime with half='lower_half').
    """
    if half not in ("lower_half", "upper_half"):
        raise ValueError(f"half must be 'lower_half' or 'upper_half', got {half!r}")

    if not matrix:
        return []

    total = sum(b["batch_size"] for b in matrix)
    if total == 0:
        return []

    # lower half: indices [0, mid-1], upper half: indices [mid, total-1]
    mid = total // 2

    if half == "lower_half":
        if mid == 0:
            return []

        result = []
        seen = 0
        for b in matrix:
            if seen >= mid:
                break
            take = min(b["batch_size"], mid - seen)
            if take == b["batch_size"]:
                result.append(b)
            else:
                # Partial batch: test only 'take' primes from batch_min_exponent.
                # Note: dict(b) is a safe full copy here because all batch
                # descriptor values are primitives (int/str).
                partial = dict(b)
                partial["batch_size"] = take

                # Keep the batch metadata self-consistent with the trimmed size.
                # The lower-half partial batch always starts at the original
                # batch_prime_start_index and takes the first 'take' primes.
                try:
                    start_idx = b["batch_prime_start_index"]
                    end_idx = start_idx + take - 1
                    bucket_primes = enumerate_bucket_primes(b["bucket_n"])
                    partial["batch_prime_end_index"] = end_idx
                    partial["batch_max_exponent"] = bucket_primes[end_idx]
                except (KeyError, IndexError):
                    # If any of the expected keys are missing or indices are
                    # out of range, leave the original metadata unchanged.
                    pass

                # If a worker_name is present, annotate it to reflect that this
                # entry covers only a partial lower batch.
                if "worker_name" in partial:
                    original_name = str(partial["worker_name"])
                    partial["worker_name"] = (
                        f"{original_name} (partial lower {take}/{b['batch_size']})"
                    )
                result.append(partial)
            seen += take
        return result

    else:  # upper
        upper_count = total - mid
        if upper_count == 0:
            return []

        result = []
        seen = 0
        for b in matrix:
            batch_end = seen + b["batch_size"]

            if batch_end <= mid:
                # Entire batch belongs to the lower half – skip it.
                seen = batch_end
                continue

            if seen < mid:
                # Straddling batch: skip the first (mid - seen) primes within it.
                skip_in_batch = mid - seen
                # Look up the actual prime value at the new start position.
                bucket_primes = enumerate_bucket_primes(b["bucket_n"])
                new_start_idx = b["batch_prime_start_index"] + skip_in_batch
                new_min_exp = bucket_primes[new_start_idx]

                # Note: dict(b) is a safe full copy here because all batch
                # descriptor values are primitives (int/str).
                partial = dict(b)
                partial["batch_min_exponent"] = new_min_exp
                partial["batch_size"] = b["batch_size"] - skip_in_batch
                partial["batch_prime_start_index"] = new_start_idx
                # Regenerate worker_name (if present) so it reflects the new start.
                old_worker_name = b.get("worker_name")
                if old_worker_name is not None:
                    partial["worker_name"] = (
                        f"{old_worker_name}-upper-half-from-exp-{new_min_exp}-idx-{new_start_idx}"
                    )
                result.append(partial)
            else:
                result.append(b)

            seen = batch_end

        return result


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("lower_half", "upper_half"):
        print(
            "Usage: echo '<batch-matrix-json>' | "
            "python3 scripts/select_prime_half.py <lower_half|upper_half>",
            file=sys.stderr,
        )
        sys.exit(1)

    half = sys.argv[1]
    data = sys.stdin.read().strip()
    matrix = json.loads(data)
    result = select_half(matrix, half)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

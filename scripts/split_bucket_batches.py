#!/usr/bin/env python3
"""
split_bucket_batches.py – Generate a GitHub Actions batch matrix for the
power-range-prime-sweep workflow.

For each selected bucket, enumerates all prime exponents and splits them into
contiguous batches of at most BATCH_SIZE (default: 1000) primes.  Outputs a
JSON array of batch descriptor objects suitable for use as a GitHub Actions
``matrix.include`` value.

Usage:
    python3 scripts/split_bucket_batches.py [options]

Options:
    --bucket-start N   First bucket (1-64, default: 1)
    --bucket-end   N   Last  bucket (1-64, default: 64)
    --batch-size   N   Max primes per batch (default: 1000)
    --dry-run          Print human-readable plan instead of JSON
    --output       FILE  Write JSON to FILE instead of stdout

Worker name format (deterministic, stable):
    bucket-{N:02d}-batch-{start_ordinal:04d}-{end_ordinal:04d}-exp-{pmin}-{pmax}

Example:
    bucket-17-batch-0001-1000-exp-65537-65867
"""

import argparse
import json
import math
import os
import sys

# Import helpers from the sibling script (same scripts/ directory).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generate_bucket_primes import bucket_range, enumerate_bucket_primes  # noqa: E402

BATCH_SIZE_DEFAULT = 1000
GITHUB_MATRIX_MAX = 256

# ── Per-bucket worst-case single-exponent time (seconds) ──────────────────
# Buckets 9–16: measured from REPORT_MERSENNE_EXPONENTS-256-65536.MD.
# Buckets 17+:  extrapolated at ×4.5 per bucket (empirical growth factor
#               observed B13→B14: ×3.5, B14→B15: ×4.0, B15→B16: ×4.4).
# Buckets 1–10: sub-millisecond; floor at 0.001 s so the cap logic works.
# ─────────────────────────────────────────────────────────────────────────
_BUCKET_WORST_SEC: dict[int, float] = {
    1:  0.001, 2:  0.001, 3:  0.001, 4:  0.001,
    5:  0.001, 6:  0.001, 7:  0.001, 8:  0.001,
    9:  0.001, 10: 0.001,            # measured: <1 ms
    11: 0.005,                        # measured
    12: 0.027,                        # measured
    13: 0.093,                        # measured
    14: 0.325,                        # measured
    15: 1.283,                        # measured
    16: 5.646,                        # measured
    17: 25.407,                       # extrapolated: 5.646 × 4.5
    18: 114.332,                      # extrapolated: 25.407 × 4.5
    19: 514.492,                      # extrapolated: 114.332 × 4.5
    20: 2315.213,                     # extrapolated: 514.492 × 4.5
    21: 10418.458,                    # extrapolated: 2315.213 × 4.5
}
_BUCKET_WORST_GROWTH = 4.5           # growth factor for buckets beyond 21
_BUCKET_WORST_BASE   = 21            # last entry in the table above

# Safety margin applied to extrapolated estimates (buckets 17+) to account for
# the uncertainty in the growth factor (observed: ×3.5, ×4.0, ×4.4, trending up).
# A 20% buffer ensures workers stay well inside the 6-hour limit even when the
# true growth slightly exceeds the 4.5× assumption.
_EXTRAPOLATED_SAFETY_MARGIN = 1.2   # only applied to buckets > 16


def _worst_sec_for_bucket(bucket_n: int) -> float:
    """Return the estimated worst-case single-exponent test time in seconds.

    For buckets 1–16 this is the measured "Longest (s)" value from the
    benchmark report.  For buckets 17+ it is an extrapolation (×4.5 per
    bucket) with a 20% safety margin applied to reflect the uncertainty in
    the growth factor (observed trend: ×3.5, ×4.0, ×4.4, still increasing).
    """
    if bucket_n in _BUCKET_WORST_SEC:
        w = _BUCKET_WORST_SEC[bucket_n]
        # Apply safety margin to extrapolated entries (bucket > 16)
        if bucket_n > 16:
            w *= _EXTRAPOLATED_SAFETY_MARGIN
        return w
    # Extrapolate beyond the table and apply the margin
    w = _BUCKET_WORST_SEC[_BUCKET_WORST_BASE]
    for _ in range(bucket_n - _BUCKET_WORST_BASE):
        w *= _BUCKET_WORST_GROWTH
    return w * _EXTRAPOLATED_SAFETY_MARGIN


def max_batch_size_for_bucket(bucket_n: int, time_limit_seconds: float) -> int:
    """Return the max batch size so a worker completes within *time_limit_seconds*.

    Uses the worst-case (longest) single-exponent time derived from benchmark
    data, applying the inverse rule of three:
        max_primes = floor(time_limit / worst_case_time_per_prime)

    The result is always ≥ 1 (one test per worker is the minimum).
    """
    worst = _worst_sec_for_bucket(bucket_n)
    if worst <= 0.0:
        return BATCH_SIZE_DEFAULT
    return max(1, int(time_limit_seconds / worst))


def make_worker_name(
    bucket_n: int,
    start_ordinal: int,    # 1-based ordinal of first prime in this batch
    end_ordinal: int,      # 1-based ordinal of last prime in this batch
    batch_min_exponent: int,
    batch_max_exponent: int,
) -> str:
    """Return a deterministic, stable worker name for one batch.

    Format:
        bucket-{N:02d}-batch-{start_ordinal:04d}-{end_ordinal:04d}-exp-{pmin}-{pmax}

    Example:
        bucket-17-batch-0001-1000-exp-65537-65867
    """
    return (
        f"bucket-{bucket_n:02d}"
        f"-batch-{start_ordinal:04d}-{end_ordinal:04d}"
        f"-exp-{batch_min_exponent}-{batch_max_exponent}"
    )


def split_bucket_into_batches(
    bucket_n: int,
    batch_size: int = BATCH_SIZE_DEFAULT,
    resume_from: int = 0,
) -> list[dict]:
    """Return a list of batch descriptor dicts for one bucket.

    Each descriptor contains all fields needed by a GitHub Actions matrix
    entry and by the worker step.

    Args:
        bucket_n:    Bucket number (1-64).
        batch_size:  Maximum number of primes per batch.
        resume_from: Skip all primes ≤ this value (0 = no skip).  The
                     batch ordinals are still reported relative to the
                     full (un-skipped) prime list so worker names stay
                     stable across resume runs.
    """
    lo, hi = bucket_range(bucket_n)
    primes = enumerate_bucket_primes(bucket_n)

    if not primes:
        return []

    total = len(primes)

    # Find the first prime that hasn't been tested yet.
    skip = 0
    if resume_from > 0:
        while skip < total and primes[skip] <= resume_from:
            skip += 1

    # Nothing left in this bucket – skip it entirely.
    if skip == total:
        return []

    remaining = primes[skip:]
    chunks = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]
    batch_count = len(chunks)

    batches: list[dict] = []
    for idx, chunk in enumerate(chunks):
        # Absolute ordinals in the full (un-skipped) bucket prime list.
        abs_start = skip + idx * batch_size
        abs_end   = abs_start + len(chunk) - 1
        start_ordinal = abs_start + 1   # 1-based
        end_ordinal   = abs_end   + 1   # 1-based
        batch_min_exp = chunk[0]
        batch_max_exp = chunk[-1]

        batches.append(
            {
                "bucket_n": bucket_n,
                "bucket_min": lo,
                "bucket_max": hi,
                "batch_index": idx,
                "batch_count": batch_count,
                "batch_prime_start_index": abs_start,
                "batch_prime_end_index": abs_end,
                "batch_min_exponent": batch_min_exp,
                "batch_max_exponent": batch_max_exp,
                "batch_size": len(chunk),
                "worker_name": make_worker_name(
                    bucket_n,
                    start_ordinal,
                    end_ordinal,
                    batch_min_exp,
                    batch_max_exp,
                ),
            }
        )

    return batches


def generate_batch_matrix(
    bucket_start: int,
    bucket_end: int,
    batch_size: int = BATCH_SIZE_DEFAULT,
    time_limit_seconds: float = 0.0,
    resume_from: int = 0,
    target_workers: int = 0,
) -> list[dict]:
    """Return the full batch matrix across all selected buckets.

    Args:
        batch_size:         Global maximum primes per batch (used when
                            *time_limit_seconds* and *target_workers* are 0).
        time_limit_seconds: When > 0, compute a per-bucket batch size via
                            the inverse rule of three and use that as a safety
                            cap.  *batch_size* also acts as an upper cap.
        resume_from:        Skip primes ≤ this value in the first affected
                            bucket; skip completed buckets entirely.
        target_workers:     When > 0, compute batch sizes so that the total
                            number of batches across all buckets is as close
                            to *target_workers* as possible.  This is done by
                            counting total remaining primes and setting
                            batch_size = ceil(total / target_workers).
                            The time-limit cap still applies per bucket so
                            no single worker can exceed the time budget.
    """
    # ── Step 1: collect all remaining primes per bucket ──────────────────
    bucket_primes: list[tuple[int, list[int]]] = []  # (bucket_n, primes_remaining)
    for n in range(bucket_start, bucket_end + 1):
        primes = enumerate_bucket_primes(n)
        if not primes:
            continue
        skip = 0
        if resume_from > 0:
            while skip < len(primes) and primes[skip] <= resume_from:
                skip += 1
        remaining = primes[skip:]
        if remaining:
            bucket_primes.append((n, remaining))

    if not bucket_primes:
        return []

    # ── Step 2: compute the global "target" batch size ────────────────────
    if target_workers > 0:
        total_remaining = sum(len(r) for _, r in bucket_primes)
        global_target_bs = max(1, -(-total_remaining // target_workers))  # ceil division
    else:
        global_target_bs = batch_size  # use explicit --batch-size as-is

    # ── Step 3: per-bucket effective batch size ───────────────────────────
    matrix: list[dict] = []
    for n, _ in bucket_primes:
        # Start with the global target
        bs = global_target_bs

        # Always cap by --batch-size (hard upper limit)
        bs = min(bs, batch_size)

        # Cap by time-limit safety (per bucket) when a budget is given
        if time_limit_seconds > 0.0:
            tl_bs = max_batch_size_for_bucket(n, time_limit_seconds)
            bs = min(bs, tl_bs)

        matrix.extend(split_bucket_into_batches(n, bs, resume_from))

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket-start", type=int, default=1,
                        metavar="N", help="First bucket (1-64, default: 1)")
    parser.add_argument("--bucket-end", type=int, default=64,
                        metavar="N", help="Last bucket (1-64, default: 64)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
                        metavar="N", help=f"Max primes per batch / upper cap when "
                                          f"--time-limit-seconds is set (default: {BATCH_SIZE_DEFAULT})")
    parser.add_argument("--time-limit-seconds", type=float, default=0.0,
                        metavar="S",
                        help="Per-worker time budget in seconds.  When set, the "
                             "batch size is calculated per-bucket using the inverse "
                             "rule of three from benchmark timing data, then capped "
                             "by --batch-size.  Overrides a plain --batch-size for "
                             "large-exponent buckets.")
    parser.add_argument("--target-workers", type=int, default=0,
                        metavar="N",
                        help="Target total number of worker batches across all buckets.  "
                             "The batch size is computed as ceil(total_primes / N) so the "
                             "sweep uses ~N parallel workers.  The per-bucket time-limit "
                             "cap still applies so no worker exceeds the time budget.  "
                             "--batch-size acts as a hard upper cap.  (default: 0 = off)")
    parser.add_argument("--resume-from-exponent", type=int, default=0,
                        metavar="N",
                        help="Skip all prime exponents ≤ N.  Batches whose highest "
                             "exponent is ≤ N are omitted entirely; the first "
                             "remaining batch starts from the next untested prime.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print human-readable plan instead of JSON")
    parser.add_argument("--output", type=str, default=None,
                        metavar="FILE", help="Write JSON to FILE instead of stdout")
    args = parser.parse_args()

    if args.bucket_start < 1 or args.bucket_start > 64:
        parser.error(f"--bucket-start must be in [1, 64], got {args.bucket_start}")
    if args.bucket_end < 1 or args.bucket_end > 64:
        parser.error(f"--bucket-end must be in [1, 64], got {args.bucket_end}")
    if args.bucket_start > args.bucket_end:
        parser.error(f"--bucket-start ({args.bucket_start}) must be <= --bucket-end ({args.bucket_end})")
    if args.batch_size < 1:
        parser.error(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.time_limit_seconds < 0:
        parser.error(f"--time-limit-seconds must be >= 0, got {args.time_limit_seconds}")
    if not math.isfinite(args.time_limit_seconds):
        parser.error(f"--time-limit-seconds must be a finite number, got {args.time_limit_seconds}")
    if args.resume_from_exponent < 0:
        parser.error(f"--resume-from-exponent must be >= 0, got {args.resume_from_exponent}")
    if args.target_workers < 0:
        parser.error(f"--target-workers must be >= 0, got {args.target_workers}")

    matrix = generate_batch_matrix(
        args.bucket_start,
        args.bucket_end,
        batch_size=args.batch_size,
        time_limit_seconds=args.time_limit_seconds,
        resume_from=args.resume_from_exponent,
        target_workers=args.target_workers,
    )

    if args.dry_run:
        print(f"bucket_start         : {args.bucket_start}")
        print(f"bucket_end           : {args.bucket_end}")
        if args.target_workers > 0:
            print(f"target_workers       : {args.target_workers}")
        if args.time_limit_seconds > 0:
            print(f"time_limit_seconds   : {args.time_limit_seconds:.0f}  "
                  f"({args.time_limit_seconds/3600:.1f} h per worker)")
            print(f"batch_size cap       : {args.batch_size}")
        else:
            print(f"batch_size           : {args.batch_size}")
        if args.resume_from_exponent > 0:
            print(f"resume_from_exponent : {args.resume_from_exponent}  "
                  f"(skipping exponents ≤ {args.resume_from_exponent})")
        print(f"total batches        : {len(matrix)}")
        print()
        for entry in matrix:
            # Per-bucket batch size annotation when time_limit is active
            if args.time_limit_seconds > 0:
                w = _worst_sec_for_bucket(entry["bucket_n"])
                size_note = (f"  size={entry['batch_size']}"
                             f"  est_worst={w:.3f}s/exp"
                             f"  est_max={entry['batch_size']*w/3600:.2f}h")
            else:
                size_note = f"  size={entry['batch_size']}"
            print(
                f"  [{entry['batch_index']:3d}/{entry['batch_count']:3d}]"
                f"  bucket={entry['bucket_n']:2d}"
                f"  primes={entry['batch_prime_start_index']:6d}-{entry['batch_prime_end_index']:6d}"
                f"  exp={entry['batch_min_exponent']}-{entry['batch_max_exponent']}"
                f"{size_note}"
                f"  name={entry['worker_name']}"
            )
        if len(matrix) > GITHUB_MATRIX_MAX:
            print(
                f"\nWARNING: {len(matrix)} batches exceeds the GitHub Actions matrix limit"
                f" of {GITHUB_MATRIX_MAX}.  Narrow the bucket range or increase --batch-size.",
                file=sys.stderr,
            )
        return

    if len(matrix) > GITHUB_MATRIX_MAX:
        print(
            f"ERROR: {len(matrix)} batches exceeds the GitHub Actions matrix limit"
            f" of {GITHUB_MATRIX_MAX}. Narrow the bucket range or increase --batch-size.",
            file=sys.stderr,
        )
        sys.exit(1)

    json_text = json.dumps(matrix)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(json_text + "\n")
        print(f"Wrote {len(matrix)} batch entries to {args.output}", file=sys.stderr)
    else:
        print(json_text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/plan_bucket_sweep.py – Unified bucket sweep planner.

Merges the functionality of generate_bucket_primes.py, split_bucket_batches.py,
and select_prime_half.py into a single script that supports auto-chunked planning
so no single GitHub Actions matrix run exceeds the 256-entry hard limit.

Usage:
    python3 scripts/plan_bucket_sweep.py [options]

Modes:
    count_chunks   Print a JSON object with total_batches, chunk_size, chunk_count.
    emit_chunk     Print a JSON array of batch descriptors for one chunk (default).

Options:
    --bucket-start N           First bucket (1-64, default: 1)
    --bucket-end   N           Last  bucket (1-64, default: 64)
    --time-limit-seconds S     Per-worker time budget in seconds (default: 0 = off)
    --target-workers N         Target number of parallel workers (default: 0 = off)
    --resume-from-exponent N   Skip exponents <= N (default: 0 = no skip)
    --prime-half {full,lower_half,upper_half}
                               Which portion of primes to plan (default: full)
    --chunk-size N             Max batches per GitHub Actions matrix run (default: 256)
    --chunk-index N            0-based index of the chunk to emit (default: 0)
    --dry-run                  Print human-readable plan instead of JSON
    --output FILE              Write JSON to FILE instead of stdout

Outputs:
    count_chunks mode:
        {"total_batches": N, "chunk_size": M, "chunk_count": K}

    emit_chunk mode:
        JSON array of up to chunk_size batch descriptor objects, each with:
            bucket_n, bucket_min, bucket_max
            batch_index, batch_count
            batch_prime_start_index, batch_prime_end_index
            batch_min_exponent, batch_max_exponent
            batch_size, worker_name

Worker names are deterministic and stable: they reflect the batch's position
in the *full* plan, not the chunk-local index, so names are identical across
runs regardless of chunk_index.

batch_index/batch_count refer to the full per-bucket batch numbering so they
remain stable across partial and resumed runs.
"""

import argparse
import json
import math
import os
import sys

# Import helpers from sibling scripts in the same scripts/ directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from generate_bucket_primes import bucket_range, enumerate_bucket_primes  # noqa: E402
from split_bucket_batches import (  # noqa: E402
    BATCH_SIZE_DEFAULT,
    GITHUB_MATRIX_MAX,
    _worst_sec_for_bucket,
    generate_batch_matrix,
    max_batch_size_for_bucket,
)
from select_prime_half import select_half  # noqa: E402

CHUNK_SIZE_DEFAULT = GITHUB_MATRIX_MAX  # 256


# ── Public helpers ─────────────────────────────────────────────────────────


def build_full_matrix(
    bucket_start: int,
    bucket_end: int,
    batch_size: int = BATCH_SIZE_DEFAULT,
    time_limit_seconds: float = 0.0,
    resume_from: int = 0,
    target_workers: int = 0,
    prime_half: str = "full",
) -> list[dict]:
    """Build the full batch matrix (all buckets, all primes, with optional half filter).

    This is the same as ``generate_batch_matrix`` from split_bucket_batches but
    with the prime-half filter integrated so callers don't need a separate step.

    Args:
        bucket_start:       First bucket number (1-64).
        bucket_end:         Last bucket number (1-64, >= bucket_start).
        batch_size:         Hard upper cap on primes per batch (default 1000).
        time_limit_seconds: When > 0, per-bucket safety cap via inverse rule of three.
        resume_from:        Skip primes <= this value (0 = start from beginning).
        target_workers:     When > 0, set global batch size = ceil(total/target_workers).
        prime_half:         'full', 'lower_half', or 'upper_half'.

    Returns:
        List of batch descriptor dicts suitable for a GitHub Actions matrix.
    """
    matrix = generate_batch_matrix(
        bucket_start,
        bucket_end,
        batch_size=batch_size,
        time_limit_seconds=time_limit_seconds,
        resume_from=resume_from,
        target_workers=target_workers,
    )

    if prime_half in ("lower_half", "upper_half"):
        matrix = select_half(matrix, prime_half)

    return matrix


def count_chunks(
    total_batches: int,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
) -> dict:
    """Return chunk planning metadata.

    Args:
        total_batches: Total number of batches in the full matrix.
        chunk_size:    Maximum number of batches per chunk (default: 256).

    Returns:
        Dict with keys: total_batches, chunk_size, chunk_count.
    """
    chunk_count = max(1, math.ceil(total_batches / chunk_size)) if total_batches > 0 else 1
    return {
        "total_batches": total_batches,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
    }


def emit_chunk(
    matrix: list[dict],
    chunk_index: int,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
) -> list[dict]:
    """Return the slice of the matrix for the given chunk.

    Args:
        matrix:      Full batch matrix (all batches, post half-filter).
        chunk_index: 0-based chunk index.
        chunk_size:  Maximum entries per chunk (default: 256).

    Returns:
        List of batch descriptors for the requested chunk.

    Raises:
        ValueError: If chunk_index is out of range.
    """
    total = len(matrix)
    if total == 0:
        if chunk_index != 0:
            raise ValueError(f"chunk_index {chunk_index} is out of range for an empty matrix")
        return []

    num_chunks = math.ceil(total / chunk_size)
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(
            f"chunk_index {chunk_index} is out of range [0, {num_chunks - 1}] "
            f"for total_batches={total} and chunk_size={chunk_size}"
        )

    start = chunk_index * chunk_size
    end = min(start + chunk_size, total)
    return matrix[start:end]


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bucket-start", type=int, default=1,
                   metavar="N", help="First bucket (1-64, default: 1)")
    p.add_argument("--bucket-end", type=int, default=64,
                   metavar="N", help="Last bucket (1-64, default: 64)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
                   metavar="N",
                   help=f"Max primes per batch / upper cap when "
                        f"--time-limit-seconds is set (default: {BATCH_SIZE_DEFAULT})")
    p.add_argument("--time-limit-seconds", type=float, default=0.0,
                   metavar="S",
                   help="Per-worker time budget in seconds.  When set, the batch size "
                        "is calculated per-bucket using the inverse rule of three.")
    p.add_argument("--target-workers", type=int, default=0,
                   metavar="N",
                   help="Target total number of worker batches.  Batch size is "
                        "ceil(total_primes / N); per-bucket time-limit cap still applies.")
    p.add_argument("--resume-from-exponent", type=int, default=0,
                   metavar="N",
                   help="Skip all prime exponents <= N (0 = no skip).")
    p.add_argument("--prime-half", type=str, default="full",
                   choices=("full", "lower_half", "upper_half"),
                   metavar="{full,lower_half,upper_half}",
                   help="Which half of primes to plan (default: full).")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT,
                   metavar="N",
                   help=f"Maximum batches per GitHub Actions matrix run "
                        f"(default: {CHUNK_SIZE_DEFAULT}).")
    p.add_argument("--chunk-index", type=int, default=0,
                   metavar="N",
                   help="0-based index of the chunk to emit in emit_chunk mode (default: 0).")
    p.add_argument("--mode", type=str, default="emit_chunk",
                   choices=("count_chunks", "emit_chunk"),
                   help="Output mode: 'count_chunks' prints planning metadata, "
                        "'emit_chunk' prints the batch matrix for one chunk (default).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print human-readable plan instead of JSON (emit_chunk mode only).")
    p.add_argument("--output", type=str, default=None,
                   metavar="FILE", help="Write JSON to FILE instead of stdout.")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if args.bucket_start < 1 or args.bucket_start > 64:
        parser.error(f"--bucket-start must be in [1, 64], got {args.bucket_start}")
    if args.bucket_end < 1 or args.bucket_end > 64:
        parser.error(f"--bucket-end must be in [1, 64], got {args.bucket_end}")
    if args.bucket_start > args.bucket_end:
        parser.error(
            f"--bucket-start ({args.bucket_start}) must be <= --bucket-end ({args.bucket_end})"
        )
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
    if args.chunk_size < 1:
        parser.error(f"--chunk-size must be >= 1, got {args.chunk_size}")
    if args.chunk_index < 0:
        parser.error(f"--chunk-index must be >= 0, got {args.chunk_index}")

    # ── Build the full matrix ─────────────────────────────────────────────
    matrix = build_full_matrix(
        args.bucket_start,
        args.bucket_end,
        batch_size=args.batch_size,
        time_limit_seconds=args.time_limit_seconds,
        resume_from=args.resume_from_exponent,
        target_workers=args.target_workers,
        prime_half=args.prime_half,
    )

    # ── count_chunks mode ─────────────────────────────────────────────────
    if args.mode == "count_chunks":
        result = count_chunks(len(matrix), chunk_size=args.chunk_size)
        json_text = json.dumps(result)
        if args.output:
            with open(args.output, "w") as fh:
                fh.write(json_text + "\n")
            print(
                f"Wrote chunk metadata to {args.output}: "
                f"total_batches={result['total_batches']}, "
                f"chunk_count={result['chunk_count']}",
                file=sys.stderr,
            )
        else:
            print(json_text)
        return

    # ── emit_chunk mode ───────────────────────────────────────────────────
    # Determine the chunk
    total = len(matrix)
    num_chunks = max(1, math.ceil(total / args.chunk_size)) if total > 0 else 1

    if args.chunk_index >= num_chunks:
        parser.error(
            f"--chunk-index {args.chunk_index} is out of range [0, {num_chunks - 1}] "
            f"(total_batches={total}, chunk_size={args.chunk_size})"
        )

    chunk = emit_chunk(matrix, args.chunk_index, chunk_size=args.chunk_size)

    # ── dry-run: human-readable output ────────────────────────────────────
    if args.dry_run:
        print(f"bucket_start         : {args.bucket_start}")
        print(f"bucket_end           : {args.bucket_end}")
        if args.target_workers > 0:
            print(f"target_workers       : {args.target_workers}")
        if args.time_limit_seconds > 0:
            print(f"time_limit_seconds   : {args.time_limit_seconds:.0f}  "
                  f"({args.time_limit_seconds / 3600:.1f} h per worker)")
            print(f"batch_size cap       : {args.batch_size}")
        else:
            print(f"batch_size           : {args.batch_size}")
        if args.resume_from_exponent > 0:
            print(f"resume_from_exponent : {args.resume_from_exponent}  "
                  f"(skipping exponents <= {args.resume_from_exponent})")
        if args.prime_half != "full":
            print(f"prime_half           : {args.prime_half}")
        print(f"total batches        : {total}")
        print(f"chunk_size           : {args.chunk_size}")
        print(f"chunk_count          : {num_chunks}")
        print(f"chunk_index          : {args.chunk_index}  "
              f"(batches {args.chunk_index * args.chunk_size + 1}–"
              f"{min((args.chunk_index + 1) * args.chunk_size, total)} of {total})")
        print()
        for entry in chunk:
            if args.time_limit_seconds > 0:
                w = _worst_sec_for_bucket(entry["bucket_n"])
                size_note = (
                    f"  size={entry['batch_size']}"
                    f"  est_worst={w:.3f}s/exp"
                    f"  est_max={entry['batch_size'] * w / 3600:.2f}h"
                )
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
        if total > CHUNK_SIZE_DEFAULT and not args.output:
            print(
                f"\nNOTE: {total} total batches split into {num_chunks} chunks of "
                f"<= {args.chunk_size}.  Use --chunk-index N to emit a specific chunk.",
                file=sys.stderr,
            )
        return

    # ── JSON output ───────────────────────────────────────────────────────
    json_text = json.dumps(chunk)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(json_text + "\n")
        print(
            f"Wrote {len(chunk)} batch entries (chunk {args.chunk_index}/{num_chunks - 1}) "
            f"to {args.output}",
            file=sys.stderr,
        )
    else:
        print(json_text)


if __name__ == "__main__":
    main()

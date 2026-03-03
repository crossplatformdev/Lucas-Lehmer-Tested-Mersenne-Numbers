#!/usr/bin/env python3
"""
scripts/generate_exponent_set.py
---------------------------------
Generate the discover-mode exponent list and print it to stdout.

Usage:
    python3 scripts/generate_exponent_set.py [options]

Options:
    --single-exponent N   Explicit first exponent (default: 0 = none)
    --min-excl N          Lower bound exclusive (default: 136279841)
    --max-incl N          Upper bound inclusive  (default: 200000000)
    --shard-count N       Total shards           (default: 1)
    --shard-index N       This shard (0-based)   (default: 0)
    --reverse             Reverse range order
    --dry-run             Print plan summary and first few entries, then exit
    --count-only          Print only the total count of exponents
    --output FILE         Write list to FILE (one per line) instead of stdout
"""

import argparse
import sys
from math import isqrt


KNOWN_MERSENNE_PRIMES = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
    2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497,
    86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269,
    2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951,
    30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281,
    77232917, 82589933, 136279841,
]
KNOWN_MERSENNE_SET = set(KNOWN_MERSENNE_PRIMES)


def is_prime(n: int) -> bool:
    """Trial-division primality test (sufficient for exponents up to ~2e8)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return n == 3
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def generate_post_known_exponents(min_excl: int, max_incl: int) -> list:
    """Return all prime p with min_excl < p <= max_incl, ascending."""
    if max_incl <= min_excl:
        return []
    result = []
    start = min_excl + 1
    if start > 2 and start % 2 == 0:
        start += 1
    p = start
    while p <= max_incl:
        if is_prime(p):
            result.append(p)
        p += 1 if p == 2 else 2
    return result


def discover_exponent_list(
    single_exp: int,
    min_excl: int,
    max_incl: int,
    reverse: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list:
    """Build the full discover-mode exponent list (mirrors C++ logic)."""
    shard_count = max(1, shard_count)
    shard_index = shard_index % shard_count

    # Generate range, remove single_exp duplicate.
    rng = generate_post_known_exponents(min_excl, max_incl)
    if single_exp:
        rng = [p for p in rng if p != single_exp]

    # Shard the range.
    if shard_count > 1:
        rng = [p for i, p in enumerate(rng) if i % shard_count == shard_index]

    # Optionally reverse.
    if reverse:
        rng = list(reversed(rng))

    # Prepend single_exp if prime.
    result = []
    if single_exp and is_prime(single_exp):
        result.append(single_exp)
    result.extend(rng)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate the discover-mode Mersenne exponent list."
    )
    parser.add_argument("--single-exponent", type=int, default=0,
                        metavar="N", help="Explicit first exponent (0=none)")
    parser.add_argument("--min-excl", type=int, default=136279841,
                        metavar="N", help="Lower bound exclusive")
    parser.add_argument("--max-incl", type=int, default=200000000,
                        metavar="N", help="Upper bound inclusive")
    parser.add_argument("--shard-count", type=int, default=1, metavar="N")
    parser.add_argument("--shard-index", type=int, default=0, metavar="N")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse range order (largest first)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary and first 20 entries, then exit")
    parser.add_argument("--count-only", action="store_true",
                        help="Print only the total count")
    parser.add_argument("--output", type=str, default=None,
                        metavar="FILE", help="Write exponents to FILE")
    args = parser.parse_args()

    exps = discover_exponent_list(
        single_exp=args.single_exponent,
        min_excl=args.min_excl,
        max_incl=args.max_incl,
        reverse=args.reverse,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )

    if args.count_only:
        print(len(exps))
        return

    if args.dry_run:
        print(f"single_exponent : {args.single_exponent}")
        print(f"min_excl        : {args.min_excl}")
        print(f"max_incl        : {args.max_incl}")
        print(f"shard           : {args.shard_index}/{args.shard_count}")
        print(f"reverse_order   : {args.reverse}")
        print(f"total_exponents : {len(exps)}")
        print()
        preview = exps[:20]
        for i, p in enumerate(preview):
            is_first = (args.single_exponent != 0 and i == 0 and p == args.single_exponent)
            known = p in KNOWN_MERSENNE_SET
            tag = " (explicit first)" if is_first else ""
            tag += " [known]" if known else ""
            print(f"  [{i:4d}] p={p:<12d}{tag}")
        if len(exps) > 20:
            print(f"  ... ({len(exps) - 20} more)")
        return

    if args.output:
        with open(args.output, "w") as fh:
            for p in exps:
                fh.write(f"{p}\n")
        print(f"Wrote {len(exps)} exponents to {args.output}", file=sys.stderr)
    else:
        for p in exps:
            print(p)


if __name__ == "__main__":
    main()

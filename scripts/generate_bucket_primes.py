#!/usr/bin/env python3
"""
generate_bucket_primes.py – Print the prime exponents in a power-of-two bucket.

Usage:
    python3 scripts/generate_bucket_primes.py <n>         # single bucket
    python3 scripts/generate_bucket_primes.py <lo> <hi>   # bucket range

Bucket definition (1-indexed, n in [1, 64]):
    B_1  = [2, 2]               (normalized: mathematical [2, 2^1-1] is empty)
    B_n  = [2^(n-1), 2^n - 1]  for n >= 2
    B_64 = [2^63, 2^64 - 1]
"""

import sys
import math


def bucket_range(n: int) -> tuple[int, int]:
    """Return (lo, hi) for bucket n (1-indexed, 1 <= n <= 64)."""
    if n < 1 or n > 64:
        raise ValueError(f"Bucket n must be in [1, 64], got {n}")
    if n == 1:
        return (2, 2)
    lo = 1 << (n - 1)
    hi = (1 << n) - 1 if n < 64 else (2**64 - 1)
    return (lo, hi)


def is_prime(n: int) -> bool:
    """Trial-division primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n == 3:
        return True
    if n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def enumerate_bucket_primes(n: int) -> list[int]:
    """Return all prime exponents in bucket n, ascending."""
    lo, hi = bucket_range(n)
    result = []
    p = lo if lo >= 2 else 2
    if p % 2 == 0 and p > 2:
        p += 1
    while p <= hi:
        if is_prime(p):
            result.append(p)
        p = p + 1 if p == 2 else p + 2
    return result


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 1:
        n_lo = n_hi = int(args[0])
    elif len(args) == 2:
        n_lo, n_hi = int(args[0]), int(args[1])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    for n in range(n_lo, n_hi + 1):
        lo, hi = bucket_range(n)
        primes = enumerate_bucket_primes(n)
        print(f"# Bucket {n}: [{lo}, {hi}] — {len(primes)} prime(s)")
        for p in primes:
            print(p)


if __name__ == "__main__":
    main()

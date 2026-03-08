#!/usr/bin/env python3
"""
ll_chain.py – Chain-state management and next-exponent scheduling for the
              find-next-largest-mersenne-prime self-chaining workflow.

Commands
--------
next-exponent   Print the next prime exponent to test.
is-prime        Check if a number is prime (exit 0 = prime, 1 = composite).

Usage
-----
  # Compute the next exponent to test after the last known Mersenne prime:
  python3 scripts/ll_chain.py next-exponent

  # Compute the first prime exponent >= 136300000:
  python3 scripts/ll_chain.py next-exponent --start 136300000

  # Compute the first prime exponent >= start, skipping exponents already
  # recorded as completed in a chain-state file:
  python3 scripts/ll_chain.py next-exponent --start 136300000 \\
      --state-file chain_state.json

  # Check if 136279843 is prime:
  python3 scripts/ll_chain.py is-prime 136279843

Exit codes
----------
  next-exponent : 0 on success, 1 on error
  is-prime      : 0 = prime, 1 = composite, 2 = error
"""

import argparse
import json
import math
import sys
from typing import Optional

# ── Known Mersenne prime exponents (must match BigNum.cpp) ──────────────────
KNOWN_MERSENNE_PRIME_EXPONENTS: list[int] = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
    2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497,
    86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269,
    2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951,
    30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281,
    77232917, 82589933, 136279841,
]

LAST_KNOWN_EXPONENT = KNOWN_MERSENNE_PRIME_EXPONENTS[-1]


# ── Primality testing (deterministic Miller-Rabin for n < 3.3 × 10²⁴) ──────
def _miller_rabin(n: int, witnesses: list[int]) -> bool:
    """Return True if n passes all Miller-Rabin rounds with the given witnesses."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Write n-1 as 2^r * d.
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def is_prime(n: int) -> bool:
    """Deterministic primality test for any positive integer n.

    Uses deterministic Miller-Rabin witness sets that are proven correct up to
    3 317 044 064 679 887 385 961 981 (≈ 3.3 × 10²⁴).  For n beyond that
    range the test falls back to a probabilistic 20-round Miller-Rabin which
    has a per-witness error probability < 4⁻²⁰ ≈ 10⁻¹².
    """
    if n < 2:
        return False
    small_primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    ]
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False

    # Deterministic witness sets (Pomerance, Selfridge, Wagstaff; Bach, et al.).
    if n < 2_047:
        witnesses = [2]
    elif n < 1_373_653:
        witnesses = [2, 3]
    elif n < 9_080_191:
        witnesses = [31, 73]
    elif n < 25_326_001:
        witnesses = [2, 3, 5]
    elif n < 3_215_031_751:
        witnesses = [2, 3, 5, 7]
    elif n < 4_759_123_141:
        witnesses = [2, 7, 61]
    elif n < 1_122_004_669_633:
        witnesses = [2, 13, 23, 1662803]
    elif n < 2_152_302_898_747:
        witnesses = [2, 3, 5, 7, 11]
    elif n < 3_474_749_660_383:
        witnesses = [2, 3, 5, 7, 11, 13]
    elif n < 341_550_071_728_321:
        witnesses = [2, 3, 5, 7, 11, 13, 17]
    elif n < 3_825_123_056_546_413_051:
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    elif n < 318_665_857_834_031_151_167_461:
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    elif n < 3_317_044_064_679_887_385_961_981:
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    else:
        # Probabilistic fallback for astronomically large n (not expected in practice).
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                     53, 59, 61, 67, 71]

    return _miller_rabin(n, witnesses)


# ── Next-exponent computation ────────────────────────────────────────────────

def next_prime_after(n: int) -> int:
    """Return the smallest prime strictly greater than n."""
    candidate = n + 1 if n < 2 else (n + 1 if n % 2 == 0 else n + 2)
    if candidate < 2:
        candidate = 2
    while not is_prime(candidate):
        candidate += 1 if candidate == 2 else 2
    return candidate


def first_prime_ge(n: int) -> int:
    """Return the smallest prime >= n."""
    if n <= 2:
        return 2
    candidate = n if n % 2 != 0 else n + 1
    while not is_prime(candidate):
        candidate += 2
    return candidate


def load_completed_exponents(state_file: str) -> set[int]:
    """Return the set of exponents already recorded as completed in a state file."""
    # Always start with the full known-Mersenne list so we never re-test them.
    completed: set[int] = set(KNOWN_MERSENNE_PRIME_EXPONENTS)
    try:
        with open(state_file) as f:
            state = json.load(f)
        # completed_exponents list (optional field in chain state)
        for exp in state.get("completed_exponents", []):
            completed.add(int(exp))
        # If the current exponent is fully done, include it too.
        status = state.get("status", "")
        if status in ("completed_composite", "completed_prime"):
            p = state.get("current_exponent")
            if p:
                completed.add(int(p))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return completed


def cmd_next_exponent(args: argparse.Namespace) -> int:
    """Print the next prime exponent to test, then exit 0."""
    # Determine the floor: the smallest p to consider.
    if args.start and int(args.start) > 0:
        floor = int(args.start)
    else:
        floor = LAST_KNOWN_EXPONENT + 1  # first candidate after last known

    # Load already-completed exponents from state file if provided.
    # Always includes the known-Mersenne list so we never re-test them.
    completed: set[int] = set(KNOWN_MERSENNE_PRIME_EXPONENTS)
    if args.state_file:
        completed |= load_completed_exponents(args.state_file)

    # Find the first prime >= floor that is not already completed.
    candidate = first_prime_ge(floor)
    while candidate in completed:
        candidate = next_prime_after(candidate)

    print(candidate)
    return 0


def cmd_is_prime(args: argparse.Namespace) -> int:
    """Exit 0 if the argument is prime, 1 if composite, 2 on error."""
    try:
        n = int(args.number)
    except (ValueError, TypeError):
        print(f"error: '{args.number}' is not a valid integer", file=sys.stderr)
        return 2
    return 0 if is_prime(n) else 1


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # next-exponent
    p_next = sub.add_parser(
        "next-exponent",
        help="Print the next prime exponent to test.",
    )
    p_next.add_argument(
        "--start",
        metavar="N",
        default="0",
        help="Start search from this exponent (0 = auto-detect after known list).",
    )
    p_next.add_argument(
        "--state-file",
        metavar="FILE",
        default="",
        help="Path to chain-state JSON to skip already-completed exponents.",
    )

    # is-prime
    p_isp = sub.add_parser("is-prime", help="Test if a number is prime.")
    p_isp.add_argument("number", help="The number to test.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "next-exponent":
        sys.exit(cmd_next_exponent(args))
    elif args.command == "is-prime":
        sys.exit(cmd_is_prime(args))
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

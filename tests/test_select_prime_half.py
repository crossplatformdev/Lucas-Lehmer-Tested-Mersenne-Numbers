#!/usr/bin/env python3
"""
tests/test_select_prime_half.py – Unit tests for scripts/select_prime_half.py.

Tests:
  1. Lower half contains the first mid primes
  2. Upper half contains the last (total - mid) primes
  3. Lower + Upper together cover all primes with no duplicates
  4. Empty matrix returns empty list for both halves
  5. Single-prime total: lower is empty, upper contains the prime
  6. Two-prime total: lower=1 prime, upper=1 prime
  7. Partial batch trimming for lower half (batch_size reduced correctly)
  8. Partial batch trimming for upper half (batch_min_exponent advanced correctly)
  9. Multi-bucket matrix splits correctly
 10. Invalid half argument raises ValueError
"""

import json
import os
import subprocess
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from select_prime_half import select_half  # noqa: E402
from split_bucket_batches import generate_batch_matrix  # noqa: E402
from generate_bucket_primes import enumerate_bucket_primes  # noqa: E402


def _all_exponents(matrix: list) -> list:
    """Flatten a batch matrix to all individual prime exponents (in order)."""
    exps = []
    for b in matrix:
        bucket_primes = enumerate_bucket_primes(b["bucket_n"])
        start = b["batch_prime_start_index"]
        count = b["batch_size"]
        exps.extend(bucket_primes[start : start + count])
    return exps


class TestSelectHalfEmpty(unittest.TestCase):
    def test_empty_matrix_lower(self):
        self.assertEqual(select_half([], "lower_half"), [])

    def test_empty_matrix_upper(self):
        self.assertEqual(select_half([], "upper_half"), [])


class TestSelectHalfInvalidArg(unittest.TestCase):
    def test_invalid_half_raises(self):
        with self.assertRaises(ValueError):
            select_half([], "full")

    def test_invalid_half_middle_raises(self):
        with self.assertRaises(ValueError):
            select_half([], "middle")


class TestSelectHalfSingleBucket(unittest.TestCase):
    """Tests using a single-bucket matrix (bucket 10: 75 primes)."""

    BUCKET = 10

    def _matrix(self, batch_size: int = 1000) -> list:
        return generate_batch_matrix(self.BUCKET, self.BUCKET, batch_size=batch_size)

    def _all_primes(self) -> list:
        return enumerate_bucket_primes(self.BUCKET)

    def test_lower_count(self):
        matrix = self._matrix()
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        lower = select_half(matrix, "lower_half")
        got = _all_exponents(lower)
        self.assertEqual(len(got), mid)

    def test_upper_count(self):
        matrix = self._matrix()
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        upper = select_half(matrix, "upper_half")
        got = _all_exponents(upper)
        self.assertEqual(len(got), len(all_primes) - mid)

    def test_lower_plus_upper_covers_all(self):
        matrix = self._matrix()
        all_primes = self._all_primes()
        lower = _all_exponents(select_half(matrix, "lower_half"))
        upper = _all_exponents(select_half(matrix, "upper_half"))
        combined = lower + upper
        self.assertEqual(combined, all_primes)

    def test_lower_is_first_half(self):
        matrix = self._matrix()
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        lower = _all_exponents(select_half(matrix, "lower_half"))
        self.assertEqual(lower, all_primes[:mid])

    def test_upper_is_second_half(self):
        matrix = self._matrix()
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        upper = _all_exponents(select_half(matrix, "upper_half"))
        self.assertEqual(upper, all_primes[mid:])

    def test_lower_no_duplicates(self):
        matrix = self._matrix()
        lower = _all_exponents(select_half(matrix, "lower_half"))
        self.assertEqual(len(lower), len(set(lower)))

    def test_upper_no_duplicates(self):
        matrix = self._matrix()
        upper = _all_exponents(select_half(matrix, "upper_half"))
        self.assertEqual(len(upper), len(set(upper)))

    def test_partial_batch_lower_batch_size_correct(self):
        """With small batch_size the last lower-half batch may be partial."""
        # Use batch_size=10 to force multiple batches across the split boundary
        matrix = self._matrix(batch_size=10)
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        lower = select_half(matrix, "lower_half")
        got = _all_exponents(lower)
        self.assertEqual(got, all_primes[:mid])

    def test_partial_batch_upper_min_exp_correct(self):
        """With small batch_size the first upper-half batch may be partial."""
        matrix = self._matrix(batch_size=10)
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        upper = select_half(matrix, "upper_half")
        got = _all_exponents(upper)
        self.assertEqual(got, all_primes[mid:])

    def test_split_boundary_exact_multiple(self):
        """When mid falls exactly on a batch boundary no partial batch needed."""
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        # Choose batch_size that divides mid exactly
        if mid > 0:
            matrix = self._matrix(batch_size=mid)
            lower = _all_exponents(select_half(matrix, "lower_half"))
            upper = _all_exponents(select_half(matrix, "upper_half"))
            self.assertEqual(lower, all_primes[:mid])
            self.assertEqual(upper, all_primes[mid:])


class TestSelectHalfTinyBuckets(unittest.TestCase):
    """Edge cases with very few primes."""

    def test_single_prime_lower_is_empty(self):
        # Bucket 1 has exactly one prime: 2
        matrix = generate_batch_matrix(1, 1, batch_size=1000)
        lower = select_half(matrix, "lower_half")
        self.assertEqual(lower, [])

    def test_single_prime_upper_has_prime(self):
        matrix = generate_batch_matrix(1, 1, batch_size=1000)
        upper = _all_exponents(select_half(matrix, "upper_half"))
        self.assertEqual(upper, [2])

    def test_two_primes_even_split(self):
        # Bucket 2 has primes: 2, 3
        matrix = generate_batch_matrix(2, 2, batch_size=1000)
        lower = _all_exponents(select_half(matrix, "lower_half"))
        upper = _all_exponents(select_half(matrix, "upper_half"))
        self.assertEqual(lower, [2])
        self.assertEqual(upper, [3])


class TestSelectHalfMultiBucket(unittest.TestCase):
    """Tests spanning multiple buckets (buckets 9–11)."""

    def _matrix(self, batch_size: int = 100) -> list:
        return generate_batch_matrix(9, 11, batch_size=batch_size)

    def _all_primes(self) -> list:
        primes = []
        for n in range(9, 12):
            primes.extend(enumerate_bucket_primes(n))
        return primes

    def test_lower_plus_upper_covers_all(self):
        matrix = self._matrix()
        all_primes = self._all_primes()
        lower = _all_exponents(select_half(matrix, "lower_half"))
        upper = _all_exponents(select_half(matrix, "upper_half"))
        self.assertEqual(lower + upper, all_primes)

    def test_lower_count(self):
        matrix = self._matrix()
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        lower = _all_exponents(select_half(matrix, "lower_half"))
        self.assertEqual(len(lower), mid)

    def test_upper_count(self):
        matrix = self._matrix()
        all_primes = self._all_primes()
        mid = len(all_primes) // 2
        upper = _all_exponents(select_half(matrix, "upper_half"))
        self.assertEqual(len(upper), len(all_primes) - mid)


class TestSelectHalfCLI(unittest.TestCase):
    """Test the command-line interface via subprocess."""

    def _run(self, half: str, matrix: list) -> list:
        result = subprocess.run(
            [sys.executable, os.path.join(_REPO_ROOT, "scripts", "select_prime_half.py"), half],
            input=json.dumps(matrix),
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_cli_lower(self):
        matrix = generate_batch_matrix(10, 10, batch_size=1000)
        all_primes = enumerate_bucket_primes(10)
        mid = len(all_primes) // 2
        lower = _all_exponents(self._run("lower_half", matrix))
        self.assertEqual(lower, all_primes[:mid])

    def test_cli_upper(self):
        matrix = generate_batch_matrix(10, 10, batch_size=1000)
        all_primes = enumerate_bucket_primes(10)
        mid = len(all_primes) // 2
        upper = _all_exponents(self._run("upper_half", matrix))
        self.assertEqual(upper, all_primes[mid:])

    def test_cli_empty_matrix(self):
        lower = self._run("lower_half", [])
        self.assertEqual(lower, [])

    def test_cli_invalid_arg_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, os.path.join(_REPO_ROOT, "scripts", "select_prime_half.py"), "full"],
            input="[]",
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

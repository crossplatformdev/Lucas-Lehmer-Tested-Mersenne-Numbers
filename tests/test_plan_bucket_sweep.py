#!/usr/bin/env python3
"""
tests/test_plan_bucket_sweep.py – Unit tests for scripts/plan_bucket_sweep.py.

Tests:
  1. count_chunks returns correct metadata for various batch counts
  2. emit_chunk returns correct slice of the full matrix
  3. all chunks together cover all batches exactly once
  4. chunk entries have the same worker_names as the full matrix
  5. prime_half filtering is applied before chunking
  6. out-of-range chunk_index raises ValueError
  7. CLI count_chunks mode
  8. CLI emit_chunk mode
  9. single-chunk case (total <= chunk_size) is unchanged
 10. build_full_matrix with prime_half integration
"""

import json
import os
import subprocess
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from plan_bucket_sweep import (  # noqa: E402
    build_full_matrix,
    count_chunks,
    emit_chunk,
    CHUNK_SIZE_DEFAULT,
)
from split_bucket_batches import generate_batch_matrix  # noqa: E402
from select_prime_half import select_half  # noqa: E402
from generate_bucket_primes import enumerate_bucket_primes  # noqa: E402


class TestCountChunks(unittest.TestCase):
    """count_chunks() returns correct planning metadata."""

    def test_zero_batches(self):
        result = count_chunks(0)
        self.assertEqual(result["total_batches"], 0)
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(result["chunk_size"], CHUNK_SIZE_DEFAULT)

    def test_exactly_chunk_size(self):
        result = count_chunks(256, chunk_size=256)
        self.assertEqual(result["total_batches"], 256)
        self.assertEqual(result["chunk_count"], 1)

    def test_one_over_limit(self):
        result = count_chunks(257, chunk_size=256)
        self.assertEqual(result["total_batches"], 257)
        self.assertEqual(result["chunk_count"], 2)

    def test_large(self):
        result = count_chunks(600, chunk_size=256)
        self.assertEqual(result["total_batches"], 600)
        self.assertEqual(result["chunk_count"], 3)  # ceil(600/256)

    def test_custom_chunk_size(self):
        result = count_chunks(100, chunk_size=30)
        self.assertEqual(result["chunk_count"], 4)  # ceil(100/30)

    def test_single_batch(self):
        result = count_chunks(1, chunk_size=256)
        self.assertEqual(result["chunk_count"], 1)


class TestEmitChunk(unittest.TestCase):
    """emit_chunk() returns the correct slice of the full matrix."""

    def _make_matrix(self, n: int) -> list:
        """Fake matrix of n entries for testing."""
        return [{"idx": i} for i in range(n)]

    def test_single_chunk_no_split(self):
        matrix = self._make_matrix(100)
        chunk = emit_chunk(matrix, chunk_index=0, chunk_size=256)
        self.assertEqual(chunk, matrix)

    def test_first_chunk(self):
        matrix = self._make_matrix(300)
        chunk = emit_chunk(matrix, chunk_index=0, chunk_size=256)
        self.assertEqual(len(chunk), 256)
        self.assertEqual(chunk[0]["idx"], 0)
        self.assertEqual(chunk[-1]["idx"], 255)

    def test_second_chunk(self):
        matrix = self._make_matrix(300)
        chunk = emit_chunk(matrix, chunk_index=1, chunk_size=256)
        self.assertEqual(len(chunk), 44)
        self.assertEqual(chunk[0]["idx"], 256)
        self.assertEqual(chunk[-1]["idx"], 299)

    def test_all_chunks_cover_full_matrix(self):
        matrix = self._make_matrix(600)
        chunk_size = 256
        import math
        n_chunks = math.ceil(600 / chunk_size)
        combined = []
        for i in range(n_chunks):
            combined.extend(emit_chunk(matrix, chunk_index=i, chunk_size=chunk_size))
        self.assertEqual(combined, matrix)

    def test_out_of_range_raises(self):
        matrix = self._make_matrix(100)
        with self.assertRaises(ValueError):
            emit_chunk(matrix, chunk_index=1, chunk_size=256)

    def test_empty_matrix_chunk_zero(self):
        chunk = emit_chunk([], chunk_index=0, chunk_size=256)
        self.assertEqual(chunk, [])

    def test_empty_matrix_chunk_nonzero_raises(self):
        with self.assertRaises(ValueError):
            emit_chunk([], chunk_index=1, chunk_size=256)


class TestBuildFullMatrix(unittest.TestCase):
    """build_full_matrix() matches generate_batch_matrix() + select_half()."""

    def test_full_matches_generate_batch_matrix(self):
        expected = generate_batch_matrix(9, 11, batch_size=100)
        got = build_full_matrix(9, 11, batch_size=100, prime_half="full")
        self.assertEqual(got, expected)

    def test_lower_half_matches_select_half(self):
        base = generate_batch_matrix(9, 11, batch_size=100)
        expected = select_half(base, "lower_half")
        got = build_full_matrix(9, 11, batch_size=100, prime_half="lower_half")
        self.assertEqual(got, expected)

    def test_upper_half_matches_select_half(self):
        base = generate_batch_matrix(9, 11, batch_size=100)
        expected = select_half(base, "upper_half")
        got = build_full_matrix(9, 11, batch_size=100, prime_half="upper_half")
        self.assertEqual(got, expected)


class TestChunkedPlanEquivalence(unittest.TestCase):
    """All chunks together cover the full matrix exactly once."""

    def _all_worker_names(self, matrix: list) -> list:
        return [e["worker_name"] for e in matrix]

    def test_chunked_covers_full_matrix(self):
        # Use buckets 9-13 to get a manageable matrix.
        full = build_full_matrix(9, 13, batch_size=50, prime_half="full")
        chunk_size = 30
        info = count_chunks(len(full), chunk_size=chunk_size)
        n_chunks = info["chunk_count"]

        combined = []
        for i in range(n_chunks):
            combined.extend(emit_chunk(full, chunk_index=i, chunk_size=chunk_size))

        self.assertEqual(len(combined), len(full))
        self.assertEqual(
            self._all_worker_names(combined),
            self._all_worker_names(full),
        )

    def test_single_chunk_unchanged(self):
        # When total <= chunk_size, chunk 0 == full matrix.
        full = build_full_matrix(9, 10, batch_size=1000, prime_half="full")
        info = count_chunks(len(full), chunk_size=256)
        self.assertEqual(info["chunk_count"], 1)
        chunk = emit_chunk(full, chunk_index=0, chunk_size=256)
        self.assertEqual(chunk, full)

    def test_chunk_worker_names_stable(self):
        """Worker names in chunks must match those in the full plan."""
        full = build_full_matrix(9, 11, batch_size=20, prime_half="full")
        chunk_size = 10
        import math
        n_chunks = math.ceil(len(full) / chunk_size)
        for i in range(n_chunks):
            chunk = emit_chunk(full, chunk_index=i, chunk_size=chunk_size)
            for j, entry in enumerate(chunk):
                expected_entry = full[i * chunk_size + j]
                self.assertEqual(
                    entry["worker_name"],
                    expected_entry["worker_name"],
                    f"Worker name mismatch at chunk {i}, entry {j}",
                )

    def test_prime_half_then_chunked(self):
        """prime_half filter is applied before chunking, so half contents are stable."""
        full_lower = build_full_matrix(9, 13, batch_size=50, prime_half="lower_half")
        chunk_size = 10
        info = count_chunks(len(full_lower), chunk_size=chunk_size)
        combined = []
        for i in range(info["chunk_count"]):
            combined.extend(emit_chunk(full_lower, chunk_index=i, chunk_size=chunk_size))
        self.assertEqual(self._all_worker_names(combined), self._all_worker_names(full_lower))


class TestPlanBucketSweepCLI(unittest.TestCase):
    """Integration tests via subprocess."""

    _SCRIPT = os.path.join(_REPO_ROOT, "scripts", "plan_bucket_sweep.py")

    def _run(self, *args, input_text: "str | None" = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, self._SCRIPT, *args],
            input=input_text,
            capture_output=True,
            text=True,
        )

    def test_count_chunks_small(self):
        result = self._run(
            "--bucket-start", "9", "--bucket-end", "11",
            "--batch-size", "100",
            "--mode", "count_chunks",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertIn("total_batches", data)
        self.assertIn("chunk_count", data)
        self.assertIn("chunk_size", data)
        self.assertGreater(data["total_batches"], 0)
        self.assertEqual(data["chunk_count"], 1)  # small buckets fit in one chunk

    def test_count_chunks_json_structure(self):
        result = self._run(
            "--bucket-start", "9", "--bucket-end", "9",
            "--mode", "count_chunks",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertIsInstance(data["total_batches"], int)
        self.assertIsInstance(data["chunk_count"], int)
        self.assertIsInstance(data["chunk_size"], int)

    def test_emit_chunk_default(self):
        result = self._run(
            "--bucket-start", "9", "--bucket-end", "11",
            "--batch-size", "100",
            "--mode", "emit_chunk",
            "--chunk-index", "0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        matrix = json.loads(result.stdout)
        self.assertIsInstance(matrix, list)
        self.assertGreater(len(matrix), 0)
        # Each entry must have all required fields.
        for entry in matrix:
            for key in ("bucket_n", "batch_index", "batch_size", "worker_name"):
                self.assertIn(key, entry, f"Missing key {key!r} in entry {entry}")

    def test_emit_chunk_all_chunks_cover_full(self):
        """All CLI-emitted chunks together reproduce the full matrix."""
        # First get total_batches and chunk_count.
        count_result = self._run(
            "--bucket-start", "9", "--bucket-end", "11",
            "--batch-size", "20",
            "--mode", "count_chunks",
            "--chunk-size", "30",
        )
        self.assertEqual(count_result.returncode, 0, count_result.stderr)
        info = json.loads(count_result.stdout)
        n_chunks = info["chunk_count"]

        combined = []
        for i in range(n_chunks):
            r = self._run(
                "--bucket-start", "9", "--bucket-end", "11",
                "--batch-size", "20",
                "--mode", "emit_chunk",
                "--chunk-size", "30",
                "--chunk-index", str(i),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            combined.extend(json.loads(r.stdout))

        # Compare against the direct Python API.
        full = build_full_matrix(9, 11, batch_size=20)
        self.assertEqual(len(combined), len(full))
        for cli_entry, py_entry in zip(combined, full):
            self.assertEqual(cli_entry["worker_name"], py_entry["worker_name"])

    def test_prime_half_lower(self):
        result = self._run(
            "--bucket-start", "10", "--bucket-end", "10",
            "--prime-half", "lower_half",
            "--mode", "emit_chunk",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        matrix = json.loads(result.stdout)
        all_primes = enumerate_bucket_primes(10)
        mid = len(all_primes) // 2
        # The total exponent count in the chunk must equal mid.
        total_exps = sum(e["batch_size"] for e in matrix)
        self.assertEqual(total_exps, mid)

    def test_prime_half_upper(self):
        result = self._run(
            "--bucket-start", "10", "--bucket-end", "10",
            "--prime-half", "upper_half",
            "--mode", "emit_chunk",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        matrix = json.loads(result.stdout)
        all_primes = enumerate_bucket_primes(10)
        mid = len(all_primes) // 2
        total_exps = sum(e["batch_size"] for e in matrix)
        self.assertEqual(total_exps, len(all_primes) - mid)

    def test_dry_run_does_not_emit_json(self):
        result = self._run(
            "--bucket-start", "9", "--bucket-end", "9",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # dry-run output should not be valid JSON (it's human-readable).
        with self.assertRaises((json.JSONDecodeError, ValueError)):
            json.loads(result.stdout)

    def test_invalid_chunk_index_errors(self):
        """Requesting a chunk_index >= chunk_count should exit with an error."""
        result = self._run(
            "--bucket-start", "9", "--bucket-end", "9",
            "--mode", "emit_chunk",
            "--chunk-index", "999",
        )
        self.assertNotEqual(result.returncode, 0)

    def test_count_chunks_prime_half(self):
        """count_chunks with lower_half should report fewer batches than full."""
        full_result = self._run(
            "--bucket-start", "9", "--bucket-end", "11",
            "--mode", "count_chunks",
        )
        half_result = self._run(
            "--bucket-start", "9", "--bucket-end", "11",
            "--prime-half", "lower_half",
            "--mode", "count_chunks",
        )
        self.assertEqual(full_result.returncode, 0)
        self.assertEqual(half_result.returncode, 0)
        full_info = json.loads(full_result.stdout)
        half_info = json.loads(half_result.stdout)
        self.assertLessEqual(half_info["total_batches"], full_info["total_batches"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
tests/test_split_bucket_batches.py – Unit tests for split_bucket_batches.py.

Tests:
  1. Prime-list splitting into batches of size <= BATCH_SIZE
  2. Exact preservation of order across batches
  3. No dropped or duplicated exponents
  4. Correct batch_min_exponent / batch_max_exponent
  5. Deterministic worker naming
  6. Matrix generation from bucket → batches
  7. batch_prime_start_index / batch_prime_end_index correctness
  8. Empty-bucket handling
  9. Single-prime-bucket handling
"""

import json
import os
import subprocess
import sys
import unittest

# Allow importing from scripts/ directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from split_bucket_batches import (  # noqa: E402
    BATCH_SIZE_DEFAULT,
    GITHUB_MATRIX_MAX,
    generate_batch_matrix,
    make_worker_name,
    split_bucket_into_batches,
)
from generate_bucket_primes import enumerate_bucket_primes, bucket_range  # noqa: E402


class TestMakeWorkerName(unittest.TestCase):
    """Deterministic worker naming."""

    def test_format(self):
        name = make_worker_name(17, 1, 1000, 65537, 65867)
        self.assertEqual(name, "bucket-17-batch-0001-1000-exp-65537-65867")

    def test_zero_padding_bucket(self):
        # Single-digit bucket should be zero-padded to 2 digits.
        name = make_worker_name(5, 1, 1, 11, 11)
        self.assertEqual(name, "bucket-05-batch-0001-0001-exp-11-11")

    def test_zero_padding_ordinals(self):
        # Ordinals zero-padded to 4 digits.
        name = make_worker_name(2, 1, 3, 2, 5)
        self.assertEqual(name, "bucket-02-batch-0001-0003-exp-2-5")

    def test_deterministic(self):
        # Calling twice with same args yields identical string.
        a = make_worker_name(10, 2001, 2437, 100003, 200003)
        b = make_worker_name(10, 2001, 2437, 100003, 200003)
        self.assertEqual(a, b)

    def test_different_batches_have_different_names(self):
        n1 = make_worker_name(17, 1, 1000, 65537, 65867)
        n2 = make_worker_name(17, 1001, 2000, 65869, 66107)
        self.assertNotEqual(n1, n2)


class TestSplitBucketIntoBatches(unittest.TestCase):
    """Core batch-splitting logic."""

    def _full_prime_list(self, bucket_n: int) -> list[int]:
        return enumerate_bucket_primes(bucket_n)

    # ------------------------------------------------------------------
    # Bucket 1: [2, 2] – exactly one prime (2)
    # ------------------------------------------------------------------
    def test_bucket1_single_prime(self):
        batches = split_bucket_into_batches(1, batch_size=1000)
        self.assertEqual(len(batches), 1)
        b = batches[0]
        self.assertEqual(b["bucket_n"], 1)
        self.assertEqual(b["batch_index"], 0)
        self.assertEqual(b["batch_count"], 1)
        self.assertEqual(b["batch_prime_start_index"], 0)
        self.assertEqual(b["batch_prime_end_index"], 0)
        self.assertEqual(b["batch_min_exponent"], 2)
        self.assertEqual(b["batch_max_exponent"], 2)
        self.assertEqual(b["batch_size"], 1)

    # ------------------------------------------------------------------
    # Bucket 2: [2, 3] – primes: 2, 3
    # ------------------------------------------------------------------
    def test_bucket2_two_primes_one_batch(self):
        batches = split_bucket_into_batches(2, batch_size=1000)
        self.assertEqual(len(batches), 1)
        b = batches[0]
        self.assertEqual(b["batch_prime_start_index"], 0)
        self.assertEqual(b["batch_prime_end_index"], 1)
        self.assertEqual(b["batch_min_exponent"], 2)
        self.assertEqual(b["batch_max_exponent"], 3)

    # ------------------------------------------------------------------
    # Batch size == 1 → each prime gets its own batch
    # ------------------------------------------------------------------
    def test_batch_size_1_bucket1(self):
        batches = split_bucket_into_batches(1, batch_size=1)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["batch_size"], 1)

    def test_batch_size_1_bucket4(self):
        # Bucket 4: [4, 7] → primes: 5, 7
        primes = self._full_prime_list(4)
        batches = split_bucket_into_batches(4, batch_size=1)
        self.assertEqual(len(batches), len(primes))
        for i, (b, p) in enumerate(zip(batches, primes)):
            self.assertEqual(b["batch_min_exponent"], p)
            self.assertEqual(b["batch_max_exponent"], p)
            self.assertEqual(b["batch_size"], 1)
            self.assertEqual(b["batch_prime_start_index"], i)
            self.assertEqual(b["batch_prime_end_index"], i)

    # ------------------------------------------------------------------
    # No dropped or duplicated exponents
    # ------------------------------------------------------------------
    def test_no_dropped_or_duplicated_bucket5(self):
        bucket_n = 5
        primes = self._full_prime_list(bucket_n)
        batches = split_bucket_into_batches(bucket_n, batch_size=3)

        # Reconstruct the full list from batches – order must be preserved.
        # We rely on batch_min/max_exponent and batch_prime_start/end_index.
        total_claimed = sum(b["batch_size"] for b in batches)
        self.assertEqual(total_claimed, len(primes))

        # Verify contiguity of index ranges.
        prev_end = -1
        for b in batches:
            self.assertEqual(b["batch_prime_start_index"], prev_end + 1)
            self.assertEqual(
                b["batch_prime_end_index"],
                b["batch_prime_start_index"] + b["batch_size"] - 1,
            )
            prev_end = b["batch_prime_end_index"]

        # Verify boundary exponents align with the source prime list.
        for b in batches:
            start = b["batch_prime_start_index"]
            end = b["batch_prime_end_index"]
            self.assertEqual(b["batch_min_exponent"], primes[start])
            self.assertEqual(b["batch_max_exponent"], primes[end])

    # ------------------------------------------------------------------
    # Order preserved across batches
    # ------------------------------------------------------------------
    def test_order_preserved_bucket6(self):
        bucket_n = 6
        primes = self._full_prime_list(bucket_n)
        batches = split_bucket_into_batches(bucket_n, batch_size=5)

        reconstructed = []
        for b in batches:
            start = b["batch_prime_start_index"]
            end = b["batch_prime_end_index"]
            reconstructed.extend(primes[start : end + 1])

        self.assertEqual(reconstructed, primes)

    # ------------------------------------------------------------------
    # Batch size boundary: exact multiple
    # ------------------------------------------------------------------
    def test_exact_multiple_batch_size(self):
        # Build a prime list and split with a batch_size that divides evenly.
        bucket_n = 5
        primes = self._full_prime_list(bucket_n)
        n = len(primes)
        # Use batch_size = n so we get exactly 1 batch.
        batches = split_bucket_into_batches(bucket_n, batch_size=n)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["batch_count"], 1)
        self.assertEqual(batches[0]["batch_size"], n)

    # ------------------------------------------------------------------
    # Last batch may be smaller than batch_size
    # ------------------------------------------------------------------
    def test_last_batch_smaller(self):
        bucket_n = 5
        primes = self._full_prime_list(bucket_n)
        n = len(primes)
        # Use batch_size that does NOT divide evenly.
        batch_size = max(1, n - 1)
        batches = split_bucket_into_batches(bucket_n, batch_size=batch_size)
        self.assertGreaterEqual(len(batches), 1)
        last = batches[-1]
        self.assertLessEqual(last["batch_size"], batch_size)

    # ------------------------------------------------------------------
    # batch_count field is consistent across all batches in same bucket
    # ------------------------------------------------------------------
    def test_batch_count_consistent(self):
        bucket_n = 5
        batches = split_bucket_into_batches(bucket_n, batch_size=3)
        if batches:
            expected_count = batches[0]["batch_count"]
            for b in batches:
                self.assertEqual(b["batch_count"], expected_count)
            self.assertEqual(expected_count, len(batches))

    # ------------------------------------------------------------------
    # batch_min_exponent / batch_max_exponent correctness
    # ------------------------------------------------------------------
    def test_min_max_exponent_correctness(self):
        bucket_n = 6
        primes = self._full_prime_list(bucket_n)
        batches = split_bucket_into_batches(bucket_n, batch_size=4)
        for b in batches:
            si = b["batch_prime_start_index"]
            ei = b["batch_prime_end_index"]
            self.assertEqual(b["batch_min_exponent"], primes[si])
            self.assertEqual(b["batch_max_exponent"], primes[ei])
            # min must be <= max
            self.assertLessEqual(b["batch_min_exponent"], b["batch_max_exponent"])

    # ------------------------------------------------------------------
    # worker_name is deterministic
    # ------------------------------------------------------------------
    def test_worker_name_deterministic(self):
        batches_a = split_bucket_into_batches(5, batch_size=3)
        batches_b = split_bucket_into_batches(5, batch_size=3)
        names_a = [b["worker_name"] for b in batches_a]
        names_b = [b["worker_name"] for b in batches_b]
        self.assertEqual(names_a, names_b)

    def test_worker_names_unique_within_bucket(self):
        batches = split_bucket_into_batches(6, batch_size=4)
        names = [b["worker_name"] for b in batches]
        self.assertEqual(len(names), len(set(names)))

    # ------------------------------------------------------------------
    # Batch size <= BATCH_SIZE_DEFAULT everywhere
    # ------------------------------------------------------------------
    def test_default_batch_size_cap(self):
        bucket_n = 10
        batches = split_bucket_into_batches(bucket_n)
        for b in batches:
            self.assertLessEqual(b["batch_size"], BATCH_SIZE_DEFAULT)

    # ------------------------------------------------------------------
    # bucket_min / bucket_max match the known bucket range
    # ------------------------------------------------------------------
    def test_bucket_range_fields(self):
        for n in [1, 2, 5, 10]:
            lo, hi = bucket_range(n)
            batches = split_bucket_into_batches(n, batch_size=1000)
            for b in batches:
                self.assertEqual(b["bucket_min"], lo)
                self.assertEqual(b["bucket_max"], hi)


class TestGenerateBatchMatrix(unittest.TestCase):
    """Matrix generation across a range of buckets."""

    def test_single_bucket(self):
        matrix = generate_batch_matrix(1, 1)
        self.assertGreater(len(matrix), 0)
        for entry in matrix:
            self.assertEqual(entry["bucket_n"], 1)

    def test_bucket_range_1_to_5(self):
        matrix = generate_batch_matrix(1, 5)
        bucket_ns = sorted({e["bucket_n"] for e in matrix})
        self.assertEqual(bucket_ns, [1, 2, 3, 4, 5])

    def test_no_duplicate_worker_names(self):
        matrix = generate_batch_matrix(1, 8, batch_size=1000)
        names = [e["worker_name"] for e in matrix]
        self.assertEqual(len(names), len(set(names)))

    def test_all_exponents_covered_bucket1_to_5(self):
        """Every prime in buckets 1-5 must appear exactly once across all batches."""
        matrix = generate_batch_matrix(1, 5, batch_size=3)

        # Build expected full prime list for buckets 1-5.
        expected: list[int] = []
        for n in range(1, 6):
            expected.extend(enumerate_bucket_primes(n))

        # Reconstruct from matrix using batch boundary fields only
        # (we don't have the exponent list itself, but we can verify counts and boundaries).
        total = sum(e["batch_size"] for e in matrix)
        self.assertEqual(total, len(expected))

    def test_matrix_entries_have_required_fields(self):
        required = {
            "bucket_n",
            "bucket_min",
            "bucket_max",
            "batch_index",
            "batch_count",
            "batch_prime_start_index",
            "batch_prime_end_index",
            "batch_min_exponent",
            "batch_max_exponent",
            "batch_size",
            "worker_name",
        }
        matrix = generate_batch_matrix(1, 3, batch_size=1000)
        for entry in matrix:
            missing = required - set(entry.keys())
            self.assertEqual(missing, set(), f"Missing fields in entry: {missing}")

    def test_batch_indices_per_bucket_start_at_zero(self):
        matrix = generate_batch_matrix(1, 5, batch_size=1)
        for bucket_n in range(1, 6):
            bucket_entries = [e for e in matrix if e["bucket_n"] == bucket_n]
            if bucket_entries:
                self.assertEqual(bucket_entries[0]["batch_index"], 0)
                for i, e in enumerate(bucket_entries):
                    self.assertEqual(e["batch_index"], i)

    def test_json_serialisable(self):
        """Matrix must be serialisable to JSON without errors."""
        matrix = generate_batch_matrix(1, 5, batch_size=1000)
        serialised = json.dumps(matrix)
        roundtripped = json.loads(serialised)
        self.assertEqual(len(roundtripped), len(matrix))


class TestCommandLineInterface(unittest.TestCase):
    """Smoke tests via subprocess for the CLI interface."""

    _SCRIPT = os.path.join(_REPO_ROOT, "scripts", "split_bucket_batches.py")

    def _run(self, *args, expect_success: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [sys.executable, self._SCRIPT, *args],
            capture_output=True,
            text=True,
        )
        if expect_success:
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        return result

    def test_dry_run_outputs_human_readable(self):
        result = self._run("--bucket-start", "1", "--bucket-end", "3", "--dry-run")
        self.assertIn("bucket_start", result.stdout)
        self.assertIn("bucket_end", result.stdout)
        self.assertIn("total batches", result.stdout)

    def test_json_output_is_valid(self):
        result = self._run("--bucket-start", "1", "--bucket-end", "3")
        parsed = json.loads(result.stdout)
        self.assertIsInstance(parsed, list)
        self.assertGreater(len(parsed), 0)

    def test_batch_size_respected(self):
        result = self._run("--bucket-start", "5", "--bucket-end", "5", "--batch-size", "2")
        parsed = json.loads(result.stdout)
        for entry in parsed:
            self.assertLessEqual(entry["batch_size"], 2)

    def test_invalid_bucket_start_errors(self):
        result = self._run("--bucket-start", "0", expect_success=False)
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_bucket_end_errors(self):
        result = self._run("--bucket-start", "5", "--bucket-end", "4", expect_success=False)
        self.assertNotEqual(result.returncode, 0)


class TestWriteSummaryMdExponentTable(unittest.TestCase):
    """The merged summary must include the per-exponent results table."""

    def setUp(self):
        import importlib
        self._mod = importlib.import_module("merge_bucket_results")

    def _run_write(self, results):
        import tempfile
        known = [r for r in results if r.get("result") == "prime"
                 and self._mod._is_true(r.get("is_known_mersenne_prime", False))]
        new   = [r for r in results if self._mod._is_true(r.get("is_new_discovery", False))]
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            path = f.name
        self._mod.write_summary_md(results, path, known, new)
        with open(path) as fh:
            content = fh.read()
        os.unlink(path)
        return content

    @staticmethod
    def _exponent_table_rows(md: str) -> list[str]:
        """Return only the data rows from the 'All Exponent Results' section."""
        in_section = False
        rows = []
        for line in md.splitlines():
            if line.startswith("## All Exponent Results"):
                in_section = True
                continue
            if in_section:
                if line.startswith("## "):   # next section reached
                    break
                if line.startswith("| ") and "Mersenne Exponent" not in line and not line.startswith("|---"):
                    rows.append(line)
        return rows

    def test_header_present(self):
        results = [
            {"exponent": "3", "bucket_n": "2", "result": "prime",
             "elapsed_sec": "0.001", "final_residue_hex": "0" * 16,
             "is_known_mersenne_prime": True, "is_new_discovery": False},
        ]
        md = self._run_write(results)
        self.assertIn("| Mersenne Exponent |", md)
        self.assertIn("| Time of the Test (s) |", md)
        self.assertIn("| Result |", md)
        self.assertIn("| Residue |", md)
        self.assertIn("| New (y/n) |", md)
        # Section heading uses the unambiguous title.
        self.assertIn("## All Exponent Results", md)

    def test_every_exponent_has_a_row(self):
        results = [
            {"exponent": "3",  "bucket_n": "2", "result": "prime",     "elapsed_sec": "0.001",
             "final_residue_hex": "0" * 16, "is_known_mersenne_prime": True,  "is_new_discovery": False},
            {"exponent": "11", "bucket_n": "4", "result": "composite", "elapsed_sec": "0.002",
             "final_residue_hex": "ab" * 8,  "is_known_mersenne_prime": False, "is_new_discovery": False},
        ]
        md = self._run_write(results)
        self.assertIn("| 3 |",  md)
        self.assertIn("| 11 |", md)

    def test_new_discovery_column_y(self):
        results = [
            {"exponent": "999999937", "bucket_n": "30", "result": "prime",
             "elapsed_sec": "9999.0", "final_residue_hex": "0" * 16,
             "is_known_mersenne_prime": False, "is_new_discovery": True},
        ]
        md = self._run_write(results)
        # Row should end with " | y |"
        self.assertIn("| y |", md)

    def test_non_discovery_column_n(self):
        results = [
            {"exponent": "11", "bucket_n": "4", "result": "composite", "elapsed_sec": "0.002",
             "final_residue_hex": "dead" * 4, "is_known_mersenne_prime": False, "is_new_discovery": False},
        ]
        md = self._run_write(results)
        self.assertIn("| n |", md)

    def test_elapsed_formatted_as_float(self):
        results = [
            {"exponent": "7", "bucket_n": "3", "result": "prime", "elapsed_sec": "1.23456789",
             "final_residue_hex": "0" * 16, "is_known_mersenne_prime": True, "is_new_discovery": False},
        ]
        md = self._run_write(results)
        # Three decimal places expected.
        self.assertIn("1.235", md)

    def test_residue_in_backticks(self):
        residue = "deadbeefcafe1234"
        results = [
            {"exponent": "5", "bucket_n": "3", "result": "prime", "elapsed_sec": "0.0",
             "final_residue_hex": residue, "is_known_mersenne_prime": True, "is_new_discovery": False},
        ]
        md = self._run_write(results)
        self.assertIn(f"`{residue}`", md)

    def test_table_row_count_matches_total(self):
        results = [
            {"exponent": str(p), "bucket_n": "3", "result": "composite", "elapsed_sec": "0.001",
             "final_residue_hex": "0" * 16, "is_known_mersenne_prime": False, "is_new_discovery": False}
            for p in [5, 7, 11, 13, 17]
        ]
        md = self._run_write(results)
        exp_rows = self._exponent_table_rows(md)
        self.assertEqual(len(exp_rows), len(results))


class TestResumeFromExponent(unittest.TestCase):
    """--resume-from-exponent: skip tested primes, preserve ordinals."""

    def test_resume_skips_primes_within_bucket(self):
        # Bucket 5: [8,15] → primes 11,13 (2 primes)
        # Resume from 11 → only 13 should remain.
        primes = enumerate_bucket_primes(5)
        self.assertGreaterEqual(len(primes), 2)
        first = primes[0]
        batches_full = split_bucket_into_batches(5, batch_size=1000)
        batches_resumed = split_bucket_into_batches(5, batch_size=1000, resume_from=first)
        # The first prime is skipped.
        all_resumed_mins = [b["batch_min_exponent"] for b in batches_resumed]
        self.assertNotIn(first, all_resumed_mins)
        # Total primes in resumed batches is one fewer.
        total_full = sum(b["batch_size"] for b in batches_full)
        total_resumed = sum(b["batch_size"] for b in batches_resumed)
        self.assertEqual(total_resumed, total_full - 1)

    def test_resume_skips_entire_bucket(self):
        # Resume from a value >= max prime in bucket 5 → empty list.
        primes = enumerate_bucket_primes(5)
        last = primes[-1]
        batches = split_bucket_into_batches(5, batch_size=1000, resume_from=last)
        self.assertEqual(batches, [])

    def test_resume_preserves_absolute_ordinals(self):
        # Ordinals in the resumed batch must still be relative to the full list.
        primes = enumerate_bucket_primes(6)
        skip_count = 3
        resume_exp = primes[skip_count - 1]   # skip first 3 primes
        batches = split_bucket_into_batches(6, batch_size=1000, resume_from=resume_exp)
        self.assertGreater(len(batches), 0)
        first_batch = batches[0]
        # start_ordinal (1-based) should be skip_count + 1
        self.assertEqual(first_batch["batch_prime_start_index"], skip_count)

    def test_resume_zero_means_no_skip(self):
        batches_no_resume = split_bucket_into_batches(5, batch_size=1000, resume_from=0)
        batches_with_resume = split_bucket_into_batches(5, batch_size=1000)
        self.assertEqual(batches_no_resume, batches_with_resume)

    def test_generate_matrix_skips_complete_buckets(self):
        # Bucket 1 has only one prime (2).  Resume from 2 → bucket 1 absent from matrix.
        matrix = generate_batch_matrix(1, 3, batch_size=1000, resume_from=2)
        bucket_ns = {e["bucket_n"] for e in matrix}
        self.assertNotIn(1, bucket_ns)

    def test_generate_matrix_resume_partial_bucket(self):
        # Bucket 5 has primes 11, 13.  Resume from 11 → only 13 remains in bucket 5.
        primes_b5 = enumerate_bucket_primes(5)
        matrix_full = generate_batch_matrix(5, 5, batch_size=1000)
        matrix_resumed = generate_batch_matrix(5, 5, batch_size=1000,
                                               resume_from=primes_b5[0])
        total_full = sum(e["batch_size"] for e in matrix_full)
        total_resumed = sum(e["batch_size"] for e in matrix_resumed)
        self.assertEqual(total_resumed, total_full - 1)


class TestTargetWorkers(unittest.TestCase):
    """--target-workers: splits total primes into ~N equal batches."""

    def test_single_bucket_target_workers(self):
        # Bucket 6 has > 4 primes.  With target_workers=2 we should get ≤ 2 batches.
        primes = enumerate_bucket_primes(6)
        matrix = generate_batch_matrix(6, 6, batch_size=1000, target_workers=2)
        self.assertLessEqual(len(matrix), 2)
        total = sum(e["batch_size"] for e in matrix)
        self.assertEqual(total, len(primes))

    def test_total_primes_unchanged(self):
        # All primes must still be covered regardless of target_workers.
        expected_total = sum(len(enumerate_bucket_primes(n)) for n in range(5, 8))
        matrix = generate_batch_matrix(5, 7, batch_size=1000, target_workers=50)
        actual_total = sum(e["batch_size"] for e in matrix)
        self.assertEqual(actual_total, expected_total)

    def test_number_of_batches_close_to_target(self):
        # With target_workers=N, len(matrix) should be >= 1 and <= N + slack
        # (small buckets may produce fewer batches than requested).
        target = 10
        matrix = generate_batch_matrix(5, 7, batch_size=1000, target_workers=target)
        self.assertGreaterEqual(len(matrix), 1)
        # Should not produce significantly more than the target.
        self.assertLessEqual(len(matrix), target * 2)

    def test_target_workers_zero_uses_batch_size(self):
        # target_workers=0 should fall back to batch_size.
        batch_size = 3
        matrix = generate_batch_matrix(5, 5, batch_size=batch_size, target_workers=0)
        for entry in matrix:
            self.assertLessEqual(entry["batch_size"], batch_size)

    def test_batch_size_is_hard_upper_cap(self):
        # Even when target_workers would compute a larger per-batch value,
        # batch_size must never be exceeded.
        cap = 2
        matrix = generate_batch_matrix(5, 5, batch_size=cap, target_workers=1)
        for entry in matrix:
            self.assertLessEqual(entry["batch_size"], cap)


class TestTimeLimitPerBucketCap(unittest.TestCase):
    """--time-limit-seconds: per-bucket batch size cap from timing data."""

    def test_large_time_limit_does_not_shrink_small_buckets(self):
        # A huge time limit should not result in 0 primes per batch.
        matrix = generate_batch_matrix(5, 5, batch_size=1000,
                                       time_limit_seconds=86400.0)
        self.assertGreater(len(matrix), 0)
        for entry in matrix:
            self.assertGreater(entry["batch_size"], 0)

    def test_small_time_limit_shrinks_large_buckets(self):
        # A very small time limit on bucket 16 (worst ≈ 5.646 s) should produce
        # batch_size = 1 (floor(1 / 5.646) = 0, clamped to 1).
        matrix = generate_batch_matrix(16, 16, batch_size=1000,
                                       time_limit_seconds=1.0)
        for entry in matrix:
            self.assertEqual(entry["batch_size"], 1)

    def test_time_limit_and_target_workers_both_apply(self):
        # With target_workers=1 (would give all primes in one batch) but a
        # tight time limit, each batch must still be small.
        matrix = generate_batch_matrix(16, 16, batch_size=1000,
                                       time_limit_seconds=1.0,
                                       target_workers=1)
        for entry in matrix:
            self.assertEqual(entry["batch_size"], 1)


class TestCLINewFlags(unittest.TestCase):
    """CLI smoke tests for --target-workers, --resume-from-exponent, --time-limit-seconds."""

    _SCRIPT = os.path.join(_REPO_ROOT, "scripts", "split_bucket_batches.py")

    def _run(self, *args, expect_success: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [sys.executable, self._SCRIPT, *args],
            capture_output=True, text=True,
        )
        if expect_success:
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        return result

    def test_target_workers_flag_accepted(self):
        result = self._run("--bucket-start", "5", "--bucket-end", "5",
                           "--target-workers", "3")
        parsed = json.loads(result.stdout)
        self.assertIsInstance(parsed, list)

    def test_target_workers_zero_accepted(self):
        # 0 means "off"; the flag must not be rejected.
        result = self._run("--bucket-start", "5", "--bucket-end", "5",
                           "--target-workers", "0")
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_resume_from_exponent_skips_primes(self):
        primes_b5 = enumerate_bucket_primes(5)
        resume = str(primes_b5[0])
        result = self._run("--bucket-start", "5", "--bucket-end", "5",
                           "--resume-from-exponent", resume)
        parsed = json.loads(result.stdout)
        for entry in parsed:
            self.assertGreater(entry["batch_min_exponent"], int(resume))

    def test_time_limit_seconds_accepted(self):
        result = self._run("--bucket-start", "5", "--bucket-end", "5",
                           "--time-limit-seconds", "21600")
        parsed = json.loads(result.stdout)
        self.assertIsInstance(parsed, list)

    def test_time_limit_seconds_inf_rejected(self):
        result = self._run("--bucket-start", "5", "--bucket-end", "5",
                           "--time-limit-seconds", "inf",
                           expect_success=False)
        self.assertNotEqual(result.returncode, 0)

    def test_time_limit_seconds_nan_rejected(self):
        result = self._run("--bucket-start", "5", "--bucket-end", "5",
                           "--time-limit-seconds", "nan",
                           expect_success=False)
        self.assertNotEqual(result.returncode, 0)

    def test_dry_run_shows_target_workers(self):
        result = self._run("--bucket-start", "5", "--bucket-end", "5",
                           "--target-workers", "5", "--dry-run")
        self.assertIn("target_workers", result.stdout)

    def test_negative_target_workers_rejected(self):
        result = self._run("--bucket-start", "5", "--bucket-end", "5",
                           "--target-workers", "-1",
                           expect_success=False)
        self.assertNotEqual(result.returncode, 0)


class TestFindResumeExponent(unittest.TestCase):
    """Unit tests for scripts/find_resume_exponent.py."""

    _SCRIPT = os.path.join(_REPO_ROOT, "scripts", "find_resume_exponent.py")

    def setUp(self):
        import importlib
        # scripts/ is already on sys.path (inserted at module level above).
        self._mod = importlib.import_module("find_resume_exponent")

    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, self._SCRIPT, *args],
            capture_output=True, text=True,
        )

    def test_missing_directory_returns_zero(self):
        result = self._mod.find_last_exponent("/tmp/does_not_exist_42abc")
        self.assertEqual(result, 0)

    def test_empty_directory_returns_zero(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = self._mod.find_last_exponent(d)
            self.assertEqual(result, 0)

    def test_non_matching_files_ignored(self):
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "README.md").touch()
            pathlib.Path(d, "bucket-17-batch-0001-0100.txt").touch()
            result = self._mod.find_last_exponent(d)
            self.assertEqual(result, 0)

    def test_single_zip_artifact_parsed(self):
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "bucket-17-batch-0001-0708-exp-65537-131071-results.zip").touch()
            result = self._mod.find_last_exponent(d)
            self.assertEqual(result, 131071)

    def test_single_directory_artifact_parsed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "bucket-17-batch-0001-0708-exp-65537-131071-results"))
            result = self._mod.find_last_exponent(d)
            self.assertEqual(result, 131071)

    def test_multiple_artifacts_returns_highest(self):
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as d:
            for fname in [
                "bucket-16-batch-0001-1000-exp-32771-65521-results.zip",
                "bucket-17-batch-0001-0708-exp-65537-131071-results.zip",
                "bucket-17-batch-0709-1416-exp-131101-196561-results.zip",
            ]:
                pathlib.Path(d, fname).touch()
            result = self._mod.find_last_exponent(d)
            self.assertEqual(result, 196561)

    def test_mixed_zip_and_directory_artifacts(self):
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "bucket-16-batch-0001-1000-exp-32771-65521-results.zip").touch()
            os.makedirs(os.path.join(d, "bucket-17-batch-0001-0708-exp-65537-262144-results"))
            result = self._mod.find_last_exponent(d)
            self.assertEqual(result, 262144)

    def test_cli_missing_dir_outputs_zero(self):
        result = self._run("/tmp/no_such_dir_xyz987")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "0")

    def test_cli_too_many_args_errors(self):
        result = self._run("a", "b")
        self.assertNotEqual(result.returncode, 0)

    def test_cli_returns_highest_exponent(self):
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as d:
            for fname in [
                "bucket-16-batch-0001-1000-exp-32771-65521-results.zip",
                "bucket-17-batch-0001-0708-exp-65537-131071-results.zip",
            ]:
                pathlib.Path(d, fname).touch()
            result = self._run(d)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), "131071")


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""
merge_bucket_results.py – Merge per-bucket CSV/JSON artifacts into a combined report.

Usage:
    python3 scripts/merge_bucket_results.py [--dir <output_dir>] [--out <output_prefix>]

The script scans <output_dir> (default: bucket_out/) for files matching
bucket_*_results.{csv,json}, merges them, and writes:
    <output_prefix>_all_results.csv
    <output_prefix>_all_results.json
    <output_prefix>_summary.md
    <output_prefix>_new_discoveries.json   (if any new Mersenne primes found)
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path


def scan_csv_files(output_dir: str) -> list[str]:
    pattern = os.path.join(output_dir, "bucket_*", "bucket_*_results.csv")
    return sorted(glob.glob(pattern))


def scan_json_files(output_dir: str) -> list[str]:
    pattern = os.path.join(output_dir, "bucket_*", "bucket_*_results.json")
    return sorted(glob.glob(pattern))


def merge_csv(csv_files: list[str]) -> list[dict]:
    rows = []
    for path in csv_files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    # Sort by exponent numerically
    rows.sort(key=lambda r: int(r["exponent"]))
    return rows


def merge_json(json_files: list[str]) -> list[dict]:
    all_results = []
    for path in json_files:
        with open(path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            all_results.append(r)
    all_results.sort(key=lambda r: int(r["exponent"]))
    return all_results


def _is_true(v) -> bool:
    """Normalize booleans that may be stored as bool or string '1'/'true'."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    return bool(v)


def _elapsed(r) -> float:
    """Return elapsed_sec as float, 0.0 if missing."""
    try:
        return float(r.get("elapsed_sec", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def write_merged_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        print(f"No rows to write to {out_path}", file=sys.stderr)
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote merged CSV: {out_path} ({len(rows)} rows)")


def write_merged_json(results: list[dict], out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump({"mode": "power_bucket_primes_merged",
                   "total_results": len(results),
                   "results": results}, f, indent=2)
    print(f"Wrote merged JSON: {out_path} ({len(results)} results)")


def write_summary_md(
    results: list[dict],
    out_path: str,
    known_verified: list[dict],
    new_discoveries: list[dict],
) -> None:
    total = len(results)
    primes = sum(1 for r in results if r.get("result") == "prime")
    composites = sum(1 for r in results if r.get("result") == "composite")
    skipped = total - primes - composites

    lines = [
        "# Power Bucket Prime Sweep – Merged Summary\n",
        f"- **Total exponents tested**: {total}",
        f"- **Prime (Mersenne candidate)**: {primes}",
        f"  - Known verified: {len(known_verified)}",
        f"  - New discoveries: {len(new_discoveries)}",
        f"- **Composite**: {composites}",
        f"- **Skipped**: {skipped}",
        "",
    ]

    # -------------------------------------------------------------------
    # Bucket timing summary
    # -------------------------------------------------------------------
    bucket_data: dict[int, dict] = {}
    for r in results:
        n = int(r.get("bucket_n", 0))
        if n not in bucket_data:
            bucket_data[n] = {"count": 0, "total_sec": 0.0, "max_sec": 0.0}
        bucket_data[n]["count"] += 1
        t = _elapsed(r)
        bucket_data[n]["total_sec"] += t
        if t > bucket_data[n]["max_sec"]:
            bucket_data[n]["max_sec"] = t

    if bucket_data:
        lines += [
            "## Bucket Timing Summary",
            "",
            "| Bucket n | Exponents tested | Total time (s)"
            " | Avg time (s) | Longest (s) |",
            "|----------|-----------------|---------------|"
            "-------------|------------|",
        ]
        for n in sorted(bucket_data):
            d = bucket_data[n]
            avg = d["total_sec"] / d["count"] if d["count"] else 0.0
            lines.append(
                f"| {n} | {d['count']}"
                f" | {d['total_sec']:.3f}"
                f" | {avg:.3f}"
                f" | {d['max_sec']:.3f} |"
            )
        lines.append("")

    # -------------------------------------------------------------------
    # Verified known Mersenne primes
    # -------------------------------------------------------------------
    lines += [
        f"## Verified Known Mersenne Primes ({len(known_verified)})",
        "",
    ]
    if not known_verified:
        lines.append("_None found in this sweep._")
        lines.append("")
    else:
        lines += [
            "| Exponent | Bucket n | Backend | Elapsed (s) | Residue |",
            "|----------|----------|---------|-------------|--------|",
        ]
        for r in known_verified:
            lines.append(
                f"| {r['exponent']}"
                f" | {r.get('bucket_n', '')}"
                f" | {r.get('backend', '')}"
                f" | {_elapsed(r):.3f}"
                f" | `{r.get('final_residue_hex', '0000000000000000')}` |"
            )
        lines.append("")

    # -------------------------------------------------------------------
    # New discoveries
    # -------------------------------------------------------------------
    lines += [
        f"## 🚨 New Mersenne Prime Discoveries ({len(new_discoveries)})",
        "",
    ]
    if not new_discoveries:
        lines.append("_No new discoveries in this sweep._")
        lines.append("")
    else:
        lines += [
            "| Exponent | Bucket n | Backend | Elapsed (s) | Residue |",
            "|----------|----------|---------|-------------|--------|",
        ]
        for d in new_discoveries:
            lines.append(
                f"| **{d['exponent']}**"
                f" | {d.get('bucket_n', '')}"
                f" | {d.get('backend', '')}"
                f" | {_elapsed(d):.3f}"
                f" | `{d.get('final_residue_hex', '')}` |"
            )
        lines.append("")

    # -------------------------------------------------------------------
    # Composite exponents with residues
    # -------------------------------------------------------------------
    composite_results = [r for r in results if r.get("result") == "composite"]
    if composite_results:
        lines += [
            f"## Composite Exponents Tested ({len(composite_results)})",
            "",
            "| Exponent | Bucket n | Backend | Elapsed (s) | Residue |",
            "|----------|----------|---------|-------------|--------|",
        ]
        for r in composite_results:
            lines.append(
                f"| {r['exponent']}"
                f" | {r.get('bucket_n', '')}"
                f" | {r.get('backend', '')}"
                f" | {_elapsed(r):.3f}"
                f" | `{r.get('final_residue_hex', '')}` |"
            )
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote summary: {out_path}")


def write_new_discoveries_json(new_discoveries: list[dict], out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump({"new_mersenne_prime_candidates": new_discoveries}, f, indent=2)
    print(f"Wrote new discoveries JSON: {out_path} ({len(new_discoveries)} found)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="bucket_out", help="Directory with bucket results")
    parser.add_argument("--out", default="bucket_out/merged", help="Output path prefix")
    args = parser.parse_args()

    output_dir = args.dir
    out_prefix = args.out

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    csv_files  = scan_csv_files(output_dir)
    json_files = scan_json_files(output_dir)

    if not csv_files and not json_files:
        print(f"No bucket result files found in '{output_dir}'.", file=sys.stderr)
        print("Run 'make bucket' or individual bucket jobs first.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) and {len(json_files)} JSON file(s).")

    rows    = merge_csv(csv_files)  if csv_files  else []
    results = merge_json(json_files) if json_files else []

    # Normalise: prefer JSON results if available; fall back to CSV rows.
    merged = results or [
        {
            "exponent":            r["exponent"],
            "bucket_n":            r.get("bucket_n", ""),
            "result":              r.get("result", ""),
            "backend":             r.get("backend", ""),
            "elapsed_sec":         r.get("elapsed_sec", ""),
            "is_known_mersenne_prime": _is_true(r.get("is_known_mersenne", "0")),
            "is_new_discovery":    _is_true(r.get("is_new_discovery", "0")),
            "final_residue_hex":   r.get("final_residue_hex", "0000000000000000"),
        }
        for r in rows
    ]

    # Classify results.
    known_verified  = [r for r in merged
                       if r.get("result") == "prime"
                       and _is_true(r.get("is_known_mersenne_prime", False))]
    new_discoveries = [r for r in merged
                       if _is_true(r.get("is_new_discovery", False))]

    if rows:
        write_merged_csv(rows, out_prefix + "_all_results.csv")
    if results:
        write_merged_json(results, out_prefix + "_all_results.json")

    write_summary_md(merged, out_prefix + "_summary.md",
                     known_verified, new_discoveries)

    if new_discoveries:
        write_new_discoveries_json(new_discoveries, out_prefix + "_new_discoveries.json")
        print("\n*** NEW MERSENNE PRIME CANDIDATES FOUND! ***")
        for d in new_discoveries:
            print(f"  M_{d['exponent']} is a new prime candidate!")
        sys.exit(3)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
find_resume_exponent.py – Find the last tested prime exponent from bucket artifacts.

Scans a directory for artifact files or zip files whose names follow the pattern:
    bucket-NN-batch-NNNN-NNNN-exp-<min>-<max>-results[.zip]

and returns the highest <max> exponent found.  If no matching files exist,
prints 0 so the caller can use that as "no resume point".

Usage:
    python3 scripts/find_resume_exponent.py [directory]

    directory: path to scan (default: bucket_artifacts/)

Output:
    A single integer on stdout: the last tested exponent (0 if none found).

Exit codes:
    0  on success (even if no artifacts exist – prints 0 in that case)
    1  on usage error
"""

import os
import re
import sys

# Matches names like:
#   bucket-17-batch-5001-5709-exp-122753-131071-results.zip
#   bucket-17-batch-5001-5709-exp-122753-131071-results   (directory)
_ARTIFACT_RE = re.compile(
    r"bucket-\d+-batch-\d+-\d+-exp-\d+-(\d+)-results(?:\.zip)?$"
)


def find_last_exponent(directory: str) -> int:
    """Return the highest max-exponent found in artifact filenames.

    Returns 0 if the directory does not exist or contains no matching files.
    """
    if not os.path.isdir(directory):
        return 0
    last = 0
    for name in os.listdir(directory):
        m = _ARTIFACT_RE.search(name)
        if m:
            exp = int(m.group(1))
            if exp > last:
                last = exp
    return last


def main() -> None:
    if len(sys.argv) > 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    directory = sys.argv[1] if len(sys.argv) == 2 else "bucket_artifacts"
    print(find_last_exponent(directory))


if __name__ == "__main__":
    main()

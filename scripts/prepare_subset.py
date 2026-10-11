#!/usr/bin/env python3
"""Stream the first N rows of HIGGS into a plain or gzip CSV subset.

Examples from the repository root:
    python scripts/prepare_subset.py data/HIGGS.csv.gz --rows 1000000 --output data/HIGGS_1M.csv.gz
    python scripts/prepare_subset.py data/HIGGS.csv.gz --rows 1000000 --output data/HIGGS_1M.csv

Output compression is inferred from the output filename: a .gz suffix creates
gzip output; any other suffix creates a plain text CSV. The script streams rows
rather than loading the entire multi-gigabyte dataset into memory.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import TextIO


def open_input(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Source HIGGS CSV or CSV.GZ")
    parser.add_argument("--rows", type=int, default=1_000_000,
                        help="Maximum number of rows to copy (default: 1000000)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV/CSV.GZ path; .gz controls compression")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows <= 0:
        raise SystemExit("--rows must be a positive integer")
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset file not found: {args.dataset}")
    if args.dataset.resolve() == args.output.resolve():
        raise SystemExit("Source and output paths must be different.")

    written = 0
    try:
        with open_input(args.dataset) as source:
            with open_output(args.output) as destination:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    if written == 0:
                        fields = line.rstrip("\r\n").split(",")
                        try:
                            label = float(fields[0].strip())
                        except (IndexError, ValueError) as exc:
                            raise SystemExit(
                                "Unexpected first row. Expected a headerless HIGGS CSV "
                                "with 29 columns: binary label + 28 features."
                            ) from exc
                        if len(fields) != 29 or label not in (0.0, 1.0):
                            raise SystemExit(
                                "Unexpected first row. Expected a headerless HIGGS CSV "
                                "with 29 columns: binary label + 28 features."
                            )
                    destination.write(line)
                    written += 1
                    if written >= args.rows:
                        break
    except (OSError, EOFError) as exc:
        raise SystemExit(f"Dataset read/write failed: {exc}") from exc

    if written == 0:
        raise SystemExit("No dataset rows were copied.")

    print(f"Input dataset : {args.dataset}")
    print(f"Rows written  : {written:,}")
    print(f"Output subset : {args.output}")
    print(f"Compression   : {'gzip' if args.output.suffix.lower() == '.gz' else 'plain CSV'}")
    if written < args.rows:
        print(f"Note: requested {args.rows:,} rows but the source contained only {written:,} nonblank rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

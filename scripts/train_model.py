#!/usr/bin/env python3
"""Train the 28 -> 32 -> 16 -> 1 HIGGS MLP used by the inference demos.

Example from the repository root:
    python scripts/train_model.py data/HIGGS.csv.gz --rows 100000

The input CSV must have no header and contain the label in column 0 followed by
28 feature columns. The model bundle is a local build artifact and is ignored by
Git. Run export_model.py after training to create model_weights.txt.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import TextIO

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "model" / "trained_model.joblib"
FEATURE_COUNT = 28


def open_dataset(path: Path) -> TextIO:
    """Open a plain or gzip-compressed, headerless CSV as text."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to HIGGS CSV or CSV.GZ (label in first column, then 28 features)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100_000,
        help="Number of initial rows used for training (default: 100000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output model bundle (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows <= 0:
        raise SystemExit("--rows must be a positive integer")
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset file not found: {args.dataset}")

    print(f"Loading the first {args.rows:,} rows from {args.dataset} ...")
    try:
        with open_dataset(args.dataset) as stream:
            values = np.loadtxt(
                stream,
                delimiter=",",
                dtype=np.float64,
                max_rows=args.rows,
                ndmin=2,
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not read HIGGS CSV: {exc}") from exc

    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT + 1:
        raise SystemExit(
            f"Expected {FEATURE_COUNT + 1} columns (label + features); "
            f"found shape {values.shape}. Check that the file has no header."
        )
    if values.shape[0] == 0:
        raise SystemExit("No dataset rows were read.")

    labels_raw = values[:, 0]
    if not np.all(np.isin(labels_raw, [0.0, 1.0])):
        raise SystemExit("The first column must contain binary labels 0 or 1.")

    X = values[:, 1:]
    y = labels_raw.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifier = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        batch_size=256,
        max_iter=20,
        random_state=42,
    )
    classifier.fit(X_scaled, y)

    training_accuracy = float(classifier.score(X_scaled, y))
    bundle = {
        "scaler": scaler,
        "classifier": classifier,
        "training_rows": int(values.shape[0]),
        "feature_count": FEATURE_COUNT,
        "config": {
            "hidden_layer_sizes": (32, 16),
            "activation": "relu",
            "solver": "adam",
            "batch_size": 256,
            "max_iter": 20,
            "random_state": 42,
            "input_file": str(args.dataset),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)

    print(f"Rows used for training : {values.shape[0]:,}")
    print(f"Training iterations    : {classifier.n_iter_}")
    print(f"Training accuracy      : {training_accuracy:.6f}")
    print(f"Saved model bundle     : {args.output}")
    print()
    print("Note: training accuracy is not a held-out test score.")
    print("Next: run python scripts/export_model.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a trained scikit-learn HIGGS MLP to model_weights.txt.

Example from the repository root:
    python scripts/export_model.py

The text layout is compatible with src/ai_sequential.cpp, src/ai_openmp.cpp,
src/ai_mpi.cpp, and src/ai_cuda.cu. Keep this generated file local; it is
ignored by Git because it is a derived model artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "model" / "trained_model.joblib"
DEFAULT_OUTPUT = ROOT / "model_weights.txt"
EXPECTED_SHAPES = ((28, 32), (32, 16), (16, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Input model bundle (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output text weights file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def write_values(stream, values: np.ndarray) -> None:
    flat = np.asarray(values).reshape(-1)
    stream.write(" ".join(format(float(value), ".17g") for value in flat))
    stream.write("\n")


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Model bundle not found: {args.model}\n"
            "Train it first with scripts/train_model.py."
        )

    bundle = joblib.load(args.model)
    scaler = bundle.get("scaler")
    classifier = bundle.get("classifier")
    if scaler is None or classifier is None:
        raise SystemExit("Model bundle must contain 'scaler' and 'classifier'.")

    if len(classifier.coefs_) != 3 or len(classifier.intercepts_) != 3:
        raise SystemExit("Expected exactly three dense layers.")
    actual_shapes = tuple(tuple(np.asarray(w).shape) for w in classifier.coefs_)
    if actual_shapes != EXPECTED_SHAPES:
        raise SystemExit(
            f"Unexpected neural-network shapes: {actual_shapes}; "
            f"expected {EXPECTED_SHAPES}."
        )
    if np.asarray(scaler.mean_).shape != (28,) or np.asarray(scaler.scale_).shape != (28,):
        raise SystemExit("Expected 28 scaler means and 28 scaler scales.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Important compatibility detail: each layer label and its dimensions/count
    # are placed on the same line. The CPU loaders skip these entire header
    # lines, while CUDA parses the dimensions/counts explicitly.
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# StandardScaler\n")
        out.write("SCALER_MEAN\n")
        write_values(out, scaler.mean_)
        out.write("SCALER_SCALE\n")
        write_values(out, scaler.scale_)

        for layer_index, (weights, biases) in enumerate(
            zip(classifier.coefs_, classifier.intercepts_), start=1
        ):
            weights = np.asarray(weights, dtype=np.float64)
            biases = np.asarray(biases, dtype=np.float64).reshape(-1)

            out.write(
                f"LAYER_{layer_index}_WEIGHTS {weights.shape[0]} {weights.shape[1]}\n"
            )
            for row in weights:
                write_values(out, row)

            out.write(f"LAYER_{layer_index}_BIASES {biases.size}\n")
            write_values(out, biases)

    print(f"Exported model weights : {args.output}")
    print(f"Architecture           : {EXPECTED_SHAPES[0][0]} -> 32 -> 16 -> 1")
    print(f"Training rows in bundle: {bundle.get('training_rows', 'unknown')}")
    print("This file is a local build artifact and is excluded from Git.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

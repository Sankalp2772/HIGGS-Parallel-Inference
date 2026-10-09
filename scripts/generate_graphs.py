#!/usr/bin/env python3
"""Generate reproducible plots and derived benchmark summaries from raw CSV data.

Run from the repository root:
    python3 scripts/generate_graphs.py

Requires matplotlib: python3 -m pip install matplotlib
Raw measurements are never modified. GPU timings are plotted separately because
they measure kernel-only time, while MPI timings exclude initial data distribution.
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "results" / "benchmark_results.csv"
GRAPHS = ROOT / "graphs"
SUMMARY_CSV = ROOT / "results" / "derived_summary.csv"


def load_rows():
    with RAW_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["dataset_rows"] = int(row["dataset_rows"])
        row["concurrency_count"] = int(row["concurrency_count"])
        row["run_index"] = int(row["run_index"])
        row["elapsed_seconds"] = float(row["elapsed_seconds"])
        row["correct_predictions"] = int(row["correct_predictions"])
        row["accuracy"] = float(row["accuracy"])
        row["average_probability"] = float(row["average_probability"])
    return rows


def grouped(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["implementation"], r["dataset_rows"],
               r["concurrency_type"], r["concurrency_count"])
        groups[key].append(r)
    result = []
    for (implementation, n, ctype, count), runs in sorted(groups.items()):
        times = [r["elapsed_seconds"] for r in runs]
        result.append({
            "implementation": implementation,
            "dataset_rows": n,
            "concurrency_type": ctype,
            "concurrency_count": count,
            "run_count": len(times),
            "mean_seconds": statistics.mean(times),
            "stddev_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
            "mean_rows_per_second": n / statistics.mean(times),
            "correct_predictions": runs[0]["correct_predictions"],
            "accuracy": runs[0]["accuracy"],
            "average_probability": runs[0]["average_probability"],
            "timing_scope": runs[0]["timing_scope"],
            "environment": runs[0]["environment"],
        })
    return result


def save_summary(summary):
    fields = [
        "implementation", "dataset_rows", "concurrency_type",
        "concurrency_count", "run_count", "mean_seconds", "stddev_seconds",
        "mean_rows_per_second", "correct_predictions", "accuracy",
        "average_probability", "timing_scope", "environment",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def make_plots(raw, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GRAPHS.mkdir(parents=True, exist_ok=True)

    # CPU inference intervals. The footnote explicitly states the MPI timing scope.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = [
        ("Sequential", 1, "Sequential"),
        ("OpenMP", 2, "OpenMP (2 threads)"),
        ("MPI", 4, "MPI (4 processes; data distribution excluded)"),
    ]
    for implementation, count, label in styles:
        vals = [s for s in summary if s["implementation"] == implementation
                and s["concurrency_count"] == count]
        vals.sort(key=lambda s: s["dataset_rows"])
        if vals:
            ax.plot([s["dataset_rows"] for s in vals],
                    [s["mean_seconds"] for s in vals],
                    marker="o", linewidth=2, label=label)
    ax.set_title("CPU Inference Timing vs Dataset Size")
    ax.set_xlabel("Dataset rows")
    ax.set_ylabel("Mean elapsed time (seconds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.text(0.01, 0.01,
             "Timing scopes are not fully identical: MPI timer starts after data distribution.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(GRAPHS / "cpu_execution_time_vs_rows.png", dpi=180)
    plt.close(fig)

    # CUDA kernel-only times, deliberately separate from CPU execution timing.
    cuda = [s for s in summary if s["implementation"] == "CUDA"]
    cuda.sort(key=lambda s: s["dataset_rows"])
    if cuda:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([s["dataset_rows"] for s in cuda],
                [s["mean_seconds"] for s in cuda],
                marker="o", linewidth=2)
        ax.set_title("CUDA GPU Kernel Time vs Dataset Size")
        ax.set_xlabel("Dataset rows")
        ax.set_ylabel("Mean kernel time (seconds)")
        ax.grid(True, alpha=0.3)
        fig.text(0.01, 0.01,
                 "Excludes CSV parsing, host-to-device transfer, and model loading.",
                 fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(GRAPHS / "cuda_kernel_time_vs_rows.png", dpi=180)
        plt.close(fig)

    # OpenMP scaling: fixed 100k dataset, sequential baseline.
    n = 100_000
    base = next((s["mean_seconds"] for s in summary
                 if s["implementation"] == "Sequential" and s["dataset_rows"] == n), None)
    omp = sorted([s for s in summary if s["implementation"] == "OpenMP"
                  and s["dataset_rows"] == n], key=lambda s: s["concurrency_count"])
    if base is not None and omp:
        xs = [s["concurrency_count"] for s in omp]
        times = [s["mean_seconds"] for s in omp]
        speedups = [base / t for t in times]
        efficiencies = [speedups[i] / xs[i] for i in range(len(xs))]
        for filename, vals, title, ylabel in [
            ("openmp_speedup_100k.png", speedups, "OpenMP Speedup (100k rows)", "Speedup vs sequential"),
            ("openmp_efficiency_100k.png", efficiencies, "OpenMP Parallel Efficiency (100k rows)", "Efficiency (speedup / threads)"),
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.plot(xs, vals, marker="o", linewidth=2)
            ax.set_xticks(xs)
            ax.set_xlabel("OpenMP thread count")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(GRAPHS / filename, dpi=180)
            plt.close(fig)

    # MPI scaling: fixed 100k rows, speedup relative to the measured 1-process MPI run.
    mpi = sorted([s for s in summary if s["implementation"] == "MPI"
                  and s["dataset_rows"] == n], key=lambda s: s["concurrency_count"])
    base_mpi = next((s["mean_seconds"] for s in mpi if s["concurrency_count"] == 1), None)
    if base_mpi is not None and mpi:
        xs = [s["concurrency_count"] for s in mpi]
        speedups = [base_mpi / s["mean_seconds"] for s in mpi]
        efficiencies = [speedups[i] / xs[i] for i in range(len(xs))]
        for filename, vals, title, ylabel in [
            ("mpi_speedup_100k.png", speedups, "MPI Speedup (100k rows)", "Speedup vs MPI 1-process baseline"),
            ("mpi_efficiency_100k.png", efficiencies, "MPI Parallel Efficiency (100k rows)", "Efficiency (speedup / processes)"),
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.plot(xs, vals, marker="o", linewidth=2)
            ax.set_xticks(xs)
            ax.set_xlabel("MPI process count")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            fig.text(0.01, 0.01, "MPI timing excludes initial data distribution.", fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(GRAPHS / filename, dpi=180)
            plt.close(fig)


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw benchmark CSV not found: {RAW_CSV}")
    raw = load_rows()
    summary = grouped(raw)
    save_summary(summary)
    make_plots(raw, summary)
    print(f"Loaded {len(raw)} raw measurement rows.")
    print(f"Wrote {len(summary)} grouped summary rows to {SUMMARY_CSV.relative_to(ROOT)}.")
    print("Generated graphs:")
    for path in sorted(GRAPHS.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

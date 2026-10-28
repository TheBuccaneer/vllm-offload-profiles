#!/usr/bin/env python3
"""Recompute the pooled Llama multiclass LOCO confusion matrix.

Run from the vllm-offload-profiles repository root:

    python3 analysis/separability/export_llama_multiclass_confusion.py

Uses the *same* data loading, StandardScaler, LogisticRegression, features,
and leave-one-concurrency-out folds as baseline_loco.py. Only the held-out
logistic-regression predictions are included in the pooled matrix.

Rows represent actual offload budgets; columns represent predicted budgets.
No measured data or existing analysis outputs are modified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from baseline_loco import load_data, make_labels, run_fold

LEVELS = (0, 2, 4, 8, 12, 16)
CONCURRENCIES = (2, 4, 8, 12, 16)
EXPECTED_MATRIX = np.array(
    [
        [25, 0, 0, 0, 0, 0],
        [0, 25, 0, 0, 0, 0],
        [0, 0, 25, 0, 0, 0],
        [0, 0, 0, 25, 0, 0],
        [0, 0, 0, 0, 7, 18],
        [0, 0, 0, 0, 5, 20],
    ],
    dtype=np.int64,
)


def calculate(input_csv: Path) -> np.ndarray:
    """Run the original LOCO classifier and pool its held-out test results."""
    df, features = load_data(str(input_csv), use_concurrency_feature=False)
    df = df.loc[df["run_concurrency"].isin(CONCURRENCIES)].reset_index(drop=True)

    if sorted(int(x) for x in df["run_concurrency"].unique()) != list(CONCURRENCIES):
        raise ValueError("Expected the five concurrency levels 2, 4, 8, 12, 16")
    if len(df) != 150:
        raise ValueError(f"Expected 150 Llama runs for c>=2; found {len(df)}")
    if sorted(int(x) for x in df["offload_gb"].unique()) != list(LEVELS):
        raise ValueError("Expected six offload classes: 0, 2, 4, 8, 12, 16")

    # The original classifier trains only on other concurrency levels.
    labels = make_labels(df, mode="multiclass")
    pooled = np.zeros((len(LEVELS), len(LEVELS)), dtype=np.int64)
    for held_out in CONCURRENCIES:
        fold = run_fold(df, features, labels, held_out, mode="multiclass")
        if fold is None:
            raise ValueError(f"The held-out fold c={held_out} was skipped")
        classes = [int(x) for x in fold["classes"]]
        if classes != list(LEVELS):
            raise ValueError(f"Unexpected classes in c={held_out}: {classes}")
        cm = np.asarray(fold["cm_lr"], dtype=np.int64)
        if cm.shape != (6, 6) or not np.all(cm.sum(axis=1) == 5):
            raise ValueError(f"Unexpected confusion counts in fold c={held_out}")
        pooled += cm

    if not np.all(pooled.sum(axis=1) == 25):
        raise ValueError("Expected exactly 25 held-out runs per offload class")
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/derived/baseline/llama/runs_summary.csv"),
        help="Existing baseline run summaries (default: Llama derived data)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/tables/separability/llama_multiclass_confusion_c2plus.csv"),
        help="Output confusion matrix CSV",
    )
    args = parser.parse_args()

    matrix = calculate(args.input)
    if not np.array_equal(matrix, EXPECTED_MATRIX):
        raise ValueError(
            "Recomputed matrix differs from the archived paper results. "
            "Check the input data and analysis environment.\n"
            f"Expected:\n{EXPECTED_MATRIX}\nObserved:\n{matrix}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["actual_offload_gb"] + [f"pred_{x}_gib" for x in LEVELS])
        for level, row in zip(LEVELS, matrix):
            writer.writerow([level] + row.tolist())

    correct = int(np.trace(matrix))
    print(f"PASS: {correct}/150 correct; {150 - correct} errors, all at 12/16 GiB")
    print(f"Balanced accuracy: {correct / 150:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

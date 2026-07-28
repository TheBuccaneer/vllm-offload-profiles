#!/usr/bin/env python3
"""Analyze within-profile and cross-profile offload reconstruction robustness."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROFILES = ["128/32", "256/64", "512/128"]
REFERENCE_PROFILE = "256/64"
FEATURE_SETS = {
    "decode": ["median_tpot_ms", "median_itl_ms"],
    "all3": ["median_tpot_ms", "median_itl_ms", "median_ttft_ms"],
}


def model_alias(name: str) -> str:
    lowered = name.lower()
    if "llama" in lowered:
        return "llama"
    if "qwen" in lowered:
        return "qwen"
    return name.replace("/", "_")


def classifier() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=42, solver="liblinear")),
    ])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def append_confusion(rows: list[dict], *, analysis: str, model: str, feature_set: str,
                     train_profile: str, test_profile: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    for i, true_label in enumerate([0, 1]):
        for j, pred_label in enumerate([0, 1]):
            rows.append({
                "analysis": analysis,
                "model": model,
                "feature_set": feature_set,
                "train_profile": train_profile,
                "test_profile": test_profile,
                "true_label": true_label,
                "pred_label": pred_label,
                "count": int(matrix[i, j]),
            })


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "model_name", "offload_gb", "run_concurrency", "run_id",
        "input_len", "output_len", *sorted({x for values in FEATURE_SETS.values() for x in values}),
    ]
    require_columns(df, required)
    for column in ["offload_gb", "run_concurrency", "run_id", "input_len", "output_len"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[
        df["offload_gb"].isin([0, 12])
        & df["run_concurrency"].isin([4, 8])
        & df["input_len"].isin([128, 256, 512])
        & df["output_len"].isin([32, 64, 128])
    ].copy()
    df["profile"] = df["input_len"].astype("Int64").astype(str) + "/" + df["output_len"].astype("Int64").astype(str)
    df = df[df["profile"].isin(PROFILES)].copy()
    df["model"] = df["model_name"].astype(str).map(model_alias)
    df["label"] = (df["offload_gb"] > 0).astype(int)
    df = df.dropna(subset=["run_id", *sorted({x for values in FEATURE_SETS.values() for x in values})])
    if df.empty:
        raise ValueError("No matching profile campaign rows remain after filtering")
    return df.sort_values(["model", "profile", "offload_gb", "run_concurrency", "run_id"]).reset_index(drop=True)


def within_profile(df: pd.DataFrame, summaries: list[dict], folds: list[dict], predictions: list[dict], confusions: list[dict]) -> None:
    for (model, profile), group in df.groupby(["model", "profile"], sort=True):
        repeat_ids = sorted(group["run_id"].astype(int).unique())
        for feature_name, features in FEATURE_SETS.items():
            all_true: list[int] = []
            all_pred: list[int] = []
            for repeat_id in repeat_ids:
                train = group[group["run_id"].astype(int) != repeat_id]
                test = group[group["run_id"].astype(int) == repeat_id]
                if train["label"].nunique() < 2 or test["label"].nunique() < 2:
                    continue
                pipe = classifier().fit(train[features], train["label"])
                pred = pipe.predict(test[features])
                metrics = evaluate_predictions(test["label"].to_numpy(), pred)
                folds.append({
                    "analysis": "within_profile_leave_one_repeat_out",
                    "model": model,
                    "feature_set": feature_name,
                    "profile": profile,
                    "held_out_repeat": int(repeat_id),
                    "n_train": int(len(train)),
                    **metrics,
                })
                for idx, predicted in zip(test.index, pred):
                    predictions.append({
                        "analysis": "within_profile_leave_one_repeat_out",
                        "model": model,
                        "feature_set": feature_name,
                        "train_profile": profile,
                        "test_profile": profile,
                        "row_index": int(idx),
                        "true_label": int(df.loc[idx, "label"]),
                        "pred_label": int(predicted),
                    })
                all_true.extend(test["label"].astype(int).tolist())
                all_pred.extend(pred.astype(int).tolist())
            if all_true:
                metrics = evaluate_predictions(np.asarray(all_true), np.asarray(all_pred))
                summaries.append({
                    "analysis": "within_profile_leave_one_repeat_out",
                    "model": model,
                    "feature_set": feature_name,
                    "train_profile": profile,
                    "test_profile": profile,
                    **metrics,
                })
                append_confusion(
                    confusions, analysis="within_profile_leave_one_repeat_out", model=model,
                    feature_set=feature_name, train_profile=profile, test_profile=profile,
                    y_true=np.asarray(all_true), y_pred=np.asarray(all_pred),
                )


def within_profile_leave_one_concurrency_out(
    df: pd.DataFrame,
    summaries: list[dict],
    folds: list[dict],
    predictions: list[dict],
    confusions: list[dict],
) -> None:
    """Within each model/profile, hold out one concurrency level.

    This mirrors the profile-robustness analysis logic more closely than
    leave-one-repeat-out: the classifier must transfer from one load level to
    the other while the request profile is fixed.
    """
    for (model, profile), group in df.groupby(["model", "profile"], sort=True):
        conc_levels = sorted(group["run_concurrency"].astype(int).unique())
        for feature_name, features in FEATURE_SETS.items():
            all_true: list[int] = []
            all_pred: list[int] = []
            for held_out_conc in conc_levels:
                train = group[group["run_concurrency"].astype(int) != held_out_conc]
                test = group[group["run_concurrency"].astype(int) == held_out_conc]
                if train["label"].nunique() < 2 or test["label"].nunique() < 2:
                    continue
                pipe = classifier().fit(train[features], train["label"])
                pred = pipe.predict(test[features])
                metrics = evaluate_predictions(test["label"].to_numpy(), pred)
                folds.append({
                    "analysis": "within_profile_leave_one_concurrency_out",
                    "model": model,
                    "feature_set": feature_name,
                    "profile": profile,
                    "held_out_concurrency": int(held_out_conc),
                    "n_train": int(len(train)),
                    **metrics,
                })
                for idx, predicted in zip(test.index, pred):
                    predictions.append({
                        "analysis": "within_profile_leave_one_concurrency_out",
                        "model": model,
                        "feature_set": feature_name,
                        "train_profile": profile,
                        "test_profile": profile,
                        "row_index": int(idx),
                        "true_label": int(df.loc[idx, "label"]),
                        "pred_label": int(predicted),
                    })
                all_true.extend(test["label"].astype(int).tolist())
                all_pred.extend(pred.astype(int).tolist())
            if all_true:
                metrics = evaluate_predictions(np.asarray(all_true), np.asarray(all_pred))
                summaries.append({
                    "analysis": "within_profile_leave_one_concurrency_out",
                    "model": model,
                    "feature_set": feature_name,
                    "train_profile": profile,
                    "test_profile": profile,
                    **metrics,
                })
                append_confusion(
                    confusions,
                    analysis="within_profile_leave_one_concurrency_out",
                    model=model,
                    feature_set=feature_name,
                    train_profile=profile,
                    test_profile=profile,
                    y_true=np.asarray(all_true),
                    y_pred=np.asarray(all_pred),
                )


def cross_profile(df: pd.DataFrame, summaries: list[dict], predictions: list[dict], confusions: list[dict]) -> None:
    for model, group in df.groupby("model", sort=True):
        train = group[group["profile"] == REFERENCE_PROFILE]
        if train.empty or train["label"].nunique() < 2:
            continue
        for target in [profile for profile in PROFILES if profile != REFERENCE_PROFILE]:
            test = group[group["profile"] == target]
            if test.empty or test["label"].nunique() < 2:
                continue
            for feature_name, features in FEATURE_SETS.items():
                pipe = classifier().fit(train[features], train["label"])
                pred = pipe.predict(test[features])
                metrics = evaluate_predictions(test["label"].to_numpy(), pred)
                summaries.append({
                    "analysis": "train_reference_test_other_profile",
                    "model": model,
                    "feature_set": feature_name,
                    "train_profile": REFERENCE_PROFILE,
                    "test_profile": target,
                    **metrics,
                })
                for idx, predicted in zip(test.index, pred):
                    predictions.append({
                        "analysis": "train_reference_test_other_profile",
                        "model": model,
                        "feature_set": feature_name,
                        "train_profile": REFERENCE_PROFILE,
                        "test_profile": target,
                        "row_index": int(idx),
                        "true_label": int(df.loc[idx, "label"]),
                        "pred_label": int(predicted),
                    })
                append_confusion(
                    confusions, analysis="train_reference_test_other_profile", model=model,
                    feature_set=feature_name, train_profile=REFERENCE_PROFILE, test_profile=target,
                    y_true=test["label"].to_numpy(), y_pred=pred,
                )


def leave_one_profile_out(df: pd.DataFrame, summaries: list[dict], predictions: list[dict], confusions: list[dict]) -> None:
    for model, group in df.groupby("model", sort=True):
        for held_out in PROFILES:
            train = group[group["profile"] != held_out]
            test = group[group["profile"] == held_out]
            if train.empty or test.empty or train["label"].nunique() < 2 or test["label"].nunique() < 2:
                continue
            train_profiles = "+".join(sorted(train["profile"].unique()))
            for feature_name, features in FEATURE_SETS.items():
                pipe = classifier().fit(train[features], train["label"])
                pred = pipe.predict(test[features])
                metrics = evaluate_predictions(test["label"].to_numpy(), pred)
                summaries.append({
                    "analysis": "leave_one_profile_out",
                    "model": model,
                    "feature_set": feature_name,
                    "train_profile": train_profiles,
                    "test_profile": held_out,
                    **metrics,
                })
                for idx, predicted in zip(test.index, pred):
                    predictions.append({
                        "analysis": "leave_one_profile_out",
                        "model": model,
                        "feature_set": feature_name,
                        "train_profile": train_profiles,
                        "test_profile": held_out,
                        "row_index": int(idx),
                        "true_label": int(df.loc[idx, "label"]),
                        "pred_label": int(predicted),
                    })
                append_confusion(
                    confusions, analysis="leave_one_profile_out", model=model,
                    feature_set=feature_name, train_profile=train_profiles, test_profile=held_out,
                    y_true=test["label"].to_numpy(), y_pred=pred,
                )


def signal_separation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = ["median_tpot_ms", "median_itl_ms", "median_ttft_ms"]
    for (model, profile, conc), group in df.groupby(["model", "profile", "run_concurrency"], sort=True):
        for metric in metrics:
            low = group.loc[group["label"] == 0, metric].median()
            high = group.loc[group["label"] == 1, metric].median()
            if pd.isna(low) or pd.isna(high) or low <= 0 or high <= 0:
                continue
            rows.append({
                "model": model,
                "profile": profile,
                "concurrency": int(conc),
                "metric": metric,
                "offload0_median": float(low),
                "offload12_median": float(high),
                "ratio_12_over_0": float(high / low),
                "log_ratio": float(math.log(high / low)),
            })
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    for model, model_df in df.groupby("model", sort=True):
        for metric in ["median_tpot_ms", "median_itl_ms"]:
            labels: list[str] = []
            values: list[np.ndarray] = []
            for profile in PROFILES:
                for offload in [0, 12]:
                    subset = model_df[(model_df["profile"] == profile) & (model_df["offload_gb"] == offload)][metric].dropna()
                    if not subset.empty:
                        labels.append(f"{profile}\noffload {offload}")
                        values.append(subset.to_numpy())
            if not values:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(values, showmeans=True)
            ax.set_xticks(range(1, len(labels) + 1), labels)
            ax.set_title(f"{model}: {metric} by request profile and offload state")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / f"{model}_{metric}_profiles.png", dpi=180)
            plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_csv", type=Path, help="Profile-robustness runs_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("profile_analysis"))
    args = parser.parse_args()

    try:
        df = prepare(args.runs_csv)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "analysis_input_filtered.csv", index=False)

    summaries: list[dict] = []
    folds: list[dict] = []
    predictions: list[dict] = []
    confusions: list[dict] = []
    within_profile(df, summaries, folds, predictions, confusions)
    within_profile_leave_one_concurrency_out(df, summaries, folds, predictions, confusions)
    cross_profile(df, summaries, predictions, confusions)
    leave_one_profile_out(df, summaries, predictions, confusions)

    pd.DataFrame(summaries).to_csv(args.output_dir / "classification_summary.csv", index=False)
    pd.DataFrame(folds).to_csv(args.output_dir / "within_profile_folds.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "classification_predictions.csv", index=False)
    pd.DataFrame(confusions).to_csv(args.output_dir / "confusion_matrices.csv", index=False)
    signal_separation(df).to_csv(args.output_dir / "signal_separation.csv", index=False)
    make_plots(df, args.output_dir)

    print(f"Filtered runs: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Profiles: {sorted(df['profile'].unique())}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
infer_offload_holdout.py

Härtere Generalisierungs-Auswertung via leave-one-concurrency-out.
Statt random split: trainiere auf allen Concurrency-Stufen außer einer,
teste auf der zurückgehaltenen Stufe — dann rotieren.

Verwendung:
    python infer_offload_holdout.py runs_summary.csv --mode binary
    python infer_offload_holdout.py runs_summary.csv --mode multiclass
    python infer_offload_holdout.py runs_summary.csv --mode multiclass --use-concurrency-feature
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "median_tpot_ms",
    "median_itl_ms",
    "median_ttft_ms",
]

VALID_OFFLOAD_CLASSES = [0, 2, 4, 8, 12, 16]


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------

def load_data(csv_path: str, use_concurrency_feature: bool) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    features = BASE_FEATURES.copy()
    if use_concurrency_feature:
        features.append("run_concurrency")

    required = ["offload_gb", "run_concurrency"] + BASE_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[FEHLER] Fehlende Spalten: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df[list(set(required + features))].dropna()
    df["offload_gb"] = df["offload_gb"].astype(int)
    df["run_concurrency"] = df["run_concurrency"].astype(int)

    # Nur bekannte Offload-Klassen behalten
    df = df[df["offload_gb"].isin(VALID_OFFLOAD_CLASSES)].reset_index(drop=True)
    return df, features


def make_labels(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "binary":
        return (df["offload_gb"] > 0).astype(int)
    else:
        return df["offload_gb"]


# ---------------------------------------------------------------------------
# Nearest-Centroid (eigene Implementierung, transparent)
# ---------------------------------------------------------------------------

class NearestCentroid:
    def __init__(self):
        self.centroids: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        for cls in np.unique(y):
            self.centroids[cls] = X[y == cls].mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        classes = np.array(list(self.centroids.keys()))
        centers = np.array(list(self.centroids.values()))
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        return classes[np.argmin(dists, axis=1)]


# ---------------------------------------------------------------------------
# Ein einzelner Holdout-Fold
# ---------------------------------------------------------------------------

def run_fold(
    df: pd.DataFrame,
    features: list[str],
    labels: pd.Series,
    held_out_conc: int,
    mode: str,
) -> dict:
    """
    Trainiert auf allen Zeilen mit run_concurrency != held_out_conc,
    testet auf allen Zeilen mit run_concurrency == held_out_conc.
    Gibt ein Dict mit allen Fold-Metriken zurück.
    """
    train_mask = df["run_concurrency"] != held_out_conc
    test_mask  = df["run_concurrency"] == held_out_conc

    X_train = df.loc[train_mask, features].values
    y_train = labels[train_mask].values
    X_test  = df.loc[test_mask,  features].values
    y_test  = labels[test_mask].values

    if len(np.unique(y_train)) < 2:
        print(f"  [WARNUNG] Fold conc={held_out_conc}: Trainingssatz hat nur eine Klasse — überspringe.",
              file=sys.stderr)
        return None

    # Skalierung (für LogReg)
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)

    # --- Nearest-Centroid ---
    nc = NearestCentroid().fit(X_train, y_train)
    y_pred_nc = nc.predict(X_test)

    classes = sorted(np.unique(np.concatenate([y_train, y_test])))

    return {
        "held_out_conc": held_out_conc,
        "n_train":       int(train_mask.sum()),
        "n_test":        int(test_mask.sum()),
        "classes":       classes,
        "y_test":        y_test,
        "y_pred_lr":     y_pred_lr,
        "y_pred_nc":     y_pred_nc,
        "acc_lr":        accuracy_score(y_test, y_pred_lr),
        "bacc_lr":       balanced_accuracy_score(y_test, y_pred_lr),
        "acc_nc":        accuracy_score(y_test, y_pred_nc),
        "bacc_nc":       balanced_accuracy_score(y_test, y_pred_nc),
        "cm_lr":         confusion_matrix(y_test, y_pred_lr, labels=classes),
        "cm_nc":         confusion_matrix(y_test, y_pred_nc, labels=classes),
        "report_lr":     classification_report(y_test, y_pred_lr, labels=classes,
                             target_names=[str(c) for c in classes], digits=4, zero_division=0),
        "report_nc":     classification_report(y_test, y_pred_nc, labels=classes,
                             target_names=[str(c) for c in classes], digits=4, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Ausgabe eines Folds
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def print_fold(fold: dict, mode: str) -> None:
    conc = fold["held_out_conc"]
    section(f"FOLD: Test auf run_concurrency = {conc}")
    print(f"\n  Train-Samples: {fold['n_train']}   Test-Samples: {fold['n_test']}")

    for model_name, y_pred, acc, bacc, cm, report in [
        ("Logistic Regression", fold["y_pred_lr"], fold["acc_lr"],
         fold["bacc_lr"], fold["cm_lr"], fold["report_lr"]),
        ("Nearest-Centroid",    fold["y_pred_nc"], fold["acc_nc"],
         fold["bacc_nc"], fold["cm_nc"], fold["report_nc"]),
    ]:
        print(f"\n  --- {model_name} ---")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {bacc:.4f}")

        classes = fold["classes"]
        if mode == "binary":
            label_names = ["offload=0", "offload>0"]
        else:
            label_names = [f"off={c}" for c in classes]

        # Confusion Matrix
        pad = max(len(n) for n in label_names) + 2
        header = f"  {'':>{pad}}" + "".join(f"  {n:>{pad}}" for n in label_names)
        print(f"\n  Confusion Matrix:")
        print(header)
        for i, row_name in enumerate(label_names):
            row = f"  {row_name:>{pad}}" + "".join(f"  {cm[i,j]:>{pad}}" for j in range(len(classes)))
            print(row)

        print(f"\n{report}")


# ---------------------------------------------------------------------------
# Zusammenfassung und CSV
# ---------------------------------------------------------------------------

def print_summary(folds: list[dict], mode: str) -> None:
    section("ZUSAMMENFASSUNG — Leave-One-Concurrency-Out")

    print(f"\n  {'Held-out conc':>15}  {'LR Acc':>8}  {'LR BAcc':>8}  {'NC Acc':>8}  {'NC BAcc':>8}")
    print(f"  {'-' * 55}")
    for f in folds:
        print(f"  {f['held_out_conc']:>15}  "
              f"{f['acc_lr']:>8.4f}  {f['bacc_lr']:>8.4f}  "
              f"{f['acc_nc']:>8.4f}  {f['bacc_nc']:>8.4f}")

    print(f"  {'-' * 55}")
    print(f"  {'MEAN':>15}  "
          f"{np.mean([f['acc_lr']  for f in folds]):>8.4f}  "
          f"{np.mean([f['bacc_lr'] for f in folds]):>8.4f}  "
          f"{np.mean([f['acc_nc']  for f in folds]):>8.4f}  "
          f"{np.mean([f['bacc_nc'] for f in folds]):>8.4f}")
    print(f"  {'STD':>15}  "
          f"{np.std([f['acc_lr']  for f in folds]):>8.4f}  "
          f"{np.std([f['bacc_lr'] for f in folds]):>8.4f}  "
          f"{np.std([f['acc_nc']  for f in folds]):>8.4f}  "
          f"{np.std([f['bacc_nc'] for f in folds]):>8.4f}")
    print()


def save_csv(folds: list[dict], output_path: Path) -> None:
    rows = []
    for f in folds:
        rows.append({
            "held_out_conc": f["held_out_conc"],
            "n_train":       f["n_train"],
            "n_test":        f["n_test"],
            "acc_lr":        round(f["acc_lr"],  4),
            "bacc_lr":       round(f["bacc_lr"], 4),
            "acc_nc":        round(f["acc_nc"],  4),
            "bacc_nc":       round(f["bacc_nc"], 4),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  [INFO] Fold-Ergebnisse gespeichert: {output_path}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-concurrency-out Generalisierungstest für Offload-Inferenz."
    )
    parser.add_argument("csv_path", help="Pfad zur runs_summary.csv")
    parser.add_argument(
        "--min-concurrency", type=int, default=2,
        help="Kleinste ausgewertete Concurrency-Stufe (Standard: 2)",
    )
    parser.add_argument(
        "--mode", choices=["binary", "multiclass"], default="binary",
        help="binary: offload=0 vs >0 | multiclass: 0/2/4/8/12/16 (Standard: binary)"
    )
    parser.add_argument(
        "--use-concurrency-feature", action="store_true",
        help="run_concurrency als zusätzliches Feature verwenden"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("holdout_results.csv"),
        help="Pfad für CSV-Ausgabe der Fold-Ergebnisse (Standard: holdout_results.csv)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Daten laden ---
    df, features = load_data(args.csv_path, args.use_concurrency_feature)
    df = df[df["run_concurrency"] >= args.min_concurrency].reset_index(drop=True)
    labels = make_labels(df, args.mode)

    concurrencies = sorted(df["run_concurrency"].unique())

    section(f"SETUP")
    print(f"\n  Modus:            {args.mode}")
    print(f"  Features:         {features}")
    print(f"  Concurrency-Werte: {concurrencies}")
    print(f"  Samples gesamt:   {len(df)}")
    print(f"\n  Klassenverteilung:")
    for cls, cnt in labels.value_counts().sort_index().items():
        label = f"offload={cls}" if args.mode == "multiclass" else ("offload=0" if cls == 0 else "offload>0")
        print(f"    {label}: {cnt}")

    # --- Leave-one-concurrency-out ---
    folds = []
    for conc in concurrencies:
        fold = run_fold(df, features, labels, conc, args.mode)
        if fold is not None:
            folds.append(fold)
            print_fold(fold, args.mode)

    if not folds:
        print("[FEHLER] Keine gültigen Folds.", file=sys.stderr)
        sys.exit(1)

    # --- Zusammenfassung ---
    print_summary(folds, args.mode)

    # --- CSV speichern ---
    save_csv(folds, args.output)


if __name__ == "__main__":
    main()

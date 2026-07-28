#!/usr/bin/env python3
"""Analyze the Fix D load-control campaign.

Primary argument: distribution SHAPE (ECDF, p99/median, CV), not just
median level, and not classification accuracy (see delta update #8).
Classification is an optional supplement only.

BLOCKER (delta update #1): if per-token ITL values ('itls' in the
--save-detailed JSON) are missing for any matched main run, this script
aborts with a clear error instead of producing a partial/misleading
analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config_fixd as cfg
import fixd_metrics as fm

CONDITION_ORDER = list(cfg.MAIN_CONDITIONS)


def load_selected(run_root: Path, model: str) -> dict | None:
    path = cfg.selected_load_path(run_root, model)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def gather_main_rows(run_root: Path, models: list[str]) -> tuple[list[dict], dict[tuple, np.ndarray], list[str], list[tuple]]:
    """Returns (per_run_rows, {(model,condition,conc): pooled_itl_array}, blocker_files, unit_sanity_flags)."""
    per_run_rows: list[dict] = []
    pooled: dict[tuple, list[np.ndarray]] = {}
    blocker_files: list[str] = []
    unit_sanity_flags: list[tuple] = []

    for model in models:
        for path in fm.discover_main_runs(run_root, model):
            parsed = fm.parse_main_filename(path.stem)
            if not parsed:
                continue
            data = fm.load_run_json(path)
            if data is None:
                blocker_files.append(f"{path} (unreadable JSON)")
                continue
            values = fm.pooled_itl_values_ms(data)
            if values is None:
                blocker_files.append(f"{path} (no usable 'itls' array)")
                continue

            metrics = fm.shape_metrics(values)
            rl = fm.run_level_fields(data)
            condition = parsed["condition"]
            conc = int(parsed["conc"])
            run_no = int(parsed["run"])

            reported_median_itl = cfg.as_float(rl["median_itl_ms"])
            unit_status, unit_ratio = fm.itl_unit_sanity_check(metrics["median_itl_ms"], reported_median_itl)
            if unit_status == "unit_mismatch_suspected":
                unit_sanity_flags.append((str(path), metrics["median_itl_ms"], reported_median_itl, unit_ratio))

            per_run_rows.append({
                "model": model,
                "condition": condition,
                "concurrency": conc,
                "run_no": run_no,
                "median_itl_ms": metrics["median_itl_ms"],
                "p95_itl_ms": metrics["p95_itl_ms"],
                "p99_itl_ms": metrics["p99_itl_ms"],
                "itl_cv": metrics["itl_cv"],
                "itl_p99_over_median": metrics["itl_p99_over_median"],
                "n_itl_samples": metrics["n_samples"],
                "median_ttft_ms": cfg.as_float(rl["median_ttft_ms"]),
                "median_tpot_ms": cfg.as_float(rl["median_tpot_ms"]),
                "ttft_over_itl_median": (
                    cfg.as_float(rl["median_ttft_ms"]) / metrics["median_itl_ms"]
                    if rl["median_ttft_ms"] and metrics["median_itl_ms"] else None
                ),
                "match_status": rl.get("match_status"),
                "bg_label": rl.get("bg_label"),
                "reported_median_itl_ms": reported_median_itl,
                "itl_unit_sanity_status": unit_status,
                "itl_unit_sanity_ratio": unit_ratio,
                "file": str(path),
            })
            pooled.setdefault((model, condition, conc), []).append(values)

    pooled_flat = {k: np.concatenate(v) for k, v in pooled.items()}
    return per_run_rows, pooled_flat, blocker_files, unit_sanity_flags


def gather_calibration_rows(run_root: Path, models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        for path in fm.discover_calibration_runs(run_root, model):
            parsed = fm.parse_calibration_filename(path.stem)
            if not parsed:
                continue
            data = fm.load_run_json(path)
            if data is None:
                continue
            failed = cfg.as_int(cfg.scalar(data, "failed"))
            values = fm.pooled_itl_values_ms(data)
            row = {
                "model": model,
                "probe_concurrency": int(parsed["conc"]),
                "run_no": int(parsed["run"]),
                "bg_label": parsed["bglabel"],
                "bg_concurrency": cfg.as_int(cfg.scalar(data, "bg_concurrency")),
                "bg_input_len": cfg.as_int(cfg.scalar(data, "bg_input_len")),
                "bg_output_len": cfg.as_int(cfg.scalar(data, "bg_output_len")),
                "failed": failed,
                "valid": failed in (0, None),
                "median_itl_ms": cfg.as_float(cfg.scalar(data, "median_itl_ms")),
                "has_shape_data": values is not None,
            }
            rows.append(row)
    return rows


def aggregate_shape(per_run: pd.DataFrame) -> pd.DataFrame:
    agg = (
        per_run.groupby(["model", "condition", "concurrency"], as_index=False)
        .agg(
            median_itl_ms=("median_itl_ms", "median"),
            p95_itl_ms=("p95_itl_ms", "median"),
            p99_itl_ms=("p99_itl_ms", "median"),
            itl_cv=("itl_cv", "median"),
            itl_p99_over_median=("itl_p99_over_median", "median"),
            median_ttft_ms=("median_ttft_ms", "median"),
            ttft_over_itl_median=("ttft_over_itl_median", "median"),
            n_repeats=("run_no", "count"),
        )
    )
    wide = agg.pivot_table(index=["model", "condition"], columns="concurrency", values="median_itl_ms")
    diffs = []
    for (model, condition), row in wide.iterrows():
        if 4 in row and 8 in row and pd.notna(row[4]) and pd.notna(row[8]):
            diffs.append({"model": model, "condition": condition, "conc8_vs_conc4_difference_ms": row[8] - row[4]})
    diff_df = pd.DataFrame(diffs)
    if not diff_df.empty:
        agg = agg.merge(diff_df, on=["model", "condition"], how="left")
    return agg


def condition_comparison(shape_agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, conc), group in shape_agg.groupby(["model", "concurrency"]):
        medians = dict(zip(group["condition"], group["median_itl_ms"]))
        normal = medians.get("gpu_only_normal")
        offload = medians.get("cpu_offload12")
        loaded = medians.get("gpu_only_loaded")
        row = {"model": model, "concurrency": conc}
        row["ratio_cpu_offload12_over_gpu_only_normal"] = (offload / normal) if normal and offload else None
        row["ratio_gpu_only_loaded_over_gpu_only_normal"] = (loaded / normal) if normal and loaded else None
        row["ratio_gpu_only_loaded_over_cpu_offload12"] = (loaded / offload) if offload and loaded else None
        rows.append(row)
    return pd.DataFrame(rows)


def make_ecdf_plots(pooled: dict[tuple, np.ndarray], models: list[str], out_dir: Path) -> None:
    colors = {"gpu_only_normal": "#4C72B0", "cpu_offload12": "#C44E52", "gpu_only_loaded": "#55A868"}
    for model in models:
        for conc in cfg.PROBE_CONCURRENCIES:
            fig, ax = plt.subplots(figsize=(7, 5))
            any_data = False
            for condition in CONDITION_ORDER:
                values = pooled.get((model, condition, conc))
                if values is None or values.size == 0:
                    continue
                x, y = fm.ecdf_xy(values)
                ax.plot(x, y, label=condition, color=colors.get(condition))
                any_data = True
            if not any_data:
                plt.close(fig)
                continue
            ax.set_xlabel("Inter-token latency (ms)")
            ax.set_ylabel("ECDF")
            ax.set_title(f"{model}: decode ITL ECDF, concurrency={conc}")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"itl_ecdf_{model}_conc{conc}.png", dpi=160)
            plt.close(fig)


def make_summary_barplot(shape_agg: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    for model in models:
        sub = shape_agg[shape_agg["model"] == model]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.35
        concs = cfg.PROBE_CONCURRENCIES
        x = np.arange(len(CONDITION_ORDER))
        for i, conc in enumerate(concs):
            vals = [sub[(sub["condition"] == c) & (sub["concurrency"] == conc)]["median_itl_ms"].mean() for c in CONDITION_ORDER]
            ax.bar(x + (i - 0.5) * width, vals, width, label=f"conc={conc}")
        ax.set_xticks(x, CONDITION_ORDER, rotation=15)
        ax.set_ylabel("median ITL (ms)")
        ax.set_title(f"{model}: median decode ITL by condition")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"itl_median_barplot_{model}.png", dpi=160)
        plt.close(fig)


def write_interpretation_notes(shape_agg: pd.DataFrame, comparison: pd.DataFrame, models: list[str], out_path: Path) -> None:
    lines = ["# Fix D interpretation notes", ""]
    lines.append(
        "This is a small control block, not a new headline result. It asks whether "
        "generic GPU-only background load can reproduce the API-visible decode-timing "
        "regime that CPU offloading (12GB) induces in the profile-robustness block."
    )
    lines.append("")
    for model in models:
        lines.append(f"## {model}")
        for conc in cfg.PROBE_CONCURRENCIES:
            comp = comparison[(comparison["model"] == model) & (comparison["concurrency"] == conc)]
            shape = shape_agg[(shape_agg["model"] == model) & (shape_agg["concurrency"] == conc)]
            if comp.empty or shape.empty:
                lines.append(f"- concurrency={conc}: insufficient data.")
                continue
            ratio_loaded_vs_offload = comp["ratio_gpu_only_loaded_over_cpu_offload12"].iloc[0]
            lines.append(f"- concurrency={conc}:")
            if ratio_loaded_vs_offload is None or pd.isna(ratio_loaded_vs_offload):
                lines.append("  - No gpu_only_loaded data available for this cell.")
                continue
            reached = 0.8 <= ratio_loaded_vs_offload <= 1.25  # informal echo of MATCH_TOLERANCE_RATIO
            if reached:
                normal_row = shape[shape["condition"] == "gpu_only_normal"]
                offload_row = shape[shape["condition"] == "cpu_offload12"]
                loaded_row = shape[shape["condition"] == "gpu_only_loaded"]
                lines.append(
                    "  - gpu_only_loaded reached a comparable median ITL to cpu_offload12 "
                    f"(ratio={ratio_loaded_vs_offload:.2f})."
                )
                if not offload_row.empty and not loaded_row.empty:
                    p99m_off = offload_row["itl_p99_over_median"].iloc[0]
                    p99m_load = loaded_row["itl_p99_over_median"].iloc[0]
                    cv_off = offload_row["itl_cv"].iloc[0]
                    cv_load = loaded_row["itl_cv"].iloc[0]
                    lines.append(f"  - p99/median ITL: cpu_offload12={p99m_off:.2f}, gpu_only_loaded={p99m_load:.2f}")
                    lines.append(f"  - ITL CV: cpu_offload12={cv_off:.2f}, gpu_only_loaded={cv_load:.2f}")
                    shape_close = abs(p99m_off - p99m_load) < 0.15 and abs(cv_off - cv_load) < 0.15
                    verdict = "is NOT clearly distinguishable" if shape_close else "IS distinguishable"
                    lines.append(f"  - Interpretation: the offload-induced regime {verdict} from severity-matched GPU-only load on these shape metrics (see plots for the full ECDF).")
            else:
                lines.append(
                    "  - GPU-only control did NOT reach a comparable median ITL to cpu_offload12 "
                    f"under the tested load grid (ratio={ratio_loaded_vs_offload:.2f})."
                )
                normal_row = shape[shape["condition"] == "gpu_only_normal"]
                loaded_row = shape[shape["condition"] == "gpu_only_loaded"]
                if not normal_row.empty and not loaded_row.empty:
                    ttft_normal = normal_row["median_ttft_ms"].iloc[0]
                    ttft_loaded = loaded_row["median_ttft_ms"].iloc[0]
                    if ttft_normal and ttft_loaded:
                        lines.append(f"  - median TTFT shift under load: {ttft_normal:.1f}ms -> {ttft_loaded:.1f}ms (check whether delay moved into waiting/TTFT/tails instead of decode ITL).")
                lines.append("  - Finding: generic GPU-only load did not reproduce the offload-level decode slowdown under the tested load grid; delay shifted elsewhere or the decode path saturated differently.")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def optional_classification_supplement(per_run: pd.DataFrame, out_path: Path) -> None:
    """Small, explicitly-labeled supplement: binary cpu_offload12 vs
    gpu_only_loaded classifier, leave-one-repeat-out. Not the main argument."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        pd.DataFrame([{"note": "scikit-learn not available; classification supplement skipped"}]).to_csv(out_path, index=False)
        return

    rows = []
    features = ["median_itl_ms", "median_tpot_ms", "median_ttft_ms"]
    for (model, conc), group in per_run.groupby(["model", "concurrency"]):
        sub = group[group["condition"].isin(["cpu_offload12", "gpu_only_loaded"])].dropna(subset=features)
        if sub["condition"].nunique() < 2:
            continue
        sub = sub.copy()
        sub["label"] = (sub["condition"] == "cpu_offload12").astype(int)
        for held_out in sorted(sub["run_no"].unique()):
            train = sub[sub["run_no"] != held_out]
            test = sub[sub["run_no"] == held_out]
            if train["label"].nunique() < 2 or test.empty:
                continue
            pipe = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
            pipe.fit(train[features], train["label"])
            pred = pipe.predict(test[features])
            rows.append({
                "model": model, "concurrency": conc, "held_out_repeat": int(held_out),
                "balanced_accuracy": float(balanced_accuracy_score(test["label"], pred)),
                "n_test": int(len(test)),
                "note": "supplement only; not the primary Fix D argument",
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("fixd_analysis"))
    parser.add_argument("--model", choices=cfg.MODEL_ALIASES, default=None)
    args = parser.parse_args()

    models = [args.model] if args.model else list(cfg.MODEL_ALIASES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows, pooled, blocker_files, unit_sanity_flags = gather_main_rows(args.run_root, models)

    if blocker_files:
        print("BLOCKER: shape metrics / ECDF are not computable for one or more matched main runs.", file=sys.stderr)
        print("Fix D cannot be analyzed as successful until every main run has a usable 'itls' array.", file=sys.stderr)
        for f in blocker_files:
            print(f"  - {f}", file=sys.stderr)
        return 2

    if unit_sanity_flags:
        print("WARNING: ITL unit mismatch suspected on one or more runs (our median(itls)*1000", file=sys.stderr)
        print("vs. vLLM's own median_itl_ms disagree by >200x). Check units before trusting results:", file=sys.stderr)
        for f, computed, reported, ratio in unit_sanity_flags:
            print(f"  - {f}: computed={computed:.3f}ms reported={reported:.3f}ms ratio={ratio:.1f}", file=sys.stderr)

    if not per_run_rows:
        print("ERROR: no Fix D main runs found under", args.run_root, file=sys.stderr)
        return 2

    per_run = pd.DataFrame(per_run_rows)
    per_run.to_csv(args.output_dir / "fixd_main_summary.csv", index=False)

    shape_agg = aggregate_shape(per_run)
    shape_agg.to_csv(args.output_dir / "fixd_shape_metrics.csv", index=False)

    comparison = condition_comparison(shape_agg)
    comparison.to_csv(args.output_dir / "fixd_condition_comparison.csv", index=False)

    calib_rows = gather_calibration_rows(args.run_root, models)
    pd.DataFrame(calib_rows).to_csv(args.output_dir / "fixd_calibration_summary.csv", index=False)

    make_ecdf_plots(pooled, models, args.output_dir)
    make_summary_barplot(shape_agg, models, args.output_dir)
    write_interpretation_notes(shape_agg, comparison, models, args.output_dir / "fixd_interpretation_notes.md")
    optional_classification_supplement(per_run, args.output_dir / "fixd_classification_supplement.csv")

    print(f"Main runs analyzed: {len(per_run)}")
    print(f"Models: {models}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

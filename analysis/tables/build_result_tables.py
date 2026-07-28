#!/usr/bin/env python3
"""Build the two result tables as CSV and LaTeX from authoritative evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import pandas as pd

from figure_common import (
    MODEL_LABELS,
    control_cells,
    find_repo_root,
    fit_table_from_base,
    load_base_runs,
    load_d2_runs,
    load_fixd_main,
    write_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--llama-csv", type=Path)
    parser.add_argument("--qwen-csv", type=Path)
    parser.add_argument("--fixd-csv", type=Path)
    parser.add_argument("--d2-llama-root", type=Path)
    parser.add_argument("--d2-qwen-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def latex_fit_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Parameters of the affine and saturating TPOT models. The affine fit uses $0$--$12\,\mathrm{GiB}$; the global linear and saturating fits use all six levels.}",
        r"    \label{tab:fit_results}",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \begin{tabular}{llrrrrr}",
        r"        \toprule",
        r"        Model & Conc. & Slope [ms/GiB] & $R^2$ affine & $R^2$ global & $G_{\mathrm{sat}}$ [GiB] & $R^2$ saturating \\",
        r"        \midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "        "
            f"{MODEL_LABELS[row['model']]} & {int(row['concurrency'])} & "
            f"{row['affine_slope_ms_per_gib']:.1f} & {row['affine_r2']:.5f} & "
            f"{row['global_linear_r2']:.5f} & {row['g_sat_gib']:.2f} & "
            f"{row['saturating_r2']:.5f} \\\\"
        )
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def make_control_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, concurrency), group in cells.groupby(["model", "concurrency"], sort=True):
        indexed = group.set_index("condition")
        rows.append(
            {
                "model": model,
                "concurrency": int(concurrency),
                "loaded_over_normal_ttft": float(indexed.loc["gpu_only_loaded", "ttft_over_normal"]),
                "loaded_over_normal_itl": float(indexed.loc["gpu_only_loaded", "itl_over_normal"]),
                "kv_over_normal_ttft": float(indexed.loc["kv_vram_pressure", "ttft_over_normal"]),
                "kv_over_normal_itl": float(indexed.loc["kv_vram_pressure", "itl_over_normal"]),
                "offload12_over_normal_ttft": float(
                    indexed.loc["cpu_offload12", "ttft_over_normal"]
                ),
                "offload12_over_normal_itl": float(
                    indexed.loc["cpu_offload12", "itl_over_normal"]
                ),
                "kv_over_offload12_itl_percent": float(
                    indexed.loc["kv_vram_pressure", "itl_over_offload12_percent"]
                ),
            }
        )
    return pd.DataFrame(rows)


def latex_control_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Ratios of the control conditions to the corresponding normal GPU-only baseline. The final column reports the ITL of the KV-/VRAM control as a percentage of the ITL at $12\,\mathrm{GiB}$ offload.}",
        r"    \label{tab:control_ratios}",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{2.1pt}",
        r"    \begin{tabular}{lcrrrrrrr}",
        r"        \toprule",
        r"        Model & Conc. & Load/base TTFT & Load/base ITL & KV/base TTFT & KV/base ITL & Off12/base TTFT & Off12/base ITL & KV/Off12 ITL [\%] \\",
        r"        \midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "        "
            f"{MODEL_LABELS[row['model']]} & {int(row['concurrency'])} & "
            f"{row['loaded_over_normal_ttft']:.2f} & {row['loaded_over_normal_itl']:.2f} & "
            f"{row['kv_over_normal_ttft']:.2f} & {row['kv_over_normal_itl']:.2f} & "
            f"{row['offload12_over_normal_ttft']:.2f} & "
            f"{row['offload12_over_normal_itl']:.2f} & "
            f"{row['kv_over_offload12_itl_percent']:.2f} \\\\"
        )
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(args.repo_root) if args.repo_root else find_repo_root()
    output_dir = args.output_dir or repo_root / "results/tables/paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    base, base_inputs = load_base_runs(repo_root, args.llama_csv, args.qwen_csv)
    fit = fit_table_from_base(base)
    fit.to_csv(output_dir / "table_fit_parameters.csv", index=False)
    (output_dir / "table_fit_parameters.tex").write_text(latex_fit_table(fit), encoding="utf-8")

    fixd, fixd_inputs = load_fixd_main(repo_root, args.fixd_csv)
    d2, d2_inputs = load_d2_runs(repo_root, args.d2_llama_root, args.d2_qwen_root)
    cells = control_cells(fixd, d2)
    controls = make_control_table(cells)
    controls.to_csv(output_dir / "table_control_ratios.csv", index=False)
    (output_dir / "table_control_ratios.tex").write_text(
        latex_control_table(controls), encoding="utf-8"
    )

    write_provenance(
        output_dir,
        "result_tables",
        repo_root,
        base_inputs + fixd_inputs + d2_inputs,
    )
    print(f"PASS: Tabellen unter {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

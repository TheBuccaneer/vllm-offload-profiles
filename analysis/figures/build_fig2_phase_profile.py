#!/usr/bin/env python3
"""Build Figure 2: normalized baseline TTFT and ITL profiles"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from figure_common import (
    MODEL_LABELS,
    MODEL_ORDER,
    PLOT_CONCURRENCIES,
    base_cell_medians,
    configure_matplotlib,
    find_repo_root,
    load_base_runs,
    save_figure,
    write_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--llama-csv", type=Path)
    parser.add_argument("--qwen-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic ratio axis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(args.repo_root) if args.repo_root else find_repo_root()
    output_dir = args.output_dir or repo_root / "results/figures/paper"
    base, inputs = load_base_runs(repo_root, args.llama_csv, args.qwen_csv)
    cells = base_cell_medians(base)

    rows: list[dict] = []
    for (model, concurrency), group in cells[
        cells["run_concurrency"].isin(PLOT_CONCURRENCIES)
    ].groupby(["model", "run_concurrency"], sort=True):
        group = group.sort_values("offload_gb")
        normal = group[group["offload_gb"] == 0].iloc[0]
        for _, row in group.iterrows():
            rows.extend(
                [
                    {
                        "model": model,
                        "concurrency": int(concurrency),
                        "offload_gb": int(row["offload_gb"]),
                        "metric": "TTFT",
                        "cell_median_ms": float(row["median_ttft_ms"]),
                        "ratio_to_offload0": float(row["median_ttft_ms"] / normal["median_ttft_ms"]),
                    },
                    {
                        "model": model,
                        "concurrency": int(concurrency),
                        "offload_gb": int(row["offload_gb"]),
                        "metric": "ITL",
                        "cell_median_ms": float(row["median_itl_ms"]),
                        "ratio_to_offload0": float(row["median_itl_ms"] / normal["median_itl_ms"]),
                    },
                ]
            )
    plot_data = pd.DataFrame(rows)

    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.95), constrained_layout=True, sharey=True)
    for axis, model in zip(axes, MODEL_ORDER):
        for metric, marker in (("TTFT", "o"), ("ITL", "s")):
            for concurrency, linestyle in ((4, "-"), (8, "--")):
                group = plot_data[
                    (plot_data["model"] == model)
                    & (plot_data["metric"] == metric)
                    & (plot_data["concurrency"] == concurrency)
                ].sort_values("offload_gb")
                axis.plot(
                    group["offload_gb"],
                    group["ratio_to_offload0"],
                    marker=marker,
                    linestyle=linestyle,
                    markersize=4,
                    linewidth=1.1,
                    label=f"{metric}, c={concurrency}",
                )
        axis.axhline(1.0, linestyle=":", linewidth=0.8)
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Configured offload budget [GiB]")
        axis.set_xticks([0, 2, 4, 8, 12, 16])
        axis.grid(axis="y", linewidth=0.4, alpha=0.35)
        if args.log_y:
            axis.set_yscale("log")
    axes[0].set_ylabel("Ratio to 0 GiB baseline")
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color="black", label="TTFT"),
        Line2D([], [], marker="s", linestyle="none", color="black", label="ITL"),
        Line2D([], [], linestyle="-", color="black", label="c=4"),
        Line2D([], [], linestyle="--", color="black", label="c=8"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4, frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(output_dir / "fig2_phase_profile_data.csv", index=False)
    save_figure(fig, output_dir, "fig2_phase_profile")
    write_provenance(
        output_dir,
        "fig2_phase_profile",
        repo_root,
        inputs,
        {"log_y": bool(args.log_y)},
    )
    plt.close(fig)
    print(f"PASS: {output_dir / 'fig2_phase_profile.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build Figure 1: TPOT over configured offload budget with both fits."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_common import (
    MODEL_LABELS,
    MODEL_ORDER,
    PLOT_CONCURRENCIES,
    base_cell_medians,
    configure_matplotlib,
    find_repo_root,
    fit_affine_saturating,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(args.repo_root) if args.repo_root else find_repo_root()
    output_dir = args.output_dir or repo_root / "results/figures/paper"
    base, inputs = load_base_runs(repo_root, args.llama_csv, args.qwen_csv)
    cells = base_cell_medians(base)

    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.85), constrained_layout=True)
    plot_rows: list[dict] = []
    fit_rows: list[dict] = []

    for axis, model in zip(axes, MODEL_ORDER):
        model_runs = base[base["model"] == model]
        model_cells = cells[cells["model"] == model]
        for offset, concurrency, marker in ((-0.07, 4, "o"), (0.07, 8, "s")):
            run_group = model_runs[model_runs["run_concurrency"] == concurrency]
            cell_group = model_cells[model_cells["run_concurrency"] == concurrency].sort_values("offload_gb")

            run_artist = axis.scatter(
                run_group["offload_gb"] + offset,
                run_group["median_tpot_ms"],
                marker=marker,
                s=11,
                alpha=0.32,
                linewidths=0,
                label=f"Run medians, c={concurrency}",
            )
            color = run_artist.get_facecolor()[0]
            axis.plot(
                cell_group["offload_gb"] + offset,
                cell_group["median_tpot_ms"],
                marker=marker,
                markersize=4.8,
                linewidth=0,
                color=color,
                label=f"Cell median, c={concurrency}",
            )

            fit = fit_affine_saturating(
                model,
                concurrency,
                cell_group["offload_gb"].to_numpy(),
                cell_group["median_tpot_ms"].to_numpy(),
            )
            fit_rows.append(fit.__dict__)
            x_unsat = np.linspace(0, 12, 160)
            y_unsat = fit.affine_intercept_ms + fit.affine_slope_ms_per_gib * x_unsat
            axis.plot(
                x_unsat,
                y_unsat,
                linestyle="-",
                linewidth=1.15,
                color=color,
                label=f"affine 0–12, c={concurrency}",
            )
            x_all = np.linspace(0, 16, 240)
            y_sat = fit.saturating_intercept_ms + fit.saturating_slope_ms_per_gib * np.minimum(
                x_all, fit.g_sat_gib
            )
            axis.plot(
                x_all,
                y_sat,
                linestyle="--",
                linewidth=1.15,
                color=color,
                label=f"saturating, c={concurrency}",
            )
            axis.axvline(fit.g_sat_gib, linestyle=":", linewidth=0.7, color=color, alpha=0.75)

            for _, row in run_group.iterrows():
                plot_rows.append(
                    {
                        "model": model,
                        "concurrency": concurrency,
                        "offload_gb": int(row["offload_gb"]),
                        "run_id": int(row["run_id"]),
                        "run_median_tpot_ms": float(row["median_tpot_ms"]),
                        "cell_median_tpot_ms": float(
                            cell_group.loc[
                                cell_group["offload_gb"] == row["offload_gb"], "median_tpot_ms"
                            ].iloc[0]
                        ),
                    }
                )

        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Configured offload budget [GiB]")
        axis.set_xticks([0, 2, 4, 8, 12, 16])
        axis.grid(axis="y", linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("TPOT [ms]")

    # Compact legend for a one-column Springer layout. Marker size is explained in the caption.
    data_handles = []
    for concurrency, marker in ((4, "o"), (8, "s")):
        artist = axes[1].collections[0 if concurrency == 4 else 1]
        color = artist.get_facecolor()[0]
        data_handles.append(
            Line2D([], [], marker=marker, linestyle="none", color=color, label=f"c={concurrency}")
        )
    style_handles = [
        Line2D([], [], linestyle="-", color="black", label="Affine fit"),
        Line2D([], [], linestyle="--", color="black", label="Saturating fit"),
    ]
    fig.legend(
        handles=data_handles + style_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plot_rows).to_csv(output_dir / "fig1_offload_fits_data.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "fit_parameters.csv", index=False)
    save_figure(fig, output_dir, "fig1_offload_fits")
    write_provenance(
        output_dir,
        "fig1_offload_fits",
        repo_root,
        inputs,
        {"models": list(MODEL_ORDER), "concurrencies": list(PLOT_CONCURRENCIES)},
    )
    plt.close(fig)
    print(f"PASS: {output_dir / 'fig1_offload_fits.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

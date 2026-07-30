#!/usr/bin/env python3
"""Build Figure 4: combined generic-load and KV/VRAM control profiles"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_common import (
    MODEL_LABELS,
    MODEL_ORDER,
    configure_matplotlib,
    control_cells,
    find_repo_root,
    load_d2_runs,
    load_fixd_main,
    save_figure,
    write_provenance,
)

CONDITIONS = (
    "gpu_only_normal",
    "gpu_only_loaded",
    "kv_vram_pressure",
    "cpu_offload12",
)
CONDITION_LABELS = {
    "gpu_only_normal": "Normal",
    "gpu_only_loaded": "GPU-only load",
    "kv_vram_pressure": "KV/VRAM",
    "cpu_offload12": "Offload 12 GiB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--fixd-csv", type=Path)
    parser.add_argument("--d2-llama-root", type=Path)
    parser.add_argument("--d2-qwen-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear instead of logarithmic ratio axis",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = (
        find_repo_root(args.repo_root)
        if args.repo_root
        else find_repo_root()
    )
    output_dir = (
        args.output_dir
        or repo_root / "results/figures/paper"
    )

    fixd, fixd_inputs = load_fixd_main(
        repo_root,
        args.fixd_csv,
    )
    d2, d2_inputs = load_d2_runs(
        repo_root,
        args.d2_llama_root,
        args.d2_qwen_root,
    )
    cells = control_cells(fixd, d2)

    condition_labels = {
        "gpu_only_normal": "Normal",
        "gpu_only_loaded": "GPU-only\nload",
        "kv_vram_pressure": "KV/VRAM\npressure",
        "cpu_offload12": "Offload\n12 GiB",
    }

    configure_matplotlib()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(5.2, 4.75),
        constrained_layout=True,
        sharex=True,
    )

    x_values = list(range(len(CONDITIONS)))

    for column, model in enumerate(MODEL_ORDER):
        for row, (metric, ratio_column) in enumerate(
            (
                ("TTFT", "ttft_over_normal"),
                ("ITL", "itl_over_normal"),
            )
        ):
            axis = axes[row, column]

            for concurrency, marker, linestyle, offset in (
                (4, "o", "-", -0.045),
                (8, "s", "--", 0.045),
            ):
                group = cells[
                    (cells["model"] == model)
                    & (cells["concurrency"] == concurrency)
                ].set_index("condition")

                values = [
                    float(group.loc[condition, ratio_column])
                    for condition in CONDITIONS
                ]

                shifted_x = [
                    x_value + offset
                    for x_value in x_values
                ]

                axis.plot(
                    shifted_x,
                    values,
                    marker=marker,
                    linestyle=linestyle,
                    markersize=4.5,
                    linewidth=1.0,
                    label=f"c={concurrency}",
                    zorder=3,
                )

            axis.axhline(
                1.0,
                linestyle=":",
                linewidth=0.8,
                zorder=1,
            )

            axis.grid(
                axis="y",
                linewidth=0.4,
                alpha=0.35,
                zorder=0,
            )

            if not args.linear_y:
                axis.set_yscale("log")

            if row == 0:
                axis.set_title(MODEL_LABELS[model])

            if column == 0:
                axis.set_ylabel(
                    f"{metric} / GPU-only baseline"
                )

            axis.set_xticks(x_values)
            axis.margins(x=0.08)

            if row == 1:
                axis.set_xticklabels(
                    [
                        condition_labels[condition]
                        for condition in CONDITIONS
                    ],
                    fontsize=7,
                    linespacing=0.9,
                )
            else:
                axis.tick_params(
                    axis="x",
                    labelbottom=False,
                )

    axes[0, 1].legend(
        frameon=False,
        loc="upper left",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cells.to_csv(
        output_dir / "fig4_control_profiles_data.csv",
        index=False,
    )

    save_figure(
        fig,
        output_dir,
        "fig4_control_profiles",
    )

    write_provenance(
        output_dir,
        "fig4_control_profiles",
        repo_root,
        fixd_inputs + d2_inputs,
        {
            "log_y": not args.linear_y,
            "lines_are_visual_guides": True,
        },
    )

    plt.close(fig)

    print(
        f"PASS: "
        f"{output_dir / 'fig4_control_profiles.pdf'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

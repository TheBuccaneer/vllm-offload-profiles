#!/usr/bin/env python3
"""Build Figure 3: offload12/offload0 robustness across request profiles"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_common import (
    MODEL_LABELS,
    MODEL_ORDER,
    PROFILE_ORDER,
    configure_matplotlib,
    find_repo_root,
    load_profile_runs,
    profile_ratios,
    save_figure,
    write_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--profile-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(args.repo_root) if args.repo_root else find_repo_root()
    output_dir = args.output_dir or repo_root / "results/figures/paper"
    profile, inputs = load_profile_runs(repo_root, args.profile_csv)
    ratios = profile_ratios(profile)
    plotted = ratios[ratios["metric"].isin(["TPOT", "ITL"])].copy()

    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.95), constrained_layout=True, sharey=True)
    x_values = list(range(len(PROFILE_ORDER)))
    for axis, model in zip(axes, MODEL_ORDER):
        for metric, marker in (("TPOT", "o"), ("ITL", "s")):
            for concurrency, linestyle in ((4, "-"), (8, "--")):
                group = plotted[
                    (plotted["model"] == model)
                    & (plotted["metric"] == metric)
                    & (plotted["concurrency"] == concurrency)
                ].sort_values("profile")
                axis.plot(
                    x_values,
                    group["ratio_offload12_over_normal"],
                    marker=marker,
                    linestyle=linestyle,
                    markersize=4.2,
                    linewidth=1.1,
                    label=f"{metric}, c={concurrency}",
                )
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Nominal request profile")
        axis.set_xticks(x_values, PROFILE_ORDER)
        axis.grid(axis="y", linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("12 GiB offload / 0 GiB offload")
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color="black", label="TPOT"),
        Line2D([], [], marker="s", linestyle="none", color="black", label="ITL"),
        Line2D([], [], linestyle="-", color="black", label="c=4"),
        Line2D([], [], linestyle="--", color="black", label="c=8"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4, frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(output_dir / "fig3_profile_robustness_data.csv", index=False)
    save_figure(fig, output_dir, "fig3_profile_robustness")
    write_provenance(output_dir, "fig3_profile_robustness", repo_root, inputs)
    plt.close(fig)
    print(f"PASS: {output_dir / 'fig3_profile_robustness.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

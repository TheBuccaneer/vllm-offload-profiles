#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


GPU_MEMORY_UTILIZATION = {
    "llama": 0.75,
    "qwen": 0.65,
}


def aggregate_condition(
    frame: pd.DataFrame,
    condition: str,
    prefix: str,
) -> pd.DataFrame:
    selected = frame[frame["condition"] == condition]

    result = (
        selected.groupby(["model", "concurrency"], as_index=False)
        .agg(
            median_itl_ms=("median_itl_ms", "median"),
            median_ttft_ms=("median_ttft_ms", "median"),
        )
        .rename(
            columns={
                "median_itl_ms": f"{prefix}_median_itl_ms",
                "median_ttft_ms": f"{prefix}_median_ttft_ms",
            }
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the KV/VRAM pressure control campaign."
    )
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument("gpu_load_summary_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    d2 = pd.read_csv(args.validation_csv)
    control = pd.read_csv(args.gpu_load_summary_csv)

    for column in (
        "concurrency",
        "median_itl_ms",
        "median_ttft_ms",
    ):
        d2[column] = pd.to_numeric(d2[column], errors="raise")
        control[column] = pd.to_numeric(control[column], errors="raise")

    d2_summary = (
        d2.groupby(["model", "concurrency"], as_index=False)
        .agg(
            kv_vram_median_itl_ms=("median_itl_ms", "median"),
            kv_vram_median_ttft_ms=("median_ttft_ms", "median"),
            repeats=("run_no", "nunique"),
        )
    )

    normal = aggregate_condition(
        control,
        "gpu_only_normal",
        "normal",
    )
    offload = aggregate_condition(
        control,
        "cpu_offload12",
        "offload12",
    )

    result = (
        d2_summary
        .merge(normal, on=["model", "concurrency"], validate="one_to_one")
        .merge(offload, on=["model", "concurrency"], validate="one_to_one")
        .sort_values(["model", "concurrency"])
        .reset_index(drop=True)
    )

    result.insert(
        2,
        "gpu_memory_utilization",
        result["model"].map(GPU_MEMORY_UTILIZATION),
    )

    result["itl_ratio_kv_vram_over_normal"] = (
        result["kv_vram_median_itl_ms"]
        / result["normal_median_itl_ms"]
    )
    result["ttft_ratio_kv_vram_over_normal"] = (
        result["kv_vram_median_ttft_ms"]
        / result["normal_median_ttft_ms"]
    )
    result["itl_ratio_kv_vram_over_offload12"] = (
        result["kv_vram_median_itl_ms"]
        / result["offload12_median_itl_ms"]
    )
    result["itl_percent_of_offload12"] = (
        100.0 * result["itl_ratio_kv_vram_over_offload12"]
    )

    if len(result) != 4:
        raise RuntimeError(f"expected 4 aggregate rows, found {len(result)}")
    if not result["repeats"].eq(3).all():
        raise RuntimeError("expected three repeats in every aggregate cell")
    if result.isna().any().any():
        raise RuntimeError("analysis result contains missing values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    display_columns = [
        "model",
        "concurrency",
        "gpu_memory_utilization",
        "kv_vram_median_itl_ms",
        "kv_vram_median_ttft_ms",
        "itl_ratio_kv_vram_over_normal",
        "ttft_ratio_kv_vram_over_normal",
        "itl_percent_of_offload12",
    ]

    print(result[display_columns].to_string(index=False))
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()

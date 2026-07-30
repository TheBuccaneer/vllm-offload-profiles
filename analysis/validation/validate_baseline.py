#!/usr/bin/env python3
"""Validate the complete two-model baseline campaign"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MODELS = ("llama", "qwen")
OFFLOAD_LEVELS = (0, 2, 4, 8, 12, 16)
CONCURRENCIES = (1, 2, 4, 8, 12, 16)
REPEATS = (1, 2, 3, 4, 5)

NAME_PATTERN = re.compile(
    r"(?:qwen_)?offload(?P<offload>\d+)"
    r"_conc(?P<concurrency>\d+)"
    r"_run(?P<run>\d+)\.json$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw and derived baseline data."
    )
    parser.add_argument(
        "raw_root",
        type=Path,
        help="Root directory containing llama/ and qwen/ raw data.",
    )
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("data/derived/baseline"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def as_int(value: object, field: str, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {field}={value!r}") from exc


def validate_model(
    model: str,
    raw_root: Path,
    derived_root: Path,
) -> str:
    model_root = raw_root / model
    files = sorted(model_root.rglob("*.json"))

    if len(files) != 180:
        raise ValueError(
            f"{model}: found {len(files)} raw JSON files instead of 180"
        )

    observed: set[tuple[int, int, int]] = set()

    for path in files:
        match = NAME_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"{model}: unexpected filename: {path}")

        cell = (
            int(match.group("offload")),
            int(match.group("concurrency")),
            int(match.group("run")),
        )

        if cell in observed:
            raise ValueError(f"{model}: duplicate cell/run: {cell}")
        observed.add(cell)

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        completed = as_int(data.get("completed"), "completed", path)
        failed = as_int(data.get("failed"), "failed", path)

        if completed != 20 or failed != 0:
            raise ValueError(
                f"{path}: completed={completed}, failed={failed}"
            )

    expected = {
        (offload, concurrency, run)
        for offload in OFFLOAD_LEVELS
        for concurrency in CONCURRENCIES
        for run in REPEATS
    }

    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"{model}: incomplete matrix; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    runs_path = derived_root / model / "runs_summary.csv"
    requests_path = derived_root / model / "requests_summary.csv"

    if not runs_path.is_file() or not requests_path.is_file():
        raise ValueError(f"{model}: derived CSV files are missing")

    runs = pd.read_csv(runs_path)
    requests = pd.read_csv(requests_path)

    if len(runs) != 180:
        raise ValueError(f"{model}: {len(runs)} derived runs instead of 180")
    if len(requests) != 3600:
        raise ValueError(
            f"{model}: {len(requests)} derived requests instead of 3600"
        )

    required = {
        "offload_gb",
        "run_concurrency",
        "run_id",
        "completed",
        "failed",
    }
    missing_columns = sorted(required - set(runs.columns))
    if missing_columns:
        raise ValueError(
            f"{model}: missing run columns: {missing_columns}"
        )

    counts = runs.groupby(
        ["offload_gb", "run_concurrency"],
        dropna=False,
    ).size()

    if len(counts) != 36 or not (counts == 5).all():
        raise ValueError(f"{model}: derived matrix is not 36 cells x 5")

    if not (pd.to_numeric(runs["completed"]) == 20).all():
        raise ValueError(f"{model}: at least one run has completed != 20")
    if not (pd.to_numeric(runs["failed"]) == 0).all():
        raise ValueError(f"{model}: at least one run has failed != 0")

    return (
        f"PASS {model}: 180 runs, 3600 requests, "
        "36 cells x 5 repeats"
    )


def main() -> int:
    args = parse_args()
    lines: list[str] = []

    try:
        for model in MODELS:
            lines.append(
                validate_model(model, args.raw_root, args.derived_root)
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1

    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

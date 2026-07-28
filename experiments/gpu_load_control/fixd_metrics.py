#!/usr/bin/env python3
"""Shape-metric and file-parsing helpers for analyze_fixd.py.

Split out of analyze_fixd.py to keep both files small. Everything here
operates on already-loaded run dicts (parsed vLLM --save-detailed JSON) or
plain lists of floats — no argparse/CLI here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import config_fixd as cfg

MAIN_NAME_RE = re.compile(
    r"(?P<model>llama|qwen)_fixd_main_(?P<condition>gpu_only_normal|cpu_offload12|gpu_only_loaded)"
    r"_profile-reference_conc(?P<conc>\d+)_run(?P<run>\d+)"
)
CALIB_NAME_RE = re.compile(
    r"(?P<model>llama|qwen)_fixd_calibration_calibration_profile-reference"
    r"_conc(?P<conc>\d+)_run(?P<run>\d+)_bg-(?P<bglabel>[a-z0-9]+)"
)


def parse_main_filename(stem: str) -> dict[str, Any] | None:
    m = MAIN_NAME_RE.fullmatch(stem)
    return m.groupdict() if m else None


def parse_calibration_filename(stem: str) -> dict[str, Any] | None:
    m = CALIB_NAME_RE.fullmatch(stem)
    return m.groupdict() if m else None


def load_run_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pooled_itl_values_ms(data: dict) -> np.ndarray | None:
    """Pool all per-token inter-token-latencies across all requests in a run,
    from the --save-detailed 'itls' field (list of lists, seconds).
    Returns None if the field is missing/empty (this is the Fix D blocker
    condition — shape metrics cannot be computed for this run)."""
    itls = data.get("itls")
    if not isinstance(itls, list) or not itls:
        return None
    flat: list[float] = []
    for per_request in itls:
        if isinstance(per_request, list):
            flat.extend(v for v in per_request if isinstance(v, (int, float)))
    if not flat:
        return None
    return np.asarray(flat, dtype=float) * 1000.0  # s -> ms


def shape_metrics(values_ms: np.ndarray) -> dict[str, float]:
    median = float(np.median(values_ms))
    p95 = float(np.percentile(values_ms, 95))
    p99 = float(np.percentile(values_ms, 99))
    mean = float(np.mean(values_ms))
    std = float(np.std(values_ms))
    return {
        "n_samples": int(values_ms.size),
        "mean_itl_ms": mean,
        "median_itl_ms": median,
        "p95_itl_ms": p95,
        "p99_itl_ms": p99,
        "itl_cv": (std / mean) if mean > 0 else float("nan"),
        "itl_p99_over_median": (p99 / median) if median > 0 else float("nan"),
    }


def ecdf_xy(values_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values_ms)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def run_level_fields(data: dict) -> dict[str, Any]:
    """Fields vLLM bench serve itself reports at the run level (used for
    sanity cross-checks and for TTFT, which is not in `itls`)."""
    out = {}
    for key in (
        "median_itl_ms", "p95_itl_ms", "p99_itl_ms",
        "median_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
        "median_tpot_ms", "failed", "completed", "num_prompts",
        "condition", "model_alias", "concurrency", "run_no",
        "bg_label", "match_status",
    ):
        out[key] = cfg.scalar(data, key)
    return out


def itl_unit_sanity_check(computed_median_ms: float | None, reported_median_ms: float | None) -> tuple[str, float | None]:
    """Compare our own computed median(pooled itls)*1000 against vLLM's own
    reported median_itl_ms for the same run. These are computed by two
    independent code paths (ours from the raw 'itls' array, vLLM's own
    from its internal aggregation), so a large, consistent disagreement is
    a strong signal of a unit bug (seconds vs milliseconds) somewhere,
    rather than of normal measurement noise.

    Returns (status, ratio):
      status is one of 'ok', 'no_reference', 'unit_mismatch_suspected'.
      A roughly 1000x (or 1/1000x) ratio is the seconds-vs-ms signature;
      this uses a looser >200x / <1/200x band so it still catches a
      partially-applied conversion without flagging ordinary jitter.
    """
    if computed_median_ms is None or reported_median_ms is None or reported_median_ms <= 0:
        return "no_reference", None
    ratio = computed_median_ms / reported_median_ms
    if ratio > 200 or ratio < (1.0 / 200):
        return "unit_mismatch_suspected", ratio
    return "ok", ratio


def discover_main_runs(run_root: Path, model: str) -> list[Path]:
    paths: list[Path] = []
    for condition in cfg.MAIN_CONDITIONS:
        d = cfg.main_output_dir(run_root, model, condition)
        if d.is_dir():
            paths.extend(sorted(d.glob(f"{model}_fixd_main_{condition}_*.json")))
    return paths


def discover_calibration_runs(run_root: Path, model: str) -> list[Path]:
    d = cfg.calibration_output_dir(run_root, model)
    if not d.is_dir():
        return []
    return sorted(d.glob(f"{model}_fixd_calibration_calibration_*.json"))

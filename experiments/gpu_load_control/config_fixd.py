#!/usr/bin/env python3
"""Central constants and small path/filename helpers for Fix D (step1d_load_control).

This module is the single documented source of truth for the numeric/naming
conventions used across the Fix D scripts. The bash scripts in this package
duplicate the small subset of these constants they need (same style as the
existing run_server.sh / run_client_profile_robustness.sh scripts) — if you
change something here, update the matching constants at the top of the bash
scripts too. This file itself is only imported by the Python scripts
(select_fixd_load.py, validate_fixd.py, analyze_fixd.py, fixd_metrics.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS: dict[str, dict[str, str]] = {
    "llama": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_tag": "llama31_8b",
    },
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_tag": "qwen25_7b",
    },
}
MODEL_ALIASES = tuple(MODELS.keys())

# ---------------------------------------------------------------------------
# Fixed probe profile (reference only — this is a control block, not a
# re-run of the full profile-robustness grid)
# ---------------------------------------------------------------------------

PROFILE_ID = "reference"
INPUT_LEN = 256
OUTPUT_LEN = 64
NUM_PROMPTS = 20
NUM_WARMUPS = 1
TEMPERATURE = 0
ENDPOINT = "/v1/chat/completions"
CAMPAIGN_VERSION = "gpu_load_control_v1"

PROBE_CONCURRENCIES = (4, 8)
MAIN_REPEATS_DEFAULT = 3
CALIBRATION_REPEATS_DEFAULT = 1

# Seconds the background load must run before a probe starts, and the
# minimum it must keep running for after the probe ends, before it may be
# stopped. Kept inside the 10-30s band required by the Fix D delta update.
BG_WARMUP_SECONDS = 15
BG_COOLDOWN_SECONDS = 5

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

# Only three condition names ever appear in filenames (delta update #2).
# Whether gpu_only_loaded is severity-matched or only max-loaded lives
# exclusively in the JSON metadata (match_status), never in the filename.
MAIN_CONDITIONS = ("gpu_only_normal", "cpu_offload12", "gpu_only_loaded")
CALIBRATION_CONDITION = "calibration"
ALL_CONDITIONS = MAIN_CONDITIONS + (CALIBRATION_CONDITION,)

CONDITION_OFFLOAD_GB = {
    "gpu_only_normal": 0,
    "cpu_offload12": 12,
    "gpu_only_loaded": 0,
    "calibration": 0,
}

MATCH_STATUS_VALUES = ("severity_matched", "not_reached_max_loaded", "insufficient_data")

# ---------------------------------------------------------------------------
# Background load grid (D0 calibration)
# ---------------------------------------------------------------------------

BG_CONCURRENCY_LEVELS = (8, 16, 32, 64)
BG_INPUT_LEN_DEFAULT = 256
BG_OUTPUT_LEN_DEFAULT = 256

# How close (relative) selected_gpu_loaded_median_itl_ms must be to
# target_offload12_median_itl_ms to count as severity_matched.
MATCH_TOLERANCE_RATIO = 0.20

# ---------------------------------------------------------------------------
# Expected run counts (used by validate_fixd.py / README bookkeeping)
# ---------------------------------------------------------------------------

EXPECTED_MAIN_RUNS_PER_MODEL = len(MAIN_CONDITIONS) * len(PROBE_CONCURRENCIES) * MAIN_REPEATS_DEFAULT  # 18
EXPECTED_MAIN_RUNS_TOTAL = EXPECTED_MAIN_RUNS_PER_MODEL * len(MODELS)  # 36

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RUN_ROOT_DEFAULT = "fixd_load_control_runs"


def main_output_dir(run_root: Path, model: str, condition: str) -> Path:
    return Path(run_root) / model / "main" / condition


def calibration_output_dir(run_root: Path, model: str) -> Path:
    return Path(run_root) / model / "calibration" / CALIBRATION_CONDITION


def background_log_dir(run_root: Path, model: str) -> Path:
    return Path(run_root) / model / "background_logs"


def selected_load_path(run_root: Path, model: str) -> Path:
    return Path(run_root) / model / "calibration" / "fixd_selected_load.json"


def calibration_manifest_path(run_root: Path, model: str) -> Path:
    return Path(run_root) / model / "calibration" / "calibration_manifest.tsv"


def main_manifest_path(run_root: Path, model: str) -> Path:
    return Path(run_root) / model / "main" / "main_manifest.tsv"


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------


def bg_label(bg_concurrency: int, bg_input_len: int, bg_output_len: int) -> str:
    return f"c{bg_concurrency}i{bg_input_len}o{bg_output_len}"


def main_filename(model: str, condition: str, conc: int, run_no: int) -> str:
    return f"{model}_fixd_main_{condition}_profile-{PROFILE_ID}_conc{conc}_run{run_no}.json"


def calibration_filename(model: str, conc: int, run_no: int, label: str) -> str:
    return (
        f"{model}_fixd_calibration_{CALIBRATION_CONDITION}_profile-{PROFILE_ID}"
        f"_conc{conc}_run{run_no}_bg-{label}.json"
    )


def background_summary_filename(model: str, phase: str, tag: str, conc: int, run_no: int) -> str:
    """tag is either a main condition name or a calibration bg_label."""
    return f"{model}_bgload_{phase}_{tag}_conc{conc}_run{run_no}_summary.json"


# ---------------------------------------------------------------------------
# Small utilities shared by the Python scripts
# ---------------------------------------------------------------------------


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scalar(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    return None if isinstance(value, (list, dict)) else value

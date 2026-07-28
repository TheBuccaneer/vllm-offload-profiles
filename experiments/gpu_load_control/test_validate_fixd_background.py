#!/usr/bin/env python3
"""Synthetic regression tests for validate_fixd.py's background-overlap check.

Covers two related bugs found during the Fix D audit:

1. The validator must require the background summary for the EXACT probe
   run_no, and must never accept a summary belonging to a different run_no
   as evidence (the old glob-based candidates[-1] fallback did).
2. bg_label (the semantic load level, e.g. c64i256o256) can legitimately
   be IDENTICAL for both probe concurrencies -- this is the expected shape
   of a not_reached_max_loaded result, where both concurrencies land on
   the same max-loaded grid point. The background summary filename must
   still be unique per (probe_concurrency, run_no) in that case, via
   bg_summary_label; the validator must accept both and must not confuse
   one probe concurrency's evidence for the other's.

Run directly:
    python3 test_validate_fixd_background.py

No pytest dependency — plain asserts, prints PASS/FAIL, exits non-zero on
any failure so it can be used as a CI/audit gate.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import config_fixd as cfg

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS = ["llama", "qwen"]
CONDITIONS = list(cfg.MAIN_CONDITIONS)  # gpu_only_normal, cpu_offload12, gpu_only_loaded
CONCS = list(cfg.PROBE_CONCURRENCIES)   # 4, 8
REPEATS = list(range(1, cfg.MAIN_REPEATS_DEFAULT + 1))  # 1, 2, 3

# Realistic D0 selection: the chosen background level generally differs by
# probe concurrency. Used for the "normal" full-grid tests (1-3); test 4/5
# below deliberately use the SAME bg_label/bg_concurrency for both probe
# concurrencies to cover the not_reached_max_loaded case.
BG_CONC_BY_PROBE_CONC = {4: 16, 8: 32}


def bg_label_for(probe_conc: int) -> str:
    return f"c{BG_CONC_BY_PROBE_CONC[probe_conc]}i256o256"


def bg_summary_label_for(bg_label: str, probe_conc: int, run_no: int) -> str:
    return f"{bg_label}_probeconc{probe_conc}_main_run{run_no}"


def make_itls(n_prompts: int = 20, n_tokens: int = 63, base: float = 0.02) -> list[list[float]]:
    return [[base] * n_tokens for _ in range(n_prompts)]


def write_probe(run_root: Path, model: str, condition: str, conc: int, run_no: int,
                 probe_start: str, probe_end: str,
                 bg_label: str | None = None, bg_conc: int | None = None) -> Path:
    out_dir = cfg.main_output_dir(run_root, model, condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "model_alias": model, "condition": condition, "profile_id": "reference",
        "input_len": cfg.INPUT_LEN, "output_len": cfg.OUTPUT_LEN, "concurrency": conc,
        "run_no": run_no, "num_prompts": 20, "completed": 20, "failed": 0,
        "itls": make_itls(), "median_itl_ms": 20.0, "median_ttft_ms": 50.0, "median_tpot_ms": 20.0,
        "probe_start_utc": probe_start, "probe_end_utc": probe_end,
    }
    if condition == "gpu_only_loaded":
        bg_label = bg_label if bg_label is not None else bg_label_for(conc)
        bg_conc = bg_conc if bg_conc is not None else BG_CONC_BY_PROBE_CONC[conc]
        data.update({
            "bg_label": bg_label, "bg_concurrency": bg_conc,
            "bg_summary_label": bg_summary_label_for(bg_label, conc, run_no),
            "bg_input_len": 256, "bg_output_len": 256, "match_status": "severity_matched",
        })
    else:
        data.update({"bg_label": "none", "bg_summary_label": "none", "bg_concurrency": "none",
                     "bg_input_len": "none", "bg_output_len": "none", "match_status": "none"})
    path = out_dir / cfg.main_filename(model, condition, conc, run_no)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_bg_summary(run_root: Path, model: str, bg_summary_label: str, bg_conc: int,
                      bg_start: str, bg_end: str, actual_duration: int = 200) -> Path:
    bg_dir = cfg.background_log_dir(run_root, model)
    bg_dir.mkdir(parents=True, exist_ok=True)
    name = f"{model}_bgload_{bg_summary_label}_conc{bg_conc}_summary.json"
    summary = {
        "label": bg_summary_label, "start_time_utc": bg_start, "end_time_utc": bg_end,
        "actual_duration_seconds": actual_duration, "exit_status": "stopped_by_signal",
        "iterations": 3, "total_completed": 60, "total_failed": 0,
    }
    path = bg_dir / name
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def write_loaded_cell(run_root: Path, model: str, conc: int, run_no: int,
                       bg_label: str, bg_conc: int) -> tuple[Path, Path]:
    """Write one gpu_only_loaded probe plus its exact matching background
    summary, with correctly overlapping timestamps."""
    probe_start = f"2026-07-26T10:{run_no:02d}:05Z"
    probe_end = f"2026-07-26T10:{run_no:02d}:20Z"
    probe_path = write_probe(run_root, model, "gpu_only_loaded", conc, run_no, probe_start, probe_end,
                              bg_label=bg_label, bg_conc=bg_conc)
    bg_start = f"2026-07-26T10:{run_no:02d}:00Z"  # before probe_start
    bg_end = f"2026-07-26T10:{run_no:02d}:30Z"    # after probe_end
    summary_path = write_bg_summary(
        run_root, model, bg_summary_label_for(bg_label, conc, run_no), bg_conc, bg_start, bg_end,
    )
    return probe_path, summary_path


def build_full_grid(run_root: Path) -> None:
    """Full 2 model x 3 condition x 2 conc x 3 run grid, all with correctly
    overlapping background summaries for the gpu_only_loaded cells."""
    for model in MODELS:
        for condition in CONDITIONS:
            for conc in CONCS:
                for run_no in REPEATS:
                    if condition == "gpu_only_loaded":
                        write_loaded_cell(run_root, model, conc, run_no, bg_label_for(conc), BG_CONC_BY_PROBE_CONC[conc])
                    else:
                        probe_start = f"2026-07-26T10:{run_no:02d}:05Z"
                        probe_end = f"2026-07-26T10:{run_no:02d}:20Z"
                        write_probe(run_root, model, condition, conc, run_no, probe_start, probe_end)


def run_validator(run_root: Path, report_path: Path) -> tuple[int, list[dict]]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "validate_fixd.py"), str(run_root), "--output", str(report_path)],
        capture_output=True, text=True,
    )
    rows: list[dict] = []
    if report_path.is_file():
        with report_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    return result.returncode, rows


def codes_for_file(rows: list[dict], filename_fragment: str) -> set[str]:
    return {r["code"] for r in rows if filename_fragment in r["file"] and r["severity"] == "ERROR"}


def test_full_grid_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "fixd_load_control_runs"
        build_full_grid(run_root)
        rc, rows = run_validator(run_root, Path(tmp) / "report.csv")
        errors = [r for r in rows if r["severity"] == "ERROR"]
        assert rc == 0, f"expected exit 0 on full valid grid, got {rc}: {errors}"
        assert not errors, f"expected 0 ERROR rows on full valid grid, got: {errors}"
    print("PASS: full 36-run grid with matching background summaries validates cleanly")


def test_missing_summary_for_one_run_is_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "fixd_load_control_runs"
        build_full_grid(run_root)
        target = (
            cfg.background_log_dir(run_root, "llama")
            / f"llama_bgload_{bg_summary_label_for(bg_label_for(4), 4, 1)}_conc{BG_CONC_BY_PROBE_CONC[4]}_summary.json"
        )
        assert target.is_file(), f"fixture setup problem: expected {target} to exist before deleting it"
        target.unlink()

        rc, rows = run_validator(run_root, Path(tmp) / "report.csv")
        assert rc == 1, f"expected exit 1 when a background summary is missing, got {rc}"
        probe_file_fragment = "llama_fixd_main_gpu_only_loaded_profile-reference_conc4_run1.json"
        codes = codes_for_file(rows, probe_file_fragment)
        assert "BACKGROUND_LOG_MISSING" in codes, f"expected BACKGROUND_LOG_MISSING for {probe_file_fragment}, got {codes}"
    print("PASS: missing background summary for one gpu_only_loaded run is flagged as ERROR")


def test_wrong_run_summary_is_not_accepted_as_fallback() -> None:
    """run1's own summary is missing, but run2/run3 summaries (same
    bg_label/bg_concurrency, different run_no) exist. The validator must
    NOT silently accept those as evidence for run1."""
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "fixd_load_control_runs"
        build_full_grid(run_root)
        bg_dir = cfg.background_log_dir(run_root, "qwen")
        label8, conc8 = bg_label_for(8), BG_CONC_BY_PROBE_CONC[8]
        target = bg_dir / f"qwen_bgload_{bg_summary_label_for(label8, 8, 1)}_conc{conc8}_summary.json"
        assert target.is_file(), f"fixture setup problem: expected {target} to exist before deleting it"
        target.unlink()
        sibling_2 = bg_dir / f"qwen_bgload_{bg_summary_label_for(label8, 8, 2)}_conc{conc8}_summary.json"
        sibling_3 = bg_dir / f"qwen_bgload_{bg_summary_label_for(label8, 8, 3)}_conc{conc8}_summary.json"
        assert sibling_2.is_file() and sibling_3.is_file()

        rc, rows = run_validator(run_root, Path(tmp) / "report.csv")
        assert rc == 1, f"expected exit 1 (old buggy code silently passed here), got {rc}"
        probe_file_fragment = "qwen_fixd_main_gpu_only_loaded_profile-reference_conc8_run1.json"
        codes = codes_for_file(rows, probe_file_fragment)
        assert "BACKGROUND_LOG_MISSING" in codes, (
            f"expected BACKGROUND_LOG_MISSING for {probe_file_fragment} even though sibling "
            f"run summaries exist on disk; got {codes}"
        )
    print("PASS: a sibling run's background summary is never accepted as a fallback for a different run_no")


def test_same_bg_level_different_probe_conc_both_accepted() -> None:
    """not_reached_max_loaded shape: probe_conc=4 and probe_conc=8 both
    select the exact same bg_label/bg_concurrency (e.g. c64i256o256/64).
    bg_summary_label must still disambiguate them; the validator must
    accept both without a BACKGROUND_LOG_MISSING false positive on either."""
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "fixd_load_control_runs"
        shared_bg_label, shared_bg_conc = "c64i256o256", 64
        write_loaded_cell(run_root, "llama", 4, 1, shared_bg_label, shared_bg_conc)
        write_loaded_cell(run_root, "llama", 8, 1, shared_bg_label, shared_bg_conc)

        summary_dir = cfg.background_log_dir(run_root, "llama")
        summaries = sorted(p.name for p in summary_dir.glob("*_summary.json"))
        assert len(summaries) == 2, f"expected 2 distinct summary files despite the shared bg_label, got: {summaries}"

        rc, rows = run_validator(run_root, Path(tmp) / "report.csv")
        conc4_codes = codes_for_file(rows, "llama_fixd_main_gpu_only_loaded_profile-reference_conc4_run1.json")
        conc8_codes = codes_for_file(rows, "llama_fixd_main_gpu_only_loaded_profile-reference_conc8_run1.json")
        assert "BACKGROUND_LOG_MISSING" not in conc4_codes, f"conc4 wrongly flagged missing: {conc4_codes}"
        assert "BACKGROUND_LOG_MISSING" not in conc8_codes, f"conc8 wrongly flagged missing: {conc8_codes}"
    print("PASS: identical bg_label/bg_concurrency for both probe concurrencies is disambiguated by bg_summary_label")


def test_same_bg_level_missing_one_summary_is_error_for_that_conc_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "fixd_load_control_runs"
        shared_bg_label, shared_bg_conc = "c64i256o256", 64
        _, summary_conc4 = write_loaded_cell(run_root, "llama", 4, 1, shared_bg_label, shared_bg_conc)
        write_loaded_cell(run_root, "llama", 8, 1, shared_bg_label, shared_bg_conc)
        summary_conc4.unlink()

        rc, rows = run_validator(run_root, Path(tmp) / "report.csv")
        conc4_codes = codes_for_file(rows, "llama_fixd_main_gpu_only_loaded_profile-reference_conc4_run1.json")
        conc8_codes = codes_for_file(rows, "llama_fixd_main_gpu_only_loaded_profile-reference_conc8_run1.json")
        assert "BACKGROUND_LOG_MISSING" in conc4_codes, f"expected BACKGROUND_LOG_MISSING for conc4, got {conc4_codes}"
        assert "BACKGROUND_LOG_MISSING" not in conc8_codes, (
            f"conc8's own (still-present) summary must not be affected by conc4's missing one; got {conc8_codes}"
        )
    print("PASS: deleting one of two same-bg-level summaries only flags the probe concurrency it belongs to")


def main() -> int:
    tests = [
        test_full_grid_passes,
        test_missing_summary_for_one_run_is_error,
        test_wrong_run_summary_is_not_accepted_as_fallback,
        test_same_bg_level_different_probe_conc_both_accepted,
        test_same_bg_level_missing_one_summary_is_error_for_that_conc_only,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {test.__name__}: {exc}")
    if failures:
        print(f"\n{failures}/{len(tests)} test(s) failed")
        return 1
    print(f"\nAll {len(tests)} background-summary validation tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

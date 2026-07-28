#!/usr/bin/env python3
"""Validate the Fix D load-control campaign (D1 main + D0 calibration)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import config_fixd as cfg
import fixd_metrics as fm


@dataclass
class ValidationRow:
    file: str
    severity: str
    code: str
    message: str


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def validate_main_run(path: Path, parsed: dict, rows: list[ValidationRow]) -> tuple | None:
    data, err = load_json(path)
    if data is None:
        rows.append(ValidationRow(str(path), "ERROR", "INVALID_JSON", err))
        return None

    model = parsed["model"]
    condition = parsed["condition"]
    conc = int(parsed["conc"])
    run_no = int(parsed["run"])

    checks = {
        "model_alias": model,
        "condition": condition,
        "profile_id": "reference",
        "input_len": cfg.INPUT_LEN,
        "output_len": cfg.OUTPUT_LEN,
        "concurrency": conc,
        "run_no": run_no,
    }
    for field, expected in checks.items():
        actual = cfg.scalar(data, field)
        actual_cmp = cfg.as_int(actual) if isinstance(expected, int) else actual
        if actual_cmp != expected:
            rows.append(ValidationRow(str(path), "ERROR", "METADATA_MISMATCH", f"{field}: expected {expected}, got {actual!r}"))

    failed = cfg.as_int(cfg.scalar(data, "failed"))
    completed = cfg.as_int(cfg.scalar(data, "completed"))
    num_prompts = cfg.as_int(cfg.scalar(data, "num_prompts"))
    if failed not in (0, None):
        rows.append(ValidationRow(str(path), "ERROR", "FAILED_REQUESTS", f"failed={failed} (main runs require failed=0)"))
    if completed is not None and num_prompts is not None and completed != num_prompts:
        rows.append(ValidationRow(str(path), "ERROR", "INCOMPLETE_RUN", f"completed={completed}, num_prompts={num_prompts}"))
    if conc not in cfg.PROBE_CONCURRENCIES:
        rows.append(ValidationRow(str(path), "ERROR", "UNEXPECTED_CONCURRENCY", f"concurrency={conc}"))

    itls = data.get("itls")
    if not isinstance(itls, list) or not itls or not any(isinstance(x, list) and x for x in itls):
        rows.append(ValidationRow(str(path), "ERROR", "MISSING_SHAPE_DATA", "No usable 'itls' array; ECDF/shape metrics not computable for this run"))

    if condition == "gpu_only_loaded":
        bg_label = cfg.scalar(data, "bg_label")
        match_status = cfg.scalar(data, "match_status")
        if not bg_label or bg_label == "none":
            rows.append(ValidationRow(str(path), "ERROR", "MISSING_BG_LABEL", "gpu_only_loaded run has no bg_label metadata"))
        if match_status not in cfg.MATCH_STATUS_VALUES:
            rows.append(ValidationRow(str(path), "WARN", "MISSING_MATCH_STATUS", f"match_status={match_status!r}"))

    return (model, condition, conc, run_no)


def check_background_overlap(run_root: Path, model: str, path: Path, data: dict, rows: list[ValidationRow]) -> None:
    """For gpu_only_loaded main runs: the background summary for THIS EXACT
    run_no AND probe_concurrency must exist, and its timestamps must
    actually cover the probe.

    The exact filename is built from bg_summary_label (which folds in
    probe_concurrency and run_no, see run_fixd_main.sh), not from bg_label
    alone -- bg_label is only the semantic load level and can legitimately
    repeat across probe concurrencies (e.g. both concurrencies landing on
    the same not_reached_max_loaded grid point). No glob/candidate
    fallback: a summary belonging to a different run_no or probe
    concurrency must never be accepted as evidence for this run.
    Missing or non-overlapping evidence undermines the severity-match
    claim, so both are ERRORs, not WARNs.
    """
    bg_label = cfg.scalar(data, "bg_label")
    if not bg_label or bg_label == "none":
        return  # already flagged as MISSING_BG_LABEL above

    bg_conc = cfg.as_int(cfg.scalar(data, "bg_concurrency"))
    run_no = cfg.as_int(cfg.scalar(data, "run_no"))
    if bg_conc is None or run_no is None:
        rows.append(ValidationRow(
            str(path), "ERROR", "BACKGROUND_METADATA_MISSING",
            "Probe JSON is missing bg_concurrency or run_no; cannot locate the exact background summary",
        ))
        return

    bg_dir = cfg.background_log_dir(run_root, model)
    bg_summary_label = cfg.scalar(data, "bg_summary_label")
    if bg_summary_label and bg_summary_label != "none":
        expected_name = f"{model}_bgload_{bg_summary_label}_conc{bg_conc}_summary.json"
    else:
        # Legacy naming (pre bg_summary_label): reconstructable only for old
        # data, and only unambiguous if bg_label/bg_concurrency happened to
        # be unique per (probe_conc, run_no) for that model -- which is not
        # guaranteed. New main runs must always carry bg_summary_label; its
        # absence is flagged as an ERROR here rather than silently accepted,
        # even though we still attempt the legacy lookup below so old
        # fixtures/tests keep validating.
        rows.append(ValidationRow(
            str(path), "ERROR", "BG_SUMMARY_LABEL_MISSING",
            "gpu_only_loaded run has no bg_summary_label metadata; falling back to legacy "
            "naming, which cannot guarantee this evidence belongs to this exact run_no+probe_concurrency",
        ))
        expected_name = f"{model}_bgload_{bg_label}_main_run{run_no}_conc{bg_conc}_summary.json"
    summary_path = bg_dir / expected_name
    if not summary_path.is_file():
        rows.append(ValidationRow(
            str(path), "ERROR", "BACKGROUND_LOG_MISSING",
            f"Expected exact background summary not found: {expected_name} "
            f"(a summary for a different run_no or probe concurrency does not count as evidence for this run)",
        ))
        return

    summary, err = load_json(summary_path)
    if summary is None:
        rows.append(ValidationRow(str(path), "ERROR", "BACKGROUND_LOG_UNREADABLE", str(err)))
        return

    exit_status = summary.get("exit_status")
    if exit_status not in ("completed_full_duration", "stopped_by_signal"):
        rows.append(ValidationRow(str(path), "WARN", "BACKGROUND_EXIT_STATUS_UNUSUAL", f"background exit_status={exit_status!r}"))

    actual_dur = summary.get("actual_duration_seconds")
    if isinstance(actual_dur, (int, float)) and actual_dur < (cfg.BG_WARMUP_SECONDS + 5):
        rows.append(ValidationRow(str(path), "ERROR", "BACKGROUND_TOO_SHORT", f"background ran only {actual_dur}s, probe overlap unlikely"))

    bg_start = summary.get("start_time_utc")
    bg_end = summary.get("end_time_utc")
    probe_start = cfg.scalar(data, "probe_start_utc")
    probe_end = cfg.scalar(data, "probe_end_utc")

    if not bg_start or not bg_end:
        rows.append(ValidationRow(str(path), "ERROR", "BACKGROUND_TIMESTAMPS_MISSING", "Background summary is missing start_time_utc/end_time_utc"))
        return
    if not probe_start:
        rows.append(ValidationRow(str(path), "WARN", "PROBE_START_UNVERIFIED", "Probe JSON is missing probe_start_utc; overlap with background not verified"))
        return

    # ISO-8601 'Z' UTC strings of fixed width sort lexicographically in
    # chronological order, so plain string comparison is sufficient here.
    if bg_start > probe_start:
        rows.append(ValidationRow(str(path), "ERROR", "BACKGROUND_STARTED_AFTER_PROBE", f"background start {bg_start} is after probe start {probe_start}"))
    if bg_end < probe_start:
        rows.append(ValidationRow(str(path), "ERROR", "BACKGROUND_ENDED_BEFORE_PROBE_START", f"background end {bg_end} is before probe start {probe_start}"))

    if not probe_end:
        rows.append(ValidationRow(str(path), "WARN", "BACKGROUND_PROBE_END_UNVERIFIED", "Probe JSON is missing probe_end_utc; full-probe-duration overlap not verified"))
    elif bg_end < probe_end:
        rows.append(ValidationRow(
            str(path), "ERROR", "BACKGROUND_ENDED_BEFORE_PROBE_END",
            f"background end {bg_end} is before probe end {probe_end}; background did not cover the full probe duration",
        ))


def validate_calibration_run(path: Path, parsed: dict, rows: list[ValidationRow]) -> None:
    data, err = load_json(path)
    if data is None:
        rows.append(ValidationRow(str(path), "WARN", "INVALID_JSON", str(err)))
        return
    failed = cfg.as_int(cfg.scalar(data, "failed"))
    if failed not in (0, None):
        rows.append(ValidationRow(str(path), "WARN", "CALIBRATION_RUN_FAILED", f"failed={failed} (expected for some load levels; excluded from selection)"))
    if not (data.get("itls")):
        rows.append(ValidationRow(str(path), "WARN", "MISSING_SHAPE_DATA_CALIBRATION", "No 'itls' array in calibration run"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--model", choices=cfg.MODEL_ALIASES, default=None, help="Restrict to one model (18 expected main runs instead of 36)")
    parser.add_argument("--runs", type=int, default=cfg.MAIN_REPEATS_DEFAULT, help="Expected repeats per main cell")
    parser.add_argument("--output", type=Path, default=Path("validation_report.csv"))
    args = parser.parse_args()

    rows: list[ValidationRow] = []
    seen_main: set[tuple] = set()

    all_json = sorted(p for p in args.run_root.rglob("*.json") if "iterations" not in p.parts)
    if not all_json:
        rows.append(ValidationRow(str(args.run_root), "ERROR", "NO_JSON", "No JSON result files found"))

    for path in all_json:
        stem = path.stem
        main_parsed = fm.parse_main_filename(stem)
        calib_parsed = fm.parse_calibration_filename(stem)

        if main_parsed:
            key = validate_main_run(path, main_parsed, rows)
            if key:
                if key in seen_main:
                    rows.append(ValidationRow(str(path), "ERROR", "DUPLICATE_CELL", f"Duplicate cell {key}"))
                seen_main.add(key)
                data, _ = load_json(path)
                if data and key[1] == "gpu_only_loaded":
                    check_background_overlap(args.run_root, key[0], path, data, rows)
        elif calib_parsed:
            validate_calibration_run(path, calib_parsed, rows)
        elif "fixd_selected_load" in stem or "summary" in stem:
            continue  # metadata artifacts, not campaign JSONs
        else:
            rows.append(ValidationRow(str(path), "WARN", "UNRELATED_NAME", "Filename does not match Fix D schema; ignored"))

    models = [args.model] if args.model else list(cfg.MODEL_ALIASES)
    expected = {
        (model, condition, conc, run_no)
        for model in models
        for condition in cfg.MAIN_CONDITIONS
        for conc in cfg.PROBE_CONCURRENCIES
        for run_no in range(1, args.runs + 1)
    }
    for missing in sorted(expected - seen_main):
        rows.append(ValidationRow(str(args.run_root), "ERROR", "MISSING_CELL", str(missing)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "severity", "code", "message"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    errors = sum(r.severity == "ERROR" for r in rows)
    warnings = sum(r.severity == "WARN" for r in rows)
    print(f"Matched main JSONs: {len(seen_main)} (expected {len(expected)})")
    print(f"Errors: {errors}; warnings: {warnings}")
    print(f"Report: {args.output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

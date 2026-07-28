#!/usr/bin/env python3
"""Validate the profile-robustness raw vLLM JSON campaign."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

EXPECTED_PROFILES = {
    "short": (128, 32),
    "reference": (256, 64),
    "long": (512, 128),
}
EXPECTED_CONCURRENCIES = {4, 8}


@dataclass
class ValidationRow:
    file: str
    severity: str
    code: str
    message: str


def scalar(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    return None if isinstance(value, (list, dict)) else value


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_name(stem: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"(?P<model>llama|qwen)_profile-(?P<profile>short|reference|long)_"
        r"offload(?P<offload>\d+)_conc(?P<conc>\d+)_run(?P<run>\d+)",
        stem,
    )
    return match.groupdict() if match else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--runs", type=int, default=3, help="Expected repeats per cell")
    parser.add_argument("--output", type=Path, default=Path("validation_report.csv"))
    args = parser.parse_args()

    rows: list[ValidationRow] = []
    seen: set[tuple[str, int, str, int, int]] = set()
    json_files = sorted(args.run_root.rglob("*.json"))
    if not json_files:
        rows.append(ValidationRow(str(args.run_root), "ERROR", "NO_JSON", "No JSON result files found"))

    for path in json_files:
        parsed = parse_name(path.stem)
        if not parsed:
            # Ignore unrelated JSONs below the selected root, but make it visible.
            rows.append(ValidationRow(str(path), "WARN", "UNRELATED_NAME", "Filename does not match campaign schema; ignored"))
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            rows.append(ValidationRow(str(path), "ERROR", "INVALID_JSON", str(exc)))
            continue

        model = str(parsed["model"])
        profile = str(parsed["profile"])
        offload = int(parsed["offload"])
        conc = int(parsed["conc"])
        run_no = int(parsed["run"])
        key = (model, offload, profile, conc, run_no)
        if key in seen:
            rows.append(ValidationRow(str(path), "ERROR", "DUPLICATE_CELL", f"Duplicate cell {key}"))
        seen.add(key)

        expected_in, expected_out = EXPECTED_PROFILES[profile]
        checks = {
            "offload_gb": offload,
            "concurrency": conc,
            "run_no": run_no,
            "input_len": expected_in,
            "output_len": expected_out,
        }
        for field, expected in checks.items():
            actual = as_int(scalar(data, field))
            if actual != expected:
                rows.append(ValidationRow(
                    str(path), "ERROR", "METADATA_MISMATCH",
                    f"{field}: expected {expected}, got {actual}",
                ))

        actual_profile = scalar(data, "profile_id")
        if actual_profile != profile:
            rows.append(ValidationRow(
                str(path), "ERROR", "PROFILE_MISMATCH",
                f"profile_id: expected {profile}, got {actual_profile!r}",
            ))

        failed = as_int(scalar(data, "failed"))
        completed = as_int(scalar(data, "completed"))
        num_prompts = as_int(scalar(data, "num_prompts"))
        if failed not in (None, 0):
            rows.append(ValidationRow(str(path), "ERROR", "FAILED_REQUESTS", f"failed={failed}"))
        if completed is not None and num_prompts is not None and completed != num_prompts:
            rows.append(ValidationRow(
                str(path), "ERROR", "INCOMPLETE_RUN",
                f"completed={completed}, num_prompts={num_prompts}",
            ))
        if conc not in EXPECTED_CONCURRENCIES:
            rows.append(ValidationRow(str(path), "ERROR", "UNEXPECTED_CONCURRENCY", f"concurrency={conc}"))
        if offload not in {0, 12}:
            rows.append(ValidationRow(str(path), "WARN", "UNEXPECTED_OFFLOAD", f"offload={offload}"))

    models = sorted({key[0] for key in seen})
    offloads = sorted({key[1] for key in seen})
    expected_keys = {
        (model, offload, profile, conc, run_no)
        for model in models
        for offload in offloads
        for profile in EXPECTED_PROFILES
        for conc in EXPECTED_CONCURRENCIES
        for run_no in range(1, args.runs + 1)
    }
    for missing in sorted(expected_keys - seen):
        rows.append(ValidationRow(str(args.run_root), "ERROR", "MISSING_CELL", str(missing)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "severity", "code", "message"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    errors = sum(row.severity == "ERROR" for row in rows)
    warnings = sum(row.severity == "WARN" for row in rows)
    matched = len(seen)
    print(f"Matched campaign JSONs: {matched}")
    print(f"Errors: {errors}; warnings: {warnings}")
    print(f"Report: {args.output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

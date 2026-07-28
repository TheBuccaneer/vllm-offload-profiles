#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

MODELS = {
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "match_status": {
            "d2_kv_pressure_gmem075_main",
            "kv_vram_control_gmem075",
        },
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "match_status": {
            "d2_kv_pressure_gmem065_qwen",
            "kv_vram_control_gmem065",
        },
    },
}

EXPECTED_CELLS = {
    (model, concurrency, run_no)
    for model in MODELS
    for concurrency in (4, 8)
    for run_no in (1, 2, 3)
}


def fail(errors, path, message):
    errors.append(f"{path}: {message}")


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Validate the KV/VRAM pressure control campaign."
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.run_root
    files = sorted(root.glob("*/main/gpu_only_loaded/*.json"))
    summaries = sorted(root.glob("*/background_logs/*_summary.json"))

    errors = []
    rows = []
    seen = set()

    if len(files) != 12:
        errors.append(f"expected 12 probe JSONs, found {len(files)}")
    if len(summaries) != 12:
        errors.append(f"expected 12 background summaries, found {len(summaries)}")

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(errors, path, f"invalid JSON: {exc}")
            continue

        model = data.get("model_alias")
        concurrency = as_int(data.get("concurrency"))
        run_no = as_int(data.get("run_no"))
        cell = (model, concurrency, run_no)

        if model not in MODELS:
            fail(errors, path, f"unexpected model_alias={model!r}")
            continue

        seen.add(cell)

        expected = MODELS[model]
        checks = {
            "model_id": data.get("model_id") == expected["model_id"],
            "condition": data.get("condition") == "gpu_only_loaded",
            "offload_gb_expected": as_int(data.get("offload_gb_expected")) == 0,
            "input_len": as_int(data.get("input_len")) == 256,
            "output_len": as_int(data.get("output_len")) == 64,
            "concurrency": concurrency in (4, 8),
            "run_no": run_no in (1, 2, 3),
            "num_prompts": as_int(data.get("num_prompts")) == 20,
            "completed": as_int(data.get("completed")) == 20,
            "failed": as_int(data.get("failed")) == 0,
            "bg_label": data.get("bg_label") == "c64i1024o512",
            "bg_concurrency": as_int(data.get("bg_concurrency")) == 64,
            "bg_input_len": as_int(data.get("bg_input_len")) == 1024,
            "bg_output_len": as_int(data.get("bg_output_len")) == 512,
            "match_status": data.get("match_status") in expected["match_status"],
        }

        for name, ok in checks.items():
            if not ok:
                fail(errors, path, f"invalid {name}: {data.get(name)!r}")

        label = data.get("bg_summary_label", "")
        summary = (
            root
            / model
            / "background_logs"
            / f"{model}_bgload_{label}_conc64_summary.json"
        )

        if not summary.is_file():
            fail(errors, path, f"missing background summary: {summary}")

        rows.append({
            "model": model,
            "concurrency": concurrency,
            "run_no": run_no,
            "median_ttft_ms": data.get("median_ttft_ms"),
            "median_itl_ms": data.get("median_itl_ms"),
            "completed": as_int(data.get("completed")),
            "failed": as_int(data.get("failed")),
            "probe_file": str(path.relative_to(root)),
            "background_summary": (
                str(summary.relative_to(root)) if summary.is_file() else ""
            ),
        })

    missing = EXPECTED_CELLS - seen
    extra = seen - EXPECTED_CELLS

    if missing:
        errors.append(f"missing cells: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected cells: {sorted(extra)}")
    if len(seen) != len(files):
        errors.append("duplicate model/concurrency/run cells detected")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "model", "concurrency", "run_no",
            "median_ttft_ms", "median_itl_ms",
            "completed", "failed",
            "probe_file",
            "background_summary",
        ]
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(sorted(
                rows,
                key=lambda row: (
                    row["model"],
                    row["concurrency"],
                    row["run_no"],
                ),
            ))

    print(f"Matched probe JSONs: {len(files)} (expected 12)")
    print(f"Matched background summaries: {len(summaries)} (expected 12)")
    print(f"Errors: {len(errors)}; warnings: 0")

    for error in errors:
        print(f"ERROR: {error}")

    if args.output:
        print(f"Report: {args.output}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

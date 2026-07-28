#!/usr/bin/env python3
"""Select the Fix D background-load level for gpu_only_loaded (D0 -> D1).

Reads D0 calibration runs for one model, compares each background-load
level's median decode ITL against a target (either the model's existing
cpu_offload12 main runs, or a manually supplied --target-itl-ms), and
writes fixd_load_control_runs/<model>/calibration/fixd_selected_load.json
with one entry per probe concurrency.

Two valid outcomes per concurrency (Fix D is not a failure either way):
  - severity_matched:       closest level is within MATCH_TOLERANCE_RATIO
                             of the target.
  - not_reached_max_loaded: no level got close; the level with the highest
                             median ITL is selected as the max-loaded control.
  - insufficient_data:      no valid (failed=0) calibration cells at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config_fixd as cfg


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def median_itl_ms(data: dict) -> float | None:
    return cfg.as_float(cfg.scalar(data, "median_itl_ms"))


def collect_calibration_cells(run_root: Path, model: str) -> dict[tuple[int, str], list[dict]]:
    """Returns {(probe_conc, bg_label): [run_json, ...]} for failed=0 runs only."""
    calib_dir = cfg.calibration_output_dir(run_root, model)
    cells: dict[tuple[int, str], list[dict]] = {}
    for path in sorted(calib_dir.glob(f"{model}_fixd_calibration_calibration_*.json")):
        data = load_json(path)
        if not data:
            continue
        failed = cfg.as_int(cfg.scalar(data, "failed"))
        if failed not in (0, None):
            continue  # invalid calibration cell, per delta rule 6/7
        conc = cfg.as_int(cfg.scalar(data, "concurrency"))
        label = cfg.scalar(data, "bg_label")
        if conc is None or not label or label == "none":
            continue
        cells.setdefault((conc, str(label)), []).append(data)
    return cells


def collect_target_from_main(run_root: Path, model: str, conc: int) -> float | None:
    main_dir = cfg.main_output_dir(run_root, model, "cpu_offload12")
    values: list[float] = []
    for path in sorted(main_dir.glob(f"{model}_fixd_main_cpu_offload12_*_conc{conc}_run*.json")):
        data = load_json(path)
        if not data:
            continue
        if cfg.as_int(cfg.scalar(data, "failed")) not in (0, None):
            continue
        m = median_itl_ms(data)
        if m is not None:
            values.append(m)
    if not values:
        return None
    values.sort()
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2.0


def bg_params_from_label(label: str, cells: dict[tuple[int, str], list[dict]], conc: int) -> tuple[int, int, int]:
    for run in cells.get((conc, label), []):
        bg_conc = cfg.as_int(cfg.scalar(run, "bg_concurrency"))
        bg_in = cfg.as_int(cfg.scalar(run, "bg_input_len"))
        bg_out = cfg.as_int(cfg.scalar(run, "bg_output_len"))
        if bg_conc is not None and bg_in is not None and bg_out is not None:
            return bg_conc, bg_in, bg_out
    raise ValueError(f"Could not recover bg params for label={label} conc={conc}")


def select_for_concurrency(
    conc: int, cells: dict[tuple[int, str], list[dict]], target: float | None
) -> dict:
    labels = sorted({label for (c, label) in cells if c == conc})
    level_medians: dict[str, float] = {}
    for label in labels:
        runs = cells[(conc, label)]
        meds = [m for m in (median_itl_ms(r) for r in runs) if m is not None]
        if meds:
            meds.sort()
            n = len(meds)
            level_medians[label] = meds[n // 2] if n % 2 else (meds[n // 2 - 1] + meds[n // 2]) / 2.0

    if not level_medians:
        return {
            "probe_concurrency": conc,
            "match_status": "insufficient_data",
            "reason": "No valid (failed=0) calibration cells found for this concurrency.",
            "target_offload12_median_itl_ms": target,
            "selected_condition_name": None,
        }

    if target is None:
        return {
            "probe_concurrency": conc,
            "match_status": "insufficient_data",
            "reason": (
                "Calibration data exists but no target_offload12_median_itl_ms is available "
                "(no cpu_offload12 main runs found and --target-itl-ms not given)."
            ),
            "target_offload12_median_itl_ms": None,
            "selected_condition_name": None,
            "candidate_levels": level_medians,
        }

    best_label = min(level_medians, key=lambda lbl: abs(level_medians[lbl] - target))
    best_ratio = level_medians[best_label] / target if target > 0 else float("inf")
    within_tolerance = abs(best_ratio - 1.0) <= cfg.MATCH_TOLERANCE_RATIO

    if within_tolerance:
        chosen_label = best_label
        match_status = "severity_matched"
        reason = (
            f"bg level {chosen_label} median ITL {level_medians[chosen_label]:.2f}ms is within "
            f"{cfg.MATCH_TOLERANCE_RATIO * 100:.0f}% of target {target:.2f}ms."
        )
    else:
        chosen_label = max(level_medians, key=lambda lbl: level_medians[lbl])
        match_status = "not_reached_max_loaded"
        reason = (
            f"No bg level reached within {cfg.MATCH_TOLERANCE_RATIO * 100:.0f}% of target "
            f"{target:.2f}ms (closest: {best_label} at {level_medians[best_label]:.2f}ms). "
            f"Selected max-loaded level {chosen_label} ({level_medians[chosen_label]:.2f}ms) as control."
        )

    bg_conc, bg_in, bg_out = bg_params_from_label(chosen_label, cells, conc)
    return {
        "probe_concurrency": conc,
        "match_status": match_status,
        "reason": reason,
        "target_offload12_median_itl_ms": target,
        "selected_bg_label": chosen_label,
        "selected_bg_concurrency": bg_conc,
        "selected_bg_input_len": bg_in,
        "selected_bg_output_len": bg_out,
        "selected_condition_name": "gpu_only_loaded",
        "selected_gpu_loaded_median_itl_ms": level_medians[chosen_label],
        "ratio_to_target": level_medians[chosen_label] / target if target > 0 else None,
        "candidate_levels": level_medians,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=cfg.MODEL_ALIASES)
    parser.add_argument("--run-root", type=Path, default=Path(cfg.RUN_ROOT_DEFAULT))
    parser.add_argument("--concurrency", default="all", help="4, 8, or all (default: all)")
    parser.add_argument(
        "--target-itl-ms",
        type=float,
        default=None,
        help="Manual override target median ITL (ms), e.g. a frozen profile-robustness value.",
    )
    args = parser.parse_args()

    concs = [4, 8] if args.concurrency == "all" else [int(args.concurrency)]
    cells = collect_calibration_cells(args.run_root, args.model)

    results = []
    for conc in concs:
        target = args.target_itl_ms
        if target is None:
            target = collect_target_from_main(args.run_root, args.model, conc)
        results.append(select_for_concurrency(conc, cells, target))

    out = {
        "model": args.model,
        "generated_by": "select_fixd_load.py",
        "match_tolerance_ratio": cfg.MATCH_TOLERANCE_RATIO,
        "per_concurrency": results,
    }
    out_path = cfg.selected_load_path(args.run_root, args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    exit_code = 0
    for r in results:
        print(f"  conc={r['probe_concurrency']}: match_status={r['match_status']}")
        print(f"    {r['reason']}")
        if r["match_status"] == "insufficient_data":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

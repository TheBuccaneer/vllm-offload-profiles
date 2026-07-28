#!/usr/bin/env bash
set -uo pipefail

# run_fixd_main.sh — D1 main-control runs.
#
# Precondition: the correct server for the requested condition is already
# running (see README). This script never starts/stops the vLLM server.
#
#   gpu_only_normal / gpu_only_loaded / ac_alternating  -> requires offload0 server
#   cpu_offload12                                       -> requires offload12 server
#
# Usage:
#   bash run_fixd_main.sh --model llama|qwen \
#     --condition gpu_only_normal|cpu_offload12|gpu_only_loaded|ac_alternating|all \
#     [options]
#
# ac_alternating (recommended for the offload0 session, see delta update #3):
#   runs gpu_only_normal and gpu_only_loaded interleaved per repeat
#   (A1, C1, A2, C2, A3, C3) for each requested concurrency, to minimize
#   session drift between the two conditions. Requires
#   fixd_selected_load.json to already exist (run calibration + selection
#   first).
#
# Options:
#   --concurrency 4|8|all   (default: all)
#   --repeats N              (default: 3)
#   --run-root DIR
#   --base-url URL / --api-key KEY
#   --probe-budget-seconds N  (default: 180; background duration for gpu_only_loaded)
#   --dry-run

MODEL_ALIAS=""
CONDITION=""
CONC_ARG="all"
REPEATS="${REPEATS:-3}"
RUN_ROOT="${RUN_ROOT:-./fixd_load_control_runs}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-pilotkey}"
PROBE_BUDGET_SECONDS="${PROBE_BUDGET_SECONDS:-180}"
WARMUP_SECONDS=15
DRY_RUN=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ALIAS="${2:?}"; shift 2 ;;
    --condition) CONDITION="${2:?}"; shift 2 ;;
    --concurrency) CONC_ARG="${2:?}"; shift 2 ;;
    --repeats) REPEATS="${2:?}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?}"; shift 2 ;;
    --base-url) BASE_URL="${2:?}"; shift 2 ;;
    --api-key) API_KEY="${2:?}"; shift 2 ;;
    --probe-budget-seconds) PROBE_BUDGET_SECONDS="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$MODEL_ALIAS" != "llama" && "$MODEL_ALIAS" != "qwen" ]]; then
  echo "ERROR: --model must be llama or qwen" >&2; exit 2
fi
case "$CONDITION" in
  gpu_only_normal|cpu_offload12|gpu_only_loaded|ac_alternating|all) ;;
  *) echo "ERROR: --condition must be gpu_only_normal|cpu_offload12|gpu_only_loaded|ac_alternating|all" >&2; exit 2 ;;
esac
if [[ "$CONC_ARG" == "all" ]]; then CONCS=(4 8)
elif [[ "$CONC_ARG" == "4" || "$CONC_ARG" == "8" ]]; then CONCS=("$CONC_ARG")
else echo "ERROR: --concurrency must be 4, 8, or all" >&2; exit 2
fi

MANIFEST="$(python3 -c "import config_fixd as c; print(c.main_manifest_path('${RUN_ROOT}', '${MODEL_ALIAS}'))" 2>/dev/null || echo "${RUN_ROOT}/${MODEL_ALIAS}/main/main_manifest.tsv")"
mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'status\tmodel\tcondition\tconcurrency\trun_no\tresult_file\tutc_time\n' > "$MANIFEST"
fi

manifest_line() {
  local status="$1" cond="$2" conc="$3" run_no="$4" file="$5"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$status" "$MODEL_ALIAS" "$cond" "$conc" "$run_no" "$file" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
}

run_unloaded() {
  # gpu_only_normal or cpu_offload12: plain probe, no background load.
  local cond="$1" conc="$2" run_no="$3"
  local outdir="${RUN_ROOT}/${MODEL_ALIAS}/main/${cond}"
  local file="${outdir}/${MODEL_ALIAS}_fixd_main_${cond}_profile-reference_conc${conc}_run${run_no}.json"
  local args=(--model "$MODEL_ALIAS" --condition "$cond" --concurrency "$conc" --run-no "$run_no" \
              --run-root "$RUN_ROOT" --base-url "$BASE_URL" --api-key "$API_KEY")
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  if bash "${SCRIPT_DIR}/run_probe_fixd.sh" "${args[@]}"; then
    manifest_line "ok" "$cond" "$conc" "$run_no" "$file"
  else
    manifest_line "failed" "$cond" "$conc" "$run_no" "$file"
  fi
}

run_loaded() {
  # gpu_only_loaded: start background load from fixd_selected_load.json, warm up, probe, stop.
  local conc="$1" run_no="$2"
  local selected="${RUN_ROOT}/${MODEL_ALIAS}/calibration/fixd_selected_load.json"
  if [[ ! -s "$selected" ]]; then
    echo "ERROR: no fixd_selected_load.json for ${MODEL_ALIAS}. Run calibration + select_fixd_load.py first." >&2
    manifest_line "failed_no_selection" "gpu_only_loaded" "$conc" "$run_no" "NONE"
    return 1
  fi

  read -r bg_label bg_conc bg_in bg_out match_status < <(python3 - "$selected" "$conc" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
conc = int(sys.argv[2])
row = next((r for r in data["per_concurrency"] if r["probe_concurrency"] == conc), None)
if not row or row.get("selected_bg_label") is None:
    print("NONE 0 0 0", row.get("match_status") if row else "insufficient_data")
else:
    print(row["selected_bg_label"], row["selected_bg_concurrency"], row["selected_bg_input_len"],
          row["selected_bg_output_len"], row["match_status"])
PYEOF
  )
  if [[ "$bg_label" == "NONE" ]]; then
    echo "ERROR: no selected load for conc=${conc} (match_status=${match_status}). Skipping." >&2
    manifest_line "failed_no_selection" "gpu_only_loaded" "$conc" "$run_no" "NONE"
    return 1
  fi

  local bg_out_dir="${RUN_ROOT}/${MODEL_ALIAS}/background_logs"
  mkdir -p "$bg_out_dir"
  local bg_duration=$(( WARMUP_SECONDS + PROBE_BUDGET_SECONDS + 5 ))
  local outdir="${RUN_ROOT}/${MODEL_ALIAS}/main/gpu_only_loaded"
  local file="${outdir}/${MODEL_ALIAS}_fixd_main_gpu_only_loaded_profile-reference_conc${conc}_run${run_no}.json"

  # bg_label is the semantic load level selected by D0 (e.g. c64i256o256)
  # and can legitimately be identical for both probe concurrencies (this is
  # the expected shape of a not_reached_max_loaded result, where both
  # concurrencies pick the same max-loaded grid point). bg_summary_label
  # additionally folds in the probe concurrency and run_no, so the
  # background summary filename is always unique per cell even when
  # bg_label repeats. run_probe_fixd.sh stores both: bg_label stays the
  # semantic level for analysis, bg_summary_label is only used to locate
  # the exact background-log evidence for this run.
  local bg_summary_label="${bg_label}_probeconc${conc}_main_run${run_no}"
  local expected_summary="${bg_out_dir}/${MODEL_ALIAS}_bgload_${bg_summary_label}_conc${bg_conc}_summary.json"
  if [[ -f "$expected_summary" && "$DRY_RUN" -ne 1 ]]; then
    # Should not happen (bg_summary_label is already unique per cell) --
    # kept as a defense-in-depth guard against a stray leftover file from a
    # previous, interrupted run of this exact cell.
    echo "ERROR: background summary already exists for this exact cell: ${expected_summary}" >&2
    echo "       Remove it (or the whole run-root cell) before re-running, to avoid mixing" >&2
    echo "       evidence from two different attempts." >&2
    manifest_line "failed_summary_collision" "gpu_only_loaded" "$conc" "$run_no" "$file"
    return 1
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: gpu_only_loaded conc=${conc} run=${run_no} bg=${bg_label} match_status=${match_status}"
    bash "${SCRIPT_DIR}/run_background_gpu_load.sh" --model "$MODEL_ALIAS" --duration-seconds "$bg_duration" \
      --concurrency "$bg_conc" --input-len "$bg_in" --output-len "$bg_out" \
      --output-dir "$bg_out_dir" --label "$bg_summary_label" --base-url "$BASE_URL" --api-key "$API_KEY" --dry-run
    bash "${SCRIPT_DIR}/run_probe_fixd.sh" --model "$MODEL_ALIAS" --condition gpu_only_loaded \
      --concurrency "$conc" --run-no "$run_no" --run-root "$RUN_ROOT" \
      --bg-label "$bg_label" --bg-summary-label "$bg_summary_label" \
      --bg-concurrency "$bg_conc" --bg-input-len "$bg_in" --bg-output-len "$bg_out" \
      --match-status "$match_status" --base-url "$BASE_URL" --api-key "$API_KEY" --dry-run
    return 0
  fi

  bash "${SCRIPT_DIR}/run_background_gpu_load.sh" --model "$MODEL_ALIAS" --duration-seconds "$bg_duration" \
    --concurrency "$bg_conc" --input-len "$bg_in" --output-len "$bg_out" \
    --output-dir "$bg_out_dir" --label "$bg_summary_label" --base-url "$BASE_URL" --api-key "$API_KEY" &
  local bg_pid=$!
  sleep "$WARMUP_SECONDS"
  if ! kill -0 "$bg_pid" 2>/dev/null; then
    echo "ERROR: background load died during warmup for conc=${conc} run=${run_no}." >&2
    wait "$bg_pid" 2>/dev/null || true
    manifest_line "failed_bg_died" "gpu_only_loaded" "$conc" "$run_no" "$file"
    return 1
  fi

  local ok=1
  bash "${SCRIPT_DIR}/run_probe_fixd.sh" --model "$MODEL_ALIAS" --condition gpu_only_loaded \
    --concurrency "$conc" --run-no "$run_no" --run-root "$RUN_ROOT" \
    --bg-label "$bg_label" --bg-summary-label "$bg_summary_label" \
    --bg-concurrency "$bg_conc" --bg-input-len "$bg_in" --bg-output-len "$bg_out" \
    --match-status "$match_status" --base-url "$BASE_URL" --api-key "$API_KEY" || ok=0

  kill -TERM "$bg_pid" 2>/dev/null || true
  wait "$bg_pid" 2>/dev/null || true

  if [[ "$ok" -eq 1 ]]; then
    manifest_line "ok" "gpu_only_loaded" "$conc" "$run_no" "$file"
  else
    manifest_line "failed" "gpu_only_loaded" "$conc" "$run_no" "$file"
  fi
}

case "$CONDITION" in
  gpu_only_normal|cpu_offload12)
    for conc in "${CONCS[@]}"; do
      for run_no in $(seq 1 "$REPEATS"); do
        run_unloaded "$CONDITION" "$conc" "$run_no"
      done
    done
    ;;
  gpu_only_loaded)
    for conc in "${CONCS[@]}"; do
      for run_no in $(seq 1 "$REPEATS"); do
        run_loaded "$conc" "$run_no"
      done
    done
    ;;
  ac_alternating)
    for conc in "${CONCS[@]}"; do
      for run_no in $(seq 1 "$REPEATS"); do
        run_unloaded "gpu_only_normal" "$conc" "$run_no"
        run_loaded "$conc" "$run_no"
      done
    done
    ;;
  all)
    echo "NOTE: --condition all runs all three conditions back to back. This only makes" >&2
    echo "sense if you are re-running everything against a server you swap manually in" >&2
    echo "between; for the offload0 session prefer --condition ac_alternating." >&2
    for conc in "${CONCS[@]}"; do
      for run_no in $(seq 1 "$REPEATS"); do
        run_unloaded "gpu_only_normal" "$conc" "$run_no"
        run_unloaded "cpu_offload12" "$conc" "$run_no"
        run_loaded "$conc" "$run_no"
      done
    done
    ;;
esac

echo "Main-control condition=${CONDITION} model=${MODEL_ALIAS} complete."
echo "Manifest: ${MANIFEST}"

#!/usr/bin/env bash
set -uo pipefail

# run_fixd_calibration.sh — D0 calibration.
#
# Precondition: the matching offload0 server for --model is already
# running (see README). This script never starts/stops the vLLM server.
#
# For every (probe_concurrency, background load level) cell it:
#   1. starts run_background_gpu_load.sh in the background
#   2. sleeps through a warmup window, then checks the background PID is alive
#   3. runs run_probe_fixd.sh --condition calibration (foreground, blocking)
#   4. stops the background load and waits for its summary to be written
#   5. records one line in calibration_manifest.tsv
#
# A failed probe or a background process that dies early only invalidates
# that one load-level cell — calibration continues with the remaining
# cells (see Fix D delta update, point 6).
#
# Usage:
#   bash run_fixd_calibration.sh --model llama|qwen [options]
#
# Options:
#   --concurrency 4|8|all       (default: all)
#   --repeats N                 (default: 1)
#   --bg-concurrencies "8,16,32,64"
#   --bg-input-len N            (default: 256)
#   --bg-output-len N           (default: 256)
#   --run-root DIR               (default: ./fixd_load_control_runs)
#   --base-url URL / --api-key KEY
#   --probe-budget-seconds N     Extra seconds of background runtime reserved
#                                 for the probe itself, beyond warmup+cooldown
#                                 (default: 180 — generous for num_prompts=20)
#   --dry-run

MODEL_ALIAS=""
CONC_ARG="all"
REPEATS="${REPEATS:-1}"
BG_CONCS="${BG_CONCS:-8,16,32,64}"
BG_INPUT_LEN="${BG_INPUT_LEN:-256}"
BG_OUTPUT_LEN="${BG_OUTPUT_LEN:-256}"
RUN_ROOT="${RUN_ROOT:-./fixd_load_control_runs}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-pilotkey}"
PROBE_BUDGET_SECONDS="${PROBE_BUDGET_SECONDS:-180}"
WARMUP_SECONDS=15
COOLDOWN_SECONDS=5
DRY_RUN=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ALIAS="${2:?}"; shift 2 ;;
    --concurrency) CONC_ARG="${2:?}"; shift 2 ;;
    --repeats) REPEATS="${2:?}"; shift 2 ;;
    --bg-concurrencies) BG_CONCS="${2:?}"; shift 2 ;;
    --bg-input-len) BG_INPUT_LEN="${2:?}"; shift 2 ;;
    --bg-output-len) BG_OUTPUT_LEN="${2:?}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?}"; shift 2 ;;
    --base-url) BASE_URL="${2:?}"; shift 2 ;;
    --api-key) API_KEY="${2:?}"; shift 2 ;;
    --probe-budget-seconds) PROBE_BUDGET_SECONDS="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$MODEL_ALIAS" != "llama" && "$MODEL_ALIAS" != "qwen" ]]; then
  echo "ERROR: --model must be llama or qwen" >&2; exit 2
fi
if [[ "$CONC_ARG" == "all" ]]; then
  CONCS=(4 8)
elif [[ "$CONC_ARG" == "4" || "$CONC_ARG" == "8" ]]; then
  CONCS=("$CONC_ARG")
else
  echo "ERROR: --concurrency must be 4, 8, or all" >&2; exit 2
fi
IFS=',' read -r -a BG_CONC_LIST <<< "$BG_CONCS"

OUT_DIR="${RUN_ROOT}/${MODEL_ALIAS}/calibration/calibration"
BG_LOG_DIR="${RUN_ROOT}/${MODEL_ALIAS}/background_logs"
mkdir -p "$OUT_DIR" "$BG_LOG_DIR"
MANIFEST="${RUN_ROOT}/${MODEL_ALIAS}/calibration/calibration_manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'status\tmodel\tprobe_conc\tbg_label\tbg_conc\tbg_input_len\tbg_output_len\trun_no\tprobe_file\tbg_summary_file\tprobe_failed\tbg_exit_status\tutc_time\n' > "$MANIFEST"
fi

BG_DURATION=$(( WARMUP_SECONDS + PROBE_BUDGET_SECONDS + COOLDOWN_SECONDS ))

run_cell() {
  local probe_conc="$1" bg_conc="$2" run_no="$3"
  local label; label="c${bg_conc}i${BG_INPUT_LEN}o${BG_OUTPUT_LEN}"
  # bgload_label disambiguates the background summary/iteration filenames by
  # probe concurrency and repeat, so different calibration cells at the same
  # bg level never overwrite each other's logs (nothing downstream parses
  # this label's structure -- select_fixd_load.py reads bg params from the
  # probe JSON's own metadata, not from the background summary filename).
  local bgload_label="${label}_probeconc${probe_conc}_run${run_no}"
  local bg_out="${BG_LOG_DIR}"
  local now; now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "--- calibration cell: model=${MODEL_ALIAS} probe_conc=${probe_conc} bg=${label} run=${run_no} ---"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: bg for ${BG_DURATION}s, sleep ${WARMUP_SECONDS}s, probe conc=${probe_conc} bg-label=${label} run=${run_no}"
    bash "${SCRIPT_DIR}/run_background_gpu_load.sh" --model "$MODEL_ALIAS" --duration-seconds "$BG_DURATION" \
      --concurrency "$bg_conc" --input-len "$BG_INPUT_LEN" --output-len "$BG_OUTPUT_LEN" \
      --output-dir "$bg_out" --label "$bgload_label" --base-url "$BASE_URL" --api-key "$API_KEY" --dry-run
    bash "${SCRIPT_DIR}/run_probe_fixd.sh" --model "$MODEL_ALIAS" --condition calibration \
      --concurrency "$probe_conc" --run-no "$run_no" --run-root "$RUN_ROOT" \
      --bg-label "$label" --bg-summary-label "$bgload_label" --bg-concurrency "$bg_conc" --bg-input-len "$BG_INPUT_LEN" --bg-output-len "$BG_OUTPUT_LEN" \
      --base-url "$BASE_URL" --api-key "$API_KEY" --dry-run
    return 0
  fi

  bash "${SCRIPT_DIR}/run_background_gpu_load.sh" --model "$MODEL_ALIAS" --duration-seconds "$BG_DURATION" \
    --concurrency "$bg_conc" --input-len "$BG_INPUT_LEN" --output-len "$BG_OUTPUT_LEN" \
    --output-dir "$bg_out" --label "$bgload_label" --base-url "$BASE_URL" --api-key "$API_KEY" &
  local bg_pid=$!

  sleep "$WARMUP_SECONDS"
  if ! kill -0 "$bg_pid" 2>/dev/null; then
    echo "WARN: background load (pid ${bg_pid}) died during warmup — this cell is invalid, skipping probe" >&2
    wait "$bg_pid" 2>/dev/null || true
    printf 'invalid\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$MODEL_ALIAS" "$probe_conc" "$label" "$bg_conc" "$BG_INPUT_LEN" "$BG_OUTPUT_LEN" "$run_no" \
      "NONE" "NONE" "n/a" "died_before_warmup" "$now" >> "$MANIFEST"
    return 0
  fi

  local probe_ok=1
  bash "${SCRIPT_DIR}/run_probe_fixd.sh" --model "$MODEL_ALIAS" --condition calibration \
    --concurrency "$probe_conc" --run-no "$run_no" --run-root "$RUN_ROOT" \
    --bg-label "$label" --bg-summary-label "$bgload_label" --bg-concurrency "$bg_conc" --bg-input-len "$BG_INPUT_LEN" --bg-output-len "$BG_OUTPUT_LEN" \
    --base-url "$BASE_URL" --api-key "$API_KEY" || probe_ok=0

  kill -TERM "$bg_pid" 2>/dev/null || true
  wait "$bg_pid" 2>/dev/null || true

  local probe_file="${OUT_DIR}/${MODEL_ALIAS}_fixd_calibration_calibration_profile-reference_conc${probe_conc}_run${run_no}_bg-${label}.json"
  local bg_summary="${bg_out}/${MODEL_ALIAS}_bgload_${bgload_label}_conc${bg_conc}_summary.json"
  local probe_failed="unknown" bg_exit="unknown"
  if [[ -s "$probe_file" ]]; then
    probe_failed=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get('failed') or 0))" "$probe_file" 2>/dev/null || echo "unknown")
  fi
  if [[ -s "$bg_summary" ]]; then
    bg_exit=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('exit_status'))" "$bg_summary" 2>/dev/null || echo "unknown")
  fi

  local status="ok"
  [[ "$probe_ok" -eq 0 || "$probe_failed" != "0" ]] && status="invalid"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$status" "$MODEL_ALIAS" "$probe_conc" "$label" "$bg_conc" "$BG_INPUT_LEN" "$BG_OUTPUT_LEN" "$run_no" \
    "$probe_file" "$bg_summary" "$probe_failed" "$bg_exit" "$now" >> "$MANIFEST"
  echo "--- cell done: status=${status} probe_failed=${probe_failed} bg_exit=${bg_exit} ---"
}

echo "Calibration output dir: $OUT_DIR"
echo "Calibration manifest:   $MANIFEST"
echo "Background load grid: concurrencies=[${BG_CONC_LIST[*]}] input_len=${BG_INPUT_LEN} output_len=${BG_OUTPUT_LEN}"
echo

for probe_conc in "${CONCS[@]}"; do
  for bg_conc in "${BG_CONC_LIST[@]}"; do
    for run_no in $(seq 1 "$REPEATS"); do
      run_cell "$probe_conc" "$bg_conc" "$run_no"
    done
  done
done

echo "Calibration complete for model=${MODEL_ALIAS}."
echo "Next: python3 select_fixd_load.py --model ${MODEL_ALIAS} --run-root ${RUN_ROOT} [--target-itl-ms N]"

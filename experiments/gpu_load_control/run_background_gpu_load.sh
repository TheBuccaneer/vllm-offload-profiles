#!/usr/bin/env bash
set -uo pipefail
# Deliberately no -e: a single failed iteration must not silently kill the
# trap/summary logic below.

# run_background_gpu_load.sh — generate stationary GPU-only background load
# against an already-running vLLM server, using vLLM's own bench client
# (no external stress tool). Intended to be launched in the background
# (`... &`) by run_fixd_calibration.sh / run_fixd_main.sh, which then sleep
# through a warmup window, run a foreground probe, and stop this process.
#
# Usage:
#   bash run_background_gpu_load.sh --model llama|qwen --duration-seconds 240 \
#     --concurrency 16 --input-len 256 --output-len 256 \
#     --output-dir DIR --label c16i256o256 [--dry-run]
#
# Behavior:
#   Runs vLLM bench serve repeatedly, back-to-back with no sleep between
#   iterations, until --duration-seconds has elapsed or it receives
#   SIGTERM/SIGINT. Each iteration writes its own JSON under
#   <output-dir>/iterations/. On exit (normal or signalled) it always
#   writes one summary JSON with PID, start/end time, elapsed seconds,
#   iteration count, completed/failed totals, and exit_status.

MODEL_ALIAS=""
DURATION_SECONDS=""
CONC=""
INPUT_LEN=""
OUTPUT_LEN=""
OUTPUT_DIR=""
LABEL=""
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-pilotkey}"
VLLM_BIN="${VLLM_BIN:-vllm}"
DRY_RUN=0
# Prompts per iteration: kept modest so iterations are short relative to
# duration and the loop can react to SIGTERM promptly.
ITER_NUM_PROMPTS="${ITER_NUM_PROMPTS:-40}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ALIAS="${2:?}"; shift 2 ;;
    --duration-seconds) DURATION_SECONDS="${2:?}"; shift 2 ;;
    --concurrency) CONC="${2:?}"; shift 2 ;;
    --input-len) INPUT_LEN="${2:?}"; shift 2 ;;
    --output-len) OUTPUT_LEN="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --label) LABEL="${2:?}"; shift 2 ;;
    --base-url) BASE_URL="${2:?}"; shift 2 ;;
    --api-key) API_KEY="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for req in MODEL_ALIAS DURATION_SECONDS CONC INPUT_LEN OUTPUT_LEN OUTPUT_DIR LABEL; do
  if [[ -z "${!req}" ]]; then
    echo "ERROR: --${req,,} is required" >&2; exit 2
  fi
done
if [[ "$MODEL_ALIAS" != "llama" && "$MODEL_ALIAS" != "qwen" ]]; then
  echo "ERROR: --model must be llama or qwen" >&2; exit 2
fi

case "$MODEL_ALIAS" in
  llama) MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
  qwen)  MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
esac

ITER_DIR="${OUTPUT_DIR}/iterations"
mkdir -p "$ITER_DIR"
SUMMARY_FILE="${OUTPUT_DIR}/${MODEL_ALIAS}_bgload_${LABEL}_conc${CONC}_summary.json"

export OPENAI_API_KEY="$API_KEY"

PID=$$
START_EPOCH=$(date +%s)
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ITERATIONS=0
TOTAL_COMPLETED=0
TOTAL_FAILED=0
EXIT_STATUS="running"
STOP_REQUESTED=0

write_summary() {
  local end_epoch end_utc elapsed
  end_epoch=$(date +%s)
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  elapsed=$(( end_epoch - START_EPOCH ))
  cat > "$SUMMARY_FILE" <<EOF
{
  "label": "${LABEL}",
  "model_alias": "${MODEL_ALIAS}",
  "model_name": "${MODEL}",
  "bg_concurrency": ${CONC},
  "bg_input_len": ${INPUT_LEN},
  "bg_output_len": ${OUTPUT_LEN},
  "pid": ${PID},
  "requested_duration_seconds": ${DURATION_SECONDS},
  "actual_duration_seconds": ${elapsed},
  "start_time_utc": "${START_UTC}",
  "end_time_utc": "${end_utc}",
  "iterations": ${ITERATIONS},
  "total_completed": ${TOTAL_COMPLETED},
  "total_failed": ${TOTAL_FAILED},
  "exit_status": "${EXIT_STATUS}"
}
EOF
  echo "[bgload] summary written: ${SUMMARY_FILE} (${EXIT_STATUS}, ${ITERATIONS} iterations, elapsed=${elapsed}s)" >&2
}

on_signal() {
  STOP_REQUESTED=1
  EXIT_STATUS="stopped_by_signal"
}
trap on_signal SIGTERM SIGINT
trap 'if [[ "$EXIT_STATUS" == "running" ]]; then EXIT_STATUS="stopped_early"; fi; write_summary' EXIT

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would loop 'vllm bench serve' for ~${DURATION_SECONDS}s (model=${MODEL_ALIAS} conc=${CONC} in=${INPUT_LEN} out=${OUTPUT_LEN} label=${LABEL})"
  EXIT_STATUS="dry_run"
  exit 0
fi

echo "[bgload] starting: model=${MODEL_ALIAS} conc=${CONC} in=${INPUT_LEN} out=${OUTPUT_LEN} duration=${DURATION_SECONDS}s pid=${PID}" >&2

while true; do
  now=$(date +%s)
  elapsed=$(( now - START_EPOCH ))
  if [[ "$STOP_REQUESTED" -eq 1 || "$elapsed" -ge "$DURATION_SECONDS" ]]; then
    break
  fi

  ITERATIONS=$((ITERATIONS + 1))
  iter_file="${ITER_DIR}/${MODEL_ALIAS}_bgload_${LABEL}_conc${CONC}_iter${ITERATIONS}.json"

  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --num-prompts "$ITER_NUM_PROMPTS" \
    --num-warmups 0 \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --max-concurrency "$CONC" \
    --temperature 0 \
    --save-result \
    --result-dir "$ITER_DIR" \
    --result-filename "$(basename "$iter_file")" \
    --metadata "role=background_load" "label=${LABEL}" "iteration=${ITERATIONS}" \
    >>"${OUTPUT_DIR}/${MODEL_ALIAS}_bgload_${LABEL}_conc${CONC}.log" 2>&1
  rc=$?

  if [[ -s "$iter_file" ]]; then
    comp=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get('completed') or 0))" "$iter_file" 2>/dev/null || echo 0)
    fail=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get('failed') or 0))" "$iter_file" 2>/dev/null || echo 0)
    TOTAL_COMPLETED=$((TOTAL_COMPLETED + comp))
    TOTAL_FAILED=$((TOTAL_FAILED + fail))
  fi

  if [[ "$rc" -ne 0 ]]; then
    echo "[bgload] iteration ${ITERATIONS} exited with code ${rc}" >&2
    EXIT_STATUS="failed"
    break
  fi
done

if [[ "$EXIT_STATUS" == "running" ]]; then
  EXIT_STATUS="completed_full_duration"
fi

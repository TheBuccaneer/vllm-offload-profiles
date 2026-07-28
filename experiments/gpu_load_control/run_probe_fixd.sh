#!/usr/bin/env bash
set -euo pipefail

# run_probe_fixd.sh — run one reference-profile probe block against an
# already-running vLLM server, for one Fix D condition/concurrency/repeat.
#
# This script only runs the probe. It never starts or stops the vLLM
# server, and it never starts background load itself — callers
# (run_fixd_calibration.sh, run_fixd_main.sh) are responsible for starting
# background load first and passing --bg-label/--match-status through.
#
# Usage:
#   bash run_probe_fixd.sh --model llama|qwen \
#     --condition gpu_only_normal|cpu_offload12|gpu_only_loaded|calibration \
#     --concurrency 4|8 --run-no N [options]
#
# Options:
#   --run-root DIR        Root for fixd_load_control_runs (default: ./fixd_load_control_runs)
#   --output-dir DIR      Override the computed output directory
#   --base-url URL        vLLM base URL (default: http://127.0.0.1:8000)
#   --api-key KEY         API key (default: pilotkey)
#   --bg-label LABEL       Semantic background load level, e.g. c16i256o256 (calibration/gpu_only_loaded only)
#   --bg-summary-label LBL  Filename-disambiguating label for the background summary
#                            (e.g. c16i256o256_probeconc4_main_run1); falls back to
#                            --bg-label if omitted (calibration only -- main callers
#                            must always pass this)
#   --bg-concurrency N     Background concurrency, for metadata only
#   --bg-input-len N       Background input length, for metadata only
#   --bg-output-len N      Background output length, for metadata only
#   --match-status STATUS  severity_matched|not_reached_max_loaded|insufficient_data (gpu_only_loaded only)
#   --session-id ID         Free-form session tag, for metadata only
#   --dry-run
#   -h, --help

usage() { sed -n '2,30p' "$0"; }

MODEL_ALIAS=""
CONDITION=""
CONC=""
RUN_NO=""
RUN_ROOT="${RUN_ROOT:-./fixd_load_control_runs}"
OUTPUT_DIR=""
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-pilotkey}"
BG_LABEL=""
BG_SUMMARY_LABEL=""
BG_CONCURRENCY=""
BG_INPUT_LEN=""
BG_OUTPUT_LEN=""
MATCH_STATUS=""
SESSION_ID="${SESSION_ID:-}"
VLLM_BIN="${VLLM_BIN:-vllm}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ALIAS="${2:?}"; shift 2 ;;
    --condition) CONDITION="${2:?}"; shift 2 ;;
    --concurrency) CONC="${2:?}"; shift 2 ;;
    --run-no) RUN_NO="${2:?}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --base-url) BASE_URL="${2:?}"; shift 2 ;;
    --api-key) API_KEY="${2:?}"; shift 2 ;;
    --bg-label) BG_LABEL="${2:?}"; shift 2 ;;
    --bg-summary-label) BG_SUMMARY_LABEL="${2:?}"; shift 2 ;;
    --bg-concurrency) BG_CONCURRENCY="${2:?}"; shift 2 ;;
    --bg-input-len) BG_INPUT_LEN="${2:?}"; shift 2 ;;
    --bg-output-len) BG_OUTPUT_LEN="${2:?}"; shift 2 ;;
    --match-status) MATCH_STATUS="${2:?}"; shift 2 ;;
    --session-id) SESSION_ID="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$MODEL_ALIAS" != "llama" && "$MODEL_ALIAS" != "qwen" ]]; then
  echo "ERROR: --model must be llama or qwen" >&2; exit 2
fi
case "$CONDITION" in
  gpu_only_normal|cpu_offload12|gpu_only_loaded|calibration) ;;
  *) echo "ERROR: --condition must be gpu_only_normal|cpu_offload12|gpu_only_loaded|calibration" >&2; exit 2 ;;
esac
if [[ "$CONC" != "4" && "$CONC" != "8" ]]; then
  echo "ERROR: --concurrency must be 4 or 8" >&2; exit 2
fi
if ! [[ "$RUN_NO" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --run-no must be a positive integer" >&2; exit 2
fi

case "$MODEL_ALIAS" in
  llama) MODEL="meta-llama/Llama-3.1-8B-Instruct"; MODEL_TAG="llama31_8b" ;;
  qwen)  MODEL="Qwen/Qwen2.5-7B-Instruct";          MODEL_TAG="qwen25_7b" ;;
esac

PROFILE_ID="reference"
INPUT_LEN="256"
OUTPUT_LEN="64"
NUM_PROMPTS="20"
NUM_WARMUPS="1"
TEMPERATURE="0"
ENDPOINT="/v1/chat/completions"
CAMPAIGN_VERSION="${CAMPAIGN_VERSION:-gpu_load_control_v1}"

case "$CONDITION" in
  gpu_only_normal|gpu_only_loaded) OFFLOAD_GB_EXPECTED=0 ;;
  cpu_offload12) OFFLOAD_GB_EXPECTED=12 ;;
  calibration) OFFLOAD_GB_EXPECTED=0 ;;
esac

if [[ "$CONDITION" == "calibration" ]]; then
  PHASE="calibration"
else
  PHASE="main"
fi

if [[ "$CONDITION" == "calibration" && -z "$BG_LABEL" ]]; then
  echo "ERROR: --bg-label is required when --condition calibration" >&2
  exit 2
fi
if [[ -z "$BG_SUMMARY_LABEL" ]]; then
  BG_SUMMARY_LABEL="$BG_LABEL"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ "$PHASE" == "calibration" ]]; then
    OUTPUT_DIR="${RUN_ROOT}/${MODEL_ALIAS}/calibration/calibration"
  else
    OUTPUT_DIR="${RUN_ROOT}/${MODEL_ALIAS}/main/${CONDITION}"
  fi
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$PHASE" == "calibration" ]]; then
  TAG="${MODEL_ALIAS}_fixd_calibration_calibration_profile-${PROFILE_ID}_conc${CONC}_run${RUN_NO}_bg-${BG_LABEL}"
else
  TAG="${MODEL_ALIAS}_fixd_main_${CONDITION}_profile-${PROFILE_ID}_conc${CONC}_run${RUN_NO}"
fi
RESULT_FILE="${OUTPUT_DIR}/${TAG}.json"

export OPENAI_API_KEY="$API_KEY"

check_server() {
  if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
  if ! curl -fsS --max-time 10 -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/v1/models" >/dev/null; then
    echo "ERROR: vLLM server not reachable at ${BASE_URL}." >&2
    echo "Start the matching server (offload${OFFLOAD_GB_EXPECTED}) for condition=${CONDITION} first." >&2
    exit 1
  fi
}
check_server

NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== PROBE model=${MODEL_ALIAS} condition=${CONDITION} phase=${PHASE} conc=${CONC} run=${RUN_NO} bg=${BG_LABEL:-none} ==="

cmd=(
  "$VLLM_BIN" bench serve
  --backend openai-chat
  --base-url "$BASE_URL"
  --endpoint "$ENDPOINT"
  --model "$MODEL"
  --num-prompts "$NUM_PROMPTS"
  --num-warmups "$NUM_WARMUPS"
  --random-input-len "$INPUT_LEN"
  --random-output-len "$OUTPUT_LEN"
  --max-concurrency "$CONC"
  --temperature "$TEMPERATURE"
  --save-result
  --save-detailed
  --result-dir "$OUTPUT_DIR"
  --result-filename "${TAG}.json"
  --metadata
  "campaign_version=${CAMPAIGN_VERSION}"
  "fixd_phase=${PHASE}"
  "condition=${CONDITION}"
  "model_alias=${MODEL_ALIAS}"
  "model_name=${MODEL}"
  "offload_gb_expected=${OFFLOAD_GB_EXPECTED}"
  "profile_id=${PROFILE_ID}"
  "input_len=${INPUT_LEN}"
  "output_len=${OUTPUT_LEN}"
  "concurrency=${CONC}"
  "num_prompts=${NUM_PROMPTS}"
  "num_warmups=${NUM_WARMUPS}"
  "temperature=${TEMPERATURE}"
  "run_no=${RUN_NO}"
  "bg_label=${BG_LABEL:-none}"
  "bg_summary_label=${BG_SUMMARY_LABEL:-none}"
  "bg_concurrency=${BG_CONCURRENCY:-none}"
  "bg_input_len=${BG_INPUT_LEN:-none}"
  "bg_output_len=${BG_OUTPUT_LEN:-none}"
  "match_status=${MATCH_STATUS:-none}"
  "session_id=${SESSION_ID:-none}"
  "probe_start_utc=${NOW}"
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,95,99
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'DRY-RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi

"${cmd[@]}"

if [[ ! -s "$RESULT_FILE" ]]; then
  echo "ERROR: expected result file missing or empty: $RESULT_FILE" >&2
  exit 1
fi

# probe_end_utc cannot be known before the run, so it can't go through
# --metadata like the other fields; inject it into the already-written
# result JSON instead. validate_fixd.py uses this (together with the
# background-load summary's start/end) to verify the background load
# covered the *entire* probe, not just its start.
PROBE_END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -c "
import json, sys
path, end_utc = sys.argv[1], sys.argv[2]
with open(path, encoding='utf-8') as f:
    data = json.load(f)
data['probe_end_utc'] = end_utc
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f)
" "$RESULT_FILE" "$PROBE_END_UTC"

echo "=== DONE ${RESULT_FILE} (probe_end_utc=${PROBE_END_UTC}) ==="

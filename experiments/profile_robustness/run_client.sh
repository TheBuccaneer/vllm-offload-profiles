#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_client_profile_robustness.sh --model llama|qwen --offload 0|12 [options]

Options:
  --out-root DIR       Root directory for results (default: ./profile_robustness_runs)
  --base-url URL       vLLM base URL (default: http://127.0.0.1:8000)
  --api-key KEY        API key (default: pilotkey)
  --num-prompts N      Requests per run (default: 20)
  --num-warmups N      Warmups per run (default: 1)
  --runs N             Repeats per cell (default: 3; campaign design expects 3)
  --dry-run            Print commands without executing them
  -h, --help           Show this help

The vLLM server must already be running with the same model and --cpu-offload-gb.
USAGE
}

MODEL_ALIAS=""
OFFLOAD_GB=""
OUT_ROOT="${OUT_ROOT:-./profile_robustness_runs}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-pilotkey}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
RUNS_PER_CELL="${RUNS_PER_CELL:-3}"
VLLM_BIN="${VLLM_BIN:-vllm}"
DRY_RUN=0
TEMP=0
ENDPOINT="/v1/chat/completions"
CAMPAIGN_VERSION="profile_robustness_v1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ALIAS="${2:?missing value for --model}"; shift 2 ;;
    --offload) OFFLOAD_GB="${2:?missing value for --offload}"; shift 2 ;;
    --out-root) OUT_ROOT="${2:?missing value for --out-root}"; shift 2 ;;
    --base-url) BASE_URL="${2:?missing value for --base-url}"; shift 2 ;;
    --api-key) API_KEY="${2:?missing value for --api-key}"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="${2:?missing value for --num-prompts}"; shift 2 ;;
    --num-warmups) NUM_WARMUPS="${2:?missing value for --num-warmups}"; shift 2 ;;
    --runs) RUNS_PER_CELL="${2:?missing value for --runs}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$MODEL_ALIAS" != "llama" && "$MODEL_ALIAS" != "qwen" ]]; then
  echo "ERROR: --model must be llama or qwen" >&2
  exit 2
fi
if ! [[ "$OFFLOAD_GB" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --offload must be a non-negative integer" >&2
  exit 2
fi
for value in "$NUM_PROMPTS" "$NUM_WARMUPS" "$RUNS_PER_CELL"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: numeric options must be non-negative integers" >&2
    exit 2
  fi
done
if [[ "$RUNS_PER_CELL" -lt 1 ]]; then
  echo "ERROR: --runs must be at least 1" >&2
  exit 2
fi

case "$MODEL_ALIAS" in
  llama)
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_TAG="llama31_8b"
    ;;
  qwen)
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    MODEL_TAG="qwen25_7b"
    ;;
esac

# The old campaign used 256/64. Short and long profiles are the only new axes.
PROFILE_IDS=("short" "reference" "long")
INPUT_LENS=(128 256 512)
OUTPUT_LENS=(32 64 128)
CONCURRENCIES=(4 8)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_ID="${CAMPAIGN_VERSION}_${MODEL_ALIAS}"
OUTDIR="${OUT_ROOT}/${MODEL_ALIAS}/offload${OFFLOAD_GB}/bench_runs_${EXPERIMENT_ID}_offload${OFFLOAD_GB}_${TIMESTAMP}"
MANIFEST="${OUTDIR}/campaign_manifest.tsv"
mkdir -p "$OUTDIR"
printf 'status\tmodel_alias\tmodel_name\toffload_gb\tprofile_id\tinput_len\toutput_len\tconcurrency\trun_no\tresult_file\tutc_time\n' > "$MANIFEST"

export OPENAI_API_KEY="$API_KEY"

check_server() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  if ! curl -fsS --max-time 10 \
      -H "Authorization: Bearer ${API_KEY}" \
      "${BASE_URL}/v1/models" >/dev/null; then
    echo "ERROR: vLLM server is not reachable at ${BASE_URL}." >&2
    echo "Start the matching existing server script first and wait until it is ready." >&2
    exit 1
  fi
}

profile_order_for_repeat() {
  local repeat="$1"
  case "$(((repeat - 1) % 3))" in
    0) echo "0 1 2" ;;
    1) echo "2 0 1" ;;
    2) echo "1 2 0" ;;
  esac
}

concurrency_order_for_repeat() {
  local repeat="$1"
  if (( repeat % 2 == 0 )); then
    echo "8 4"
  else
    echo "4 8"
  fi
}

run_bench() {
  local profile_idx="$1"
  local conc="$2"
  local run_no="$3"
  local profile_id="${PROFILE_IDS[$profile_idx]}"
  local input_len="${INPUT_LENS[$profile_idx]}"
  local output_len="${OUTPUT_LENS[$profile_idx]}"
  local tag="${MODEL_ALIAS}_profile-${profile_id}_offload${OFFLOAD_GB}_conc${conc}_run${run_no}"
  local result_file="${OUTDIR}/${tag}.json"
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  printf 'started\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$MODEL_ALIAS" "$MODEL" "$OFFLOAD_GB" "$profile_id" "$input_len" "$output_len" \
    "$conc" "$run_no" "$result_file" "$now" >> "$MANIFEST"

  echo "=== START model=${MODEL_ALIAS} offload=${OFFLOAD_GB} profile=${profile_id} (${input_len}/${output_len}) conc=${conc} repeat=${run_no} ==="

  cmd=(
    "$VLLM_BIN" bench serve
    --backend openai-chat
    --base-url "$BASE_URL"
    --endpoint "$ENDPOINT"
    --model "$MODEL"
    --num-prompts "$NUM_PROMPTS"
    --num-warmups "$NUM_WARMUPS"
    --random-input-len "$input_len"
    --random-output-len "$output_len"
    --max-concurrency "$conc"
    --temperature "$TEMP"
    --save-result
    --save-detailed
    --result-dir "$OUTDIR"
    --result-filename "${tag}.json"
    --metadata
    "campaign_version=${CAMPAIGN_VERSION}"
    "experiment_id=${EXPERIMENT_ID}"
    "server_config_label=${MODEL_TAG}_offload${OFFLOAD_GB}"
    "model_alias=${MODEL_ALIAS}"
    "model_name=${MODEL}"
    "offload_gb=${OFFLOAD_GB}"
    "profile_id=${profile_id}"
    "concurrency=${conc}"
    "num_prompts=${NUM_PROMPTS}"
    "input_len=${input_len}"
    "output_len=${output_len}"
    "temperature=${TEMP}"
    "num_warmups=${NUM_WARMUPS}"
    "run_no=${run_no}"
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,95,99
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
    if [[ ! -s "$result_file" ]]; then
      echo "ERROR: expected result file missing or empty: $result_file" >&2
      exit 1
    fi
  fi

  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'completed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$MODEL_ALIAS" "$MODEL" "$OFFLOAD_GB" "$profile_id" "$input_len" "$output_len" \
    "$conc" "$run_no" "$result_file" "$now" >> "$MANIFEST"
  echo "=== DONE ${result_file} ==="
  echo
}

check_server

echo "Output directory: $OUTDIR"
echo "Campaign manifest: $MANIFEST"
echo

for run_no in $(seq 1 "$RUNS_PER_CELL"); do
  read -r -a profile_order <<< "$(profile_order_for_repeat "$run_no")"
  read -r -a conc_order <<< "$(concurrency_order_for_repeat "$run_no")"
  for profile_idx in "${profile_order[@]}"; do
    for conc in "${conc_order[@]}"; do
      run_bench "$profile_idx" "$conc" "$run_no"
    done
  done
done

echo "All profile-robustness runs completed."
echo "Results: $OUTDIR"

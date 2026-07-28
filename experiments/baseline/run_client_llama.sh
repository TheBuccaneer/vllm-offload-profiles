#!/usr/bin/env bash
set -euo pipefail

# Nutzung:
#   OFFLOAD_GB=8 bash run_client_llama.sh
#
# Voraussetzung:
# - vLLM-Server läuft bereits auf http://127.0.0.1:8000
# - Server wurde mit genau diesem OFFLOAD_GB gestartet
# - dieses Skript startet nur den Client-Benchmark

MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_URL="http://127.0.0.1:8000"
ENDPOINT="/v1/chat/completions"
API_KEY="pilotkey"

EXPERIMENT_ID="offload_baseline_llama"
OFFLOAD_GB="${OFFLOAD_GB:?Bitte OFFLOAD_GB setzen, z.B. OFFLOAD_GB=8}"
SERVER_LABEL="llama31_8b_offload${OFFLOAD_GB}"

NUM_PROMPTS=20
NUM_WARMUPS=1
INPUT_LEN=256
OUTPUT_LEN=64
TEMP=0

# Deine aktuelle Matrix, inklusive der schon ergänzten Concurrency-Werte
CONCURRENCIES=(1 2 4 8 12 16)
RUNS_PER_CELL=5

# Ergebnisordner pro Offload
OUTDIR="bench_runs_${EXPERIMENT_ID}_offload${OFFLOAD_GB}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

export OPENAI_API_KEY="$API_KEY"

run_bench() {
  local conc="$1"
  local run_no="$2"
  local tag="offload${OFFLOAD_GB}_conc${conc}_run${run_no}"

  echo "=== START: offload=${OFFLOAD_GB}, concurrency=${conc}, run=${run_no} ==="

  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --max-concurrency "$conc" \
    --temperature "$TEMP" \
    --save-result \
    --save-detailed \
    --result-dir "$OUTDIR" \
    --result-filename "${tag}.json" \
    --metadata \
      experiment_id="${EXPERIMENT_ID}" \
      server_config_label="${SERVER_LABEL}" \
      model_name="${MODEL}" \
      offload_gb="${OFFLOAD_GB}" \
      concurrency="${conc}" \
      num_prompts="${NUM_PROMPTS}" \
      input_len="${INPUT_LEN}" \
      output_len="${OUTPUT_LEN}" \
      temperature="${TEMP}" \
      num_warmups="${NUM_WARMUPS}" \
      run_no="${run_no}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99

  echo "=== DONE: ${OUTDIR}/${tag}.json ==="
  echo
}

for conc in "${CONCURRENCIES[@]}"; do
  for run_no in $(seq 1 "$RUNS_PER_CELL"); do
    run_bench "$conc" "$run_no"
  done
done

echo "Alle Benchmarks abgeschlossen."
echo "Ergebnisse liegen in: ${OUTDIR}"

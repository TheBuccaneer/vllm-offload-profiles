#!/usr/bin/env bash
set -euo pipefail

# Nutzung:
#   bash run_server_qwen.sh 8
#
# Beispiel:
#   bash run_server_qwen.sh 0
#   bash run_server_qwen.sh 2
#   bash run_server_qwen.sh 4
#   bash run_server_qwen.sh 8
#   bash run_server_qwen.sh 12
#   bash run_server_qwen.sh 16

if [[ $# -lt 1 ]]; then
  echo "Fehler: Bitte cpu-offload-gb angeben, z.B. 8"
  echo "Nutzung: bash run_server_qwen.sh <offload_gb>"
  exit 1
fi

OFFLOAD_GB="$1"

MODEL="Qwen/Qwen2.5-7B-Instruct"
API_KEY="pilotkey"
GPU_MEM_UTIL="0.90"
TP_SIZE="1"
MAX_MODEL_LEN="8192"

LOGDIR="server_logs_qwen"
mkdir -p "$LOGDIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="${LOGDIR}/server_qwen_offload${OFFLOAD_GB}_${TIMESTAMP}.log"

export OPENAI_API_KEY="$API_KEY"

echo "Starte vLLM-Server mit Qwen und cpu-offload-gb=${OFFLOAD_GB}"
echo "Model: ${MODEL}"
echo "Logdatei: ${LOGFILE}"
echo

vllm serve "$MODEL" \
  --api-key "$API_KEY" \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --cpu-offload-gb "$OFFLOAD_GB" \
  2>&1 | tee "$LOGFILE"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
GPU_LOAD_DIR="$REPO_ROOT/experiments/gpu_load_control"
export CAMPAIGN_VERSION="kv_vram_control_v1"
LOG_DIR="$REPO_ROOT/results/reports/run_logs"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

LOG="$LOG_DIR/llama_kv_vram_gmem075_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "START $(date -Is)"

RUNROOT="${RUNROOT:-$REPO_ROOT/data/raw/kv_vram_control_reruns/llama_$(date +%Y%m%d_%H%M%S)}"
MODEL="llama"
BG_LABEL="c64i1024o512"
BG_CONC=64
BG_IN=1024
BG_OUT=512
BG_DURATION=300
BG_WARMUP=20

BG_PID=""

cleanup_bg() {
  if [[ -n "${BG_PID:-}" ]]; then
    kill "$BG_PID" 2>/dev/null || true
    wait "$BG_PID" 2>/dev/null || true
    BG_PID=""
  fi
}
trap cleanup_bg EXIT INT TERM

for RUNNO in 1 2 3; do
  for PROBE_CONC in 4 8; do
    BG_SUMMARY_LABEL="${BG_LABEL}_probeconc${PROBE_CONC}_kv_vram_gmem075_run${RUNNO}"

    echo "=== KV/VRAM gmem075 run=${RUNNO} probe_conc=${PROBE_CONC} bg=${BG_LABEL} ==="

    bash "$GPU_LOAD_DIR/run_background_gpu_load.sh" \
      --model "$MODEL" \
      --duration-seconds "$BG_DURATION" \
      --concurrency "$BG_CONC" \
      --input-len "$BG_IN" \
      --output-len "$BG_OUT" \
      --output-dir "$RUNROOT/$MODEL/background_logs" \
      --label "$BG_SUMMARY_LABEL" &

    BG_PID=$!
    echo "background pid=$BG_PID"

    sleep "$BG_WARMUP"

    bash "$GPU_LOAD_DIR/run_probe_fixd.sh" \
      --model "$MODEL" \
      --condition gpu_only_loaded \
      --concurrency "$PROBE_CONC" \
      --run-no "$RUNNO" \
      --run-root "$RUNROOT" \
      --bg-label "$BG_LABEL" \
      --bg-summary-label "$BG_SUMMARY_LABEL" \
      --bg-concurrency "$BG_CONC" \
      --bg-input-len "$BG_IN" \
      --bg-output-len "$BG_OUT" \
      --match-status kv_vram_control_gmem075

    cleanup_bg

    echo "=== DONE run=${RUNNO} probe_conc=${PROBE_CONC} ==="
  done
done

echo "END $(date -Is)"
echo "LOG: $LOG"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
GPU_LOAD_DIR="$REPO_ROOT/experiments/gpu_load_control"
export CAMPAIGN_VERSION="kv_vram_control_v1"
LOG_DIR="$REPO_ROOT/results/reports/run_logs"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

LOG="$LOG_DIR/qwen_kv_vram_gmem065_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "START $(date -Is)"

RUNROOT="${RUNROOT:-$REPO_ROOT/data/raw/kv_vram_control_reruns/qwen_$(date +%Y%m%d_%H%M%S)}"
MODEL="qwen"

BG_LABEL="c64i1024o512"
BG_CONC=64
BG_IN=1024
BG_OUT=512
BG_DURATION=300
BG_WARMUP=20

# Standardmäßig alle sechs finalen Zellen.
# Für den ersten kontrollierten Start beispielsweise:
# CELLS="1:4" bash run_qwen.sh
CELLS="${CELLS:-1:4 1:8 2:4 2:8 3:4 3:8}"

BG_PID=""

cleanup_bg() {
    if [[ -n "${BG_PID:-}" ]]; then
        kill "$BG_PID" 2>/dev/null || true
        wait "$BG_PID" 2>/dev/null || true
        BG_PID=""
    fi
}

trap cleanup_bg EXIT INT TERM

for CELL in $CELLS; do
    IFS=: read -r RUNNO PROBE_CONC <<< "$CELL"

    if [[ ! "$RUNNO" =~ ^[1-3]$ ]]; then
        echo "ERROR: invalid run number in cell: $CELL" >&2
        exit 2
    fi

    if [[ "$PROBE_CONC" != "4" && "$PROBE_CONC" != "8" ]]; then
        echo "ERROR: invalid probe concurrency in cell: $CELL" >&2
        exit 2
    fi

    BG_SUMMARY_LABEL="${BG_LABEL}_probeconc${PROBE_CONC}_kv_vram_gmem065_run${RUNNO}"

    RESULT_FILE="${RUNROOT}/${MODEL}/main/gpu_only_loaded/"\
"${MODEL}_fixd_main_gpu_only_loaded_profile-reference_conc${PROBE_CONC}_run${RUNNO}.json"

    SUMMARY_FILE="${RUNROOT}/${MODEL}/background_logs/"\
"${MODEL}_bgload_${BG_SUMMARY_LABEL}_conc${BG_CONC}_summary.json"

    if [[ -e "$RESULT_FILE" || -e "$SUMMARY_FILE" ]]; then
        echo "ERROR: refusing to overwrite an existing cell:"
        echo "  result:  $RESULT_FILE"
        echo "  summary: $SUMMARY_FILE"
        exit 1
    fi

    echo
    echo "=== Qwen KV/VRAM run=${RUNNO} probe_conc=${PROBE_CONC} bg=${BG_LABEL} ==="

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
        --match-status kv_vram_control_gmem065 \
        --session-id qwen_kv_vram_gmem065

    cleanup_bg

    test -s "$RESULT_FILE"
    test -s "$SUMMARY_FILE"

    echo "=== DONE run=${RUNNO} probe_conc=${PROBE_CONC} ==="
done

echo
echo "END $(date -Is)"
echo "LOG: $LOG"

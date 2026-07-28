#!/usr/bin/env bash
set -uo pipefail

# run_analysis_fixd.sh — run validate_fixd.py then analyze_fixd.py into a
# single timestamped output directory:
#   fixd_load_control_analysis_<timestamp>/
#     validation_report.csv
#     results/
#       fixd_main_summary.csv, fixd_shape_metrics.csv, ...
#
# By default, analysis is refused if validation reports any ERROR — pass
# --force to run analyze_fixd.py anyway (e.g. while iterating). The
# analyzer has its own independent blocker check for missing shape data,
# so --force does not bypass that.
#
# Usage:
#   bash run_analysis_fixd.sh --run-root DIR [--model llama|qwen] [--force]

RUN_ROOT="${RUN_ROOT:-./fixd_load_control_runs}"
MODEL_ARG=""
FORCE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:?}"; shift 2 ;;
    --model) MODEL_ARG="${2:?}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="fixd_load_control_analysis_${TIMESTAMP}"
mkdir -p "${OUT_DIR}/results"

VALIDATE_ARGS=("$RUN_ROOT" --output "${OUT_DIR}/validation_report.csv")
[[ -n "$MODEL_ARG" ]] && VALIDATE_ARGS+=(--model "$MODEL_ARG")

echo "=== validate_fixd.py ==="
# python3 automatically puts the script's own directory on sys.path, so
# config_fixd/fixd_metrics import correctly regardless of the caller's CWD.
python3 "${SCRIPT_DIR}/validate_fixd.py" "${VALIDATE_ARGS[@]}"
VALIDATE_RC=$?

if [[ "$VALIDATE_RC" -ne 0 && "$FORCE" -ne 1 ]]; then
  echo "Validation reported ERRORs (see ${OUT_DIR}/validation_report.csv)." >&2
  echo "Refusing to analyze invalid data. Re-run with --force to override." >&2
  exit 1
fi

ANALYZE_ARGS=("$RUN_ROOT" --output-dir "${OUT_DIR}/results")
[[ -n "$MODEL_ARG" ]] && ANALYZE_ARGS+=(--model "$MODEL_ARG")

echo "=== analyze_fixd.py ==="
python3 "${SCRIPT_DIR}/analyze_fixd.py" "${ANALYZE_ARGS[@]}"
ANALYZE_RC=$?

echo
echo "Output directory: ${OUT_DIR}"
if [[ "$ANALYZE_RC" -ne 0 ]]; then
  echo "analyze_fixd.py exited with code ${ANALYZE_RC} (see BLOCKER message above)." >&2
fi
exit "$ANALYZE_RC"

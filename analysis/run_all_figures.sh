#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"

if [[ ! -d "$REPO_ROOT/data/derived" ||
      ! -d "$REPO_ROOT/experiments" ]]; then
  echo "Usage: bash analysis/run_all_figures.sh [repository-root]" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
INCLUDE_FIG2="${INCLUDE_FIG2:-0}"

FIGURE_DIR="$REPO_ROOT/results/figures/paper"
TABLE_DIR="$REPO_ROOT/results/tables/paper"

mkdir -p "$FIGURE_DIR" "$TABLE_DIR"

BASE_ARGS=(
  --repo-root "$REPO_ROOT"
  --llama-csv "$REPO_ROOT/data/derived/baseline/llama/runs_summary.csv"
  --qwen-csv "$REPO_ROOT/data/derived/baseline/qwen/runs_summary.csv"
)

PROFILE_ARGS=(
  --repo-root "$REPO_ROOT"
  --profile-csv "$REPO_ROOT/data/derived/profile_robustness/runs_summary.csv"
)

CONTROL_ARGS=(
  --repo-root "$REPO_ROOT"
  --fixd-csv "$REPO_ROOT/results/tables/gpu_load_control/fixd_main_summary.csv"
  --d2-llama-root "$REPO_ROOT/data/raw/kv_vram_control/llama"
  --d2-qwen-root "$REPO_ROOT/data/raw/kv_vram_control/qwen"
)

"$PYTHON" "$SCRIPT_DIR/figures/build_fig1_offload_fits.py" \
  "${BASE_ARGS[@]}" --output-dir "$FIGURE_DIR"

if [[ "$INCLUDE_FIG2" == "1" ]]; then
  "$PYTHON" "$SCRIPT_DIR/figures/build_fig2_phase_profile.py" \
    "${BASE_ARGS[@]}" --output-dir "$FIGURE_DIR"
fi

"$PYTHON" "$SCRIPT_DIR/figures/build_fig3_profile_robustness.py" \
  "${PROFILE_ARGS[@]}" --output-dir "$FIGURE_DIR"

"$PYTHON" "$SCRIPT_DIR/figures/build_fig4_control_profiles.py" \
  "${CONTROL_ARGS[@]}" --output-dir "$FIGURE_DIR"

"$PYTHON" "$SCRIPT_DIR/tables/build_result_tables.py" \
  "${BASE_ARGS[@]}" "${CONTROL_ARGS[@]}" \
  --output-dir "$TABLE_DIR"

echo "PASS: paper figures and tables generated"

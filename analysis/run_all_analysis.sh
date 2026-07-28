#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PYTHON="${PYTHON:-python3}"

cd "$REPO_ROOT"

PROFILE_TMP="$(mktemp -d)"
GPU_TMP="$(mktemp -d)"

cleanup() {
  rm -rf "$PROFILE_TMP" "$GPU_TMP"
}
trap cleanup EXIT

mkdir -p \
  data/derived/baseline/llama \
  data/derived/baseline/qwen \
  data/derived/profile_robustness \
  results/reports \
  results/tables/separability \
  results/tables/profile_robustness \
  results/figures/profile_robustness \
  results/tables/gpu_load_control \
  results/figures/gpu_load_control \
  results/tables/kv_vram_control

echo "[1/11] Verify raw-data checksums"
bash analysis/verify_checksums.sh "$REPO_ROOT"

echo "[2/11] Validate GPU-memory configuration provenance"
"$PYTHON" analysis/validation/validate_gpu_memory_configuration_provenance.py

echo
echo "[3/11] Extract baseline datasets"
"$PYTHON" analysis/extraction/extract_baseline.py \
  data/raw/baseline/llama \
  --outdir data/derived/baseline/llama

"$PYTHON" analysis/extraction/extract_baseline.py \
  data/raw/baseline/qwen \
  --outdir data/derived/baseline/qwen

echo
echo "[4/11] Validate baseline campaign"
"$PYTHON" analysis/validation/validate_baseline.py \
  data/raw/baseline \
  --derived-root data/derived/baseline \
  --output results/reports/baseline_validation.txt

echo
echo "[5/11] Validate and extract profile-robustness campaign"
"$PYTHON" analysis/validation/validate_profile_robustness.py \
  data/raw/profile_robustness \
  --runs 3 \
  --output results/reports/profile_robustness_validation.csv \
  | tee results/reports/profile_robustness_validation.txt

"$PYTHON" analysis/extraction/extract_baseline.py \
  data/raw/profile_robustness \
  --outdir data/derived/profile_robustness

echo
echo "[6/11] Analyze profile robustness"
"$PYTHON" analysis/separability/profile_robustness.py \
  data/derived/profile_robustness/runs_summary.csv \
  --output-dir "$PROFILE_TMP" \
  | sed "s|$PROFILE_TMP|results/profile_robustness|g" \
  | tee results/reports/profile_robustness_analysis.txt

rm -f \
  results/tables/profile_robustness/*.csv \
  results/figures/profile_robustness/*.png

for file in \
  analysis_input_filtered.csv \
  classification_predictions.csv \
  classification_summary.csv \
  confusion_matrices.csv \
  signal_separation.csv \
  within_profile_folds.csv
do
  mv "$PROFILE_TMP/$file" \
    "results/tables/profile_robustness/$file"
done

for file in \
  llama_median_itl_ms_profiles.png \
  llama_median_tpot_ms_profiles.png \
  qwen_median_itl_ms_profiles.png \
  qwen_median_tpot_ms_profiles.png
do
  mv "$PROFILE_TMP/$file" \
    "results/figures/profile_robustness/$file"
done

echo
echo "[7/11] Run baseline separability analyses"
"$PYTHON" analysis/separability/baseline_loco.py \
  data/derived/baseline/llama/runs_summary.csv \
  --mode binary \
  --min-concurrency 2 \
  --output results/tables/separability/llama_binary_loco_c2plus.csv \
  | tee results/reports/llama_binary_loco_c2plus.txt

"$PYTHON" analysis/separability/baseline_loco.py \
  data/derived/baseline/qwen/runs_summary.csv \
  --mode binary \
  --min-concurrency 2 \
  --output results/tables/separability/qwen_binary_loco_c2plus.csv \
  | tee results/reports/qwen_binary_loco_c2plus.txt

"$PYTHON" analysis/separability/baseline_loco.py \
  data/derived/baseline/llama/runs_summary.csv \
  --mode multiclass \
  --min-concurrency 1 \
  --output results/tables/separability/llama_multiclass_loco_all.csv \
  | tee results/reports/llama_multiclass_loco_all.txt

"$PYTHON" analysis/separability/baseline_loco.py \
  data/derived/baseline/llama/runs_summary.csv \
  --mode multiclass \
  --min-concurrency 2 \
  --output results/tables/separability/llama_multiclass_loco_c2plus.csv \
  | tee results/reports/llama_multiclass_loco_c2plus.txt

echo
echo "[8/11] Validate and analyze generic GPU-load control"
"$PYTHON" experiments/gpu_load_control/validate_fixd.py \
  data/raw/gpu_load_control \
  --runs 3 \
  --output results/reports/gpu_load_control_validation.csv \
  | tee results/reports/gpu_load_control_validation.txt

"$PYTHON" experiments/gpu_load_control/analyze_fixd.py \
  data/raw/gpu_load_control \
  --output-dir "$GPU_TMP" \
  | sed "s|$GPU_TMP|results/gpu_load_control|g" \
  | tee results/reports/gpu_load_control_analysis.txt

rm -f \
  results/tables/gpu_load_control/*.csv \
  results/figures/gpu_load_control/*.png

for file in \
  fixd_calibration_summary.csv \
  fixd_classification_supplement.csv \
  fixd_condition_comparison.csv \
  fixd_main_summary.csv \
  fixd_shape_metrics.csv
do
  mv "$GPU_TMP/$file" \
    "results/tables/gpu_load_control/$file"
done

for file in \
  itl_ecdf_llama_conc4.png \
  itl_ecdf_llama_conc8.png \
  itl_ecdf_qwen_conc4.png \
  itl_ecdf_qwen_conc8.png \
  itl_median_barplot_llama.png \
  itl_median_barplot_qwen.png
do
  mv "$GPU_TMP/$file" \
    "results/figures/gpu_load_control/$file"
done

echo
echo "[9/11] Validate and analyze KV/VRAM control"
"$PYTHON" experiments/kv_vram_control/validate.py \
  data/raw/kv_vram_control \
  --output results/reports/kv_vram_control_validation.csv \
  | tee results/reports/kv_vram_control_validation.txt

"$PYTHON" experiments/kv_vram_control/analyze.py \
  results/reports/kv_vram_control_validation.csv \
  results/tables/gpu_load_control/fixd_main_summary.csv \
  --output results/tables/kv_vram_control/kv_vram_control_summary.csv \
  | tee results/reports/kv_vram_control_analysis.txt

echo
echo "[10/11] Generate paper figures and tables"
rm -rf \
  results/figures/paper \
  results/tables/paper

bash analysis/run_all_figures.sh "$REPO_ROOT"

echo
echo "[11/11] Verify key outputs"
EXPECTED=(
  results/reports/baseline_validation.txt
  results/reports/profile_robustness_validation.csv
  results/reports/gpu_load_control_validation.csv
  results/reports/kv_vram_control_validation.csv
  results/reports/gpu_memory_configuration_provenance.md
  results/tables/separability/llama_binary_loco_c2plus.csv
  results/tables/separability/qwen_binary_loco_c2plus.csv
  results/tables/paper/table_fit_parameters.csv
  results/tables/paper/table_control_ratios.csv
  results/figures/paper/fig1_offload_fits.pdf
  results/figures/paper/fig2_phase_profile.pdf
  results/figures/paper/fig2_phase_profile.png
  results/figures/paper/fig2_phase_profile_data.csv
  results/figures/paper/fig2_phase_profile_provenance.json
  results/figures/paper/fig3_profile_robustness.pdf
  results/figures/paper/fig4_control_profiles.pdf
)

for file in "${EXPECTED[@]}"; do
  [[ -s "$file" ]] || {
    echo "FAIL: missing or empty output: $file" >&2
    exit 1
  }
done

echo
echo "============================================================"
echo "PASS: complete analysis pipeline"
echo "============================================================"

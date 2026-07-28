# Artifact Manifest

This manifest maps each evaluated campaign to its canonical raw data,
analysis code, and generated outputs.

## 1. Baseline CPU-offload sweep

Design:

- models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct
- offload budgets: `0, 2, 4, 8, 12, 16 GiB`
- concurrency: `1, 2, 4, 8, 12, 16`
- repetitions: 5 per cell
- request profile: 256 input / 64 output tokens
- total: 360 runs and 7,200 requests

Artifacts:

- raw data: `data/raw/baseline/`
- runners: `experiments/baseline/`
- extraction: `analysis/extraction/extract_baseline.py`
- validation: `analysis/validation/validate_baseline.py`
- derived data: `data/derived/baseline/`
- validation report: `results/reports/baseline_validation.txt`
- separability outputs: `results/tables/separability/`
- checksums: `data/provenance/baseline_sha256.txt`

## 2. Request-profile robustness

Design:

- offload budgets: 0 and 12 GiB
- profiles: `128/32`, `256/64`, and `512/128`
- concurrency: 4 and 8
- repetitions: 3 per cell
- total: 72 runs and 1,440 requests

Artifacts:

- raw data: `data/raw/profile_robustness/`
- runner: `experiments/profile_robustness/run_client.sh`
- extraction: `analysis/extraction/extract_baseline.py`
- validation: `analysis/validation/validate_profile_robustness.py`
- analysis: `analysis/separability/profile_robustness.py`
- derived data: `data/derived/profile_robustness/`
- figures: `results/figures/profile_robustness/`
- tables: `results/tables/profile_robustness/`
- reports: `results/reports/profile_robustness_*`
- checksums: `data/provenance/profile_robustness_sha256.txt`

## 3. Generic GPU-only background-load control

Design:

- conditions: normal, GPU-only loaded, and 12 GiB CPU offload
- concurrency: 4 and 8
- repetitions: 3 per cell
- total main campaign: 36 runs and 720 requests
- `gpu-memory-utilization`: 0.86

Artifacts:

- raw data: `data/raw/gpu_load_control/`
- runners and analysis: `experiments/gpu_load_control/`
- figures: `results/figures/gpu_load_control/`
- tables: `results/tables/gpu_load_control/`
- reports: `results/reports/gpu_load_control_*`
- checksums: `data/provenance/gpu_load_control_sha256.txt`

## 4. KV-cache and VRAM-pressure control

Design:

- CPU model offloading: 0 GiB
- probe profile: 256 input / 64 output tokens
- probe concurrency: 4 and 8
- background profile: concurrency 64, 1,024 input / 512 output tokens
- `gpu-memory-utilization`: 0.75 for Llama and 0.65 for Qwen
- total: 12 probe runs and 240 probe requests

Artifacts:

- raw data: `data/raw/kv_vram_control/`
- runners and analysis: `experiments/kv_vram_control/`
- summary: `results/tables/kv_vram_control/kv_vram_control_summary.csv`
- reports: `results/reports/kv_vram_control_*`
- checksums: `data/provenance/kv_vram_control_sha256.txt`

## 5. GPU-memory configuration provenance

Scope:

- documents the campaign-specific `gpu-memory-utilization` settings,
- records the matched-comparison rule across campaign blocks,
- retains curated successful and failed server-start evidence,
- documents the Llama and Qwen GPU model-loading memory values,
- distinguishes model-specific near-boundary settings from equal-pressure claims.

Artifacts:

- evidence excerpts: `data/provenance/gpu_memory_configuration/`
- historical source mapping: `data/provenance/gpu_memory_configuration/source_map.tsv`
- checksums: `data/provenance/gpu_memory_configuration_sha256.txt`
- validation: `analysis/validation/validate_gpu_memory_configuration_provenance.py`
- report: `results/reports/gpu_memory_configuration_provenance.md`

## Shared analysis pipeline

- shared loading and fitting: `analysis/common/figure_common.py`
- paper figures: `analysis/figures/`
- paper tables: `analysis/tables/`
- figure entry point: `analysis/run_all_figures.sh`
- complete entry point: `analysis/run_all_analysis.sh`
- checksum verification: `analysis/verify_checksums.sh`
- dependencies: `environment/requirements-analysis.txt`

Publication-ready outputs:

- `results/figures/paper/`
- `results/tables/paper/`

## Excluded material

The artifact excludes superseded experiments, unrelated branches,
temporary archives, duplicated outputs, obsolete intermediate results,
and planning documents.

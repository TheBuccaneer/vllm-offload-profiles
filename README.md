# vLLM Offload Profiles

Experimental artifacts for the study:

**Phase-Aware Characterization of CPU Model Offloading in vLLM Serving**

This repository contains the raw measurements, experiment runners,
validation code, analysis pipeline, and publication-ready outputs used
to characterize how vLLM's CPU model-parameter offloading changes
client-observed serving phases.

## Study scope

The evaluated setup includes:

- vLLM 0.17.1
- NVIDIA RTX 3090 with 24 GiB GPU memory
- `bfloat16` model precision
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- configured CPU-offload budgets of `0, 2, 4, 8, 12, 16 GiB`
- concurrency levels of `1, 2, 4, 8, 12, 16`
- client-observed TTFT, TPOT, ITL, and E2EL

The artifact covers four measurement blocks:

1. baseline CPU-offload sweep,
2. request-profile robustness,
3. generic GPU-only background-load control,
4. near-boundary KV-cache and VRAM-pressure control.

## Repository layout

- `experiments/`: experiment runners and campaign-specific utilities
- `analysis/`: extraction, validation, fitting, separability, and plotting
- `data/raw/`: canonical raw measurement outputs
- `data/derived/`: reproducibly generated run- and request-level datasets
- `data/provenance/`: SHA-256 manifests and campaign provenance
- `results/figures/`: generated figures
- `results/tables/`: generated tables and analysis datasets
- `results/reports/`: validation and analysis reports
- `environment/`: analysis dependencies and environment documentation
- `paper/`: publication-related material

See `MANIFEST.md` for the complete mapping between campaigns, source
data, analysis scripts, and outputs.

## Reproduce the analysis

Create an isolated Python environment and install the analysis
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements-analysis.txt

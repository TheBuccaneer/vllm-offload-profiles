# vLLM Offload Profiles

This repository contains the experimental artifacts for the study:

**Phase-Aware Characterization of CPU Model Offloading in vLLM Serving**

The study examines how vLLM's budget-based CPU model-parameter
offloading affects client-observed phases of streamed LLM responses.

## Study scope

The evaluated setup uses:

- vLLM 0.17.1
- NVIDIA RTX 3090 with 24 GiB GPU memory
- bfloat16 precision
- meta-llama/Llama-3.1-8B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- configured CPU-offload budgets from 0 to 16 GiB
- client-observed TTFT, TPOT, ITL, and E2EL

The repository covers four evaluated measurement blocks:

1. baseline offload sweep,
2. request-profile robustness,
3. GPU-only background-load control,
4. KV/VRAM-pressure control.

## Repository layout

- `experiments/` contains the canonical experiment runners and configurations.
- `analysis/` contains extraction, validation, fitting, classification, and plotting code.
- `data/raw/` contains the canonical raw measurement outputs.
- `data/derived/` contains reproducibly generated datasets.
- `data/provenance/` documents configurations, checksums, and source information.
- `results/` contains publication-ready figures, tables, and validation reports.
- `environment/` documents the hardware and software environment.
- `paper/` contains publication-related material.

## Reproduction workflow

The intended analysis path is:

    raw measurements
        -> validation
        -> extraction
        -> aggregation
        -> fitting and separability analysis
        -> figures and tables

Detailed reproduction instructions will be added after the canonical
files have been imported and independently validated.

## Artifact policy

Only files used by the final study are included.

Exploratory experiments, superseded scripts, temporary archives,
internal handover documents, and unrelated project material are
intentionally excluded.

## Citation

Citation metadata is provided in `CITATION.cff`.

## License

Code and data licenses will be added before the public artifact release.

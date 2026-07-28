# Analysis

This directory contains the reproducible analysis pipeline from raw
measurements to the reported figures and tables.

## Complete pipeline

Run the complete analysis from the repository root:

    bash analysis/run_all_analysis.sh

The pipeline:

1. verifies the raw-data SHA-256 manifests,
2. extracts the baseline and profile-robustness datasets,
3. validates all four measurement campaigns,
4. runs the separability and robustness analyses,
5. analyzes the GPU-load and KV/VRAM controls,
6. fits the reported offload models,
7. regenerates the paper figures and tables,
8. verifies that the required outputs exist.

## Directory structure

- `common/`: shared evidence loading, aggregation, and fitting functions
- `extraction/`: conversion of raw JSON measurements into CSV datasets
- `validation/`: campaign-level structural and semantic validation
- `separability/`: leave-one-concurrency-out and robustness analyses
- `figures/`: publication-figure builders
- `tables/`: publication-table builders

## Entry points

Complete analysis:

    bash analysis/run_all_analysis.sh

Paper figures and tables only:

    bash analysis/run_all_figures.sh

Verify canonical raw inputs:

    bash analysis/verify_checksums.sh

## Generated outputs

The pipeline writes generated artifacts beneath:

- `data/derived/`
- `results/reports/`
- `results/tables/`
- `results/figures/`

Temporary analysis directories are removed automatically. Files beneath
`data/raw/` are treated as immutable inputs and are not modified.

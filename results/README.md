# Results

This directory contains generated validation reports, analysis
datasets, figures, and publication tables.

## Reports

`reports/` contains human-readable and machine-readable validation and
analysis reports for all four measurement campaigns.

## Tables

`tables/` contains:

- leave-one-concurrency-out results,
- request-profile robustness summaries,
- GPU-load-control summaries,
- KV/VRAM-control summaries,
- publication-ready CSV and LaTeX tables.

The final paper tables are stored in:

    results/tables/paper/

## Figures

`figures/` contains diagnostic plots and publication-ready figures.

The final paper figures are stored in:

    results/figures/paper/

Each paper-output group includes a provenance JSON recording its input
files, SHA-256 checksums, and the Git revision of the analysis code.

All outputs can be regenerated with:

    bash analysis/run_all_analysis.sh

## License

Figures, tables, reports, and generated analysis datasets in this
directory are licensed under the Creative Commons Attribution 4.0
International License. See `../LICENSE-DATA`.

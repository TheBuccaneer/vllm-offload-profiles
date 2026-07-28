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
- NVIDIA GeForce RTX 3090 with 24 GiB GPU memory
- `bfloat16` model precision
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- CPU-offload budgets of `0, 2, 4, 8, 12, 16 GiB`
- concurrency levels of `1, 2, 4, 8, 12, 16`
- client-observed TTFT, TPOT, ITL, and E2EL

The artifact covers four measurement blocks:

1. baseline CPU-offload sweep,
2. request-profile robustness,
3. generic GPU-only background-load control,
4. KV-cache and VRAM-pressure control.

## Repository layout

- `experiments/`: experiment runners and campaign utilities
- `analysis/`: extraction, validation, fitting, and plotting
- `data/raw/`: canonical raw measurements
- `data/derived/`: reproducibly generated datasets
- `data/provenance/`: SHA-256 manifests and retained configuration evidence
- `results/figures/`: generated figures
- `results/tables/`: generated tables and analysis datasets
- `results/reports/`: validation and analysis reports
- `environment/`: environment and dependency documentation
- `paper/`: publication-related material

See `MANIFEST.md` for the mapping between campaigns, data, analysis
scripts, and outputs.

## Reproduce the analysis

Create an isolated Python environment:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r environment/requirements-analysis.txt

Run the complete pipeline from the repository root:

    bash analysis/run_all_analysis.sh

The pipeline:

1. verifies the raw-data SHA-256 manifests,
2. validates the retained GPU-memory configuration evidence,
3. extracts the baseline and robustness datasets,
4. validates all four measurement campaigns,
5. runs the separability and robustness analyses,
6. analyzes the GPU-load and KV/VRAM controls,
7. fits the reported TPOT models,
8. regenerates the paper figures and tables,
9. verifies the required outputs.

A successful execution ends with:

    PASS: complete analysis pipeline

## Main paper outputs

Figures:

- `results/figures/paper/fig1_offload_fits.pdf`
- `results/figures/paper/fig2_phase_profile.pdf`
- `results/figures/paper/fig3_profile_robustness.pdf`
- `results/figures/paper/fig4_control_profiles.pdf`

Tables:

- `results/tables/paper/table_fit_parameters.csv`
- `results/tables/paper/table_fit_parameters.tex`
- `results/tables/paper/table_control_ratios.csv`
- `results/tables/paper/table_control_ratios.tex`

Supporting configuration-provenance report:

- `results/reports/gpu_memory_configuration_provenance.md`

Paper-output provenance JSON files record the input checksums and the
Git revision of the analysis code used to generate the outputs.


## Data integrity

Verify all canonical raw inputs independently with:

    bash analysis/verify_checksums.sh

Verify the retained GPU-memory configuration evidence and report with:

    python3 analysis/validation/validate_gpu_memory_configuration_provenance.py

Files beneath `data/raw/` are treated as immutable inputs. Generated
datasets and results are written beneath `data/derived/` and `results/`.

## Artifact policy

The repository contains only material used by the final study.
Superseded experiments, temporary archives, duplicated outputs,
unrelated project material, and planning documents are excluded.

Historical identifiers inside original measurement records are
retained where required to preserve provenance.

## Citation

Citation metadata is provided in `CITATION.cff`.

## License

Source code and executable scripts are licensed under the MIT License.
See `LICENSE`.

Measurement data, generated figures, tables, reports, and repository
documentation are licensed under the Creative Commons Attribution 4.0
International License. See `LICENSE-DATA`.

Third-party software, model weights, model licenses, and other external
materials are not relicensed by this repository. See `LICENSING.md`.

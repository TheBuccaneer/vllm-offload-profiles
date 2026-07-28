# Generic GPU Load Control

This experiment distinguishes CPU model offloading from generic GPU-only background load.

## Conditions

- `normal`: no additional background load
- `gpu_only_loaded`: concurrent GPU-only workload
- `offload12`: 12 GiB CPU model offloading

## Design

- Models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct
- Probe concurrency: 4 and 8
- Repeats: 3 per cell
- Requests: 20 per probe run
- Total main campaign: 36 probe runs and 720 requests
- GPU memory utilization: 0.86 for all three conditions

Calibration runs and background-load summaries document the selection and execution of the GPU-only load levels.

Historical campaign identifiers inside the original JSON files are preserved for provenance.

## Validation

```bash
python3 experiments/gpu_load_control/validate_fixd.py data/raw/gpu_load_control --runs 3
```

## Analysis

```bash
python3 experiments/gpu_load_control/analyze_fixd.py data/raw/gpu_load_control --output-dir results/gpu_load_control
```

The analysis generates condition comparisons, run-level summaries, shape metrics, calibration summaries, and ITL figures.

The calibration status `not_reached_max_loaded` is preserved as measurement metadata. It does not indicate a failed or invalid probe run.

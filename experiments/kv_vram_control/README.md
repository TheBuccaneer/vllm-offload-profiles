# KV and VRAM Pressure Control

This control evaluates whether strong GPU-only KV-cache and VRAM pressure reproduces the decode cadence observed under CPU model offloading.

## Design

- CPU model offloading: 0 GiB
- Probe profile: 256 input and 64 output tokens
- Probe concurrency: 4 and 8
- Repeats: 3 per cell
- Background workload: concurrency 64, 1,024 input and 512 output tokens
- Total: 12 probe runs and 240 probe requests

## Server configurations

- Llama-3.1-8B-Instruct: `--gpu-memory-utilization 0.75`
- Qwen2.5-7B-Instruct: `--gpu-memory-utilization 0.65`

The next lower tested settings, 0.70 for Llama and 0.60 for Qwen, did not provide sufficient KV-cache capacity to start the intended workload.

The experiment runners expect the corresponding vLLM server to be running already. They reuse the probe and background-load utilities from `experiments/gpu_load_control/`.

By default, new measurements are written beneath `data/raw/kv_vram_control_reruns/`. Set `RUNROOT` explicitly to choose another output root.

## Validation

```bash
python3 experiments/kv_vram_control/validate.py data/raw/kv_vram_control --output results/reports/kv_vram_control_validation.csv
```

## Analysis

```bash
python3 experiments/kv_vram_control/analyze.py results/reports/kv_vram_control_validation.csv results/tables/gpu_load_control/fixd_main_summary.csv --output results/tables/kv_vram_control/kv_vram_control_summary.csv
```

## Result

Across both models, near-boundary GPU-only KV/VRAM pressure increased median TTFT by 4.64–7.54x relative to normal serving, while median ITL increased by only 1.03–1.08x. The resulting ITL remained 1.36–1.62% of the corresponding CPU-offload12 cadence.

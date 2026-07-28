# Baseline offload sweep

The baseline campaign evaluates two instruction-tuned model families
under six configured CPU-offload budgets and six concurrency levels.

## Models

- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

## Experiment matrix

- CPU offload: `0, 2, 4, 8, 12, 16 GiB`
- Concurrency: `1, 2, 4, 8, 12, 16`
- Nominal request profile: `256/64`
- Measured requests per run: `20`
- Warm-up requests per run: `1`
- Repetitions per cell: `5`
- Precision: `bfloat16`
- `gpu-memory-utilization`: `0.90`

The complete campaign contains 180 runs per model and 360 runs in
total.

## Files

- `run_server_llama.sh`: historical Llama server runner
- `run_client_llama.sh`: historical Llama client runner
- `run_server_qwen.sh`: historical Qwen server runner
- `run_client_qwen.sh`: historical Qwen client runner

Canonical raw measurement outputs are stored under:

- `data/raw/baseline/llama/`
- `data/raw/baseline/qwen/`

The imported runners are preserved as experiment provenance. Paths may
need adjustment before an independent rerun.

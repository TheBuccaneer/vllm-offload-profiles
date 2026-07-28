# Experimental Environment

This directory documents the environment used for the reported
measurements and the dependencies required to reproduce the analysis.

## Measurement host

Recorded hardware:

- CPU: AMD Ryzen Threadripper 3970X
- GPU: NVIDIA GeForce RTX 3090
- GPU memory: 24 GiB

Recorded software environment:

- operating system: Kubuntu 25.10
- Linux kernel: 6.17.0-12-generic
- NVIDIA driver: 580.159.03
- CUDA driver capability: 13.0
- Python: 3.13.7
- vLLM: 0.17.1

## Models

- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

Both models were served using `bfloat16` precision.

Model weights are not redistributed by this repository and must be
obtained separately under their respective licenses and access terms.

## Campaign-specific GPU-memory settings

Baseline CPU-offload sweep:

- `gpu-memory-utilization=0.90`

Request-profile robustness:

- `gpu-memory-utilization=0.90`

Generic GPU-only background-load control:

- `gpu-memory-utilization=0.86`

KV-cache and VRAM-pressure control:

- Llama: `gpu-memory-utilization=0.75`
- Qwen: `gpu-memory-utilization=0.65`

The lower attempted settings of 0.70 for Llama and 0.60 for Qwen did
not provide sufficient KV-cache capacity for the configured background
workload and are not included as successful measurement conditions.

## Analysis environment

The analysis does not require a GPU or installed model weights.

Create an isolated environment and install the recorded analysis
dependencies with:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r environment/requirements-analysis.txt

Then run:

    bash analysis/run_all_analysis.sh

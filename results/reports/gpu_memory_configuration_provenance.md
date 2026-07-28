# GPU-memory configuration provenance

## Purpose

This report documents why the experimental campaign blocks use different
`gpu_memory_utilization` settings and anchors those settings to retained
runner configurations, curated server-log excerpts, and campaign data.

The settings are not intended to represent equal memory pressure across
campaigns or models. Each block serves a different experimental purpose.

## Comparison rule

All reported contrasts and ratios are formed only between matched conditions
within the same campaign block, model, request profile, concurrency, and
GPU-memory-utilization setting.

Absolute measurements from different campaign blocks must not be interpreted
as though their GPU-memory configurations were identical.

## Campaign configurations

| Campaign block | Model | Setting | Rationale and evidence |
|---|---|---:|---|
| Baseline sweep | Llama and Qwen | 0.90 | Main reference configuration. The committed baseline runners explicitly pin `GPU_MEM_UTIL=0.90`. |
| Profile robustness | Llama and Qwen | 0.90 | Reuses the baseline memory configuration so that only the request profile changes. |
| GPU-load control, Fix D | Llama and Qwen | 0.86 | A preliminary Llama GPU-only start using the default 0.90 setting failed during sampler warm-up with a CUDA out-of-memory error. The value 0.86 was then applied consistently to both models and to all matched Fix-D conditions. |
| KV/VRAM control | Llama | 0.75 | Retained near-boundary configuration. A lower test at 0.70 failed because only 0.95 GiB of KV-cache memory was available while 1.0 GiB was required for the configured maximum sequence length of 8192. |
| KV/VRAM control | Qwen | 0.65 | Retained near-boundary campaign configuration. A successful start provided a GPU KV-cache capacity of 9,088 tokens for a maximum model length of 8192. |
| Qwen lower-boundary test | Qwen | 0.60 | Failed during engine initialization. The server reported -0.70 GiB of available KV-cache memory and could not allocate cache blocks. |

## Fix-D adjustment from 0.90 to 0.86

The initial Fix-D Llama GPU-only server start used the default
`gpu_memory_utilization=0.90`.

The model loaded successfully, but sampler warm-up with 256 dummy requests
failed while attempting an additional 126 MiB allocation. The server reported
only about 140 MiB of free GPU memory and terminated engine initialization.

The adjustment to 0.86 was therefore an execution-stability change for the
Fix-D block. It was not applied selectively to a single result condition:
both models and all Fix-D comparison conditions used the same 0.86 value.

Evidence:

- `data/provenance/gpu_memory_configuration/fixd_llama_gmem090_offload0_failed.log`
- `data/provenance/gpu_memory_configuration/fixd_llama_gmem086_offload0.log`
- `data/provenance/gpu_memory_configuration/fixd_llama_gmem086_offload12.log`
- `data/provenance/gpu_memory_configuration/fixd_qwen_gmem086_offload0.log`
- `data/provenance/gpu_memory_configuration/fixd_qwen_gmem086_offload12.log`

## KV/VRAM boundary tests

### Llama

At 0.70, vLLM reported:

- available KV-cache memory: 0.95 GiB;
- required KV-cache memory for one 8192-token request: 1.0 GiB;
- estimated supported maximum model length: 7760 tokens.

The engine therefore did not start. The retained campaign used 0.75.

Evidence:

- `data/provenance/gpu_memory_configuration/kv_vram_llama_gmem075_offload0.log`
- `data/provenance/gpu_memory_configuration/kv_vram_llama_gmem070_offload0_failed.log`

### Qwen

At 0.60, vLLM reported:

- model loading memory: 14.25 GiB;
- available KV-cache memory: -0.70 GiB;
- no available memory for cache-block allocation.

The engine therefore did not start. The retained campaign used 0.65.

A successful 0.65 start reported a GPU KV-cache capacity of 9,088 tokens.
During boundary exploration, another 0.65 start failed narrowly because
0.44 GiB of KV-cache memory was required but only 0.41 GiB was available.
The retained campaign setting should therefore be interpreted as a
successfully used near-boundary configuration with limited startup headroom,
not as a universal stability guarantee or as evidence that both models
experienced quantitatively identical memory pressure.

Evidence:

- `data/provenance/gpu_memory_configuration/kv_vram_qwen_gmem065_offload0.log`
- `data/provenance/gpu_memory_configuration/kv_vram_qwen_gmem065_offload0_failed.log`
- `data/provenance/gpu_memory_configuration/kv_vram_qwen_gmem060_offload0_failed.log`

## Model-loading memory

The retained server logs repeatedly report the following GPU model-loading
memory values for GPU-only execution:

| Model | Reported model-loading memory |
|---|---:|
| Llama-3.1-8B-Instruct | 14.99 GiB |
| Qwen2.5-7B-Instruct | 14.25 GiB |

Representative evidence:

- `data/provenance/gpu_memory_configuration/baseline_llama_gmem090_offload0.log`
- `data/provenance/gpu_memory_configuration/baseline_qwen_gmem090_offload0.log`

These values are vLLM server-log measurements and should not be interpreted
as complete measurements of every GPU allocation made during execution.

## Interpretation

The use of different values across campaign blocks is intentional:

- 0.90 preserves the primary baseline configuration;
- 0.86 provides a common executable setting for the matched GPU-load control;
- 0.75 and 0.65 create model-specific near-boundary KV/VRAM controls.

The control results therefore support within-block comparisons. They do not
claim that 0.75 for Llama and 0.65 for Qwen represent quantitatively identical
memory pressure.

## Integrity and source mapping

The files beneath `data/provenance/gpu_memory_configuration/` are curated
excerpts rather than complete server logs. Only configuration, model-loading,
KV-cache, successful-start, and initialization-failure lines are retained.
Request-access lines, local ports, dummy API-key values, and unrelated
stack-trace content are excluded.

For every excerpt, `source_map.tsv` records the historical source path,
source-repository revision, and SHA-256 checksum of the complete original
log.

Checksums for the retained evidence files are stored in:

`data/provenance/gpu_memory_configuration_sha256.txt`

The mapping between neutral artifact filenames and their historical source
paths is stored in:

`data/provenance/gpu_memory_configuration/source_map.tsv`

The historical provenance audit was collected from the former working
repository at commit:

`72b9272c59ca0738ffe7fb012b9a33169b3b941f`

The artifact repository state immediately before adding this report was:

`71dcc103094a71418dbaadc8ec9704a82b24a2f6`

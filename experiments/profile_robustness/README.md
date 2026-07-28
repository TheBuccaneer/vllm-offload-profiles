# Profile Robustness Campaign

This campaign evaluates whether the offload-induced serving profile remains stable across request-length configurations.

## Design

- Models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct
- CPU offload: 0 GiB and 12 GiB
- Profiles: 128/32, 256/64, and 512/128 input/output tokens
- Concurrency: 4 and 8
- Repeats: 3 per cell
- Requests: 20 per run
- Total: 72 runs and 1,440 requests

The historical raw campaign identifier is preserved inside the original JSON files. New runs use the neutral identifier `profile_robustness_v1`.

## Validation

```bash
python3 analysis/validation/validate_profile_robustness.py data/raw/profile_robustness --runs 3
```

## Extraction

```bash
python3 analysis/extraction/extract_baseline.py data/raw/profile_robustness --outdir data/derived/profile_robustness
```

## Analysis

```bash
python3 analysis/separability/profile_robustness.py data/derived/profile_robustness/runs_summary.csv --output-dir results/profile_robustness
```

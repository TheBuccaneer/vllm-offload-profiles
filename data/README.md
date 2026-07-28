# Data

This directory separates canonical measurement inputs from
reproducibly generated analysis datasets.

## Raw data

`raw/` contains the original measurement outputs for the four
evaluated campaigns:

- `baseline/`
- `profile_robustness/`
- `gpu_load_control/`
- `kv_vram_control/`

These files are treated as immutable inputs. Historical identifiers
inside original records are retained where required for provenance.

## Derived data

`derived/` contains run-level and request-level CSV datasets generated
from the raw measurements.

The derived datasets can be regenerated from the repository root with:

    bash analysis/run_all_analysis.sh

## Provenance

`provenance/` contains SHA-256 manifests for every canonical raw-data
block:

- `baseline_sha256.txt`
- `profile_robustness_sha256.txt`
- `gpu_load_control_sha256.txt`
- `kv_vram_control_sha256.txt`

Additional retained configuration evidence is stored in:

- `gpu_memory_configuration/`: curated server-log excerpts
- `gpu_memory_configuration/source_map.tsv`: historical source mapping
- `gpu_memory_configuration_sha256.txt`: excerpt checksums

Verify all canonical raw inputs with:

    bash analysis/verify_checksums.sh

Verify the retained GPU-memory configuration evidence with:

    python3 analysis/validation/validate_gpu_memory_configuration_provenance.py

## License

The measurement data in this directory are licensed under the Creative
Commons Attribution 4.0 International License. See `../LICENSE-DATA`.

Third-party model weights and software are not included in this license.

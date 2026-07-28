#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / "data/provenance"
EVIDENCE = PROVENANCE / "gpu_memory_configuration"
MANIFEST = PROVENANCE / "gpu_memory_configuration_sha256.txt"
REPORT = ROOT / "results/reports/gpu_memory_configuration_provenance.md"


EXPECTED = {
    "baseline_llama_gmem090_offload0.log": [
        "Model loading took 14.99 GiB memory",
    ],
    "baseline_qwen_gmem090_offload0.log": [
        "Model loading took 14.25 GiB memory",
    ],
    "fixd_llama_gmem090_offload0_failed.log": [
        "CUDA out of memory occurred when warming up sampler with 256 dummy requests",
        "Engine core initialization failed",
    ],
    "fixd_llama_gmem086_offload0.log": [
        "'gpu_memory_utilization': 0.86",
    ],
    "fixd_llama_gmem086_offload12.log": [
        "'gpu_memory_utilization': 0.86",
        "'cpu_offload_gb': 12.0",
    ],
    "fixd_qwen_gmem086_offload0.log": [
        "'gpu_memory_utilization': 0.86",
    ],
    "fixd_qwen_gmem086_offload12.log": [
        "'gpu_memory_utilization': 0.86",
        "'cpu_offload_gb': 12.0",
    ],
    "kv_vram_llama_gmem075_offload0.log": [
        "'gpu_memory_utilization': 0.75",
        "Model loading took 14.99 GiB memory",
    ],
    "kv_vram_llama_gmem070_offload0_failed.log": [
        "'gpu_memory_utilization': 0.7",
        "available KV cache memory (0.95 GiB)",
        "max seq len (8192)",
    ],
    "kv_vram_qwen_gmem065_offload0.log": [
        "'gpu_memory_utilization': 0.65",
        "GPU KV cache size: 9,088 tokens",
    ],
    "kv_vram_qwen_gmem065_offload0_failed.log": [
        "'gpu_memory_utilization': 0.65",
        "available KV cache memory (0.41 GiB)",
        "max seq len (8192)",
    ],
    "kv_vram_qwen_gmem060_offload0_failed.log": [
        "'gpu_memory_utilization': 0.6",
        "Available KV cache memory: -0.7 GiB",
        "No available memory for the cache blocks",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"GPU MEMORY PROVENANCE: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


if not REPORT.is_file():
    fail(f"missing report: {REPORT.relative_to(ROOT)}")

if not MANIFEST.is_file():
    fail(f"missing checksum manifest: {MANIFEST.relative_to(ROOT)}")

for filename, required_strings in EXPECTED.items():
    path = EVIDENCE / filename

    if not path.is_file():
        fail(f"missing evidence file: {path.relative_to(ROOT)}")

    content = path.read_text(encoding="utf-8", errors="replace")

    for required in required_strings:
        if required not in content:
            fail(f"{filename}: missing expected text: {required!r}")


for path in sorted(EVIDENCE.glob("*.log")):
    content = path.read_text(encoding="utf-8", errors="replace")

    if not content.startswith(
        "# Curated GPU-memory configuration evidence excerpt"
    ):
        fail(f"{path.name}: missing curated-excerpt header")

    if path.stat().st_size > 50_000:
        fail(f"{path.name}: evidence excerpt unexpectedly large")

    for forbidden in (
        "pilotkey",
        "127.0.0.1",
        "/home/",
        "projects/porto",
        "paper1_workspace",
        "POST /v1/",
        "GET /metrics",
    ):
        if forbidden in content:
            fail(f"{path.name}: forbidden retained content: {forbidden!r}")

source_map = EVIDENCE / "source_map.tsv"
source_map_text = source_map.read_text(encoding="utf-8")

for required_column in (
    "source_repository_commit",
    "source_sha256",
):
    if required_column not in source_map_text.splitlines()[0]:
        fail(f"source_map.tsv missing column: {required_column}")

manifest_entries: dict[str, str] = {}

for raw_line in MANIFEST.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()

    if not line:
        continue

    try:
        expected_hash, relative_path = line.split(maxsplit=1)
    except ValueError:
        fail(f"malformed checksum line: {raw_line!r}")

    relative_path = relative_path.lstrip("*")
    manifest_entries[relative_path] = expected_hash

for path in sorted(EVIDENCE.iterdir()):
    if not path.is_file():
        continue

    relative_path = str(path.relative_to(PROVENANCE))
    expected_hash = manifest_entries.get(relative_path)

    if expected_hash is None:
        fail(f"missing checksum entry: {relative_path}")

    actual_hash = sha256(path)

    if actual_hash != expected_hash:
        fail(
            f"checksum mismatch for {relative_path}: "
            f"expected {expected_hash}, got {actual_hash}"
        )

report_text = REPORT.read_text(encoding="utf-8")

for required in (
    "All reported contrasts and ratios are formed only between matched conditions",
    "0.90 preserves the primary baseline configuration",
    "0.86 provides a common executable setting",
    "0.75 and 0.65 create model-specific near-boundary",
):
    if required not in report_text:
        fail(f"report missing required statement: {required!r}")

print(
    "GPU MEMORY PROVENANCE: PASS "
    f"({len(EXPECTED)} evidence logs, checksums and report verified)"
)

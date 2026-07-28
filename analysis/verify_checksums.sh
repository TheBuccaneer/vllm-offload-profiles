#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"

cd "$REPO_ROOT"

BLOCKS=(
  baseline
  profile_robustness
  gpu_load_control
  kv_vram_control
)

for block in "${BLOCKS[@]}"; do
  manifest="data/provenance/${block}_sha256.txt"
  raw_root="data/raw/${block}"

  [[ -f "$manifest" ]] || {
    echo "FAIL: missing manifest: $manifest" >&2
    exit 1
  }

  raw_count="$(find "$raw_root" -type f | wc -l)"
  hash_count="$(wc -l < "$manifest")"

  if [[ "$raw_count" -ne "$hash_count" ]]; then
    echo "FAIL: $block raw=$raw_count manifest=$hash_count" >&2
    exit 1
  fi

  sha256sum --quiet --check "$manifest"
  printf 'PASS %-22s files=%s\n' "$block" "$raw_count"
done

echo "PASS: all raw-data checksums verified"

#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "tie_word_embeddings": False,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "attention_bias": False,
        "mlp_bias": False,
        "qk_norm": False,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_type": "qwen2",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "vocab_size": 152064,
        "tie_word_embeddings": False,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "qk_norm": False,
    },
}

BYTES_PER_PARAMETER = 2  # bfloat16, unquantized weights
GIB = 1024**3


def load_config(path: Path | None, fallback: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return dict(fallback)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def count_decoder_params(config: dict[str, Any], family: str) -> dict[str, Any]:
    if family not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model family: {family}")
    expected_type = MODEL_CONFIGS[family]["model_type"]
    if config.get("model_type") != expected_type:
        raise ValueError(f"Expected model_type={expected_type}, got {config.get('model_type')!r}")

    h = int(config["hidden_size"])
    i = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    q_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    if not all(x > 0 for x in (h, i, layers, q_heads, kv_heads)):
        raise ValueError("Model dimensions must be positive")
    if h % q_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if q_heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    head_dim = int(config.get("head_dim") or h // q_heads)
    if bool(config.get("qk_norm", False)):
        raise ValueError("qk_norm=True is not supported by this study-specific counter")

    if family == "llama":
        qkv_bias = bool(config.get("attention_bias", False))
        out_bias = bool(config.get("attention_bias", False))
        mlp_bias = bool(config.get("mlp_bias", False))
    else:
        # vLLM 0.17.1 Qwen2Attention constructs QKVParallelLinear(bias=True),
        # and o_proj, gate_up_proj and down_proj with bias=False.
        qkv_bias = True
        out_bias = False
        mlp_bias = False

    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    pieces = {
        "attention_qkv_weights": h * (q_dim + 2 * kv_dim),
        "attention_o_weights": q_dim * h,
        "mlp_gate_up_down_weights": 3 * h * i,
        "input_and_post_attention_norm": 2 * h,
        "attention_qkv_biases": (q_dim + 2 * kv_dim) if qkv_bias else 0,
        "attention_o_bias": h if out_bias else 0,
        "mlp_biases": (2 * i + h) if mlp_bias else 0,
    }
    per_layer = sum(pieces.values())
    total = per_layer * layers
    # For both studied models, embeddings and lm_head are distinct matrices.
    # They and the final RMSNorm are outside make_layers() and are not wrapped
    # by the UVA offloader. This also supplies an independent total-size check.
    vocab = int(config["vocab_size"])
    tied = bool(config.get("tie_word_embeddings", False))
    nondecoder_params = vocab * h * (1 if tied else 2) + h
    model_total = total + nondecoder_params
    return {
        "model_id": MODEL_CONFIGS[family]["model_id"],
        "model_type": expected_type,
        "num_hidden_layers": layers,
        "parameters_per_decoder_layer": per_layer,
        "eligible_decoder_parameters": total,
        "bytes_per_parameter": BYTES_PER_PARAMETER,
        "eligible_decoder_bytes": total * BYTES_PER_PARAMETER,
        "eligible_decoder_gib": total * BYTES_PER_PARAMETER / GIB,
        "not_uva_eligible_parameters": nondecoder_params,
        "not_uva_eligible_gib": nondecoder_params * BYTES_PER_PARAMETER / GIB,
        "model_parameters_total": model_total,
        "model_parameter_gib": model_total * BYTES_PER_PARAMETER / GIB,
        "per_layer_breakdown": pieces,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-config", type=Path, help="Local HF config.json for Llama")
    ap.add_argument("--qwen-config", type=Path, help="Local HF config.json for Qwen")
    ap.add_argument("--json-output", type=Path, help="Write full report to JSON")
    ap.add_argument("--csv-output", type=Path, help="Write summary to CSV")
    args = ap.parse_args()

    results = [
        count_decoder_params(load_config(args.llama_config, MODEL_CONFIGS["llama"]), "llama"),
        count_decoder_params(load_config(args.qwen_config, MODEL_CONFIGS["qwen"]), "qwen"),
    ]
    report = {
        "calculation": "Analytical count of decoder-layer parameters eligible for UVA offloading; not measured traffic",
        "implementation": "vLLM v0.17.1, tensor parallel size 1, unquantized bfloat16",
        "excluded": "Token embeddings, final model norm, and output projection (lm_head)",
        "results": results,
    }

    for row in results:
        print(f"{row['model_id']}: {row['eligible_decoder_parameters']:,} params "
              f"= {row['eligible_decoder_gib']:.8f} GiB BF16 "
              f"(per decoder layer {row['parameters_per_decoder_layer']:,}); "
              f"model total {row['model_parameter_gib']:.8f} GiB")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model_id", "num_hidden_layers", "parameters_per_decoder_layer",
                "eligible_decoder_parameters", "bytes_per_parameter",
                "eligible_decoder_bytes", "eligible_decoder_gib",
                "not_uva_eligible_parameters", "not_uva_eligible_gib",
                "model_parameters_total", "model_parameter_gib",
            ])
            writer.writeheader()
            for row in results:
                writer.writerow({field: row[field] for field in writer.fieldnames})


if __name__ == "__main__":
    main()

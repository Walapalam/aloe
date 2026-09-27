"""Probe mlx-lm to answer Q3 (docs/DECISIONS.md):

Can mlx-lm target individual modules by name with per-module rank and alpha?

Run with:  python scripts/probe_mlx.py

Expected outcomes:
- If "per_layer_rank_scale" succeeds, MLX is a first-class typed backend.
- If only "linear_names filter" succeeds, MLX can filter targets but uses uniform
  rank/alpha; typed runs need manual parameter surgery or a CUDA backend.
- If both fail, MLX is baseline-only.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers


MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
OUT_PATH = Path("runs/mlx_probe_report.json")


def _sample_lora_keys(params: dict, n: int = 10) -> list[str]:
    return [k for k in params.keys() if "lora" in k.lower()][:n]


def _try_attach(model, **kwargs) -> dict:
    try:
        linear_to_lora_layers(model, **kwargs)
        return {
            "status": "ok",
            "lora_key_count": len(_sample_lora_keys(model.parameters(), n=10_000)),
            "sample_keys": _sample_lora_keys(model.parameters()),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": repr(e)}


def main() -> None:
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)

    sig = inspect.signature(linear_to_lora_layers)
    print("linear_to_lora_layers signature:", sig)

    report = {
        "model_id": MODEL_ID,
        "linear_to_lora_layers_signature": str(sig),
        "per_module_attempts": [],
    }

    # Attempt 1: global uniform LoRA (the baseline case).
    report["per_module_attempts"].append(
        {"name": "uniform", "kwargs": {"rank": 8, "scale": 2.0}}
    )
    report["per_module_attempts"][-1].update(_try_attach(model, rank=8, scale=2.0))

    # Attempt 2: filter by linear name only (same rank/scale everywhere).
    report["per_module_attempts"].append(
        {"name": "linear_names_filter", "kwargs": {"rank": 8, "scale": 2.0, "linear_names": ["q_proj", "v_proj"]}}
    )
    report["per_module_attempts"][-1].update(
        _try_attach(model, rank=8, scale=2.0, linear_names=["q_proj", "v_proj"])
    )

    # Attempt 3: per-layer rank/scale dict, if supported.
    # This is the make-or-break call for architecture-typed search on MLX.
    per_layer_kwargs = {
        "rank": {0: 8, 1: 16, 2: 32},
        "scale": {0: 2.0, 1: 1.0, 2: 0.5},
    }
    report["per_module_attempts"].append(
        {"name": "per_layer_rank_scale", "kwargs": per_layer_kwargs}
    )
    report["per_module_attempts"][-1].update(_try_attach(model, **per_layer_kwargs))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {OUT_PATH}")
    for a in report["per_module_attempts"]:
        print(f"  {a['name']}: {a['status']}")


if __name__ == "__main__":
    main()

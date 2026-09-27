#!/usr/bin/env python3
"""ALOE Stage 1 / R1: Backend Capability Probe.

Empirically probes PEFT on MPS and Apple MLX for per-group rank and alpha LoRA support.
Writes runs/probe/backends.json adhering to Result Schema Envelope (schema_version: "1.0").

Adheres to:
- ADR-002: Core framework isolation
- ADR-003: Two execution backends (PEFT on MPS and MLX)
- Gate 1: Backend capability verification
- Gate 6: Standard Envelope (schema_version: "1.0")
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import inspect
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

try:
    import mlx.nn as mnn
    from mlx_lm.tuner import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "UNKNOWN"


def get_cpu_chip() -> str:
    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return platform.processor() or "Apple Silicon"


def get_environment_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "chip": get_cpu_chip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": "torch_mps" if torch.backends.mps.is_available() else "cpu",
    }


class ToyAttention(nn.Module):
    def __init__(self, d_model: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q + k + v)


class ToyMLP(nn.Module):
    def __init__(self, d_model: int = 32, d_ff: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class ToyTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 32, d_ff: int = 64):
        super().__init__()
        self.self_attn = ToyAttention(d_model)
        self.mlp = ToyMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class ToyModel(nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 32, d_ff: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([ToyTransformerBlock(d_model, d_ff) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def probe_peft_mps() -> dict[str, Any]:
    mps_available = torch.backends.mps.is_available()
    device = torch.device("mps" if mps_available else "cpu")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    rank_pattern = {"gate_proj": 16, "up_proj": 16, "down_proj": 16}
    alpha_pattern = {"gate_proj": 32, "up_proj": 32, "down_proj": 32}

    base_model = ToyModel(num_layers=2).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=False,
    )

    peft_model = get_peft_model(base_model, lora_config)

    verified_modules: list[dict[str, Any]] = []
    observed_ranks = set()
    observed_alphas = set()

    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            r = module.r["default"] if isinstance(module.r, dict) else module.r
            alpha = module.lora_alpha["default"] if isinstance(module.lora_alpha, dict) else module.lora_alpha
            scaling = module.scaling["default"] if isinstance(module.scaling, dict) else module.scaling
            shape = list(module.lora_A["default"].weight.shape)

            observed_ranks.add(r)
            observed_alphas.add(alpha)
            target_name = name.split(".")[-1]
            verified_modules.append({
                "name": name,
                "target": target_name,
                "actual_r": r,
                "actual_alpha": alpha,
                "actual_scaling": float(scaling),
                "param_shape": shape,
            })

    x = torch.randn(2, 4, 32, device=device, requires_grad=True)
    out = peft_model(x)
    loss = out.sum()
    loss.backward()

    adapter_param_gradients: list[dict[str, Any]] = []
    backward_success = True

    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            if param.grad is None:
                backward_success = False
                norm_val = 0.0
            else:
                norm_val = float(param.grad.norm().item())
                if norm_val <= 1e-8:
                    backward_success = False
            adapter_param_gradients.append({"param": name, "norm": norm_val})

    persistence_ok = False
    tmp_dir = tempfile.mkdtemp()
    try:
        peft_model.save_pretrained(tmp_dir)
        fresh_base = ToyModel(num_layers=2).to(device)
        reloaded_model = PeftModel.from_pretrained(fresh_base, tmp_dir)
        reloaded_ranks = {
            module.r["default"] if isinstance(module.r, dict) else module.r
            for _, module in reloaded_model.named_modules()
            if hasattr(module, "lora_A") and "default" in module.lora_A
        }
        if reloaded_ranks == observed_ranks:
            persistence_ok = True
    except (RuntimeError, OSError, ValueError):
        persistence_ok = False
    finally:
        shutil.rmtree(tmp_dir)

    distinct_ranks = len(observed_ranks)
    distinct_alphas = len(observed_alphas)
    test_passed = (
        mps_available
        and distinct_ranks > 1
        and distinct_alphas > 1
        and backward_success
        and persistence_ok
    )

    return {
        "name": "peft",
        "version": importlib.metadata.version("peft"),
        "device": "mps" if mps_available else "cpu",
        "supports_per_group_rank_alpha": test_passed,
        "mechanism": (
            "peft.LoraConfig(rank_pattern=..., alpha_pattern=...) maps module name regex/suffix "
            "patterns to specific rank and alpha values during layer creation in LoraModel._create_and_replace."
        ),
        "empirical_test": {
            "model_architecture": "ToyModel (2-layer Attention + MLP)",
            "target_modules": target_modules,
            "requested_rank_pattern": rank_pattern,
            "requested_alpha_pattern": alpha_pattern,
            "verified_modules": verified_modules,
            "gradient_flow": {
                "backward_pass_success": backward_success,
                "adapter_param_gradients": adapter_param_gradients,
            },
            "distinct_ranks_count": distinct_ranks,
            "distinct_alphas_count": distinct_alphas,
            "test_passed": test_passed,
        },
        "status": "PASS" if test_passed else "FAIL",
    }


def probe_mlx_lm() -> dict[str, Any]:
    if not MLX_AVAILABLE:
        return {
            "name": "mlx-lm",
            "version": "NOT_INSTALLED",
            "supports_per_group_rank_alpha": False,
            "source_inspection": {
                "entrypoint": "mlx_lm.tuner.linear_to_lora_layers",
                "signature": "NOT_DETERMINED",
                "keys_filtering_supported": False,
                "per_module_rank_supported": False,
                "evidence": "mlx_lm is not installed in the environment.",
            },
            "empirical_test": {
                "uniform_lora": "NOT_RUN",
                "key_filtering_lora": "NOT_RUN",
                "per_module_dict_rank_lora": "NOT_RUN",
                "error_message": "mlx_lm not installed",
            },
            "status": "NOT_INSTALLED",
        }

    mlx_lm_version = importlib.metadata.version("mlx-lm")
    sig = str(inspect.signature(linear_to_lora_layers))

    evidence = (
        "mlx_lm.tuner.linear_to_lora_layers takes config['rank'] (int) and config['scale'] (float) "
        "as global parameters and passes them to LoRALayer.from_base. The inner helper to_lora(layer) "
        "does not receive module names, making per-module dictionary rank lookup impossible. "
        "Passing a dictionary to config['rank'] causes a TypeError when constructing mx.random.uniform."
    )

    class MLXDumTransformer(mnn.Module):
        def __init__(self):
            super().__init__()
            class Layer(mnn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = mnn.Linear(16, 16)
                    self.gate_proj = mnn.Linear(16, 32)
            self.layers = [Layer() for _ in range(2)]

    m1 = MLXDumTransformer()
    try:
        linear_to_lora_layers(m1, num_layers=2, config={"rank": 8, "scale": 16.0, "dropout": 0.0})
        uniform_res = "PASS"
    except (KeyError, ValueError, TypeError, RuntimeError) as e:
        uniform_res = f"FAIL: {e}"

    m2 = MLXDumTransformer()
    try:
        linear_to_lora_layers(m2, num_layers=2, config={"rank": 8, "scale": 16.0, "dropout": 0.0, "keys": ["q_proj"]})
        key_filter_res = "PASS"
    except (KeyError, ValueError, TypeError, RuntimeError) as e:
        key_filter_res = f"FAIL: {e}"

    m3 = MLXDumTransformer()
    dict_rank_res = "PASS"
    err_msg: str | None = None
    try:
        linear_to_lora_layers(m3, num_layers=2, config={"rank": {"q_proj": 8, "gate_proj": 16}, "scale": 16.0, "dropout": 0.0})
    except (KeyError, ValueError, TypeError, RuntimeError) as e:
        dict_rank_res = f"FAIL: {type(e).__name__}"
        err_msg = f"{type(e).__name__}: {e}"

    supports_per_group = (dict_rank_res == "PASS")

    return {
        "name": "mlx-lm",
        "version": mlx_lm_version,
        "supports_per_group_rank_alpha": supports_per_group,
        "source_inspection": {
            "entrypoint": "mlx_lm.tuner.linear_to_lora_layers",
            "signature": sig,
            "keys_filtering_supported": (key_filter_res == "PASS"),
            "per_module_rank_supported": supports_per_group,
            "evidence": evidence,
        },
        "empirical_test": {
            "uniform_lora": uniform_res,
            "key_filtering_lora": key_filter_res,
            "per_module_dict_rank_lora": dict_rank_res,
            "error_message": err_msg,
        },
        "status": "PASS" if supports_per_group else "UNIFORM_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe backend LoRA capabilities on Apple Silicon.")
    parser.add_argument(
        "--out-file",
        default="runs/probe/backends.json",
        help="Target path for backends.json (default: runs/probe/backends.json)",
    )
    args = parser.parse_args()

    run_id = str(uuid.uuid4())
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_commit = get_git_commit()
    env_info = get_environment_info()

    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Probing PEFT on MPS capability...")
    peft_probe = probe_peft_mps()

    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Probing Apple MLX (mlx-lm) capability...")
    mlx_probe = probe_mlx_lm()

    at_least_one = peft_probe["supports_per_group_rank_alpha"] or mlx_probe["supports_per_group_rank_alpha"]
    gate_1_status = "PASS" if at_least_one else "FAIL"

    passing_backend = None
    if peft_probe["supports_per_group_rank_alpha"]:
        passing_backend = "peft_mps"
    elif mlx_probe["supports_per_group_rank_alpha"]:
        passing_backend = "mlx_lm"

    if gate_1_status == "PASS":
        summary = (
            "Per-group rank and alpha assignment IS possible on this machine, via PEFT on MPS, "
            "evidenced by runs/probe/backends.json:backends.peft_mps.empirical_test.status PASS."
        )
        fallback_options = None
    else:
        summary = "Neither PEFT on MPS nor MLX supports per-group rank and alpha assignment."
        fallback_options = [
            "Option (a): Constrain search space to uniform rank across all target modules.",
            "Option (b): Implement custom MLX module surgery to wrap individual linears with differing rank.",
            "Option (c): Escalate to supervisor for external CUDA access.",
        ]

    failures = []
    if gate_1_status == "FAIL":
        failures.append("Gate 1 failed: No backend supports per-group rank and alpha targeting.")

    envelope = {
        "schema_version": "1.0",
        "run_id": run_id,
        "timestamp_utc": now_utc,
        "script": "probe_backends.py",
        "git_commit": git_commit,
        "environment": env_info,
        "gate_1": {
            "gate_id": "Gate 1",
            "name": "Backend Capability Gate",
            "status": gate_1_status,
            "summary_sentence": summary,
            "passing_backend": passing_backend,
            "fallback_options": fallback_options,
        },
        "backends": {
            "peft_mps": peft_probe,
            "mlx_lm": mlx_probe,
        },
        "status": gate_1_status,
        "failures": failures,
        "not_determined": [],
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2)

    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Wrote backends probe envelope to {out_path}")
    print("\n" + "=" * 60)
    print("BACKEND PROBE SUMMARY:")
    print(f"  Gate 1 Status:     {gate_1_status}")
    print(f"  Passing Backend:   {passing_backend}")
    print(f"  PEFT on MPS:       {peft_probe['status']}")
    print(f"  MLX (mlx-lm):      {mlx_probe['status']}")
    print(f"  Summary Sentence:  {summary}")
    print("=" * 60)

    if gate_1_status != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()

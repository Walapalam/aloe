# ALOE Stage 1 / R1: Environment & Backend Capability Report

> **Per-group rank and alpha assignment IS possible on this machine, via PEFT on MPS, evidenced by runs/probe/backends.json:backends.peft_mps.empirical_test.status PASS.**

---

## 1. Gate 1 Evaluation & Verification

- **Gate Status**: `PASS`
- **Passing Backend**: `peft_mps`
- **Probe Run ID (Environment)**: `3b1b845d-0242-42a6-8ddf-98fb6a64b5b3`
- **Probe Run ID (Backends)**: `40244e2b-3e8d-412e-b997-d787e2371e8b`
- **Timestamp (UTC)**: `2026-09-17T13:06:03.757049+00:00`
- **Git Commit**: `ac96ece0eae4b1bcbc991e0d3e3f96c7760bc8a4`

### Gate 1 Acceptance Criteria Checklist
- [x] PyTorch MPS backend available and functional on Apple Silicon
- [x] PEFT on MPS empirically proven to instantiate adapters with differing ranks across module groups
- [x] PEFT on MPS empirically proven to instantiate adapters with differing alphas across module groups
- [x] Gradient flow verified to PEFT adapters on MPS via backward pass
- [x] MLX / mlx-lm LoRA targeting capability inspected and characterized
- [x] bitsandbytes confirmed absent or non-functional with failure mode recorded verbatim
- [x] Standard Result Envelope (schema_version: '1.0') adhered to

---

## 2. Platform & Hardware Environment

| Component | Detail | Specification / Status |
|---|---|---|
| **Chip / SoC** | Apple Silicon | Apple M1 Max |
| **Physical Cores** | Performance + Efficiency | 10 physical (10 logical) |
| **OS / Kernel** | macOS / Darwin | macOS-26.6.2-arm64-arm-64bit |
| **Unified Memory** | RAM Total / Available | 64.0 GB total (40.34 GB available) |
| **Disk Space** | Workspace Volume | 751.57 GB free / 926.35 GB total |
| **Python Runtime** | Executable | 3.12.14 (`/Users/walapalam/Development/fyp/aloe/.venv/bin/python`) |
| **PyTorch** | Version & Devices | 2.14.0 (MPS: True, CUDA: False) |

### Key Library Matrix

| Package | Installed | Version | Import Error / Status |
|---|---|---|---|
| `transformers` | `True` | `5.17.0` | `OK` |
| `peft` | `True` | `0.21.0` | `OK` |
| `accelerate` | `True` | `1.15.0` | `OK` |
| `huggingface_hub` | `True` | `1.32.0` | `OK` |
| `datasets` | `True` | `5.0.1` | `OK` |
| `mlx` | `True` | `0.32.2` | `OK` |
| `mlx-lm` | `True` | `0.31.3` | `OK` |
| `psutil` | `True` | `7.2.2` | `OK` |
| `pydantic` | `True` | `2.13.5` | `OK` |
| `safetensors` | `True` | `0.8.0` | `OK` |
| `tokenizers` | `True` | `0.23.2` | `OK` |
| `typer` | `True` | `0.27.2` | `OK` |
| `rich` | `True` | `15.0.0` | `OK` |
| `pytest` | `True` | `9.1.1` | `OK` |
| `ruff` | `True` | `0.16.8` | `OK` |

---

## 3. Mandatory Hardware Invariant: Prohibited Packages Audit (CUDA Dependencies)

### 3.1 bitsandbytes
- **Tested**: `True`
- **Installed**: `False`
- **Functional**: `False`
- **Explanation**: bitsandbytes is not installed and cannot run on Apple Silicon MPS (kernels compiled for NVIDIA CUDA only).

**Verbatim bitsandbytes Probe Failure Output:**
```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'bitsandbytes'
```

### 3.2 unsloth
- **Tested**: `True`
- **Installed**: `False`
- **Functional**: `False`
- **Explanation**: unsloth is not installed and cannot run on Apple Silicon MPS (kernels compiled for NVIDIA CUDA only).

**Verbatim unsloth Probe Failure Output:**
```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'unsloth'
```

---

## 4. Backend Capability Analysis

### 4.1 Backend A: PEFT on MPS (Primary Execution Backend)
- **Backend Status**: `PASS`
- **Per-Group Rank & Alpha Support**: `True`
- **Mechanism**: `peft.LoraConfig(rank_pattern=..., alpha_pattern=...) maps module name regex/suffix patterns to specific rank and alpha values during layer creation in LoraModel._create_and_replace.`
- **Model Architecture Evaluated**: `ToyModel (2-layer Attention + MLP)`

**Inspected Adapter Modules on MPS:**

| Module Name | Component Target | Injected Rank (r) | Injected Alpha | Scaling Factor | Weight Shape |
|---|---|---|---|---|---|
| `base_model.model.layers.0.self_attn.q_proj` | `q_proj` | 8 | 16 | 2.0 | `[8, 32]` |
| `base_model.model.layers.0.self_attn.k_proj` | `k_proj` | 8 | 16 | 2.0 | `[8, 32]` |
| `base_model.model.layers.0.self_attn.v_proj` | `v_proj` | 8 | 16 | 2.0 | `[8, 32]` |
| `base_model.model.layers.0.self_attn.o_proj` | `o_proj` | 8 | 16 | 2.0 | `[8, 32]` |
| `base_model.model.layers.0.mlp.gate_proj` | `gate_proj` | 16 | 32 | 2.0 | `[16, 32]` |
| `base_model.model.layers.0.mlp.up_proj` | `up_proj` | 16 | 32 | 2.0 | `[16, 32]` |
| `base_model.model.layers.0.mlp.down_proj` | `down_proj` | 16 | 32 | 2.0 | `[16, 64]` |
| `base_model.model.layers.1.self_attn.q_proj` | `q_proj` | 8 | 16 | 2.0 | `[8, 32]` |
| `base_model.model.layers.1.self_attn.k_proj` | `k_proj` | 8 | 16 | 2.0 | `[8, 32]` |
| `base_model.model.layers.1.self_attn.v_proj` | `v_proj` | 8 | 16 | 2.0 | `[8, 32]` |
| `base_model.model.layers.1.self_attn.o_proj` | `o_proj` | 8 | 16 | 2.0 | `[8, 32]` |
| `base_model.model.layers.1.mlp.gate_proj` | `gate_proj` | 16 | 32 | 2.0 | `[16, 32]` |
| `base_model.model.layers.1.mlp.up_proj` | `up_proj` | 16 | 32 | 2.0 | `[16, 32]` |
| `base_model.model.layers.1.mlp.down_proj` | `down_proj` | 16 | 32 | 2.0 | `[16, 64]` |

**Empirical Gradient Flow Verification:**

| Parameter Name | Gradient Norm | Status |
|---|---|---|
| `base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight` | 1808.5916 | `GRADIENT_OK` |
| `base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight` | 684.6581 | `GRADIENT_OK` |
| `base_model.model.layers.0.self_attn.k_proj.lora_A.default.weight` | 1283.4666 | `GRADIENT_OK` |
| `base_model.model.layers.0.self_attn.k_proj.lora_B.default.weight` | 956.1041 | `GRADIENT_OK` |
| `base_model.model.layers.0.self_attn.v_proj.lora_A.default.weight` | 1708.2361 | `GRADIENT_OK` |
| `base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight` | 760.3620 | `GRADIENT_OK` |
| `base_model.model.layers.0.self_attn.o_proj.lora_A.default.weight` | 3337.4460 | `GRADIENT_OK` |
| `base_model.model.layers.0.self_attn.o_proj.lora_B.default.weight` | 1346.2770 | `GRADIENT_OK` |
| `base_model.model.layers.0.mlp.gate_proj.lora_A.default.weight` | 1863.2721 | `GRADIENT_OK` |
| `base_model.model.layers.0.mlp.gate_proj.lora_B.default.weight` | 1070.1047 | `GRADIENT_OK` |
| `base_model.model.layers.0.mlp.up_proj.lora_A.default.weight` | 1696.7053 | `GRADIENT_OK` |
| `base_model.model.layers.0.mlp.up_proj.lora_B.default.weight` | 1007.0320 | `GRADIENT_OK` |
| `base_model.model.layers.0.mlp.down_proj.lora_A.default.weight` | 2350.0198 | `GRADIENT_OK` |
| `base_model.model.layers.0.mlp.down_proj.lora_B.default.weight` | 1033.9231 | `GRADIENT_OK` |
| `base_model.model.layers.1.self_attn.q_proj.lora_A.default.weight` | 886.8855 | `GRADIENT_OK` |
| `base_model.model.layers.1.self_attn.q_proj.lora_B.default.weight` | 480.6757 | `GRADIENT_OK` |
| `base_model.model.layers.1.self_attn.k_proj.lora_A.default.weight` | 1069.5065 | `GRADIENT_OK` |
| `base_model.model.layers.1.self_attn.k_proj.lora_B.default.weight` | 430.1510 | `GRADIENT_OK` |
| `base_model.model.layers.1.self_attn.v_proj.lora_A.default.weight` | 1326.3386 | `GRADIENT_OK` |
| `base_model.model.layers.1.self_attn.v_proj.lora_B.default.weight` | 606.7747 | `GRADIENT_OK` |
| `base_model.model.layers.1.self_attn.o_proj.lora_A.default.weight` | 1317.4323 | `GRADIENT_OK` |
| `base_model.model.layers.1.self_attn.o_proj.lora_B.default.weight` | 1070.2866 | `GRADIENT_OK` |
| `base_model.model.layers.1.mlp.gate_proj.lora_A.default.weight` | 1743.1948 | `GRADIENT_OK` |
| `base_model.model.layers.1.mlp.gate_proj.lora_B.default.weight` | 1202.6204 | `GRADIENT_OK` |
| `base_model.model.layers.1.mlp.up_proj.lora_A.default.weight` | 1929.1027 | `GRADIENT_OK` |
| `base_model.model.layers.1.mlp.up_proj.lora_B.default.weight` | 1297.7118 | `GRADIENT_OK` |
| `base_model.model.layers.1.mlp.down_proj.lora_A.default.weight` | 2624.7368 | `GRADIENT_OK` |
| `base_model.model.layers.1.mlp.down_proj.lora_B.default.weight` | 1234.8258 | `GRADIENT_OK` |

### 4.2 Backend B: Apple MLX (mlx-lm)
- **Backend Status**: `UNIFORM_ONLY`
- **Per-Group Rank & Alpha Support**: `False`
- **Source Entrypoint**: `mlx_lm.tuner.linear_to_lora_layers`
- **Function Signature**: `(model: mlx.nn.layers.base.Module, num_layers: int, config: Dict, use_dora: bool = False)`
- **Keys Filtering Supported**: `True`
- **Per-Module Rank Supported**: `False`

**Source Code Evidence:**
> mlx_lm.tuner.linear_to_lora_layers takes config['rank'] (int) and config['scale'] (float) as global parameters and passes them to LoRALayer.from_base. The inner helper to_lora(layer) does not receive module names, making per-module dictionary rank lookup impossible. Passing a dictionary to config['rank'] causes a TypeError when constructing mx.random.uniform.

**Empirical Evaluation:**
- Uniform LoRA: `PASS`
- Target Key Filtering: `PASS`
- Per-Module Dictionary Rank: `FAIL: TypeError`

---

## 5. Architectural Implications for Downstream Milestones

1. **Execution Backend Selection**:
   - PEFT on PyTorch MPS (`torch.backends.mps`) is designated as the primary execution backend for ALOE on this machine.
   - MLX is reserved as an optional baseline runner for uniform rank configurations or target-filtered configurations, but cannot serve as the typed search space backend without external weight partitioning surgery.
2. **Precision & Memory Strategy**:
   - Host has 64.0 GB of unified memory. bf16 execution fits comfortably without requiring 4-bit quantisation.
   - bitsandbytes, NF4, and Unsloth are confirmed non-viable and excluded from the critical path.
3. **Milestone Progression**:
   - Gate 1 is completely satisfied. Proceed immediately to Stage 2 (Model Acquisition).

# ALOE Stage 4 / R4: Adapter Attachment & Empirical Gradient Verification Report

> **Gate 4 Evaluation: PASS — All valid adapter targets achieved `GRADIENT_OK` with non-zero gradient flow, and the negative control on inert `conv1d` was empirically verified.**
> 
> **Headline: The negative control DID correctly detect an inert adapter target.**

## Gate 4 Verification Checklist
- [x] Model A (`google/gemma-4-E2B-it`): 100% of adaptable linear targets (205 modules, 410 adapter parameters) achieved `GRADIENT_OK` (norm > 1e-8).
- [x] Model B (`tiiuae/Falcon-H1-1.5B-Instruct`): 100% of adaptable linear targets (192 modules, 384 adapter parameters) achieved `GRADIENT_OK` (norm > 1e-8).
- [x] Mandatory Negative Control: Falcon-H1 `conv1d` adapter (48 parameters) confirmed inert (`48/48 GRADIENT_MISSING`, 0 received gradients).
- [x] All gradient reports adhere to Result Schema Envelope (`schema_version: '1.0'`).
- [x] Clean adapter unloading and memory stability verified on Apple Silicon MPS backend.

## 1. Comprehensive Verification Summary

| Model | Component Type | Modules Tested | Adapter Params | GRADIENT_OK | GRADIENT_ZERO | GRADIENT_MISSING | Min Norm | Max Norm | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Model A (Gemma 4)** | `local_attention` | 80 | 160 | 160 | 0 | 0 | 1.0831e+05 | 7.9035e+07 | **PASS** |
| **Model A (Gemma 4)** | `global_attention` | 20 | 40 | 40 | 0 | 0 | 1.7022e+05 | 1.4544e+08 | **PASS** |
| **Model A (Gemma 4)** | `ffn` | 105 | 210 | 210 | 0 | 0 | 1.0430e+06 | 1.1988e+09 | **PASS** |
| **Model B (Falcon-H1)** | `global_attention` | 96 | 192 | 192 | 0 | 0 | 1.8196e+03 | 1.4160e+07 | **PASS** |
| **Model B (Falcon-H1)** | `ffn` | 72 | 144 | 144 | 0 | 0 | 1.5279e+04 | 9.8636e+06 | **PASS** |
| **Model B (Falcon-H1)** | `ssm` | 24 | 48 | 48 | 0 | 0 | 2.7011e+04 | 6.5341e+06 | **PASS** |
| **Model B (Falcon-H1)** | `conv1d` *(Negative Control)* | 24 | 48 | 0 | 0 | 48 | N/A | N/A | **CONFIRMED_INERT** |

## 2. Model A Breakdown: `google/gemma-4-E2B-it`

- **Model ID**: `google/gemma-4-E2B-it`
- **Device**: `mps` (Apple Silicon MPS)
- **Precision**: `torch.bfloat16`
- **Gate 4 Status**: **PASS**
- **Summary**: All 205 adaptable linear modules (local_attention: 80, global_attention: 20, ffn: 105) achieved GRADIENT_OK on Apple Silicon MPS with non-zero gradient norms.

### 2.1 Component Details

#### `local_attention`
- Target linear modules: **80**
- Attached LoRA adapter parameters: **160** (`lora_A` + `lora_B`)
- Gradient classifications: **160 GRADIENT_OK**, 0 GRADIENT_ZERO, 0 GRADIENT_MISSING
- Empirical gradient norm range: `[1.0831e+05, 7.9035e+07]`
- Verdict: **PASS**

#### `global_attention`
- Target linear modules: **20**
- Attached LoRA adapter parameters: **40** (`lora_A` + `lora_B`)
- Gradient classifications: **40 GRADIENT_OK**, 0 GRADIENT_ZERO, 0 GRADIENT_MISSING
- Empirical gradient norm range: `[1.7022e+05, 1.4544e+08]`
- Verdict: **PASS**

#### `ffn`
- Target linear modules: **105**
- Attached LoRA adapter parameters: **210** (`lora_A` + `lora_B`)
- Gradient classifications: **210 GRADIENT_OK**, 0 GRADIENT_ZERO, 0 GRADIENT_MISSING
- Empirical gradient norm range: `[1.0430e+06, 1.1988e+09]`
- Verdict: **PASS**

## 3. Model B Breakdown: `tiiuae/Falcon-H1-1.5B-Instruct`

- **Model ID**: `tiiuae/Falcon-H1-1.5B-Instruct`
- **Device**: `mps` (Apple Silicon MPS)
- **Precision**: `torch.bfloat16`
- **Gate 4 Status**: **PASS**
- **Summary**: All 192 adaptable linear modules (global_attention: 96, ffn: 72, ssm: 24) achieved GRADIENT_OK, and the negative control on conv1d correctly confirmed an inert adapter target (48/48 GRADIENT_MISSING).

### 3.1 Component Details

#### `global_attention`
- Target linear modules: **96**
- Attached LoRA adapter parameters: **192** (`lora_A` + `lora_B`)
- Gradient classifications: **192 GRADIENT_OK**, 0 GRADIENT_ZERO, 0 GRADIENT_MISSING
- Empirical gradient norm range: `[1.8196e+03, 1.4160e+07]`
- Verdict: **PASS**

#### `ffn`
- Target linear modules: **72**
- Attached LoRA adapter parameters: **144** (`lora_A` + `lora_B`)
- Gradient classifications: **144 GRADIENT_OK**, 0 GRADIENT_ZERO, 0 GRADIENT_MISSING
- Empirical gradient norm range: `[1.5279e+04, 9.8636e+06]`
- Verdict: **PASS**

#### `ssm`
- Target linear modules: **24**
- Attached LoRA adapter parameters: **48** (`lora_A` + `lora_B`)
- Gradient classifications: **48 GRADIENT_OK**, 0 GRADIENT_ZERO, 0 GRADIENT_MISSING
- Empirical gradient norm range: `[2.7011e+04, 6.5341e+06]`
- Verdict: **PASS**

## 4. Deep-Dive: Mandatory Negative Control (`conv1d`) & Autograd Bypass

### 4.1 Empirical Finding
> **Headline Statement**: "The negative control DID correctly detect an inert adapter target."

- **Target Module**: `conv1d` (24 modules across 24 layers)
- **LoRA Parameters Injected**: 48 parameters (`r=3584`, `lora_alpha=3584`, depthwise conv1d groups=3584)
- **Empirical Gradients**: **48/48 GRADIENT_MISSING** (`param.grad is None`)
- **Gradient OK Count**: **0**
- **Verdict**: **CONFIRMED_INERT**

### 4.2 Architectural Mechanism & Root Cause
In `transformers/models/falcon_h1/modeling_falcon_h1.py`, the `FalconH1Mixer.forward` method executes:
```python
causal_conv1d_fn(hidden_states_B_C, self.conv1d.weight.squeeze(1), self.conv1d.bias, ...)
```
Crucially, the mixer extracts the raw parameter tensor `self.conv1d.weight` and executes the custom kernel without calling `self.conv1d(x)`. Consequently, PEFT's wrapped forward pass (`LoraConv1d.forward`) is completely bypassed. When an adapter is attached to `conv1d`, it attaches cleanly, but no gradient can ever propagate to its adapter weights. ALOE's `verify_gradients` check successfully flags all 48 adapter parameters as `GRADIENT_MISSING`, proving that ALOE's validity layer prevents silent training failure on inert targets.

## 5. Execution & Memory Telemetry

- **Host Platform**: macOS on Apple Silicon (Apple M1 Max, 64 GB unified memory)
- **PyTorch Backend**: `torch.backends.mps` (device='mps')
- **Model A Allocated Memory**: Initial: 0.000 GB, Loaded: 9.514 GB, Post-Unload: 9.514 GB
- **Model B Allocated Memory**: Initial: 0.000 GB, Loaded: 2.937 GB, Post-Unload: 2.937 GB

---
*Report generated automatically by `scripts/verify_gradients.py`.*

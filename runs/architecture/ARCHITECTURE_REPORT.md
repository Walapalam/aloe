# ALOE Stage 3 / R3: Architecture Verification & Claims Report

> **Gate 3 Evaluation: PASS — All architectural claims for Model A (`google/gemma-4-E2B-it`) and Model B (`tiiuae/Falcon-H1-1.5B-Instruct`) verified against loaded configs and meta-device ASTs.**

## Gate 3 Verification Checklist
- [x] Model A: 35 decoder layers confirmed (`text_config.num_hidden_layers`)
- [x] Model A: Local/global attention pattern extracted from named config field (`text_config.layer_types`)
- [x] Model A: KV-cache sharing verified (20 shared layers; `k_proj`/`v_proj` absent on layers 15..34)
- [x] Model B: Parallel hybrid attention + Mamba-2 SSM structure verified
- [x] Model B: 24 `conv1d` and 24 `out_proj` modules enumerated with shapes and excluded under ADR-005
- [x] Both models loaded on PyTorch `meta` device without RAM weight allocation
- [x] All JSON trial files adhere to Result Schema Envelope (`schema_version: '1.0'`)

## 1. Summary of Pilot Model Architectures

| Attribute | Model A (`google/gemma-4-E2B-it`) | Model B (`tiiuae/Falcon-H1-1.5B-Instruct`) |
|---|---|---|
| **Family** | `gemma4` | `falcon_h1` |
| **Topology** | `interleaved` | `parallel_hybrid` |
| **Decoder Layers** | 35 | 24 |
| **Total Parameters** | 5,506,950,688 | 1,554,863,488 |
| **Module Leaves** | 988 | 339 |
| **Pinned Revision** | `3e22461f65e8...` | `80ebc50d7799...` |
| **Inspection Device** | `meta` (0 bytes RAM) | `meta` (0 bytes RAM) |
| **Gate 3 Status** | **PASS** | **PASS** |

---

## 2. Model A Verification: Interleaved Attention & KV Sharing

**Model ID**: `google/gemma-4-E2B-it`  
**Topology**: `interleaved`  
**Language Core Parameters**: 4,628,569,344 params (Text Model) / 5,506,950,688 params (Total Multimodal Checkpoint)  

### 2.1 Decoder Layer Count & Config Attribution
- **Claim**: Model A language core contains exactly 35 decoder layers.
- **Status**: **PASS**
- **Config Field**: `text_config.num_hidden_layers` = `35`
- **AST Verification**: `model.language_model.layers` contains exactly `35` `Gemma4TextDecoderLayer` instances (indices 0 through 34).

### 2.2 Local / Global Attention Distribution
- **Claim**: Attention layers follow a repeating 4-local (sliding) + 1-global (full) pattern across 35 layers (28 local, 7 global), extracted from text_config.layer_types.
- **Status**: **PASS**
- **Named Config Field**: `text_config.layer_types` (Length: 35)
- **Sliding Window Size**: `512` tokens
- **Pattern**: Repeating 5-layer macro block: 4 local (`sliding_attention`) + 1 global (`full_attention`), repeated 7 times.

| Cycle | Layers | Local Layers (`sliding_attention`) | Global Layer (`full_attention`) | Local Head Dim | Global Head Dim |
|---|---|---|---|---|---|
| Cycle 1 | 0..4 | 0, 1, 2, 3 | 4 | 256 | 512 |
| Cycle 2 | 5..9 | 5, 6, 7, 8 | 9 | 256 | 512 |
| Cycle 3 | 10..14 | 10, 11, 12, 13 | 14 | 256 | 512 |
| Cycle 4 | 15..19 | 15, 16, 17, 18 | 19 | 256 | 512 |
| Cycle 5 | 20..24 | 20, 21, 22, 23 | 24 | 256 | 512 |
| Cycle 6 | 25..29 | 25, 26, 27, 28 | 29 | 256 | 512 |
| Cycle 7 | 30..34 | 30, 31, 32, 33 | 34 | 256 | 512 |

### 2.3 KV-Cache Sharing & Parameter Distribution
- **Claim**: KV-cache sharing is active across 20 layers (text_config.num_kv_shared_layers: 20); k_proj and v_proj exist only on layers 0..14 (15 layers) and are completely absent on layers 15..34.
- **Status**: **PASS**
- **Named Config Field**: `text_config.num_kv_shared_layers` = `20`
- **Mechanism**: Layers 0..14 compute and store KV tensors (layer 13 for sliding, layer 14 for full). Layers 15..34 share those tensors via shared_kv_states without local K/V projection weights.

| Attention Projection | Layers 0..14 (15 Layers) | Layers 15..34 (20 Layers) | Total Parameter Modules |
|---|---|---|---|
| `q_proj` | Present (15 layers) | Present (20 layers) | 35 |
| `k_proj` | **Present (15 layers)** | **ABSENT (0 layers)** | 15 |
| `v_proj` | **Present (15 layers)** | **ABSENT (0 layers)** | 15 |
| `o_proj` | Present (15 layers) | Present (20 layers) | 35 |

> **Search Space Consequence (ADR-002)**: A blind search grammar targeting 7 projections across all 35 layers would crash on layers 15..34. ALOE's architecture typing enumerates only physically instantiated modules, guaranteeing structural validity.

---

## 3. Model B Verification: Parallel Hybrid & SSM Exclusions

**Model ID**: `tiiuae/Falcon-H1-1.5B-Instruct`  
**Topology**: `parallel_hybrid`  
**Total Model Parameters**: 1,554,863,488 params  

### 3.1 Parallel Hybrid Structure
- **Claim**: Model B is a 24-layer parallel hybrid model combining self-attention and Mamba-2 SSM branches operating in parallel on the same normalized hidden state and summed before the FFN.
- **Status**: **PASS**
- **SSM Configuration**:
  - `mamba_d_ssm`: `3072`
  - `mamba_n_heads`: `48`
  - `mamba_d_head`: `64`
  - `mamba_d_state`: `256`
  - `mamba_d_conv`: `4`
- **Dataflow Graph**:
  ```
  hidden_states ───────────────────┬────────────────────────────────┐
                                   ▼                                ▼
                           [input_layernorm]                [input_layernorm]
                                   │                                │
                                   ▼                                ▼
                           [FalconH1Mixer]                  [FalconH1Attention]
                            (Mamba-2 SSM)                     (Self-Attention)
                                   │                                │
                                   └───────────────┬────────────────┘
                                                   ▼
                                        sum (mamba + attention)
                                                   │
                                                   ▼
                                           + residual connection
                                                   │
                                                   ▼
                                          [pre_ff_layernorm]
                                                   │
                                                   ▼
                                             [FalconH1MLP]
                                                   │
                                                   ▼
                                           + residual connection
  ```

### 3.2 ADR-005 Excluded Mamba Modules Enumeration
- **Claim**: Model B contains exactly 24 conv1d modules (430,080 params) and 24 out_proj modules (150,994,944 params) in the Mamba mixer path; all 48 modules (151,425,024 params) are marked Validity.EXCLUDED under ADR-005.
- **Status**: **PASS**
- **Rationale**: Borobia et al. (arXiv:2604.22127) Sec. 3.2 identified that `conv1d` feeds a custom autograd kernel bypassing LoRA and causes training instability; `out_proj` shares this path.
- **Combined Excluded SSM Parameters**: 151,425,024 params

#### Enumerated `conv1d` Modules (24 Layers)
- **Weight Shape**: `[3584, 1, 4]` | **Bias Shape**: `[3584]`
- **Params per Module**: 17,920 | **Total Across 24 Layers**: 430,080 params
- **Validity Verdict**: `excluded`
- **Exclusion Reason**: Feeds a custom autograd Mamba kernel that bypasses the LoRA forward pass; adapter is silently inert. Borobia et al. (arXiv:2604.22127) Sec. 3.2 exclude it after observing training instability.

| Module Name | Layer | Class | Weight Shape | Bias Shape | Params | Validity |
|---|---|---|---|---|---|---|
| `model.layers.0.mamba.conv1d` | 0 | `Conv1d` | `[3584, 1, 4]` | `[3584]` | 17,920 | `excluded` |
| `model.layers.1.mamba.conv1d` | 1 | `Conv1d` | `[3584, 1, 4]` | `[3584]` | 17,920 | `excluded` |
| `model.layers.2.mamba.conv1d` | 2 | `Conv1d` | `[3584, 1, 4]` | `[3584]` | 17,920 | `excluded` |
| `model.layers.3.mamba.conv1d` | 3 | `Conv1d` | `[3584, 1, 4]` | `[3584]` | 17,920 | `excluded` |
| `model.layers.4.mamba.conv1d` | 4 | `Conv1d` | `[3584, 1, 4]` | `[3584]` | 17,920 | `excluded` |
| `model.layers.5.mamba.conv1d` | 5 | `Conv1d` | `[3584, 1, 4]` | `[3584]` | 17,920 | `excluded` |
| *... [18 additional conv1d modules for layers 6..23 identical in shape]...* | 6..23 | `Conv1d` | `[3584, 1, 4]` | `[3584]` | 17,920 each | `excluded` |

#### Enumerated `out_proj` Modules (24 Layers)
- **Weight Shape**: `[2048, 3072]` | **Bias**: None
- **Params per Module**: 6,291,456 | **Total Across 24 Layers**: 150,994,944 params
- **Validity Verdict**: `excluded`
- **Exclusion Reason**: Same custom-kernel path as conv1d. Excluded in Borobia et al. Sec. 3.2.

| Module Name | Layer | Class | Weight Shape | Bias | Params | Validity |
|---|---|---|---|---|---|---|
| `model.layers.0.mamba.out_proj` | 0 | `Linear` | `[2048, 3072]` | None | 6,291,456 | `excluded` |
| `model.layers.1.mamba.out_proj` | 1 | `Linear` | `[2048, 3072]` | None | 6,291,456 | `excluded` |
| `model.layers.2.mamba.out_proj` | 2 | `Linear` | `[2048, 3072]` | None | 6,291,456 | `excluded` |
| `model.layers.3.mamba.out_proj` | 3 | `Linear` | `[2048, 3072]` | None | 6,291,456 | `excluded` |
| `model.layers.4.mamba.out_proj` | 4 | `Linear` | `[2048, 3072]` | None | 6,291,456 | `excluded` |
| `model.layers.5.mamba.out_proj` | 5 | `Linear` | `[2048, 3072]` | None | 6,291,456 | `excluded` |
| *... [18 additional out_proj modules for layers 6..23 identical in shape]...* | 6..23 | `Linear` | `[2048, 3072]` | None | 6,291,456 each | `excluded` |

---

## 4. Adaptable LoRA Search Space Groups Comparison

| Model ID | Topology | Adaptable Component Types | Valid Search Target Count | Excluded Modules Count |
|---|---|---|---|---|
| `google/gemma-4-E2B-it` | `interleaved` | `local_attention, ffn, global_attention` | 205 | 783 |
| `tiiuae/Falcon-H1-1.5B-Instruct` | `parallel_hybrid` | `ffn, ssm, global_attention` | 240 | 99 |

### Detailed Target Breakdown by Component Type

| Component Type | Model A Target Count | Model B Target Count | Adaptation Handling |
|---|---|---|---|
| `local_attention` | 80 (`q_proj`, `o_proj`, and present `k_proj`/`v_proj`) | 0 | Standard LoRA (rank/alpha) |
| `global_attention` | 20 (`q_proj`, `o_proj`, and present `k_proj`/`v_proj`) | 96 (24 layers x `q`, `k`, `v`, `o`) | Standard LoRA (rank/alpha) |
| `ffn` | 105 (35 layers x `gate`, `up`, `down`) | 72 (24 layers x `gate`, `up`, `down`) | Standard LoRA (rank/alpha) |
| `ssm` | 0 | 72 (24 `in_proj` linear + 48 mixer/recurrent) | Standard LoRA for `in_proj`; special handler for mixer |
| `embedding` | 0 (Excluded by convention) | 0 (Excluded by convention) | EXCLUDED |
| `norm` | 0 (Excluded by convention) | 0 (Excluded by convention) | EXCLUDED |
| `other` | 0 (Multimodal / lm_head excluded) | 0 (lm_head excluded) | EXCLUDED |

---

## 5. Gate 3 Pass / Fail Attestation

- **Status**: **PASS**
- **Attestation Date**: 2026-09-17
- **Hardware Tested**: Apple M1 Max (64 GB unified memory, macOS, PyTorch `meta` device)
- **Enforcement**: Gate 3 requires halting if `layer_types` cannot be extracted from config; the field was extracted verbatim without fallback or guessing.
- **Artifact Paths**:
  - `runs/architecture/google--gemma-4-E2B-it/inspection.json`
  - `runs/architecture/google--gemma-4-E2B-it/claims.json`
  - `runs/architecture/tiiuae--Falcon-H1-1.5B-Instruct/inspection.json`
  - `runs/architecture/tiiuae--Falcon-H1-1.5B-Instruct/claims.json`
  - `runs/architecture/ARCHITECTURE_REPORT.md`

# September/October Pilot Report: Multi-Objective LoRA & Parameter-Budget Parity (±5%) Evaluation

- **Generated At**: `2026-10-28 09:12:10Z`
- **Git Commit SHA**: `215d2e5de5e74a4740cdd3b09ae5168638211a15` (Identical across all 8 pilot runs)
- **Status**: **PASS (Milestone Gate C + Budget-Parity ±5%, Left-Truncated `-100` Prompt Masking & Full AST Evaluation)**
- **Dataset Split**: `data/manifests/xlam_pilot_v1.json` (Strictly Tool-Disjoint: $\mathcal{T}_{\text{train}} \cap \mathcal{T}_{\text{val}} = \emptyset$)
- **SFT Prompt Masking (`FR06`)**: Left-truncated schema prefixes (`full_ids[-max_seq_len:]`) with `-100` loss masking on 100% of prompt/schema tokens (`25/25` training examples preserve assistant JSON tool-call tokens)
- **Evaluation Set**: 8 unseen tool validation samples (`max_length=1024`, `max_new_tokens=96`, evaluated via `ASTToolCallEvaluator` alongside Zero-Shot Base Models)
- **Hardware & Precision**: Apple M1 Max (64 GB Unified Memory), macOS, PyTorch MPS (`peft_mps`), **BF16 Base Weights + FP32 LoRA Adapters**
- **Primary Memory Objective (ADR-004)**: **Peak Resident Set Size (Process RSS)** sampled at 100 ms intervals (explicitly distinguished from 4-bit NF4 QLoRA, MPS Driver-Allocated Memory, and NVIDIA CUDA Peak VRAM)

> [!IMPORTANT]
> **STATISTICAL VALIDITY & PARAMETER-BUDGET PARITY NOTICE**:
> 1. **Parameter-Budget Parity ($\pm 5\%$, FR04)**: To prevent parameter-count confounding (Schneider et al., 2025), this report explicitly separates **Strictly Parameter-Budget-Matched ($\pm 5\%$) Comparisons** (`P4_Matched` vs. `P1` on Gemma 4 E2B at `-4.66%`; `P2` vs. `P1` at `+1.95%` and `P4_SSM_Matched` vs. `P1` at `-3.12%` on Falcon-H1 1.5B) from **Unmatched Capacity Ablations** (`P2`/`P3` on Gemma at `-16.73%`/`+66.73%`, `P3` on Falcon at `+36.97%`, and the 2-step FFN-pruned smoke ablation at `-67.79%`).
> 2. **Identical-Commit Single-Seed Pilot Scope ($N=1$ on 25/12-step pilot; $N=2$ on 2-step smoke ablation)**: All 8 primary pilot trials were executed at the exact same commit (`215d2e5de5e74a4740cdd3b09ae5168638211a15`) with identical initial training losses (`2.8117` on Gemma 4 E2B; `1.5534` on Falcon-H1 1.5B). They verify pipeline execution, pre-training parameter parity enforcement, latency asymmetry, and hierarchical AST tool-calling telemetry prior to full 30–50 trial `qLogNEHVI` campaigns with $\ge 3$ random seeds and 10,000-iteration paired bootstrap significance testing.

---

## 1. Execution Budget, Precision & Hardware Telemetry Distinction

### 1.1 Hardware & Precision Separation
| Execution Environment | Precision & Adapter Mode | Primary Memory Metric | Secondary Diagnostic Metric | Purpose in ALOE |
| :--- | :--- | :--- | :--- | :--- |
| **Apple M1 Max (64 GB Unified Memory, MPS — Primary Pilot)** | **BF16 Base Weights + FP32 LoRA Adapters** (`peft_mps`) | **Peak Process RSS (GB)** sampled at 100 ms (`psutil`) | PyTorch MPS Driver Allocated Buffer (GB) | Local air-gapped pilot execution & Pareto memory objective (ADR-004) |
| **Apple M1 Max (MPS — 2-Step FFN-Pruning Smoke Ablation)** | **BF16 Base Weights + FP32 LoRA Adapters** | **MPS Driver Allocated Memory (GB)** (`torch.mps.driver_allocated_memory`) | Host Process RSS (GB) | Secondary unmatched capacity-pruning smoke ablation (`runs/pilot_cuda/SMOKE_20261012T095504Z/`) |

### 1.2 Step Latency Asymmetry (Apple Silicon MPS)
Due to Apple Silicon PyTorch MPS fallback for Mamba-2 SSM chunk scanning, step latencies exhibit a ~9× asymmetry:
- **Gemma 4 E2B**: **748–789 ms/step** (~3,786 steps fit within a 45-minute trial budget) $\rightarrow$ **25 training steps** per pilot trial.
- **Falcon-H1 1.5B**: **6,654–6,891 ms/step** (~390 steps fit within a 45-minute trial budget) $\rightarrow$ **12 training steps** per pilot trial.

All configurations within each model family share identical step counts, optimizer settings (`lr=2e-4`, `batch_size=2`, `grad_accum=4`, `max_seq_len=128`), dataset manifests, and random seeds (`seed=42`).

---

## 2. Results: Gemma 4 E2B (`google/gemma-4-E2B-it`) — Budget-Matched (±5%) vs. Unmatched Ablations

| Config | Comparison Regime | Layer-Role Allocation | Trainable Params ($\Delta$ vs. P1) | Init $\rightarrow$ Final SFT Loss | AST Parse / Tool / Arg-Name / Exact Arg-Value | Peak Process RSS (MPS Alloc) | Train Time (Adapter) | Pareto Front (Matched $\pm 5\%$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Zero-Shot)** | *Untuned Reference* | No LoRA adapters attached | 0 (`-100%`) | — | 75.0% / 75.0% / 62.5% / **37.5%** | — | — | Reference |
| **P1** | **Baseline (100%)** | Uniform `r=8, α=16` (`local_attn`, `global_attn`, `ffn`) | **12,079,104** (`0.00%`) | 2.8117 $\rightarrow$ 0.9538 | **87.5% / 87.5% / 75.0% / 75.0%** | 11.31 GB (1.43 GB) | 20.0s (46.17 MB) | Dominated by `P4_Matched` |
| **P4\_Matched** | **✅ Matched ($\pm 5\%$ Parity)** | `local_attn`: `r=16, α=32`; `global_attn`: `r=32, α=64`; `ffn`: `r=4, α=8` | **11,515,904** (**`-4.66%`**) | 2.8117 $\rightarrow$ **0.9338** | **87.5% / 87.5% / 75.0% / 75.0%** | **11.28 GB** (1.71 GB) | **19.8s (44.03 MB)** | ⭐ **Non-dominated Winner** |
| **P2** | *Unmatched Ablation* | `local_attn`: `r=16, α=32`; `global_attn`: `r=16, α=32`; `ffn`: `r=4, α=8` | 10,057,728 (`-16.73%`) | 2.8117 $\rightarrow$ 0.9317 | 87.5% / 87.5% / 75.0% / 75.0% | 11.32 GB (1.13 GB) | 19.1s (38.46 MB) | Unmatched (`-16.73%`) |
| **P3** | *Unmatched Ablation* | `local_attn`: `r=4, α=64`; `global_attn`: `r=4, α=64`; `ffn`: `r=16, α=32` | 20,140,032 (`+66.73%`) | 2.8117 $\rightarrow$ 0.9499 | 87.5% / 87.5% / 75.0% / 75.0% | 11.39 GB (1.28 GB) | 19.1s (76.92 MB) | Unmatched (`+66.73%`) |

### Key Takeaway on Gemma 4 E2B (Strict $\pm 5\%$ Parameter-Budget Parity)
- Both **`P1`** and **`P4_Matched`** double **AST Exact Argument-Value Accuracy** over the untuned Zero-Shot Base Model (`37.5%` $\rightarrow$ **`75.0%`**, `6/8` unseen tool calls exact) and improve AST Tool Selection from `75.0%` to **`87.5%`**.
- Comparing **`P4_Matched`** (`11,515,904` params, **`-4.66%` vs. `P1`**, strictly within $\pm 5\%$ budget parity) against **`P1`** (`12,079,104` params) from the identical initial loss (`2.8117`):
  - **Final SFT Loss** decreases from `0.9538` to **`0.9338`**.
  - **Peak Process RSS** decreases from `11.31 GB` to **`11.28 GB`**, while **adapter disk footprint** shrinks from `46.17 MB` to **`44.03 MB`** (`-4.66%`).
  - **`P4_Matched` strictly Pareto-dominates `P1`** across trainable parameter count, Peak Process RSS, and final SFT loss while matching `87.5%` Tool Selection and `75.0%` Exact AST Match.

---

## 3. Results: Falcon-H1 1.5B (`tiiuae/Falcon-H1-1.5B-Instruct`) — Budget-Matched (±5%) vs. Unmatched Ablation

| Config | Comparison Regime | Layer-Role Allocation | Trainable Params ($\Delta$ vs. P1) | Init $\rightarrow$ Final SFT Loss | AST Parse / Tool / Arg-Name / Exact Arg-Value | Peak Process RSS (MPS Alloc) | Train Time (Adapter) | Pareto Front (Matched $\pm 5\%$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Zero-Shot)** | *Untuned Reference* | No LoRA adapters attached | 0 (`-100%`) | — | 87.5% / 87.5% / 75.0% / 62.5% | — | — | Reference |
| **P1** | **Baseline (100%)** | Uniform `r=8, α=16` (`global_attn`, `ssm`, `ffn`) | **7,578,624** (`0.00%`) | 1.5534 $\rightarrow$ 0.1922 | 87.5% / 87.5% / 75.0% / 62.5% | **4.58 GB** (1.88 GB) | 87.4s (28.98 MB) | ⭐ **Non-dominated (Lowest RSS)** |
| **P2** (`P2_Matched`) | **✅ Matched ($\pm 5\%$ Parity)** | `global_attn`: `r=16, α=32`; `ssm`: `r=8, α=16`; `ffn`: `r=4, α=8` | **7,726,080** (**`+1.95%`**) | 1.5534 $\rightarrow$ **0.1769** | 87.5% / 87.5% / **87.5%** / 62.5% | 4.60 GB (1.97 GB) | 87.5s (29.55 MB) | ⭐ **Non-dominated (Lowest Loss & +12.5% Arg-Name)** |
| **P4\_SSM\_Matched** | **✅ Matched ($\pm 5\%$ Parity)** | `global_attn`: `r=8, α=16`; `ssm`: `r=16, α=32`; `ffn`: `r=4, α=8` | **7,342,080** (**`-3.12%`**) | 1.5534 $\rightarrow$ **0.1899** | 87.5% / 87.5% / **87.5%** / 62.5% | 4.61 GB (1.95 GB) | **87.3s (28.08 MB)** | ⭐ **Non-dominated (Smallest Params & +12.5% Arg-Name)** |
| **P3** | *Unmatched Ablation* | `global_attn`: `r=4, α=64`; `ssm`: `r=8, α=16`; `ffn`: `r=16, α=32` | 10,380,288 (`+36.97%`) | 1.5534 $\rightarrow$ 0.1413 | **100% / 100% / 100% / 75.0%** | 5.07 GB (1.98 GB) | 85.3s (39.67 MB) | Unmatched (`+36.97%`) |

### Key Takeaway on Falcon-H1 1.5B (Strict $\pm 5\%$ Parameter-Budget Parity)
- Under strict $\pm 5\%$ parameter-budget parity, both **`P2` (`+1.95%`)** and **`P4_SSM_Matched` (`-3.12%`)** outperform uniform **`P1`** on **Final SFT Loss** (`0.1769` and `0.1899` vs. `0.1922`) and improve **AST Argument-Name Accuracy from `75.0%` (`6/8`) to `87.5%` (`7/8`, `+12.5%`)**.
- Specifically, **`P4_SSM_Matched`** (`global_attn: r=8, α=16; ssm: r=16, α=32; ffn: r=4, α=8`) achieves `+12.5%` higher AST Argument-Name Accuracy and lower SFT loss (`0.1899` vs. `0.1922`) while using **`3.12%` fewer trainable parameters** (`7,342,080` vs. `7,578,624`) and a smaller adapter footprint (**`28.08 MB` vs. `28.98 MB`**).

---

## 4. Secondary Unmatched FFN-Pruning 2-Step Apple MPS Smoke Ablation

Recorded in `runs/pilot_cuda/SMOKE_20261012T095504Z/` across two random seeds (`42` and `137`, 2 training steps, $n=3$ tool-calling validation queries) on `google/gemma-4-E2B-it` (**BF16 Base Weights + FP32 LoRA Adapters**, **MPS Driver-Allocated Memory**):

| Config | Regime | Layer-Role Allocation | Trainable Params ($\Delta$ vs. Baseline) | Mean Final SFT Loss (Seeds 42, 137) | Peak MPS Driver-Allocated Memory |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Config A (Uniform Baseline)** | Baseline (`100%`) | Uniform `r=8, α=16` (`local_attn`, `global_attn`, `ffn`) | 12,079,104 (`0.00%`) | 0.5819 | 20.61 GB |
| **Config B (FFN-Pruned Ablation)** | *Unmatched Capacity-Pruning Ablation* | `local_attn`: `r=4, α=8`; `global_attn`: `r=32, α=64`; `ffn`: Disabled (`r=0`) | **3,891,200 (`-67.79%`)** | **0.4822** | **18.90 GB (`-1.71 GB`)** |

*Methodological Note:* Because `Config B` reduces total trainable parameters by `67.79%` on a 2-step smoke run, it is reported strictly as a **secondary unmatched capacity-pruning ablation** demonstrating FFN redundancy and memory savings when FFN adapters are disabled (`r=0`), whereas **`P4_Matched` (`-4.66%`)**, **`P2_Matched` (`+1.95%`)**, and **`P4_SSM_Matched` (`-3.12%`)** serve as the **primary parameter-budget-matched ($\pm 5\%$) comparisons**.

---

## 5. Pre-Train Verification & Failed Configuration Log

- **Pre-Training Budget Parity Validator (`TrialConfig.validate_parameter_budget_parity`)**: Analytically verifies exact parameter counts across all 205 (`Gemma 4 E2B`) and 192 (`Falcon-H1 1.5B`) linear targets prior to model instantiation (`tests/test_budget_matching.py`).
- **Negative Control (`Falcon-H1 conv1d`)**: An intentional trial targeting `model.layers.0.mamba.conv1d` was rejected **before** allocation by `to_peft_lora_config()`, and dynamically trapped with `GRADIENT_MISSING` in `verify_gradients()`. Zero training steps or compute were wasted.
- **Active Trials**: All 8 primary MPS pilot runs (`P1, P4_Matched, P2, P3` on Gemma 4 E2B; `P1, P2, P4_SSM_Matched, P3` on Falcon-H1 1.5B) completed at the identical commit (`215d2e5de5e74a4740cdd3b09ae5168638211a15`), stored their weights, passed fresh subprocess reload verification, and recorded full AST tool-calling metrics.

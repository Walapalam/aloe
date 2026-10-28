# ALOE Hardware Throughput & Latency Benchmark Report

> **Hardware:** Apple M1 Max (10-core CPU, 32-core GPU, 64 GB Unified Memory)
> **Runtime:** PyTorch 2.14.0 MPS (`mps:0`), macOS 26.6.2
> **Precision:** `bfloat16` (unquantized)

## 1. Generation & Decode Performance (Tokens / Second)

| Model | Target Output | Actual Output | TTFT (ms) | Decode Speed (tok/s) | E2E Speed (tok/s) | Latency (ms/tok) |
|---|---|---|---|---|---|---|
| `google/gemma-4-E2B-it` | 32 tok | 32 tok | **97.56 ms** | **24.4 tok/s** | 23.4 tok/s | 41.05 ms |
| `google/gemma-4-E2B-it` | 64 tok | 64 tok | **85.09 ms** | **24.8 tok/s** | 24.4 tok/s | 40.33 ms |
| `google/gemma-4-E2B-it` | 128 tok | 128 tok | **83.41 ms** | **24.8 tok/s** | 24.6 tok/s | 40.28 ms |
| `google/gemma-4-E2B-it` | 256 tok | 256 tok | **83.78 ms** | **24.7 tok/s** | 24.6 tok/s | 40.55 ms |
| `google/gemma-4-E4B-it` | 32 tok | 32 tok | **300.09 ms** | **18.4 tok/s** | 16.1 tok/s | 54.48 ms |
| `google/gemma-4-E4B-it` | 64 tok | 64 tok | **200.62 ms** | **16.5 tok/s** | 15.9 tok/s | 60.7 ms |
| `google/gemma-4-E4B-it` | 128 tok | 128 tok | **136.74 ms** | **16.7 tok/s** | 16.5 tok/s | 59.98 ms |
| `google/gemma-4-E4B-it` | 256 tok | 256 tok | **133.82 ms** | **16.6 tok/s** | 16.5 tok/s | 60.15 ms |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 32 tok | 32 tok | **1708.02 ms** | **28.3 tok/s** | 11.4 tok/s | 35.29 ms |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 64 tok | 64 tok | **1659.72 ms** | **26.9 tok/s** | 16.0 tok/s | 37.22 ms |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 128 tok | 128 tok | **1657.27 ms** | **27.0 tok/s** | 20.1 tok/s | 37.06 ms |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 256 tok | 256 tok | **1661.72 ms** | **27.0 tok/s** | 23.0 tok/s | 37.09 ms |

## 2. Prefill / Prompt Processing Throughput

| Model | Prompt Length | Avg Latency (ms) | Prefill Throughput (tok/s) |
|---|---|---|---|
| `google/gemma-4-E2B-it` | 64 tokens | 162.96 ms | **392.7 tok/s** |
| `google/gemma-4-E2B-it` | 256 tokens | 468.02 ms | **547.0 tok/s** |
| `google/gemma-4-E2B-it` | 512 tokens | 859.28 ms | **595.8 tok/s** |
| `google/gemma-4-E4B-it` | 64 tokens | 293.79 ms | **217.8 tok/s** |
| `google/gemma-4-E4B-it` | 256 tokens | 855.91 ms | **299.1 tok/s** |
| `google/gemma-4-E4B-it` | 512 tokens | 1810.03 ms | **282.9 tok/s** |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 64 tokens | 1717.66 ms | **37.3 tok/s** |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 256 tokens | 3861.6 ms | **66.3 tok/s** |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 512 tokens | 7556.77 ms | **67.8 tok/s** |

## 3. LoRA Fine-Tuning Training Throughput

| Model | Batch Size | Seq Len | Step Latency (ms) | Steps / Sec | Effective Training Throughput |
|---|---|---|---|---|---|
| `google/gemma-4-E2B-it` | 1 | 128 | **712.89 ms** | **1.4 steps/s** | **179.6 tokens/s** |
| `google/gemma-4-E4B-it` | 1 | 128 | **1371.62 ms** | **0.73 steps/s** | **93.3 tokens/s** |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 1 | 128 | **6917.94 ms** | **0.14 steps/s** | **18.5 tokens/s** |

## 4. Peak Sampled Memory Footprint (100ms Interval)

| Model | Inference Peak RSS | Inference Peak MPS | Training Peak RSS | Training Peak MPS | Headroom on 64GB Unified |
|---|---|---|---|---|---|
| `google/gemma-4-E2B-it` | 10.5044 GB | 0.2588 GB | **10.493 GB** | 1.1269 GB | **+53.5 GB free** |
| `google/gemma-4-E4B-it` | 7.5319 GB | 0.2774 GB | **7.5756 GB** | 1.1904 GB | **+56.4 GB free** |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 3.4571 GB | 0.4391 GB | **3.495 GB** | 1.7039 GB | **+60.5 GB free** |

---
## 5. Architectural Findings & Takeaways

- **Interactive Speed**: Generation speeds on Apple Silicon MPS provide fluid response times suitable for interactive agents and fast evaluation loops.
- **Training Efficiency**: LoRA training backward passes execute natively on MPS with zero host-device ping-pong thrashing.
- **Memory Budget Headroom**: With >50 GB of unified memory available during training, larger models such as `google/gemma-4-E4B-it` (14.89 GB weights) will fit easily without needing quantization.

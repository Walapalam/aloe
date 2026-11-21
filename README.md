# ALOE: Architecture-aware LoRA Optimization Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-117%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)]()
[![Release](https://img.shields.io/badge/release-v0.1.0--alpha-orange.svg)]()

**ALOE** (**A**rchitecture-aware **L**oRA **O**ptimisation **E**ngine) is a research framework for **architecture-typed, multi-objective LoRA optimisation** across heterogeneous Small Language Models (up to 3B effective parameters) for local agentic tool calling on a single GPU.

Rather than applying uniform rank-8 LoRA across every layer, ALOE automatically inspects a model's module tree, groups adaptable linear projections into 3–4 functional layer types, excludes zero-gradient modules via a 1-step gradient probe, and searches per-type `(rank r, alpha α)` configurations within an equal `±5%` parameter budget (`qLogNEHVI`).

---

## Key Features

- **Automated Layer Typing (`aloe inspect`)**: Introspects model module graphs on zero-RAM `meta` tensors and groups adaptable linear projections by functional role (`local_attention`, `global_attention`, `ssm`, `ffn`).
- **Three Target Model Families**:
  - **Gemma 4-E2B** (`google/gemma-4-E2B-it`): Interleaved local/global attention Transformer (`205` adaptable modules across 3 types).
  - **Falcon-H1 1.5B** (`tiiuae/Falcon-H1-1.5B-Instruct`): Parallel Attention + Mamba-2 SSM hybrid (`192` adaptable modules across 3 types; `24` inert `conv1d` excluded).
  - **LFM3-1.7B** (`LiquidAI/LFM3-1.7B`): Sequential ShortConv + local/global attention tri-hybrid (`142` adaptable modules across 4 types; `13` inert `ShortConv` excluded).
- **1-Step Gradient Probe (`aloe probe`)**: Verifies non-zero gradient flow across role groups and blocks `GRADIENT_MISSING` custom-kernel bypasses before training.
- **Equal `±5%` Budget Gate & Guided BO (`aloe search`)**: Analytically enforces `±5%` parameter parity against uniform rank-8 LoRA (`430` Falcon-H1, `550` Gemma 4, `11,375` LFM3 coupled states) and proposes candidate configs via BoTorch `qLogNEHVI`.
- **4-Step AST/JSON Tool-Call Evaluator & Paired Bootstrap (`aloe evaluate`)**: Offline, non-executing structural verification (`Valid JSON → Tool Name → Argument Names → Argument Values`) and `10,000`-resample paired bootstrap 95% CIs.
- **Reproducible Trial Runner (`aloe run`)**: Executes PEFT trials with `-100` answer-only prompt masking, `100 ms` peak RSS memory profiling, and fresh-subprocess reload verification.

---

## Architecture Overview

```
                      ┌────────────────────────┐
                      │ Base Model Weights     │
                      │ (Gemma 4 / Falcon-H1)  │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Architecture Inspector │
                      │ (Semantic Layer Typing)│
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Search Grammar Engine  │
                      │ (Role-Derived Bounds)  │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Pre-Flight Trap        │
                      │ (Gradient Verification)│
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Multi-Objective Search │
                      │ (Accuracy vs RSS/VRAM) │
                      └────────────────────────┘
```

---

## Installation

ALOE requires **Python 3.12** and uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management:

```bash
# Clone the repository
git clone https://github.com/walapalam/aloe.git
cd aloe

# Create virtual environment and synchronize dependencies
uv sync
```

---

## Quickstart & 5-Command CLI Usage

ALOE provides five reproducible CLI commands (`inspect`, `probe`, `search`, `evaluate`, `run`):

```bash
# 1. Inspect model layer groups & search space cardinality (0-RAM meta tensors)
uv run aloe inspect google/gemma-4-E2B-it
uv run aloe inspect tiiuae/Falcon-H1-1.5B-Instruct
uv run aloe inspect LiquidAI/LFM3-1.7B

# 2. Run 1-step autograd sanity check & inspect role-group gradient probe telemetry
uv run aloe probe --model tiiuae/Falcon-H1-1.5B-Instruct

# 3. Enumerate ±5% parameter-parity states & generate 20 guided BO trial configs
uv run aloe search --model google/gemma-4-E2B-it --trials 20 --parity-tol 0.05

# 4. Evaluate pilot tool-call outputs & compute paired bootstrap CIs
uv run aloe evaluate --run-dir runs/pilot --bootstrap 10000

# 5. Execute a single TrialConfig end to end
uv run aloe run configs/falcon_baseline.json
```

---

## Running Tests

ALOE includes a comprehensive unit and regression test suite:

```bash
uv run pytest tests/ -v
```

---

## Documentation

Comprehensive architectural and engineering specifications are located in [`docs/`](docs/):

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System components, module boundaries, and execution pipeline.
- [`ENGINEERING_SPECIFICATION.md`](docs/ENGINEERING_SPECIFICATION.md) — Mathematical formulations, layer typing rules, and surrogate objectives.
- [`DECISIONS.md`](docs/DECISIONS.md) — Architectural Decision Records (`ADR-001` through `ADR-015`).
- [`DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) — Step-by-step 10-minute CLI demonstration protocol.
- [`GCP_L4_EXECUTION_GUIDE.md`](docs/GCP_L4_EXECUTION_GUIDE.md) — Setup and execution guide for NVIDIA L4 Cloud GPU offloading.

---

## Citation & Reference

If you utilize ALOE in academic research or technical benchmarks, please cite:

```bibtex
@misc{rameez2026aloe,
  author = {Rameez, Raqeeb},
  title  = {ALOE: Architecture-aware LoRA Optimization Engine for Heterogeneous Small Language Models},
  year   = {2026},
  url    = {https://github.com/walapalam/aloe}
}
```

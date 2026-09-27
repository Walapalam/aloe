# ALOE: Architecture-aware LoRA Optimization Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen.svg)]()
[![Release](https://img.shields.io/badge/release-v0.1.0--alpha-orange.svg)]()

**ALOE** is a research framework for **architecture-typed Low-Rank Adaptation (PEFT)** and multi-objective optimization across heterogeneous Small Language Models (<3B parameters) deployed on resource-constrained consumer hardware.

Rather than applying uniform heuristic ranks across all Transformer and State Space Model (SSM) blocks, ALOE automatically inspects the underlying neural topology, assigns semantic layer roles (e.g., Attention, MLP, SSM Conv1D, SSM State Matrix, Gating), and derives architecture-specialized search spaces for parameter-efficient adaptation.

---

## Key Features

- **Automated Architecture Inspection**: Introspects model weight graphs and classifies layers by mathematical functional role (`aloe inspect`).
- **Heterogeneous Architecture Support**:
  - Pure Transformer: **Gemma 4-E2B** (Multi-Head / Grouped Query Attention + MLP blocks).
  - Hybrid SSM-Transformer: **Falcon-H1 1.5B** (Mamba-2 Structured State Space Duality + Attention + Convolutional projections).
- **Architecture-Derived Search Grammars**: Maps layer semantics into bounded rank ($r \in [4, 64]$) and scaling factor ($\alpha \in [8, 128]$) dimensions (`aloe grammar`).
- **Hardware-Tiered Metric Profiling**:
  - Apple Silicon Unified Memory: Process RSS memory tracking via MPS backend (`ADR-005`, `ADR-015`).
  - NVIDIA Cloud GPU: CUDA VRAM and compiled kernel acceleration for Mamba-2 chunk scanning (`ADR-012`).
- **Compute Safety Trap**: Pre-flight zero-gradient verification using random orthogonal probes aborts invalid allocations in milliseconds before training begins (`ADR-004`).
- **Tool-Disjoint Generalization**: Built-in split verification on function-calling benchmarks (xLAM-60k) ensuring $\text{Train} \cap \text{Val} = \emptyset$ across API names.

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

## Quickstart & CLI Usage

ALOE provides a unified command-line interface for architecture inspection, search grammar generation, and trial execution:

### 1. Inspect Model Architecture
```bash
# Inspect Gemma 4-E2B layer topology
uv run aloe inspect configs/gemma_baseline.json

# Inspect Falcon-H1 1.5B hybrid SSM topology
uv run aloe inspect configs/falcon_baseline.json
```

### 2. Generate Architecture-Derived Search Grammar
```bash
uv run aloe grammar configs/gemma_baseline.json -o runs/grammars/gemma_grammar.json
```

### 3. Run Pre-Flight Gradient Verification
```bash
uv run python scripts/verify_gradients.py
```

### 4. Execute a QLoRA Smoke Training Trial
```bash
uv run python scripts/smoke_train.py
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

---

## License

This project is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

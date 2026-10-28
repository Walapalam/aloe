# ALOE — Core Engineering Specification & Implementation Protocol

**Author:** Raqeeb Rameez (IIT ID: 20230934 / UoW ID: w2084794)  
**Project:** Architecture-aware LoRA Optimization Engine (ALOE)  
**Date:** September 2026  
**Status:** Authoritative Core Engineering Specification  

---

## 1. System Overview & Contribution Boundary

ALOE inspects a language model's internal structure, automatically derives a LoRA search space grouped by **ARCHITECTURAL ROLE** rather than by numerical importance score, and searches that space under multiple objectives for agentic tool-calling performance on consumer hardware.

The execution pipeline consists of 6 sequential stages:
```text
  model_id
    -> 1. INVENTORY   walk the module tree, record every leaf with parameters
    -> 2. CLASSIFY    assign a ComponentType to each module (per-family adapter)
    -> 3. VALIDATE    (architecture x backend) -> can this module receive a gradient?
    -> 4. EMIT        generate a SearchGrammar: enabled x rank x alpha, per group
    -> 5. EXECUTE     train one configuration, measure real cost
    -> 6. SCORE       tool-use metrics + peak memory + duration + adapter size
```

Steps 1 and 2 establish module discovery. Borobia et al. (arXiv:2604.22127) Section 3.2 automated module discovery and component-type classification. Steps 3 and 4 are where that prior work stops and where ALOE's contribution begins: empirical validity by gradient verification and architectural grammar emission.

---

## 2. Core Research Principles

### 2.1 The Validity Principle
**Validity means "this module can receive a non-zero gradient", NOT "the adapter attached successfully".**

Borobia et al. verified that adapters attached where intended. However, manual exclusion was still required for Falcon-H1's `conv1d` and `out_proj` because those weights feed a custom autograd Mamba kernel that bypasses the LoRA forward pass entirely. An adapter placed there attaches without raising an error, but never updates during backpropagation.

Therefore: every execution backend implements `verify_gradients()`, running a synthetic forward-backward pass to confirm non-zero gradients reach each adapter target. A target that attaches but receives no gradient is marked `EXCLUDED`, not `VALID`.

### 2.2 Hardware Architecture & Multi-Backend Profiling
Development and execution are partitioned across targeted tiers:
1. **Apple Silicon Unified Memory (MPS):** PyTorch MPS backend with native unified memory management.
2. **Cloud GPU Acceleration (CUDA):** HuggingFace PEFT + bitsandbytes NF4 / compiled kernels (`causal-conv1d`, `mamba-ssm`) for full hybrid training.

**Rules:**
- `aloe/architecture/` and `aloe/search_space/` import `torch` and `pydantic` only (core isolation).
- Backend selection occurs at runtime via `ExecutionBackend.available()`.
- Validity rules evaluate backend-specific constraints.
- Metric pooling across backends is prohibited; separate Pareto frontiers are maintained for unified memory vs dedicated VRAM.

---

## 3. Component Architecture & Repository State

| Module | Purpose | Status |
| :--- | :--- | :--- |
| `aloe/architecture/schema.py` | `ComponentType`, `Validity`, `ModuleRecord`, `ArchitectureDescription` | Complete |
| `aloe/architecture/inspector.py` | Walks PyTorch module tree, records leaf parameters | Complete |
| `aloe/architecture/typing.py` | Model adapter protocol and family registry | Complete |
| `aloe/architecture/constraints.py` | Empirical exclusion database and validation rules | Complete |
| `aloe/search_space/schema.py` | `GroupDimensions`, `SearchGrammar` | Complete |
| `aloe/search_space/grammar.py` | Generates typed search spaces | Complete |
| `aloe/execution/backends/` | Backend implementations (`base.py`, `peft_mps.py`) | Complete |
| `aloe/cli/main.py` | CLI entry points (`inspect`, `run`) | Complete |
| `aloe/pipeline.py` | End-to-end inspection and grammar emission pipeline | Complete |

---

## 4. Sequential Implementation Roadmap

### T1. Backend Granularity Probe
Probed adapter targeting capabilities and module-level rank/alpha granularity across candidate execution engines. Resolved in ADR-010.

### T2. Core Pipeline Assembly (`aloe/pipeline.py`)
Implements `inspect_to_grammar(model_id, backend)`:
1. Inspects module hierarchy via `inspector.inspect_model()`.
2. Classifies modules using model-specific adapters via `typing.classify()`.
3. Applies gradient validity constraints via `constraints.apply_validity(backend)`.
4. Emits typed search grammar via `grammar.generate()`.

### T3. Gemma 4 Architecture Adapter (`aloe/architecture/adapters/gemma4.py`)
Classifies modules into `LOCAL_ATTENTION`, `GLOBAL_ATTENTION`, `FFN`, `EMBEDDING`, `NORM`. Derives local vs global sliding window patterns directly from `model.config` (`layer_types` or `sliding_window_pattern`). Handles KV-cache sharing where projection matrices vary across layers.

### T4. Falcon-H1 Architecture Adapter (`aloe/architecture/adapters/falcon_h1.py`)
Classifies modules into `ATTENTION`, `SSM`, `FFN`, `NORM`. Models parallel hybrid blocks with attention and Mamba-2 branches. Enforces explicit exclusions on `conv1d` and `out_proj`.

### T5. Validity & Gradient Verification Test Suite
Automated regression tests asserting:
- Every module has a non-empty `validity_reason`.
- No module is marked `VALID` without concrete justification.
- Known incompatible layers are safely `EXCLUDED`.
- Backward pass gradient verification catches inert adapters.

### T6. Structural Grammar Differentiation (Gate B)
Ensures structural differentiation holds: Gemma and Falcon produce distinct component classifications and distinct search grammar groups under identical engine parameters.

### T7. Dataset Preparation: Tool-Disjoint xLAM Split (`aloe/datasets/xlam.py`)
Builds a frozen, tool-disjoint benchmark split where APIs in the training set do not appear in validation. Stratifies across single-call, multi-candidate-tool, and no-tool scenarios. Exports SHA-256 verified manifests.

### T8. Execution Backend & Verification (`aloe/execution/backends/peft_mps.py`)
Implements `run_trial()` and `verify_gradients()`. Captures peak process memory (RSS) and verifies gradient flow across all target modules.

### T9. Tool-Calling Evaluation Engine (`aloe/evaluation/resources.py`)
Evaluates JSON validity, tool selection, argument schema conformity, and correct tool abstention.

### T10. Resource Profiler & Reproducible Trial Store (`aloe/tracking/trial_store.py`)
Records complete trial envelope: model ID, dataset manifest hash, seed, per-group hyperparameters, training duration, peak memory, adapter parameter counts, and full hardware metadata.

### T11. Empirical Pilot Validation
Executes pilot configurations across Gemma 4-E2B and Falcon-H1 1.5B (global LoRA baselines and architecture-typed configurations) to verify end-to-end pipeline execution.

---

## 5. Engineering Standards & Quality Gates

1. **Fail Loudly:** Never use bare `except` blocks or default fallbacks when inspecting model configurations. Missing required configuration fields must raise explicit exceptions.
2. **Evidence-Based Validity:** Every exclusion rule in `KNOWN_EXCLUSIONS` must reference a published paper citation or empirical experiment ID.
3. **Core Architectural Isolation:** No hardware or vendor-specific library imports (`peft`, `bitsandbytes`, `mamba_ssm`) are permitted inside `aloe/architecture/` or `aloe/search_space/`.
4. **Deterministic Reproducibility:** Fixed seeds across PyTorch, NumPy, and Python runtime. Detailed hardware telemetry logged with every trial.
5. **Rigorous Typing & Coverage:** Pydantic models for cross-module interfaces; 100% test coverage across critical paths.

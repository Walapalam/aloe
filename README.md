# ALOE: Architecture-aware LoRA Optimization Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-126%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen.svg)]()
[![Release](https://img.shields.io/badge/release-v0.2.0-brightgreen.svg)]()

**ALOE** (**A**rchitecture-aware **L**oRA **O**ptimisation **E**ngine) is a research framework for **architecture-typed, multi-objective LoRA optimisation** across heterogeneous Small Language Models (up to 3B effective parameters) for local agentic tool calling on a single GPU or Apple Silicon workstation.

Rather than applying uniform rank-8 LoRA across every layer, ALOE automatically inspects a model's module tree, groups adaptable linear projections into 3–4 functional layer types, excludes zero-gradient custom kernels via a 1-step gradient probe, and searches per-type `(rank r, alpha α)` configurations within an exact `±5%` parameter budget (`5` scrambled-Sobol initial points + `15` discrete `qLogNEHVI` acquisitions).

---

## Key Features (`protocol-v1` · `v0.2.0`)

- **Automated Layer Typing (`aloe inspect`)**: Introspects model module graphs on zero-RAM `meta` tensors and groups adaptable linear projections by functional role (`local_attention`, `global_attention`, `ssm`, `ffn`).
- **Three Heterogeneous Target Model Families**:
  - **Gemma 4-E2B** (`google/gemma-4-E2B-it`): Interleaved local/global attention Transformer (`205` adaptable modules across 3 types; `550` parity states).
  - **Falcon-H1 1.5B** (`tiiuae/Falcon-H1-1.5B-Instruct`): Parallel Attention + Mamba-2 SSM hybrid (`192` adaptable modules across 3 types; `24` custom-kernel `conv1d` excluded; `430` parity states).
  - **LFM3-1.7B** (`LiquidAI/LFM3-1.7B`): Sequential ShortConv + local/global attention tri-hybrid (`142` adaptable modules across 4 types; `13` custom-kernel `conv.conv` + `13` scope-excluded `gate_proj`; `11,375` parity states).
- **Frozen 3-Way Tool-Disjoint Benchmark (`data/manifests/xlam_ipd_v1.json`)**: Category-stratified `1,000 train / 200 validation / 400 sealed test` split of `Salesforce/xlam-function-calling-60k` with **zero pairwise tool-name overlap** across splits.
- **Equal `±5%` Budget Gate, Discrete `qLogNEHVI` & Controlled Baselines (`aloe search --execute`)**:
  - **`aloe`**: `5` scrambled-Sobol + `15` discrete `qLogNEHVI` (`optimize_acqf_discrete`) over the exact shape-computed feasible pool.
  - **`b0` / `b1` / `b2` / `borobia_*`**: Matched 20-trial uniform `r=8` `(alpha, lr)` grid (`b0`), exact-feasible random search (`b1`), PEFT `AdaLoRA` `update_and_allocate` (`b2`), and exact-integer-rank single-operator ablations (`borobia_attn`, `borobia_ssm`, `borobia_ffn`).
- **Strict Schema-Aware Multi-Call Evaluator & Step-Down Statistics (`aloe evaluate`)**:
  - `ASTToolCallEvaluator(strict=True)` (`v2.0.0-strict`): full multi-call matching, strict JSON value equality (`True != 1`, `"001" != 1`), `[]` abstention support, duplicate-key/NaN rejection, and JSON Schema validation.
  - `10,000`-resample paired query bootstrap (`BCa` + percentile 95% CIs), exact two-sided McNemar test, and step-down **Holm-Bonferroni** correction (`α = 0.05`) over persistent `.jsonl` `PredictionStore` logs.
- **Crash-Resumable Isolated Worker & Colab Pro+ A100 Orchestrator**:
  - Atomic `CampaignLedger`, 25-step mid-trial `checkpoint_state.pt` recovery, isolated subprocess `worker_entry.py` (100% VRAM/RSS release between trials), `--pinned-gpu A100` hardware enforcement (`ADR-017`), and Google Drive mirror sync (`scripts/run_colab_a100_campaign.py`).
- **Zero-Dependency Stdlib Audit Scripts**: `scripts/verify_campaign.py` and `scripts/recompute_metrics.py` independently verify split disjointness, `±5%` parity, SHA-256 digests, and AST exact-match accuracy using **only the Python 3.12 standard library**.

---

## Installation

ALOE requires **Python 3.12** and uses [`uv`](https://github.com/astral-sh/uv) for deterministic dependency management:

```bash
git clone https://github.com/walapalam/aloe.git
cd aloe
uv sync --all-extras
```

---

## Quickstart & Reproducible Commands

```bash
# 1. Inspect model layer groups & exact ±5% parity search space (0-RAM meta tensors)
uv run aloe inspect google/gemma-4-E2B-it
uv run aloe inspect tiiuae/Falcon-H1-1.5B-Instruct
uv run aloe inspect LiquidAI/LFM3-1.7B

# 2. Run 1-step autograd sanity check & inspect role-group gradient probe telemetry
uv run aloe probe --model tiiuae/Falcon-H1-1.5B-Instruct

# 3. Preview Sobol parity-feasible design (plan-only) OR execute full 20-trial campaign
uv run aloe search --model google/gemma-4-E2B-it --trials 20 --parity-tol 0.05
uv run aloe search --model google/gemma-4-E2B-it --arm aloe --trials 20 --execute

# 4. Run resumable Colab Pro+ NVIDIA A100 hybrid campaign with Google Drive mirror
python3 scripts/run_colab_a100_campaign.py \
  --model falcon --phase search --arm all \
  --drive-root /content/drive/MyDrive/aloe_ipd_v1

# 5. Compute strict AST metrics, paired bootstrap CIs, McNemar & Holm-Bonferroni
uv run aloe evaluate --run-dir runs/ipd_v1 --split validation --bootstrap 10000

# 6. Zero-dependency offline verification (Python stdlib only — no torch/botorch needed)
python3 scripts/verify_campaign.py --manifest data/manifests/xlam_ipd_v1.json
```

---

## Running Tests

```bash
uv run pytest tests/ --cov=aloe --cov-fail-under=86 -q
```

---

## Documentation

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System components, module boundaries, and execution pipeline.
- [`ENGINEERING_SPECIFICATION.md`](docs/ENGINEERING_SPECIFICATION.md) — Mathematical formulations, layer typing rules, and surrogate objectives.
- [`DECISIONS.md`](docs/DECISIONS.md) — Architectural Decision Records (`ADR-001` through `ADR-017`).
- [`PPRS_FEEDBACK_ACTION_MATRIX.md`](docs/PPRS_FEEDBACK_ACTION_MATRIX.md) — Traceability matrix from proposal & review feedback to code artifacts.
- [`CLAIMS.md`](docs/CLAIMS.md) — Explicit scope, hardware boundaries, and empirical claim guardrails.
- [`DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) — Step-by-step CLI demonstration protocol.

---

## Citation

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

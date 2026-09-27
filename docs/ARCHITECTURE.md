# ALOE — Architecture

**A Reusable Architecture-Typed, Multi-Objective LoRA Optimisation Framework for Resource-Constrained Local Agentic Tool Use in Small Language Models**

---

## The one-sentence version

ALOE inspects a language model's internal structure, automatically derives a LoRA search space grouped by *architectural role* rather than by numerical importance score, and searches that space under multiple objectives for agentic tool-calling performance on consumer hardware.

---

## Pipeline

```
model_id
   │
   ▼
┌──────────────────┐   architecture/inspector.py      backend-agnostic
│ 1. INVENTORY     │   walk nn.Module tree, record every leaf with params
└────────┬─────────┘   → ArchitectureDescription
         ▼
┌──────────────────┐   architecture/typing.py + adapters/
│ 2. CLASSIFY      │   assign ComponentType per module, one adapter per family
└────────┬─────────┘   local_attn / global_attn / linear_attn / ssm / ffn / router / expert
         ▼
┌──────────────────┐   architecture/constraints.py    ◄── THE CONTRIBUTION
│ 3. VALIDATE      │   (architecture × backend) → VALID | SPECIAL | UNSUPPORTED | EXCLUDED
└────────┬─────────┘   "can this receive a gradient?", not "did it attach?"
         ▼
┌──────────────────┐   search_space/grammar.py
│ 4. EMIT GRAMMAR  │   one group per adaptable type; enabled × rank × alpha per group
└────────┬─────────┘   → SearchGrammar (never hand-written)
         ▼
┌──────────────────┐   execution/executor.py → backends/{cuda_peft,mlx}
│ 5. EXECUTE TRIAL │   train one config, measure cost
└────────┬─────────┘   → TrialResult
         ▼
┌──────────────────┐   evaluation/ + tracking/
│ 6. SCORE & STORE │   tool-use metrics + peak memory + duration + adapter size
└────────┬─────────┘
         ▼
     [ October: multi-objective search over the grammar ]
```

Steps 1–6 are September. The optimiser is October, and the pipeline is built so that adding it changes nothing upstream.

---

## Why the layers are split where they are

**The boundary that matters is step 4 → step 5.** Everything above it is the research; everything below it is engineering. The grammar is a plain data object with no framework types in it, so the same grammar can be executed by PEFT on CUDA or by MLX on Apple Silicon and produce comparable *structure*, even though the measured costs are not comparable across backends.

**Steps 1 and 2 are prior art.** Borobia et al. (arXiv:2604.22127) §3.2 already automate module discovery and component-type classification. Concede this cleanly — the contribution is steps 3 and 4, where they stop.

---

## Component types

| Type | Appears in | Notes |
|---|---|---|
| `LOCAL_ATTENTION` | Gemma 4 | Sliding-window. Pattern must be read from config, never assumed |
| `GLOBAL_ATTENTION` | Gemma 4, Qwen3.5 | Full softmax |
| `LINEAR_ATTENTION` | Qwen3.5 | GatedDeltaNet. 18 of 24 layers, 3:1 ratio |
| `SSM` | Falcon-H1 | Mamba-2. Parallel to attention, outputs summed pre-MLP |
| `FFN` | all | |
| `ROUTER` / `EXPERT` | MoE (stretch goal) | 4-bit skips experts — see ADR-003 |
| `EMBEDDING` / `NORM` / `OTHER` | all | Excluded by convention |

Four families, four distinct taxonomies, one of them (Qwen3-1.7B) deliberately flat as a control.

---

## The three documented silent failures

These are why step 3 exists. Each is a case an architecture-blind pipeline gets wrong **without raising an error**.

| Model | Failure | Consequence |
|---|---|---|
| Falcon-H1 | `conv1d` and `out_proj` feed a custom autograd Mamba kernel that bypasses the LoRA forward pass | Adapter is silently inert; trials burn budget on configurations that cannot learn |
| Gemma 4 | KV-cache sharing across some layers | `k_proj`/`v_proj` absent on some layers; assuming all seven projections on all 35 produces an invalid grammar |
| Any MoE | bitsandbytes 4-bit replaces only `nn.Linear`, skipping experts | The bulk of the weights stay in full precision; naive QLoRA silently fails to quantise the thing consuming memory |

Borobia et al. hit the first one and fixed it by hand after observing training instability. That is the cost of not deriving it.

---

## Execution backends

| | `cuda_peft` | `mlx` |
|---|---|---|
| Stack | HuggingFace PEFT + bitsandbytes | Apple MLX + mlx-lm |
| Quantisation | NF4 + double quant + paged optimisers | MLX native 4-bit |
| Memory metric | Peak VRAM (GB) | Peak unified memory (GB) |
| Reference hardware | RTX 4060 8 GB (Avinash, arXiv:2509.12229) | M1 Max 64 GB |
| Known defaults | PagedAdamW > AdamW (+25% throughput); fp16 > bf16 | TBD — measure, do not assume |

**Never pool metrics across backends.** Two Pareto fronts, reported separately.

---

## Repository layout

```
aloe/
├── architecture/
│   ├── schema.py          ComponentType, Validity, ModuleRecord, ArchitectureDescription
│   ├── inspector.py       stage 1 — works today
│   ├── typing.py          stage 2 — ModelAdapter protocol + registry
│   ├── constraints.py     stage 3 — validity rules as evidenced data
│   └── adapters/          one file per model family
├── search_space/
│   ├── schema.py          GroupDimensions, SearchGrammar
│   └── grammar.py         stage 4
├── execution/
│   ├── executor.py        stage 5 orchestration
│   └── backends/          base.py (contract) + cuda_peft.py + mlx_backend.py
├── datasets/xlam.py       tool-disjoint split
├── evaluation/
│   ├── tool_calling.py    AST / arg correctness / no-tool decision
│   └── resources.py       peak memory, duration, adapter size
├── tracking/trial_store.py
└── cli/main.py            aloe inspect | aloe run
```

---

## Milestone definition

**ALOE Core v0.1** — inspects two heterogeneous architectures, semantically types their components, validates PEFT compatibility per backend, generates architecture-specific search grammars, executes reproducible LoRA/QLoRA configurations, and records agentic-performance and resource measurements.

It does **not** claim to have found an optimal configuration. That is October.

**The test that decides whether v0.1 succeeded:**

> Run `aloe inspect` on a structurally different model and show that the generated grammar differs — **without editing the optimiser, the executor, or the grammar generator.** Only a new file in `adapters/`.

If that requires touching anything outside `adapters/`, the abstraction has failed and SQ2 is not yet supported.

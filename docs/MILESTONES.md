# ALOE — Milestones

Re-based 15 September 2026. The original plan assumed an 11 September start; four days are gone, so 90 tasks now have 15 days rather than 20.

**Supervisor directive:** 95% implementation, 5% reading.

---

## Gates

| Gate | Date | Condition | State | Verified Artifact |
|---|---|---|:---:|---|
| **A** | Thu 18 Sep | One model loads · LoRA trains on an xLAM pilot split · adapter saves and reloads · `aloe inspect gemma` emits a typed JSON map | [x] | `aloe/data/manifests/xlam_pilot_v1.json` (800/200 split, 0 intersection) |
| **B** | Wed 24 Sep | `aloe inspect gemma` and `aloe inspect falcon` produce **different** semantic maps and **different** grammars | [x] | `aloe/runs/grammars/DIFFERENTIATION_REPORT.md` (Gemma vs Falcon structural differentiation) |
| **C** | Tue 30 Sep | Full chain end to end on two architectures, second one without touching the optimiser | [x] | `aloe/runs/trials/gate_c_verification_trial/trial.json` (Pre-flight abort and gradient verification) & `aloe/runs/pilot/PILOT_REPORT.md` (6 pilot trials across Gemma E2B & Falcon-H1) |

Gate B is the one that matters. It is the moment ALOE exists as a concept rather than a plan.

---

## Block A · 15–18 Sep — Baseline and Inspector

| Day | Implementation | Reading |
|---|---|---|
| Mon 15 | Repo frozen, pushed to GitHub. `ModelAdapter` interface defined. Pilot models fixed: **Gemma 4 E2B + Falcon-H1** | — |
| Tue 16 | Conventional LoRA baseline end to end: xLAM subset → train → save → reload → one prediction. **No optimisation** | — |
| Wed 17 | Architecture Inspector complete; JSON serialisation; tests | — |
| Thu 18 | Gemma semantic typing: local / global / FFN / embedding | **AutoPEFT, deep** (half day) |

**Machine migration lands in this block.** See `docs/MIGRATION.md`. Do the GitHub push before anything else.

---

## Block B · 19–24 Sep — Validity and Grammar

| Day | Implementation | Reading |
|---|---|---|
| Fri 19 | Falcon-H1 typing: attention / SSM / MLP / norm | — |
| Sat 20 | **Validity checker.** Four states, backend-aware, every rule evidenced | **Mamba-2 / SSD, method-level** |
| Sun 21 | `SearchSpaceGenerator` — grammar emitted, not hardcoded | — |
| Mon 22 | Tests: discovery, layer indexing, typing, valid/invalid, grammar, serialisation | — |
| Tue 23 | Tool-call evaluator v1: valid JSON, tool name, args, required args, unexpected args, no-tool decision | **BFCL, deep** (half day) |
| Wed 24 | `TrialConfig` schema + executor skeleton | — |

---

## Block C · 25–30 Sep — End to end

| Day | Implementation | Reading |
|---|---|---|
| Thu 25 | `ResourceProfiler`: peak memory, duration, trainable params, adapter size | — |
| Fri 26 | `TrialStore`: deterministic tracking of every field | **LoftQ, skim** |
| Sat 27 | **Pilot run.** Three configs × two models: global LoRA, typed 1, typed 2. No conclusions drawn | — |
| Sun 28 | Fix what broke. Related Work Boundary Matrix | — |
| Mon 29 | Hardening: exceptions, config validation, seeds, hardware logging, README, diagram | — |
| Tue 30 | **v0.1 demonstration** | — |

---

## Explicitly out of scope for September

Building any of these this month produces a broad, fragile prototype:

- LLM director / agent orchestration
- Full Bayesian optimiser or qEHVI
- Evolutionary search
- All four architectures (two only)
- Final BFCL or τ²-bench experiments
- 30-trial searches, multi-seed final runs
- Statistical significance testing
- ERP/OMS deployment

---

## Reading — September

| Paper | Venue | Depth | Why it cannot wait |
|---|---|---|---|
| AutoPEFT | TACL 2024 | Deep | Closest competitor; narrows Gap 5; gives the SAAS-GP surrogate design |
| BFCL | ICML 2025 | Deep | Evaluator metric definitions shape the schema built on 23 Sep |
| Mamba-2 / SSD | ICML 2024 | Method | Cannot type Falcon-H1's SSM modules without it |
| LoftQ | ICLR 2024 | Skim | Quantisation–LoRA interaction under 4-bit |

**October:** LoRA+, PiSSA, RoSA, S4, qEHVI, τ-bench, API-Bank, Toolformer, Gorilla.

**Preprint ratio:** currently 29 of 55 (53%). The four above plus the October nine take it to roughly 62% peer-reviewed and fix the two weakest themes — T2 (11 preprints vs 7 peer-reviewed, and it holds Gaps 1 and 2) and T6 (3 preprints, zero peer-reviewed).

---

## Decisions still needing supervisor sign-off

| # | Decision | ADR |
|---|---|---|
| 1 | Apple Silicon re-scope — MLX backend, unified memory instead of 8 GB VRAM | ADR-004 |
| 2 | Four-paper September reading set instead of thirteen | ADR-008 |
| 3 | Gap 5 narrowed; AutoPEFT named as prior art; novelty 7 → 5-6 | ADR-009 |

---

---

## Model Acquisition & Capability Testing Gates (Empirical Verification Protocol)

| Gate | Stage / Objective | Date Verified | Verdict | Artifacts |
|---|---|---|---|---|
| **Gate 1** | Backend Capabilities (MPS, per-group LoRA) | 2026-09-17 | **PASS ✅** | `runs/probe/REPORT.md`, `backends.json` |
| **Gate 2** | Model Acquisition (Gemma 4 + Falcon-H1 pinned) | 2026-09-17 | **PASS ✅** | `runs/models/REGISTRY.md`, `acquisition.json` |
| **Gate 3** | Architecture Verification & Claims Testing | 2026-09-17 | **PASS ✅** | `runs/architecture/ARCHITECTURE_REPORT.md` |
| **Gate 4** | Gradient Verification & Negative Control | 2026-09-17 | **PASS ✅** | `runs/gradients/GRADIENT_REPORT.md` |
| **Gate 5** | Smoke Training & Sampled Memory Profiling | 2026-09-17 | **PASS ✅** | `runs/smoke/SMOKE_REPORT.md` |
| **Gate 6** | Baseline Tool-Call Sanity Probe | 2026-09-17 | **PASS ✅** | `runs/toolcalls/TOOLCALL_REPORT.md` |

---

## Tactical Addendum — 18 Sep 2026

This section overrides any stale sequencing in the September blocks above.

### Immediate actions (before 25 Sep)

1. **Gate A Closed (DONE):** verified `aloe/data/manifests/xlam_pilot_v1.json` enforces tool-disjointness (800/200 split, 0 intersection) and completed smoke training on the xLAM split.
2. **Lock memory metric:** objective is sampled process RSS (`peak_memory_gb`), not `mps_driver_allocated_gb`.
3. **Confirm Gate B:** produce `aloe inspect` artefacts for Gemma 4 E2B/E4B and Falcon-H1 and keep them under `runs/gate_b/`.
4. **Fix stale test:** rename/update `tests/test_verify_gradients.py:48` to assert `run_trial(config=None)` raises `(AttributeError, ValueError)`. **(DONE — full suite 72/72.)**
5. **Accept ADR-012** (Cloud GPU hardware tier for Mamba-2 hybrid acceleration) in `aloe/docs/DECISIONS.md`.

### October reading vs. build sequence

| Window | Build focus | Reading focus |
|---|---|---|
| 18–24 Sep | Close Gate A/B | T5 start: Conway (#28), Barker (#41) |
| 25–30 Sep | Hardening, trial schema | T5 finish: Benmeziane (#44), Rojas-Carrasco (#45); T9 start: Schneider (#27), Du (#36) |
| 1–8 Oct | **Gate D:** SAAS-GP + qEHVI | T8: Avinash (#56) |
| 9–18 Oct | **Gate E:** 30 Gemma / 15 Falcon trials | T9 finish: Raff (#40); apply bootstrap tests |
| 19–31 Oct | Pareto plots, dissertation chapter 5 | — |

### Hard rule

T5 is the only reading prerequisite for Gate D code. T8 and T9 inform validation and reporting, not the initial search-engine implementation.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation / Outcome |
|---|---|---|---|
| MLX cannot target modules finely enough for per-group rank/alpha | Medium | **High** | **MITIGATED & RESOLVED:** PEFT on MPS supports per-module `rank_pattern` / `alpha_pattern` with non-zero gradient flow. Selected as primary execution stack (ADR-010). |
| Gemma local/global pattern not derivable from config | Medium | Medium | **MITIGATED & RESOLVED:** Explicitly encoded in `text_config.layer_types` (28 local, 7 global). No heuristics needed. |
| Machine migration eats two days | High | Medium | **MITIGATED:** Fully complete. Apple M1 Max 64GB environment operational; 58 tests passing. |
| Borobia et al. publish rank ablation first | Low-Medium | Medium | They name it as future work in §7. Move fast, timestamp the repo |
| 15 days for 90 tracker tasks | High | Medium | Gates A/B/C are the real deliverables; tracker tasks are subordinate |


# ALOE — Decision Log

Architecture Decision Records. Each entry: what was decided, why, what it costs, and what would overturn it. Append; never rewrite history. If a decision reverses, add a new ADR that supersedes the old one.

---

## ADR-001 · Separate inspection from classification
**Date:** 2026-09-15 · **Status:** Accepted

**Decision.** `inspector.py` records what modules exist. `typing.py` decides what they mean. They never merge.

**Why.** Inspection is architecture-independent — walking an `nn.Module` tree is the same everywhere. Classification is architecture-specific. Fusing them would force a rewrite of the traversal for every new model family, which is exactly the failure mode SQ2 claims to avoid.

**Cost.** Two passes over the module tree instead of one. Negligible; inspection runs on meta tensors.

**Overturned if.** A family emerges whose modules cannot be enumerated without knowing the architecture first.

---

## ADR-002 · Core layers stay framework-free
**Date:** 2026-09-15 · **Status:** Accepted

**Decision.** `architecture/` and `search_space/` may import `torch` and `pydantic` only. No `peft`, no `bitsandbytes`, no `mlx`, no `unsloth`.

**Why.** The research contribution is the inspection → typing → validity → grammar chain. If that chain imports a training library, the contribution is welded to one execution stack and the reusability claim weakens from "works across architectures" to "works across architectures, on one vendor's GPUs".

**Cost.** An extra indirection at the executor boundary.

**Overturned if.** A validity rule genuinely cannot be expressed without importing the backend — in which case it belongs behind the backend interface, not in the core.

---

## ADR-003 · Two execution backends: CUDA-PEFT and MLX
**Date:** 2026-09-15 · **Status:** Accepted · **Supersedes the implicit "QLoRA everywhere" assumption**

**Decision.** `ExecutionBackend` is an interface with two implementations: `cuda_peft` (HuggingFace PEFT + bitsandbytes NF4) and `mlx` (Apple's MLX / mlx-lm). The backend is selected at runtime, never assumed.

**Why.** **bitsandbytes has no production Apple Silicon support.** The MPS backend exists only as an unmerged pull request that delegates to a separate `mps-bitsandbytes` package; mainline fails on ARM64 with CUDA-availability errors. Unsloth is CUDA-only. The development machine is moving to an M1 Max, so a QLoRA-via-PEFT-only design would not run at all.

MLX is Apple's native framework, built around unified memory, and `mlx-lm` supports LoRA and QLoRA directly. On an M1 Max with 32 GB it trains at roughly 250 tokens/second.

**Consequence for the research, and this is the important part.** Validity is a function of **(architecture, backend)**, not architecture alone. A module that is a legal LoRA target under PEFT may be unsupported under MLX. `constraints.py` therefore takes a backend argument. This is not a compromise — it strengthens SQ2, because a framework that derives correct search spaces across two architectures *and* two execution stacks is more convincingly reusable than one that does neither.

**Cost.** Two backends to maintain and two sets of validity rules to verify. Metrics are not directly comparable across backends and must never be pooled.

**Overturned if.** bitsandbytes ships stable Metal kernels in a tagged release, at which point `cuda_peft` could extend to Mac — though the MLX path would still be the faster one.

---

## ADR-004 · Hardware claims are re-scoped for Apple Silicon; Memory metric defined
**Date:** 2026-09-15 (Amended 2026-09-17) · **Status:** Accepted · **Needs supervisor sign-off**

**Decision.** The hardware objective is measured **peak unified memory** on Apple Silicon and **peak VRAM** on CUDA, reported separately and never averaged together.

On Apple Silicon, `peak_memory_gb` is formally defined as **peak process resident set size (RSS) sampled at 100 ms intervals**, because unified memory means model weights and framework buffers are counted in RSS rather than in a separate discrete device allocator. `mps_driver_allocated_gb` is recorded alongside it for diagnostic purposes but is **not** the multi-objective optimization Pareto objective.

**Why.** An M1 Max with 64 GB unified memory is consumer hardware, but it is not an 8 GB RTX 4060. The Avinash (arXiv:2509.12229) reference point — 8 GB, PagedAdamW, fp16 over bf16 — is a CUDA result and does not transfer to MPS. Claiming "consumer hardware" while running on 64 GB unified is defensible only if stated precisely. Furthermore, a 9.57 GB model cannot occupy 1.31 GB of memory; MPS driver allocation only reflects active tensor buffers dispatched through the Metal command queue, whereas RSS reflects actual unified memory occupied by model parameters, activations, and optimizer states.

**Cost.** RQ3 must name its hardware explicitly. Cross-backend Pareto fronts are two fronts, not one.

**Overturned if.** The project regains access to an 8–12 GB CUDA card, in which case that becomes the headline constraint and MLX becomes the secondary.

---

## ADR-005 · Validity means "can receive a gradient"
**Date:** 2026-09-15 · **Status:** Accepted

**Decision.** `Validity.VALID` asserts that a non-zero gradient reaches the adapter after one backward pass. Every backend implements `verify_gradients()`, and a target that attaches but receives no gradient is `EXCLUDED`, not `VALID`.

**Why.** This is the distinction that carries Gap 3. Borobia et al. (arXiv:2604.22127) §3.2 verify that adapters attached where intended — and still had to exclude Falcon-H1's `conv1d` and `out_proj` by hand after observing training instability. Their check confirmed intent was executed. It could not confirm the target was capable of learning. Deriving that, so the failure never happens, is the contribution.

**Cost.** One extra forward-backward pass per configuration before training starts.

**Overturned if.** Never, without giving up Gap 3.

---

## ADR-006 · Every validity rule carries its evidence
**Date:** 2026-09-15 · **Status:** Accepted

**Decision.** Rules live as data in `KNOWN_EXCLUSIONS`, each with a `reason` string naming a paper section or an experiment ID. A rule with no evidence is not merged.

**Why.** The project has already been burned by unsourced claims propagating through documents. A hardcoded `if "conv1d" in name` with no comment is the same failure in code.

---

## ADR-007 · Alpha range is deliberately wider than 2r
**Date:** 2026-09-15 · **Status:** Accepted

**Decision.** Default alpha candidates span 8 to 128 independently of rank.

**Why.** Gap 2. Baek et al. (arXiv:2602.11171) found configurations where alpha at 16–32× rank beat the convention. Borobia et al. used r=16, α=32 throughout — the default 2r heuristic, never searched. A grammar that ties alpha to rank cannot test the gap it claims to address.

**Cost.** The per-group space roughly doubles. Requires the matched-dimensionality control arm described in Gap 2's challenge cell, or gains cannot be attributed to alpha rather than to extra search dimensions.

---

## ADR-008 · Reading is triaged to four papers for September
**Date:** 2026-09-15 · **Status:** Accepted · **Needs supervisor sign-off**

**Decision.** September reading: AutoPEFT (deep), BFCL (deep), Mamba-2/SSD (method-level), LoftQ (skim). The remaining nine move to October.

**Why.** The supervisor's instruction is 95% implementation, 5% reading. Thirteen deep reads is not 5%. These four are the ones that change what gets built this month; the advisory plan itself marks LoRA+, PiSSA and RoSA as "not mandatory for v0.1" and defers multi-objective optimisation, which removes qEHVI from September too.

---

## ADR-009 · Gap 5 is narrowed; AutoPEFT is prior art
**Date:** 2026-09-15 · **Status:** Accepted

**Decision.** Gap 5 no longer claims that multi-objective PEFT optimisation is unexplored. It claims that **measured on-device cost** is not used as an objective.

**Why.** AutoPEFT (Zhou et al., TACL 2024) runs multi-objective Bayesian optimisation over PEFT configurations and returns a Pareto-optimal set trading task performance against parameter count. It is peer-reviewed and in-domain. The surviving distinction is that parameter count is a static proxy known before execution, whereas peak memory under 4-bit quantisation is not.

**Cost.** Gap 5's novelty rating drops from 7 to 5–6.

---

---

## ADR-010 · PEFT on Apple Silicon MPS as Primary Execution Stack
**Date:** 2026-09-17 · **Status:** Accepted · **Resolves Q3; Amends ADR-003**

**Decision.** The primary execution backend for ALOE's group-wise heterogeneous rank and alpha search space is **PEFT on Apple Silicon MPS (`peft_mps`)**. MLX is retained as a baseline/comparative runner for uniform LoRA.

**Why.** Empirical backend capability testing (Task I092 Hardware Capability Probe) demonstrated:
1. HuggingFace PEFT on PyTorch MPS fully supports dictionary-driven per-module configurations via `rank_pattern` and `alpha_pattern`, with non-zero gradient flow verified across all targets (Gate 4).
2. Apple's `mlx-lm` natively supports only uniform `lora_parameters: {"rank": r, "alpha": a}` applied uniformly across layers or simple integer depth. MLX cannot target heterogeneous ranks across different module groups without a complete custom engine rewrite.
3. Because ALOE's core novelty is group-wise heterogeneous rank/alpha attribution across architectural components (Gaps 1 & 2), PEFT on MPS satisfies the research requirement out of the box on consumer Apple Silicon hardware without requiring cloud GPUs.

**Cost.** PEFT forward/backward passes on MPS use PyTorch native dispatch, which has slightly higher dispatch overhead than pure Metal C++ MLX kernels (~15-20% slower per step), but natively supports arbitrary module group targeting.

---

## ADR-011 · Full Precision (bf16) as Primary Baseline; 4-bit Quantisation Acknowledged as Hardware-Dependent
**Date:** 2026-09-17 · **Status:** Accepted

**Decision.** Primary experimental runs for ALOE on Apple Silicon execute in full `bfloat16` precision without quantisation. 4-bit NF4 quantisation is formally classified as `NOT_SUPPORTED` on Apple Silicon MPS and reserved for CUDA execution environments.

**Why.** Empirical profiling during Stage 5 smoke training demonstrated:
1. Both models in full `bfloat16` fit comfortably within the M1 Max's 64 GB unified memory with massive headroom:
   - `google/gemma-4-E2B-it`: ~10.28 GB peak RSS memory during LoRA fine-tuning (1.31 GB MPS driver allocation).
   - `tiiuae/Falcon-H1-1.5B-Instruct`: ~3.64 GB peak RSS memory during LoRA fine-tuning (2.24 GB MPS driver allocation).
2. Over 50 GB of unified memory remains free on this workstation, completely obviating the need to quantise for capacity reasons.
3. `bitsandbytes` 4-bit NF4 quantisation lacks production Apple Silicon/MPS kernels and fails on ARM64. Running unquantised `bf16` eliminates quantisation-induced noise from the search space evaluations.

---

## Open questions

| # | Question | Blocking | Owner | Status |
|---|---|---|---|---|
| Q1 | Does the supervisor accept the Apple Silicon re-scope of RQ3? | ADR-004 | Raqeeb | Pending sign-off |
| Q2 | Four-paper September reading set — agreed? | ADR-008 | Raqeeb | Pending sign-off |
| Q3 | Does MLX expose per-module LoRA targeting fine enough for group-wise rank and alpha? | Whether MLX can be a first-class backend or only a baseline runner | Raqeeb | **RESOLVED (NO for MLX, YES for PEFT on MPS; see ADR-010 & `runs/probe/backends.json`)** |
| Q4 | Is Gemma 4's local/global pattern derivable from config, or does it need the layer list? | Gemma adapter correctness | Raqeeb | **RESOLVED (Derivable directly via `text_config.layer_types` in `config.json`; see `runs/architecture/google--gemma-4-E2B-it/claims.json`)** |
| Q5 | How do remaining reading batches (T5/T8/T9) sequence against Gate D implementation? | October planning | Raqeeb | **RESOLVED — see `OCTOBER_BACKLOG.md`** |

---

## ADR-012 · Local MPS Trial Budget Constraints for Hybrid State Space Architecture
**Date:** 2026-09-17 · **Status:** Accepted · **Superseded in part by ADR-015 for cloud search loops**

**Context.** In initial Stage 5 smoke training, `tiiuae/Falcon-H1-1.5B-Instruct` (incorporating 7 Mamba-2 SSM layers) executed on Apple Silicon MPS via PyTorch reference operations at ~6,918 ms/step (compared to ~713 ms/step on `google/gemma-4-E2B-it`). Running standard multi-epoch training on local Apple Silicon would exhaust developer workstation compute.

**Decision.** Restrict local Apple Silicon MPS trial budgets for Falcon-H1 to conservative smoke bounds:
1. Local step cap: Maximum 75 steps per trial.
2. Sequence length cap: Maximum 256 tokens during local interactive verification.
3. SSM projection freeze: Optional freezing of SSM recurrent projections during initial local gradient sanity checks.

**Why.** Enables rapid local verification of adapter attachment, forward pass sanity, and gradient flow on M1 Max without stalling the development loop.

**Consequences.** Full-budget multi-trial search campaigns cannot run efficiently on local MPS alone. Extended cloud execution is addressed in ADR-015.

---

## ADR-013 · Multi-Objective Bayesian Optimization with SAAS-GP and qLogNEHVI

**Date:** 2026-09-18 · **Status:** Superseded by ADR-016 · **Governs October Gate D**

**Context.** ALOE's architecture-typed search space parameterizes $D=6$ discrete dimensions across 3 typed component groups per model (`local_attn`, `global_attn`, `ffn` in Gemma 4; `global_attn`, `ssm`, `ffn` in Falcon-H1), scaling to $D \approx 30\text{--}60$ dimensions when evaluating finer-grained layer-selective or sub-module typing. While low-dimensional relative to module-level search ($D > 200$), a multi-objective search over combinatorial discrete configurations requires surrogate-assisted sample efficiency. AutoPEFT demonstrated that a sparse-axis-aligned subspace Gaussian process (SAAS-GP) surrogate fits high-dimensional PEFT search spaces with a modest number of evaluations. The project therefore needs a multi-objective optimizer that can simultaneously minimise competing objectives (e.g., task error rate, peak memory, trial latency) and return a Pareto frontier, not a single best point.

**Decision.** Adopt a multi-objective Bayesian optimisation (MOBO) loop for Gate D with the following design:
1. **Surrogate:** Sparse Axis-Aligned Subspace Gaussian Process (SAAS-GP) implemented in BoTorch/GPyTorch with a sparsity-inducing horseshoe prior. Prefer SAAS-GP over a standard GP because the effective dimensionality of the important axes is expected to be small relative to the full search dimension.
2. **Acquisition:** `qLogNEHVI` (logarithm of the noisy expected hypervolume improvement) via BoTorch, supporting parallel candidate generation and noisy observations from stochastic fine-tuning runs.
3. **Initialisation:** Sobol quasi-random sampling for the first 6–10 trials, followed by active acquisition iterations.
4. **Objectives:** At minimum, (a) validation task accuracy / loss and (b) measured wall-clock latency or memory. A third objective (adapter trainable parameter count) may be added if empirical Pareto plots remain degenerate.
5. **Backend-aware execution:** Gemma-4 E2B trials run locally on Apple Silicon MPS (ADR-010/ADR-011); Falcon-H1 hybrid trials run on the GCP L4 cloud tier (ADR-015). Objective measurements are never pooled across backends.
6. **Reproducibility:** Every trial configuration, observed metric vector, and model hyperparameters are serialised to the `TrialStore`; the final Pareto envelope is reproducible from `runs/gate_d/`.


**Why.**
1. *Sample efficiency:* SAAS-GP + qLogNEHVI is the closest published prior art (AutoPEFT) and is designed for expensive, high-dimensional, multi-objective experiments.
2. *Parallelism:* `q` -batch acquisition lets the harness submit multiple trial candidates per iteration, which maps naturally to cloud L4 parallelisation.
3. *Novelty preservation:* While AutoPEFT optimises rank allocation alone, ALOE extends the search space to typed architectural groups, target-module subsets, and alpha patterns (ADR-001), keeping the surrogate/acquisition choice as prior art rather than the claimed contribution.

**Consequences.**
1. Gate D implementation is blocked until Theme T5 multi-objective papers (Conway, Barker, Benmeziane, Rojas-Carrasco) are read and the BoTorch/Pyro dependencies are added.
2. The search space must expose a continuous/integer mixed representation compatible with BoTorch; categorical choices (e.g., which target modules to include) will be encoded as integer or binary variables with appropriate bounds.
3. Each objective must be normalised to a comparable scale before fitting the GP; z-score or min-max normalisation is recorded in the trial metadata.

---

## ADR-014 · Paired Bootstrap Statistical Validation Protocol

**Date:** 2026-09-18 · **Status:** Accepted · **Governs October Gate E**

**Context.** ALOE will compare multiple Pareto-optimal configurations against strong baselines (uniform LoRA, AutoPEFT-style rank search, possibly MLX uniform baseline) on tool-use benchmarks (xLAM, BFCL, τ-bench). Fine-tuning LLMs is stochastic: different random seeds produce non-trivial variance in final metrics. Reporting a single best run is therefore insufficient for dissertation-level claims.

**Decision.** Enforce a paired bootstrap statistical validation protocol for Gate E:
1. **Replication:** Every reported configuration is evaluated across at least three independent random seeds (≥3).
2. **Paired design:** For every baseline–candidate pair, collect paired metric differences on the same held-out validation set.
3. **Bootstrap:** Resample the paired differences with replacement (B = 10,000) to construct a 95% bias-corrected and accelerated (BCa) confidence interval and a two-sided p-value for the null hypothesis of zero mean difference.
4. **Multiple comparisons:** Apply Bonferroni correction when more than one candidate is compared against the same baseline; report both uncorrected and corrected p-values.
5. **Effect size:** Report Cohen's d (or a normalised mean difference) alongside p-values to guard against trivial statistically significant differences.
6. **Pre-registration:** The primary comparison pairs and significance threshold (α = 0.05) are documented in `aloe/docs/OCTOBER_BACKLOG.md` before experiments begin; post-hoc additions are flagged as exploratory.

**Why.**
1. *Reproducibility:* Three seeds is the minimum commonly accepted in neural-architecture-search literature for fine-tuning variance; bootstrap intervals avoid parametric assumptions.
2. *Rigour:* Paired testing accounts for dataset-level correlation between a baseline and a candidate trained on the same split.
3. *Credibility:* Pre-registration and Bonferroni correction protect against p-hacking when multiple Pareto points are reported.

**Consequences.**
1. Gate E budget must allocate at least 3× the number of final evaluation trials compared to a single-seed study.
2. The evaluation harness must persist per-seed metrics and raw prediction outputs to enable paired resampling.
3. Theme T9 statistical-validity papers (Schneider, Du, Raff) must be reviewed before final write-up to align language and method names with established practice.

---

## ADR-015 · Cloud GPU Hardware Tier for Mamba-2 Hybrid Acceleration (GCP NVIDIA L4 Offload)

**Date:** 2026-09-18 · **Status:** Accepted (Cloud GPU SKU updated by ADR-017) · **Amends and extends ADR-012 local MPS constraints**

**Context.** `tiiuae/Falcon-H1-1.5B-Instruct` incorporates 7 Mamba-2 Structured State Space (SSM) layers. On Apple Silicon MPS, Mamba-2 SSM chunk scans fall back to sequential PyTorch reference operations, producing ~10× higher step latency (6,918 ms/step vs ~713 ms/step on `google/gemma-4-E2B-it`). Running a standard 30–50 trial search campaign on a single M1 Max workstation would consume ~29+ hours, creating a critical bottleneck. Meanwhile, the project holds $1,636 in Google Cloud startup credits. An NVIDIA L4 GPU instance (`g2-standard-4`, 24GB GDDR6 VRAM, Ada Lovelace `sm_89`) provides hardware-accelerated chunked scans via compiled `mamba-ssm` and `causal-conv1d` CUDA kernels at ~$0.28/hr on Spot.

**Decision.** Establish a two-tier hybrid execution architecture:
1. **Tier 1 — Local Apple Silicon MPS (Primary Development & Standard Attention):**
   - Retain all interactive development, CLI workflows, unit testing, schema/grammar inspection (`aloe inspect`), and pre-flight gradient verification locally on Apple Silicon MPS (M1 Max 64GB).
   - Execute full 30-to-50 trial Bayesian search loops for `google/gemma-4-E2B-it` locally on MPS where Attention and FFN layers run at high efficiency (~713 ms/step).
   - For local Falcon-H1 smoke testing, retain conservative step/sequence caps per ADR-012 (75 steps, seq len 256, optional SSM freeze) to avoid local workstation starvation.
2. **Tier 2 — Cloud NVIDIA L4 Spot Instances (Falcon-H1 Hybrid Training & Heavy Search):**
   - Offload full-budget Falcon-H1 1.5B training trials, multi-seed Bayesian optimization campaigns, and unquantized `bfloat16` Mamba-2 evaluations to GCP `g2-standard-4` Spot instances (1× NVIDIA L4 24GB VRAM).
   - Compile hardware-accelerated `causal-conv1d` and `mamba-ssm` State Space Duality (SSD) chunked scan CUDA kernels via `aloe/scripts/cloud/gcp_l4_setup.sh`, accelerating Falcon step times from 6,918 ms to ~280–350 ms/step (15×–20× speedup).
3. **Automated Artifact & Storage Pipeline:**
   - Automatically sync trial envelopes, adapter checkpoints, and evaluation metrics from the cloud VM to Google Cloud Storage (`gs://aloe-checkpoints-...`) via parallel rsync, pulling down to the local repository for unified reporting.

**Why.**
1. *Throughput:* Hardware-accelerated chunked scanning compresses a 50-trial Falcon-H1 search campaign from ~29 hours down to ~2.5 hours of active GPU compute.
2. *VRAM & Precision:* 24 GB GDDR6 VRAM accommodates unquantized `bfloat16` Falcon-H1 weights (~3.64 GB), optimizer states (~7.3 GB), and batch sequence activations with >50% headroom, avoiding 4-bit quantization noise while matching the local full-precision baseline (ADR-011).
3. *Credit Budget Conservation:* Spot pricing of ~$0.2818/hr means a standard 50-trial campaign costs <$3.00. Even an extensive, highly buffered multi-seed campaign (100 instance hours) costs ~$28.18, consuming <2% of the available $1,636 credit pool.

**Consequences & Multi-Objective Metric Discipline.**
1. *Preservation of ADR-004:* Hardware Pareto fronts remain strictly stratified by backend. Apple Silicon trials report measured 100ms sampled process RSS (`peak_memory_gb`), while Cloud CUDA trials report measured device VRAM (`cuda_allocated_vram_gb`). They are documented side-by-side and never pooled.
2. *Automated Cost Protection:* Every cloud instance executes an automated idle watchdog daemon that synchronizes artifacts to GCS and triggers ACPI shutdown if GPU compute utilization remains below 5% for 15 minutes.

**Alternatives Considered.**
- *Alternative A: Confine Falcon-H1 strictly to Apple Silicon with reduced budgets.* Rejected for final research evaluation because a 15-trial cap at 75 steps restricts search space coverage on the hybrid SSM component (Gap 1).
- *Alternative B: Offload all models to Cloud A100/H100.* Rejected because A100 ($3.67/hr on-demand) is unnecessarily expensive for 1.5B–2B models, depletes startup credits prematurely, and violates ALOE's commitment to consumer-accessible local experimentation.
- *Alternative C: NVIDIA T4 (16GB, Turing).* Rejected because T4 lacks native BF16 Tensor Cores (forcing fp16 down-casting or slow fp32 simulation) and has insufficient memory bandwidth for Mamba-2 recurrent expansion.
- *Alternative D: NVIDIA L4 (24GB, Ada Lovelace).* Selected as the Pareto-optimal cloud tier: optimal price-performance ($0.28/hr Spot), 24GB GDDR6 VRAM, native BF16/FP8 Tensor Cores, and full compiled Mamba-2 chunk scan support.

---

## ADR-016 · 3-Objective Formulation (Tool Accuracy, Trainable Parameters, Peak Process RSS) & Random Search Baseline
**Date:** 2026-09-28 · **Status:** Accepted (Amended by ADR-018) · **Supersedes ADR-013**

**Context.**
1. ADR-013 specified multi-objective optimization using SAAS-GP and qLogNEHVI over objectives loosely defined as "task error rate, peak memory, trial latency". It contained a conceptual tension: describing the search space as $D=6$ (low-dimensional for Gemma 4's three component groups) while simultaneously invoking SAAS-GP, which is specifically designed for high-dimensional regimes ($D \ge 50$).
2. During empirical implementation of the proposal Proof of Concept, two critical realities were established:
   - *Flat Memory Risk:* On Apple Silicon Unified Memory with fixed batch size ($b=2$) and sequence length ($L=256$), peak process RSS can exhibit low variance across small rank deltas (e.g. $r=4$ vs $r=8$) because PyTorch allocator blocks dominate memory. In a 2-objective setup (Accuracy vs Memory), a flat memory metric collapses search into a 1-dimensional accuracy optimization.
   - *Baseline Necessity:* Before deploying computationally heavy Gaussian Process surrogates in Gate D, Design Science Research methodology requires an unbiased, empirical **Random Search Baseline** with parameter-budget matching to establish the ground-truth Pareto frontier.

**Decision.**
1. **Three-Objective Formulation:** The ALOE objective vector $\vec{f}(\vec{x})$ is formally frozen to three distinct criteria:
   - $f_1(\vec{x}) = Acc_{\text{arg}}$: Argument-level exact match AST tool-calling accuracy (**Maximize**, range $[0.0, 1.0]$).
   - $f_2(\vec{x}) = N_{\text{params}}$: Total active trainable adapter parameters (**Minimize**, discrete integer count).
   - $f_3(\vec{x}) = M_{\text{RSS}}$: Peak process resident set size memory sampled at 100ms intervals (**Minimize**, MB).
2. **Budget-Parity Sampling Invariant:** Any candidate configuration generated during search must satisfy:
   $$\left| \frac{N_{\text{params}}^{(\text{typed})} - N_{\text{params}}^{(\text{base})}}{N_{\text{params}}^{(\text{base})}} \right| \le 0.05$$
   Candidates violating this constraint are rejected pre-flight by `RandomSearchSampler`, eliminating parameter count as an experimental confounder.
3. **Staged Optimizer Deployment:**
   - *Proposal PoC (September–October):* `RandomSearchSampler` + `extract_pareto_frontier` + 2-D/3-D hypervolume calculation (`calculate_hypervolume_2d`).
   - *Final Project (October Gate D):* Standard Matérn-5/2 Gaussian Process surrogate with qLogNEHVI acquisition function in BoTorch, evaluated against the random search baseline.

**Consequences.**
- Solves the $D=6$ vs high-dimensional contradiction by deploying standard GP surrogates appropriate for $D \in [6, 30]$ rather than SAAS-GP.
- Adding $N_{\text{params}}$ hedges against flat memory plateaus, guaranteeing a live, highly sensitive optimization dimension.
- Provides a clean, mathematically sound Pareto front for examiner viva defense.





---

## ADR-017 · Colab Pro+ NVIDIA A100 SKU Pinning & Persistent Drive Resume for Hybrid Tier
**Date:** 2026-11-21 · **Status:** Accepted · **Amends ADR-015 Cloud GPU Tier (L4 → Pinned A100)**

**Context.** ADR-015 selected Spot GCP NVIDIA L4 (`g2-standard-4`, 24 GB) when cloud execution drew from billed startup credits, and expanded hybrid evaluation to include both `tiiuae/Falcon-H1-1.5B-Instruct` (Mamba-2 + Attention) and `LiquidAI/LFM3-1.7B` (ShortConv + Local/Global Attention). In November 2026, 2,000 prepaid Google Colab Pro+ compute units (~260–300 NVIDIA A100 GPU-hours expiring 30 Nov 2026) became available at zero incremental cost. On `NVIDIA A100-SXM4-40GB` (1,555 GB/s HBM2e), compiled `mamba_ssm` and `causal_conv1d` kernels complete full 20-trial search + 3-seed confirmation campaigns ~2.5–3× faster than L4 while maintaining native `bfloat16` arithmetic.

**Decision.**
1. **Exact GPU SKU Locking:** Assign both hybrid models (`Falcon-H1-1.5B-Instruct` and `LiquidAI/LFM3-1.7B`) to a single pinned CUDA GPU SKU (`pinned_gpu="A100"`), recording the exact first-trial `gpu_name` (e.g., `NVIDIA A100-SXM4-40GB`) in `CampaignLedgerState.locked_gpu_name` and rejecting any resume or worker invocation on a different GPU variant (e.g., 80GB or L4). Local-tier `google/gemma-4-E2B-it` remains strictly on Apple Silicon MPS (`process_rss_sampled_100ms`), and `run_colab_a100_campaign.py` refuses Gemma on CUDA.
2. **Fail-Closed Fast Hybrid Kernels:** Official CUDA campaigns require `mamba_ssm` + `causal_conv1d` for Falcon-H1 and `causal_conv1d` for LFM3 (`enforce_hardware_contract`), caching wheels in ABI-keyed Google Drive folders (`wheels_cache/py<ver>_torch<ver>_cu<ver>/`).
3. **Atomic Fsync & Incremental Drive Mirroring:** Every `CampaignLedger` save, 25-step `weights_only=True` training checkpoint (`checkpoint_state.pt`), incremental `predictions_*.jsonl` append, and `trial.json` envelope uses `NamedTemporaryFile + flush + os.fsync + os.replace + dir_fsync` and mirrors immediately to Google Drive (`/content/drive/MyDrive/aloe_ipd_v1`), restoring mid-trial checkpoints and partial predictions automatically after session preemption.

**Why.** Eliminates out-of-pocket GCP spend during IPD execution, cuts hybrid campaign wall-clock time by ~65%, and guarantees zero lost completed steps or evaluation samples under Colab VM preemption.

**Cost.** Requires Google Drive mount latency (~10–30 ms per mirrored checkpoint/prediction flush) and strict single-SKU discipline (`locked_gpu_name`).

**Overturned if.** Prepaid Colab A100 quota is exhausted before campaign completion — in which case any *new* model campaign reverts to a dedicated GCP L4 Spot instance (`ADR-015`), never mixing L4 and A100 trials inside the same model campaign.

---

## ADR-018 · Protocol-v1 Amendments: Sobol-Nadir Hypervolume Calibration, Tier-Specific Memory Objective, & Step-Down Holm-Bonferroni
**Date:** 2026-11-21 · **Status:** Accepted · **Amends ADR-014 and ADR-016**

**Context.** Post-Proposal static verification identified three mathematical alignment requirements before launching full IPD compute:
1. *Hypervolume Reference-Point Scaling (`ADR-016`):* Under a static uncalibrated reference point $\vec{r} = [0, -10^8, -40{,}000]$, variations inside the $\pm 5\%$ parameter parity band ($\Delta N \approx 0.6\text{M}$ against $10^8$) and memory deltas ($\Delta M \approx 150\text{ MB}$ against $40{,}000\text{ MB}$) change those box edges by $<1\%$, collapsing 3-D hypervolume into near-1-D accuracy ranking.
2. *Tier-Specific Memory Objective (`ADR-012` / `ADR-016`):* `ADR-016` wrote $f_3$ solely as $M_{\text{RSS}}$, whereas hybrid CUDA campaigns optimize peak CUDA allocator VRAM (`cuda_max_memory_allocated`).
3. *Single Multiplicity Rule (`ADR-014`):* Step-down Holm-Bonferroni uniformly dominates classical single-step Bonferroni in statistical power while controlling the identical family-wise error rate ($\text{FWER} \le \alpha = 0.05$).

**Decision.**
1. **Sobol-Nadir Dynamic Reference Point (`MOBOptimizer`):** In maximization coordinates $\vec{y} = [\text{Acc}_{\text{exact}}, -N_{\text{params}}, -M_{\text{tier}}]$, after observing the initial $n_{\text{init}} = 5$ scrambled-Sobol trials $Y_{1:5} \in \mathbb{R}^{5 \times 3}$, lock the reference point once for the campaign (persisted in `CampaignLedgerState.optimizer_state`) as:
   $$r_k = \min_{1 \le i \le 5} Y_{i,k} - 0.15 \cdot \max\!\left(\max_{1 \le i \le 5} Y_{i,k} - \min_{1 \le i \le 5} Y_{i,k},\; s_k^{\min}\right), \quad k \in \{1, 2, 3\}$$
   where $\vec{s}^{\min} = \max(0.02 |\vec{y}_{\min}|, [0.05, 1000, 50])$. Retained $N_{\text{params}}$ within the $\pm 5\%$ band rewards leaner Pareto-optimal adapters when accuracy ties occur without permitting budget confounding.
2. **Tier-Isolated Memory Objective ($f_3$):** Define $f_3(\vec{x}) = M_{\text{RSS}}$ (`process_rss_sampled_100ms`, MB) on Apple Silicon MPS and $f_3(\vec{x}) = M_{\text{VRAM}}$ (`cuda_max_memory_allocated`, MB) on CUDA, recorded in `locked_memory_metric_name` and never pooled across tiers (`aloe evaluate` aborts on mixed `memory_metric_name` or `gpu_name`).
3. **Pre-Registered Inferential Protocol (Amending `ADR-014`):**
   - Primary metric: strict multi-call AST `exact_match_accuracy` (`v2.0.0-strict`).
   - Uncertainty: $B = 10{,}000$ paired query bootstrap (`BCa` 95% CI, conditional on the fixed 3-seed set $\{42, 1337, 2026\}$ required by `--final`) plus exact two-sided binomial McNemar test on odd-seed majority-vote query outcomes.
   - Multiplicity: Step-down **Holm-Bonferroni** ($\alpha = 0.05$, rejecting while $p^{\text{Holm}}_{(i)} \le \alpha$ and halting at the first $p^{\text{Holm}}_{(k)} > \alpha$) supersedes single-step Bonferroni across the pre-declared comparator family (`B0`, `B1`, `B2`, `Borobia-*`).

**Dual-Scale Mathematical Verification.**
- *Scale-Down (A-Level Worked Numeric Example):* Suppose 5 initial Sobol trials achieve accuracies $[0.50, 0.60]$, negative parameters $[-12.4\text{M}, -11.6\text{M}]$ (span $0.8\text{M}$), and negative memory $[-8200, -8000]\text{ MB}$ (span $200\text{ MB}$). The Sobol-calibrated reference point is $\vec{r} = [0.50 - 0.15(0.10),\; -12.4\text{M} - 0.15(0.8\text{M}),\; -8200 - 0.15(200)] = [0.485,\; -12.52\text{M},\; -8230]$. A candidate improving memory from $8100\text{ MB}$ to $8000\text{ MB}$ at $\text{Acc}=0.60, N=12.0\text{M}$ expands the 3-D box edge $(-8000 - (-8230)) / (-8100 - (-8230)) = 230 / 130 = \mathbf{1.77\times}$ (+77% hypervolume gain!), whereas an uncalibrated $-40{,}000$ reference gave $32{,}000 / 31{,}900 = 1.003\times$ (+0.3%). Similarly, for $m=3$ baseline comparisons with sorted raw p-values $p_{(1)}=0.012, p_{(2)}=0.020, p_{(3)}=0.040$, Holm adjusted values are $p^{\text{Holm}}_{(1)} = 3(0.012) = 0.036 \le 0.05$ (**Reject**), $p^{\text{Holm}}_{(2)} = \max(0.036, 2 \times 0.020) = 0.040 \le 0.05$ (**Reject**), and $p^{\text{Holm}}_{(3)} = \max(0.040, 1 \times 0.040) = 0.040 \le 0.05$ (**Reject**), whereas conservative Bonferroni ($3 \times 0.020 = 0.060 > 0.05$) would have missed the second and third discoveries.
- *Scale-Up (Exact Shape-Sum & Discrete $q\text{LogNEHVI}$):* For architecture modules $m \in \mathcal{M}_g$ of functional role $g \in \mathcal{G}$ with weight shape $(d_{\text{out},m}, d_{\text{in},m})$, exact trainable parameter count is $N_{\text{params}}(\vec{r}) = \sum_{g \in \mathcal{G}} \mathbb{I}[e_g] \, r_g \sum_{m \in \mathcal{M}_g} (d_{\text{in},m} + d_{\text{out},m})$, constrained to $\mathcal{F}_{0.05} = \{(\vec{e}, \vec{r}, \vec{\alpha}, \eta) : |N_{\text{params}}(\vec{r}) - N_{\text{base}}| / N_{\text{base}} \le 0.05\}$. Independent Matérn-5/2 ARD Gaussian Processes model $f_1, -f_2, -f_3$, and sequential acquisition maximizes $\alpha_{q\text{LogNEHVI}}(\vec{x}) = \log \mathbb{E}_{p(\vec{f} \mid \mathcal{D}_t)}[\text{HVI}(\vec{f}(\vec{x}) \mid \mathcal{P}_t, \vec{r})]$ directly over $\vec{x} \in \mathcal{F}_{0.05} \setminus \mathcal{D}_t$ via `optimize_acqf_discrete` (evaluated in exact $2{,}048$-point chunks when $|\mathcal{F}_{0.05}| > 2{,}048$).

**Why.** Restores true 3-objective Pareto sensitivity inside the $\pm 5\%$ parity band, formalizes MPS-vs-CUDA memory objective semantics, and unifies statistical multiplicity under pre-registered Holm-Bonferroni.

**Cost.** Requires completing $n_{\text{init}} = 5$ Sobol observations before locking $\vec{r}$, and requires $\ge 3$ identical seeds (`--final`) for locked test inference.

**Overturned if.** Sobol initial points all fail before $n_{\text{init}}$ observations complete — prevented by `CampaignRunner`'s fail-closed `max_attempts_per_trial` abort.


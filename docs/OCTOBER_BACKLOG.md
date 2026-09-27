# ALOE October Engineering Backlog & Architecture Specification

**Document Version:** 1.0.0  
**Target Milestone:** October 2026 Sprint (Gates D & E)  
**Author:** Raqeeb Rameez (IIT ID: 20230943 / UoW ID: w2082238)  
**Status:** **APPROVED FOR IMPLEMENTATION**  
**Core Dependencies:** `aloe.search_space.schema.SearchGrammar`, `aloe.execution.trial_config.TrialConfig`, BoTorch $\ge 0.11$, Ax-Platform $\ge 0.4$, GPyTorch $\ge 1.13$.

---

## 1. Overview & Objective

Having frozen the core architecture and validated Gates A, B, and C in ALOE v0.1.0-alpha, the October milestone shifts from infrastructure establishment to **autonomous multi-objective exploration** (Gate D) and **statistically grounded validation** (Gate E).

The October backlog defines three interconnected technical pillars:
1. **Gate D — Multi-Objective Bayesian Optimization (MOBO):** Implementation of a noisy, parallel hypervolume improvement loop utilizing a Sparse Axis-Aligned Subspaces Gaussian Process (SAAS-GP) surrogate model and the $q\text{LogNEHVI}$ acquisition function.
2. **Gate E — Statistical Significance & Rigorous Validation:** A nested multi-seed evaluation protocol employing paired bootstrap testing across $\ge 3$ random seeds, confidence interval reporting, and EvalCards reproducibility standards.
3. **Multi-Benchmark Evaluation Harnesses:** Expansion beyond the pilot xLAM split to integrate standardized agentic tool-use benchmarks (Berkeley Function Calling Leaderboard [BFCL] and $\tau$-bench).

---

## 2. Gate D: Multi-Objective Bayesian Optimization (MOBO)

### 2.1 Theoretical Formulation & Mathematical Objectives

ALOE frames parameter-efficient adaptation as a constrained, dual-objective black-box optimization problem over the discrete/continuous hybrid search space defined by `SearchGrammar`:

$$\max_{\mathbf{x} \in \mathcal{X}} \; \mathbf{f}(\mathbf{x}) = \begin{bmatrix} f_{\text{accuracy}}(\mathbf{x}) \\ - f_{\text{memory}}(\mathbf{x}) \end{bmatrix}$$

where:
- $\mathbf{x} \in \mathcal{X}$ is a concrete configuration vector derived from the architecture-specific grammar.
- $f_{\text{accuracy}}(\mathbf{x}) \in [0.0, 1.0]$ is the agentic tool-calling accuracy (AST and argument match score on the held-out search validation set).
- $f_{\text{memory}}(\mathbf{x}) \in \mathbb{R}^+$ is the peak Resident Set Size (RSS) in GB sampled at 100ms intervals on Apple Silicon unified memory (or peak VRAM on CUDA). Formulated as a maximization of negative memory to align with BoTorch conventions.
- $f_{\text{latency}}(\mathbf{x})$ is recorded as an auxiliary tracking metric (mean ms/step) and used as a feasibility constraint ($f_{\text{latency}}(\mathbf{x}) \le \tau_{\max}$).

### 2.2 Search Space Parameterization (`aloe/search_space/bo_adapter.py`)

The discrete choices in `GroupDimensions` are mapped into a standardized parameter space for BoTorch/Ax:

| Parameter | Domain | Type | Description |
| :--- | :---: | :---: | :--- |
| `enabled_{group}` | $\{0, 1\}$ | Binary | Whether the architectural group receives LoRA adapters. |
| `rank_{group}` | $\{4, 8, 16, 32\}$ | Ordinal / Log2 | Rank allocation for the active component group. |
| `alpha_{group}` | $\{8, 16, 32, 64, 128\}$ | Ordinal / Log2 | Scaling factor (decoupled from $2r$ heuristic per Gap 2). |

When an architectural group is disabled (`enabled=0`), rank and alpha selections collapse to an inactive configuration ($1$ operational state). When enabled (`enabled=1`), there are $|R| \times |A| = 4 \times 5 = 20$ configurations, yielding $1 + 20 = 21$ distinct operational states per group. For Gemma 4-E2B (3 groups: `local_attn`, `global_attn`, `ffn`), the non-trivial discrete search space cardinality is $21^3 - 1 = 9,260$ points ($9,261$ including the trivial all-disabled baseline). For Falcon-H1 1.5B (3 groups: `global_attn`, `ssm`, `ffn`), the non-trivial cardinality is likewise $9,260$ points.


### 2.3 Surrogate Model: Sparse Axis-Aligned Subspace Gaussian Process (SAAS-GP)

Standard Gaussian Processes with Matérn or RBF kernels degrade rapidly when the search space exceeds 10–15 dimensions (the curse of dimensionality). In architectural PEFT, while the total parameter space has $D \approx 12\text{--}24$ dimensions, only a sparse subset of architectural groups typically drive the trade-off frontier.

To model this phenomena, Gate D deploys the **SAAS-GP** prior (Eriksson & Jankowiak, AISTATS 2021):

1. **Hierarchical Half-Cauchy Shrinkage Prior:**
   $$\tau \sim \mathcal{HC}(\tau_0), \quad \lambda_d \sim \mathcal{HC}(1), \quad \ell_d = \tau \lambda_d \quad \text{for } d = 1, \dots, D$$
   where $\ell_d$ is the lengthscale for dimension $d$, and $\tau$ is a global shrinkage parameter.
2. **Effect:**
   When a dimension $d$ has negligible impact on tool accuracy or memory, its lengthscale $\ell_d \to \infty$ (inverse lengthscale $\to 0$), effectively removing that dimension from the kernel computation without requiring discrete feature selection.
3. **Inference Engine:**
   Markov Chain Monte Carlo (MCMC) sampling via No-U-Turn Sampler (NUTS) implemented in Pyro / BoTorch, generating 4 chains of 256 posterior samples.

### 2.4 Acquisition Function: $q\text{LogNEHVI}$

Candidate selection is governed by **Quasi-Noisy Expected Hypervolume Improvement in Log-Space** ($q\text{LogNEHVI}$, Daulton et al., NeurIPS 2020/2021; P15 in ALOE literature matrix):

$$\alpha_{q\text{LogNEHVI}}(\mathbf{X}) = \log \mathbb{E}_{\mathbf{Y} \sim p(\mathbf{f}(\mathbf{X}) | \mathcal{D})} \left[ \text{HVI}(\mathbf{Y}, \mathcal{P}, \mathbf{r}) \right]$$

**Why $q\text{LogNEHVI}$ is mandatory for ALOE:**
1. **Gradient Flow in Flat Regions:** Standard $q\text{EHVI}$ suffers from vanishing gradients when evaluating points far from the current empirical Pareto frontier. Log-space evaluation maintains smooth gradient propagation throughout the entire domain.
2. **Observation Noise Tolerance:** Tool accuracy on small validation sets exhibits empirical sampling variance. $q\text{LogNEHVI}$ treats function evaluations as noisy observations ($y_i \sim \mathcal{N}(f(\mathbf{x}_i), \sigma_i^2)$), integrating over the posterior distribution rather than assuming deterministic outputs.
3. **Batch Evaluation ($q \ge 1$):** Allows selecting batches of configurations for concurrent execution across available execution resources.

**Reference Point Setting:**
- Accuracy reference: $r_1 = 0.0$ (0.0% tool accuracy).
- Memory reference: $r_2 = -24.0$ GB (representing the maximum allowable memory budget on a 32GB/64GB workstation).

### 2.5 Optimization Loop Architecture & Budget Allocation

```text
               ┌────────────────────────────────────────────────────────┐
               │ 1. INITIALIZATION: Sobol Quasi-Random (N_init = 6)     │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │ 2. REPRODUCIBLE EXECUTION: PeftMPSBackend / TrialStore │
               │    Evaluates candidate, samples 100ms RSS, stores JSON │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │ 3. SURROGATE FIT: Fit SAAS-GP via NUTS (BoTorch/Pyro)  │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │ 4. ACQUISITION: Optimize qLogNEHVI -> Next x*          │
               └───────────────────────────┬────────────────────────────┘
                                           │
                            [Repeat until Budget Exhausted]
```

**Trial Budgets (Adhering to ADR-012):**
- **Gemma 4-E2B (Workstation Native):**
  - 6 initial Sobol trials + 24 active BO iterations = **30 total trials**.
  - Budget: 200 steps per trial, seq len 512.
  - Expected duration: 30 trials $\times$ 2.4 min = **~1.2 hours**.
- **Falcon-H1 1.5B (Asymmetric Budget / Cloud Hybrid):**
  - Workstation mode: 6 initial Sobol trials + 9 active BO iterations = **15 total trials**, capped at 75 steps, seq len 256 per ADR-012 (~1.8 hours).
  - Cloud offload mode: Optional execution on GCP NVIDIA L4 GPU via `aloe/scripts/cloud/gcp_l4_setup.sh`, removing the step cap.

---

## 3. Gate E: Statistical Significance & Validation Protocol

To guarantee that reported Pareto improvements are genuine and not artifacts of seed selection or dataset noise (addressing literature findings #27 Schneider, #36 Du, and #40 Raff), Gate E enforces a three-stage statistical verification protocol.

### 3.1 Multi-Seed Evaluation Matrix

The final non-dominated Pareto configurations, alongside uniform LoRA and full-rank baselines, must be evaluated across at least three distinct random seeds:
$$\mathcal{S} = \{42, 1337, 2026\}$$

All seeds govern:
1. Dataset minibatch sampling order.
2. LoRA weight initialization ($A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$).
3. Dropout and generation sampling (greedy decoding $\tau=0.0$ used for final evaluation).

### 3.2 Paired Bootstrap Significance Protocol (`aloe/evaluation/bootstrap.py`)

Point estimates of benchmark accuracy (e.g., "62.5% vs. 50.0%") are statistically incomplete. Gate E mandates paired bootstrap testing:

1. Let $\mathcal{D}_{\text{test}} = \{(q_i, \mathcal{T}_i, a_i^*)\}_{i=1}^M$ be the held-out test set ($M \ge 200$).
2. For candidate model $\mathcal{M}_A$ and baseline model $\mathcal{M}_B$, record binary per-query correctness:
   $$c_{A, i} \in \{0, 1\}, \quad c_{B, i} \in \{0, 1\} \quad \text{for } i = 1, \dots, M$$
3. Perform $B = 10,000$ bootstrap resamples:
   - For $b = 1, \dots, B$:
     - Sample $M$ indices with replacement: $\mathcal{I}^{(b)} = \{j_1, \dots, j_M\} \sim \text{Uniform}(1, M)$.
     - Calculate sample mean accuracies:
       $$\bar{acc}_A^{(b)} = \frac{1}{M} \sum_{k \in \mathcal{I}^{(b)}} c_{A, k}, \quad \bar{acc}_B^{(b)} = \frac{1}{M} \sum_{k \in \mathcal{I}^{(b)}} c_{B, k}$$
     - Record accuracy difference: $\Delta^{(b)} = \bar{acc}_A^{(b)} - \bar{acc}_B^{(b)}$.
4. **95% Confidence Interval:**
   $$\text{CI}_{95\%} = \left[ \text{Percentile}(\Delta, 2.5), \; \text{Percentile}(\Delta, 97.5) \right]$$
5. **Empirical $p$-Value Calculation:**
   $$p = 2 \cdot \min \left( \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{(b)} \le 0), \; \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{(b)} \ge 0) \right)$$
6. **Acceptance Rule:**
   A candidate configuration is reported as statistically superior to baseline *if and only if* $p < 0.05$ AND the lower bound of $\text{CI}_{95\%} > 0.0$.

### 3.3 EvalCards Reproducibility Standard (`aloe/evaluation/eval_card.py`)

Every evaluated candidate configuration generates a machine-readable `EvalCard.json` complying with the Raff et al. (#40) reproducibility specification:

```json
{
  "eval_card_version": "1.0",
  "model": {
    "model_id": "google/gemma-4-E2B-it",
    "base_commit": "9cecf17",
    "parameters_total": 2488422400,
    "trainable_parameters": 10057728,
    "parameter_fraction": 0.00404
  },
  "search_point": {
    "group_settings": {
      "local_attention": {"enabled": true, "rank": 16, "alpha": 32},
      "global_attention": {"enabled": true, "rank": 16, "alpha": 32},
      "ffn": {"enabled": true, "rank": 4, "alpha": 8}
    }
  },
  "metrics": {
    "tool_accuracy_mean": 0.625,
    "tool_accuracy_std": 0.018,
    "ci_95_bootstrap": [0.035, 0.125],
    "p_value_vs_uniform": 0.012,
    "peak_memory_rss_gb": 11.95,
    "memory_metric": "process_rss_unified"
  },
  "environment": {
    "hardware": "Apple M1 Max (64 GB)",
    "os": "macOS 15.x",
    "backend": "peft_mps",
    "torch_version": "2.x",
    "seeds_evaluated": [42, 1337, 2026]
  }
}
```

---

## 4. Multi-Benchmark Evaluation Harnesses

ALOE v0.1 verified the internal xLAM tool-disjoint split. In October, the evaluation engine will be coupled to two standard external agentic benchmarks:

### 4.1 Berkeley Function Calling Leaderboard (BFCL) Integration

- **Target Benchmark:** BFCL v3 (Patil et al., 2025; P10 in ALOE literature matrix).
- **Categories Evaluated:**
  1. *Simple Function Calling:* Single tool invocation with string/integer/boolean parameters.
  2. *Multiple Function Calling:* Disambiguation among 5–10 available tools.
  3. *Parallel Tool Calling:* Emitting multiple independent tool calls in a single response turn.
  4. *Irrelevance / No-Tool Turns:* Suppressing tool calls when user prompt is chit-chat or general reasoning.
- **Implementation Design (`aloe/evaluation/harnesses/bfcl_adapter.py`):**
  - Adapts ALOE model outputs to BFCL AST parser.
  - Generates standardized JSON error classification: `WRONG_FUNCTION_NAME`, `MISSING_ARGUMENT`, `TYPE_MISMATCH`, `SYNTAX_ERROR`, `FALSE_POSITIVE_INVOCATION`.

### 4.2 $\tau$-bench (Tau-Bench) Agentic Interaction Harness

- **Target Benchmark:** $\tau$-bench (Yao et al., 2025; P11 in ALOE literature matrix).
- **Evaluation Paradigm:** Multi-turn conversation with dynamic environment state tracking in retail and airline domains.
- **Core Requirements:**
  - Evaluates not just one-shot function emission, but the agent's ability to act upon API return values and user corrections over multi-turn dialogues.
  - Pass@1 task completion metric under simulated user personas.
- **Implementation Design (`aloe/evaluation/harnesses/tau_adapter.py`):**
  - Lightweight local environment runner without external API dependencies.
  - Execution sandbox checking database state mutations.

---

## 5. Work Breakdown Structure (WBS) & Milestones

```text
October 2026
├── Week 1 (Oct 01 – Oct 08): Gate D Implementation
│   ├── Task D1: Create aloe/search_space/bo_adapter.py (SearchGrammar <-> BoTorch mapping)
│   ├── Task D2: Implement SAAS-GP surrogate model with Pyro NUTS MCMC sampler
│   ├── Task D3: Implement qLogNEHVI acquisition loop with Sobol quasi-random initialization
│   └── Task D4: Write tests/test_mobo_loop.py with mock execution backend
│
├── Week 2 (Oct 09 – Oct 18): Gate E Execution & Statistical Validation
│   ├── Task E1: Implement aloe/evaluation/bootstrap.py (paired bootstrap, CIs, p-values)
│   ├── Task E2: Execute 30-trial search on Gemma 4-E2B across seeds {42, 1337, 2026}
│   ├── Task E3: Execute 15-trial search on Falcon-H1 1.5B (MPS / GCP L4 hybrid)
│   └── Task E4: Generate automated EvalCards and Pareto frontier artifacts
│
└── Week 3 (Oct 19 – Oct 31): Benchmarks & Dissertation Chapter 5 Synthesis
    ├── Task B1: Integrate BFCL simple & parallel function calling harnesses
    ├── Task B2: Integrate tau-bench multi-turn simulation harness
    ├── Task B3: Construct cross-model Pareto frontier comparisons
    └── Task B4: Author Dissertation Chapter 5 (Empirical Evaluation & Research Impact)
```

---

## 6. Acceptance Criteria for October Milestone

- [ ] `tests/test_mobo_loop.py` executes a synthetic 5-iteration MOBO optimization run without failure.
- [ ] Gemma 4-E2B completes 30 trials; non-dominated Pareto frontier exhibits a non-trivial trade-off curve between tool accuracy and process RSS.
- [ ] All reported Pareto improvements over uniform LoRA carry empirical paired-bootstrap 95% confidence intervals and two-tailed $p$-values ($p < 0.05$).
- [ ] Each non-dominated model configuration possesses a validated, reproducible `EvalCard.json`.
- [ ] Falcon-H1 cross-architecture comparison confirms search grammar reuse without manual optimizer modifications.

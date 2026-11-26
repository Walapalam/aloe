# ALOE Protocol-v1 Pre-Unseal Statistical & Multi-Objective Analysis Plan (`IPD-PROTOCOL-V1`)

## 1. Scope & Pre-Registration Commitment
This document freezes the exact statistical endpoints, hypothesis families, multiplicity adjustments, composite non-inferiority decision rule, execution retry policy, hypervolume coordinate bounds, and zero-shot baseline rules **prior to unsealing** the `400`-item unseen-tool `test` split (`splits["test"]` in `data/manifests/xlam_ipd_v1.json`, `manifest_sha256 = f8832777fd2cd61a68ed3ea875031bacac11c33da8e762cc775f191965291ce1`).

No winner configuration, hyperparameter, prompt template, decoding budget (`max_new_tokens = 192`, greedy `do_sample = False`), or statistical comparison rule may be altered after the write-once Seal Ceremony commit is published.

---

## 2. Frozen Evaluation Batch (`63` Receipts Total) & Execution Retry Policy
1. **Trained Winner Evaluations (`60` Receipts):**
   - `google/gemma-4-E2B-it` (`Apple M1 Max` MPS, `process_rss_sampled_100ms`, `git_commit = 8187255cae076f583dce455acaf5019b20bad1e4`): `6` frozen arm winners (`aloe`, `b0`, `b1`, `b2`, `borobia_attn`, `borobia_ffn`) × `3` training seeds `[42, 1337, 2026]` = **`18` runs**.
   - `tiiuae/Falcon-H1-1.5B-Instruct` (`NVIDIA L4` CUDA, `cuda_max_memory_allocated`, `git_commit = f0d0ea2b56b441e18d475366ea249617afe65883`): `7` frozen arm winners (`aloe`, `b0`, `b1`, `b2`, `borobia_attn`, `borobia_ssm`, `borobia_ffn`) × `3` training seeds `[42, 1337, 2026]` = **`21` runs**.
   - `LiquidAI/LFM3-1.7B` (`NVIDIA L4` CUDA, `cuda_max_memory_allocated`, `git_commit = ab9877ce5110f3889d2d2f028a4d3ff559bd48b2`): `7` frozen arm winners × `3` training seeds `[42, 1337, 2026]` = **`21` runs**.
   - Every run executes the uniform canonical worker path (`aloe.execution.worker_entry --eval-split test --final --seal-file test_unsealed.json`) with `max_steps = 125` on `splits["train"]`, fresh-process reload verification, and strict `400`-item evaluation on `splits["test"]`.
2. **Hardware-Matched Deterministic Base Zero-Shot Evaluations (`3` Receipts):**
   - One unmodified base-model (`0` adapters) greedy evaluation (`do_sample = False`, `max_new_tokens = 192`) per model on its canonical hardware tier (`Apple M1 Max` MPS for `Gemma-4-E2B`; `NVIDIA L4` CUDA for `Falcon-H1-1.5B` and `LFM3-1.7B`).
   - **Pre-Declared `LFM3` Zero-Shot Disclosure:** Under the frozen `max_new_tokens = 192` protocol, unmodified `LiquidAI/LFM3-1.7B` emits unprompted chain-of-thought text that frequently truncates prior to closing JSON tool calls (`10.5%` validation parse rate). We retain the uniform `192`-token protocol without post-hoc modification and interpret `LFM3`'s zero-shot-to-PEFT delta as joint adaptation of concise non-CoT format compliance and unseen-tool invocation.
3. **Comparator Budget Classification (`LoRA` vs `AdaLoRA` `B2`):**
   - All LoRA arms (`aloe`, `b0`, `b1`, `borobia_*`) enforce exact analytical 2D-shape `±5%` parameter parity against `B0` (`r = 8` uniform all-linear).
   - `B2` (`AdaLoRA`, Zhang et al., 2023) is pre-classified as a **nominal rank-budget comparator** (`target_r = 8` unweighted singular-value budget). Because `AdaLoRA`'s unweighted `RankAllocator` retains more singular triplets in wide FFN matrices than narrow GQA projections, both its nominal `target_r = 8` budget and its actual post-pruning `adalora_final_active_params` (alongside `adalora_init_params`) are reported side-by-side.
4. **Execution Retry & Append-Only Attempt-Log Policy:**
   - Every sealed `final_test` and `zero_shot_test` invocation generates one invocation `attempt_id`, binds `seal_sha256`, and publishes remote-first append-only `LAUNCHED` (`seq = 1`) and terminal `COMPLETED` or `FAILED` (`seq = 2`) records under `final_test_attempts/<slug>/` before persisting the local mirror copy.
   - Infrastructure preemption or transient runtime retries are permitted **solely** when a prior `attempt_id` terminated prior to `COMPLETED`, using the **exact same** read-only `test_unsealed.json` (`--expected-seal-sha256`), frozen winner configuration, and seed. Launching an uncached run after any `COMPLETED` record exists is blocked pre-launch; if a process crashes after local receipt verification, restart recovery replays pending attempt files and re-uploads the verified receipt idempotently without re-training or re-evaluating on `splits["test"]`.
   - `verify_complete_63_final_test_gate` audits `final_test_attempts/` across all `63` canonical `run_id`s: every terminal record must link to a valid `LAUNCHED` `attempt_id` bound to the model's `seal_sha256`, any `run_id` with `launched > 1` is recorded in `retried_runs` for mandatory thesis disclosure, and any `run_id` with `completed != 1`, unclosed/dangling launches (`launched != completed + failed`), or a `LAUNCHED` event ordered after `COMPLETED` **fails the gate closed**.

---

## 3. Primary & Secondary Statistical Endpoints on `splits["test"]` (`M = 400` Items)
1. **Unit of Analysis (Anti-Pseudoreplication Rule):**
   - All paired tests operate on the **`M = 400` unique test items** (`sample_id` aligned). The `1,200` `item × seed` observations are **never** pooled as independent samples.
   - For each trained arm $A$ and item $i \in \{1,\dots,400\}$, let $\bar{y}_{A,i} = \frac{1}{3}\sum_{s \in \{42,1337,2026\}} \mathbb{I}[\text{EM}_{A,i,s} = 1] \in \{0, \frac{1}{3}, \frac{2}{3}, 1\}$.
   - For deterministic greedy `zero_shot`, the single binary outcome $y_{\text{ZS},i} \in \{0,1\}$ is subtracted directly from $\bar{y}_{\text{ALOE},i}$ ($d_i = \bar{y}_{\text{ALOE},i} - y_{\text{ZS},i}$), never treated as three independent runs.
2. **Paired Bootstrap Confidence Intervals & Two-Sided p-Values:**
   - Resample the `400` paired differences $d_i = \bar{y}_{\text{ALOE},i} - \bar{y}_{B,i}$ with replacement for locked **`B = 10,000`** iterations (**`rng_seed = 42`**, `alpha = 0.05`). Official test evaluation rejects any override of `B` or `rng_seed`.
   - **Primary CI:** Bias-corrected and accelerated (**`95%` BCa CI**; official evaluation fails closed if `ci_method != "bca"`, never substituting a silent percentile fallback). **Sensitivity CI:** `95%` percentile CI.
3. **Discordant-Pair Binary Test:**
   - **Primary McNemar:** Exact two-sided binomial McNemar test on the **3-seed per-item majority vote** ($\hat{y}_{A,i} = \mathbb{I}[\bar{y}_{A,i} > 0.5]$, tie-free since $K=3$ is odd).
   - **Secondary Diagnostic:** Per-seed (`42`, `1337`, `2026`) discordant pair counts $(n_{11}, n_{10}, n_{01}, n_{00})$.
4. **Multiplicity Control (ADR-018):**
   - **Primary Hypothesis Family (per model, `m = 3`):** `ALOE vs B0` (Uniform All-Linear), `ALOE vs B1` (Budget-Matched Typed Random), and `ALOE vs B2` (AdaLoRA SVD), adjusted via step-down **Holm-Bonferroni** at family-wise $\alpha = 0.05$.
   - **Secondary / Exploratory Comparisons:** `ALOE vs {borobia_attn, borobia_ssm, borobia_ffn, zero_shot}` reported with unadjusted `95%` BCa CIs and exact McNemar p-values, explicitly labeled secondary.
5. **Exact Composite Multi-Objective & Non-Inferiority Decision Rule (`composite_claim`):**
   - Evaluated strictly on the primary comparator family $c \in \{\text{B0}, \text{B1}, \text{B2}\}$ ($m = 3$ per model) using unrounded full-precision statistics, where resource metrics are compared strictly within each model's locked hardware tier (`process_rss_sampled_100ms` on MPS; `cuda_max_memory_allocated` on CUDA), ties (`==`) do **not** count as resource gains, and exact physical parameter/memory deltas are always reported alongside `composite_claim` (note: step-down Holm-Bonferroni controls family-wise $\alpha = 0.05$ for superiority, while each `-2.0 pp` non-inferiority check uses its individual two-sided `95%` BCa lower bound, i.e., one-sided $\alpha = 0.025$):
     1. **`SUPERIOR_EM`:** Holm-adjusted two-sided paired bootstrap $p_{\text{Holm}} \le 0.05$ **and** `95%` BCa $\text{CI}_{\text{low}} > 0.0$.
     2. **`NON_INFERIOR_EFFICIENCY_WIN`:** Not `SUPERIOR_EM`, `95%` BCa $\text{CI}_{\text{low}} \ge -0.02$ (`-2.0 pp`), **and** at least one strict resource reduction holds:
        - **Strict Parameter Reduction:** `nominal_trainable_params(ALOE) < nominal_trainable_params(c)` (and for `c = B2`, additionally `nominal_trainable_params(ALOE) < adalora_mean_retained_active_params(B2)`), **OR**
        - **Strict Peak-Memory Reduction:** 3-seed mean `search_memory_mb(ALOE) < search_memory_mb(c)`.
     3. **`INCONCLUSIVE_OR_INFERIOR`:** Any primary comparison failing both (1) and (2) (either `95%` BCa $\text{CI}_{\text{low}} < -0.02$ or non-inferior EM with zero strict parameter or memory reduction).

---

## 4. Frozen Search-Phase Hypervolume Normalization Rule
- Search-phase 3D Pareto hypervolume (`exact_match`, `-trainable_parameters`, `-search_memory_mb`, `nadir_pad = 0.15`) is frozen strictly from validation search ledgers prior to unsealing:
  1. **Canonical All-Arm Search (`seed = 42`):** Shared raw bounds across all completed arms of that model (`6` arms on `Gemma-4`, `7` arms on `Falcon-H1` and `LFM3`), accompanied by physical-unit Pareto front tables.
  2. **Symmetric 3-Seed Paired Search (`seeds = [42, 1337, 2026]` on `Falcon-H1` & `LFM3`):** Evaluated under **one invariant per-model coordinate box** across all 3 seeds (`paired_aloe_b1_union_3seed` primary, `all_arms_union_3seed` sensitivity) rather than per-directory bounds. Test-split outcomes never enter HV normalization.

---

## 5. Frozen Execution & Statistical Code Digest Binding
Each pre-committed `final_sealed_test/<slug>/test_unsealed.json` binds and enforces the exact SHA-256 digests of `scripts/run_colab_shard.py` (`colab_bootstrap_sha256`), `scripts/run_local_mps_final.py` (`mps_bootstrap_sha256`), their combined digest (`bootstrap_sha256`), the runtime worker `sitecustomize.py` (`shim_sha256`), `docs/PRE_UNSEAL_ANALYSIS_PLAN.md` (`analysis_plan_sha256`), `scripts/final_test_audit.py` (`audit_script_sha256`), `scripts/evaluate_final_test_suite.py` (`evaluator_script_sha256`), and `aloe/evaluation/bootstrap.py` (`bootstrap_engine_sha256`), alongside mandatory `--expected-seal-sha256` verification.

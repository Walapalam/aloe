# Model Acquisition & Capability Testing — Engineering Specification

Six verification phases executed in sequential order. Each phase enforces a strict acceptance gate.

**What this actually is.** It looks like environment setup. It is not. Phases 1, 3 and 4 answer the three core architectural questions that decide whether ALOE is buildable as designed:

1. Can this machine set a *different rank and alpha per module group*? (Phase 1)
2. Does the architecture match what the design assumes? (Phase 3)
3. Do adapters on the interesting modules *actually receive gradients*? (Phase 4)

If any answer is no, you need to know in week one, not week four.

---

# ⚠️ Before you start: two things I cannot confirm for you

**Model IDs.** I have not verified the exact HuggingFace repository names. Do not guess them — a hallucinated model ID produces a confusing 404 rather than an obvious error. Search HuggingFace yourself, confirm the four IDs, and verify against the specification below. Approximate shapes to look for:

| Role | Look for | Notes |
|---|---|---|
| Model A | Gemma 4 E2B instruction-tuned | **Gated** — you must accept the licence on HuggingFace and authenticate with a token |
| Model B | Falcon-H1 1.5B Instruct | Parallel hybrid, the structural opposite of A |
| Model C (Oct) | Qwen3.5 0.8B | Sequential hybrid |
| Model D (Oct) | Qwen3 1.7B | Uniform — the control |

**Disk.** Four models at bf16 is roughly 12–16 GB, plus a duplicate if you also pull 4-bit MLX conversions. Check you have 60 GB free before starting.

---

## System Execution Environment & Architectural Context

```text
PROJECT: ALOE — a framework that inspects a language model's structure,
automatically derives a LoRA search space grouped by architectural role, and
searches it under multiple objectives for agentic tool-calling performance.
BSc Computer Science Final Year Project. The repository is at ~/Development/fyp/aloe.

HARDWARE: Apple M1 Max, 64 GB unified memory, macOS. Apple Silicon — NOT NVIDIA.

CRITICAL HARDWARE FACTS:
- bitsandbytes does NOT work here. Its kernels are compiled CUDA. The MPS backend
  exists only as an unmerged pull request. Do not try to install or import it.
- Unsloth is CUDA-only. Do not use it.
- torch.cuda.* does not exist. Use torch.backends.mps and torch.mps.*.
- torch.mps gives CURRENT allocated memory, not peak. To report a peak you must
  sample repeatedly during execution and keep the maximum. A single reading taken
  at the end is not a peak and must never be labelled as one.
- 64 GB unified memory means quantisation may be unnecessary for a 2B model.
  Measure bf16 before assuming 4-bit is required.

MODEL IDS (verified by the student, use exactly these):
  MODEL_A = "<paste verified Gemma 4 E2B id>"
  MODEL_B = "<paste verified Falcon-H1 1.5B id>"

RULES THAT OVERRIDE CONVENIENCE:
1. NEVER invent a number, a model ID, a config field name or a benchmark result.
   If something cannot be determined, write "NOT DETERMINED" and say why. This
   project has already been damaged by fabricated figures and will not tolerate more.
2. FAIL LOUD. No bare except. No silent defaults. If a config field is missing,
   raise an exception naming the field rather than guessing a value.
3. Every claim in a report must be traceable to a command whose output you can show.
4. Write results as machine-readable JSON first, human-readable Markdown second,
   generated FROM the JSON. Never hand-write the Markdown.
5. Do not modify anything under aloe/architecture/ or aloe/search_space/ without
   saying so explicitly — those are the research core and are covered by ADR-002.
```

---

## Phase 1 — Environment and backend capability probe

**This is tracker task I092 and it is the highest-priority task in the project.** Nothing downstream is safe until it is answered.

```text
[PASTE MASTER CONTEXT]

TASK: Determine what this machine can actually do, and answer one question that
decides the project's architecture.

Create scripts/probe_environment.py and scripts/probe_backends.py.

=== PART A: environment ===
Record and print:
- Python version, platform, processor, macOS version
- Total and available unified memory (psutil)
- Free disk space
- torch version; torch.backends.mps.is_available(); torch.backends.mps.is_built()
- Whether mlx and mlx_lm import, and their versions
- Whether peft imports, and its version
- Confirm bitsandbytes is ABSENT or non-functional. Do not try to make it work.
  Record the failure mode verbatim.

=== PART B: THE DECIDING QUESTION ===
ALOE's entire design assumes you can assign a DIFFERENT LoRA rank and alpha to
DIFFERENT groups of modules in the same model. For example: rank 8 on local
attention, rank 16 on global attention, rank 4 on the feed-forward blocks, all in
one training run.

Test whether each available backend supports this. Use the smallest model you can
find — a 0.5B model is fine, this is a capability test not a performance test.

B1. peft on MPS:
    Build a LoraConfig using rank_pattern and alpha_pattern (these take regex-keyed
    dictionaries mapping module patterns to per-module rank and alpha). Apply it to
    a model moved to device "mps". Then enumerate every adapter that was created and
    print its actual rank and alpha.
    PASS = at least two adapter groups exist with genuinely different rank values,
    confirmed by inspecting the created modules, not by reading back the config.

B2. mlx-lm:
    Investigate whether mlx-lm's LoRA implementation can target modules by name with
    per-module rank. Read its actual source or API — do not rely on blog posts.
    Report precisely what it supports: per-model rank only, per-layer-range rank, or
    true per-module rank.

=== OUTPUT ===
Write runs/probe/environment.json and runs/probe/backends.json, then generate
runs/probe/REPORT.md from them.

The report must answer this question in one unambiguous sentence at the top:

  "Per-group rank and alpha assignment IS / IS NOT possible on this machine,
   via <backend>, evidenced by <file>:<line of output>."

=== GATE ===
If NEITHER backend supports per-group rank and alpha, STOP. Do not proceed to
Phase 2. Report the finding and state the three options: (a) restrict ALOE to
per-group enable/disable with uniform rank, (b) implement a custom LoRA injection
layer in pure PyTorch, (c) rent a CUDA machine for typed runs. Do not choose —
that is the student's decision with the supervisor.
```

**What I expect to happen, written down so the agent can prove me wrong:** B1 passes. `peft`'s `rank_pattern`/`alpha_pattern` are pure PyTorch and should work on MPS. If so, `peft`-on-MPS in bf16 is your primary backend and MLX becomes an optional speed optimisation. That is the lower-risk path to Gate B.

---

## Phase 2 — Model acquisition

```text
[PASTE MASTER CONTEXT]

TASK: Download and verify the two pilot models. Record everything needed to prove
later which exact weights were used.

Create scripts/download_models.py and aloe/datasets/model_registry.py.

For MODEL_A and MODEL_B:
1. Download via huggingface_hub.snapshot_download. Gemma is gated — if auth fails,
   report the exact error and the URL where the licence must be accepted. Do not
   attempt a workaround.
2. Pin the exact revision. Record the commit SHA, not just the branch. A model ID
   without a revision is not reproducible.
3. Record: total size on disk, file list with sizes, config.json in full,
   the declared licence, and the download timestamp.
4. Load config with transformers.AutoConfig and record every field. Do not filter
   to fields you think matter — record all of them. You do not yet know which
   fields the architecture adapters will need.
5. Verify the tokenizer loads and that a chat template exists. Record the template
   verbatim. If there is no chat template, record NOT PRESENT — this matters for
   tool-call formatting later.

=== OUTPUT ===
runs/models/<model_slug>/acquisition.json per model, containing everything above.
runs/models/REGISTRY.md — a table of model, revision SHA, size, licence, gated
yes/no, chat template present yes/no.

=== GATE ===
Both models must load their config and tokenizer. If either fails, stop and report.
Do NOT substitute a different model to keep going.
```

---

## Phase 3 — Architecture verification

This is where the test harness becomes a research instrument. You are checking whether the model matches what your design documents claim.

```text
[PASTE MASTER CONTEXT]

TASK: Verify that each model's actual structure matches what ALOE's design assumes.
Several design documents make claims about these architectures. Your job is to
confirm or refute them from the loaded model, not to assume they are correct.

Create scripts/verify_architecture.py using aloe.architecture.inspector.

For each model, load with device_map="meta" where possible so you do not need to
fit the weights in memory, then:

1. Run inspect_model() and record: total modules, modules with parameters, number
   of distinct layer indices, total parameter count.

2. CLAIMS TO TEST — report each as CONFIRMED, REFUTED, or NOT DETERMINED, with the
   evidence. Do not report CONFIRMED without showing the value you read.

   MODEL A (Gemma 4 E2B):
   a. "35 decoder layers"
   b. "Interleaved local and global attention in a repeating pattern of 4 local
       to 1 global, giving 28 local and 7 global"
      → Find the config field that encodes this. It may be layer_types,
        sliding_window_pattern, or something else entirely. NAME THE FIELD YOU
        USED. If no such field exists, report NOT DETERMINED — do not infer the
        pattern from a modulo of the layer index.
   c. "KV-cache sharing means k_proj and v_proj do not exist on every layer"
      → Count k_proj and v_proj occurrences per layer index. Report the actual
        distribution. If they appear on all 35 layers, this claim is REFUTED and
        a design document needs correcting.

   MODEL B (Falcon-H1):
   d. "Parallel hybrid: each block contains an attention branch and a Mamba-2 SSM
       branch whose outputs are summed before the MLP"
   e. "conv1d and out_proj modules exist in the SSM path"
      → List every module whose name contains conv1d or out_proj, with shapes.

3. Produce a per-layer module map: for each layer index, which module types are
   present. Save as a table.

=== OUTPUT ===
runs/architecture/<model_slug>/inspection.json — the full ArchitectureDescription
runs/architecture/<model_slug>/claims.json — each claim with verdict and evidence
runs/architecture/ARCHITECTURE_REPORT.md generated from those

=== GATE ===
If claim (b) is NOT DETERMINED — if you cannot find a config field that states the
local/global pattern — STOP and report. The Gemma adapter cannot be written
correctly without it, and guessing would silently corrupt every downstream search
group with no test able to catch it.
```

**Why the gate is there.** If the pattern is wrong, `aloe inspect` still runs, the grammar still generates, training still completes, and every result is quietly meaningless. There is no error to catch. That class of failure is exactly what Gap 3 exists to prevent, so the project should not tolerate it in its own tooling.

---

## Phase 4 — Adapter attachment and gradient verification

**The research-critical one.** This is ADR-005 and tracker task I057.

```text
[PASTE MASTER CONTEXT]

TASK: Prove that adapters attached to each module type actually receive gradients.

BACKGROUND YOU NEED: Borobia et al. (arXiv:2604.22127) Section 3.2 built a pipeline
that verifies "all expected modules received adapters and no forbidden modules were
modified". That checks whether an adapter ATTACHED. It does not check whether the
adapted module can LEARN. They still had to exclude Falcon-H1's conv1d and out_proj
by hand, after training instability, because those weights feed a custom Mamba
kernel that bypasses the LoRA forward pass. An adapter there attaches perfectly and
never receives a gradient.

ALOE's contribution is catching that automatically. This script is that check.

Create scripts/verify_gradients.py and aloe/execution/backends/<chosen>.py.

Procedure, per model, per component type:
1. Attach a LoRA adapter to the modules of ONE component type only.
2. Run a single forward pass on one short batch, then backward().
3. For every adapter parameter, record: does .grad exist, and what is its L2 norm?
4. Classify:
     GRADIENT_OK       grad exists and norm > 1e-8
     GRADIENT_ZERO     grad exists but norm is effectively zero
     GRADIENT_MISSING  grad is None — the module is not in the autograd path
5. Report per component type, with counts.

MANDATORY NEGATIVE CONTROL: deliberately attach an adapter to a module you expect
to be inert — Falcon-H1's conv1d is the documented case — and confirm the check
returns GRADIENT_MISSING or GRADIENT_ZERO for it.

A check that has never failed is not evidence that it works. If the negative
control comes back GRADIENT_OK, either the exclusion documented in the literature
does not apply to this checkpoint, or your check is broken. Investigate which, and
report. Do not assume the literature is right about this specific checkpoint.

=== OUTPUT ===
runs/gradients/<model_slug>/gradient_report.json
runs/gradients/GRADIENT_REPORT.md

The Markdown must contain a table: component type | modules tested | GRADIENT_OK |
GRADIENT_ZERO | GRADIENT_MISSING | verdict.

And one headline sentence:
  "The negative control DID / DID NOT correctly detect an inert adapter target."

=== GATE ===
If the negative control does not fail as expected, stop and report. Everything in
ALOE's validity layer rests on this check being able to detect a non-learning target.
```

---

## Phase 5 — Smoke training and memory profiling

```text
[PASTE MASTER CONTEXT]

TASK: Train each model for a handful of steps and measure what it actually costs.
This is not about quality. It is about whether training completes and what it uses.

Create scripts/smoke_train.py and aloe/evaluation/resources.py.

Configurations to test, per model:
  C1. bf16, no quantisation, uniform LoRA rank 8 alpha 16 on attention only
  C2. bf16, no quantisation, uniform LoRA rank 16 alpha 32 on all valid targets
  C3. 4-bit if the backend supports it, otherwise record NOT SUPPORTED and skip

For each: 20 training steps on a tiny fixed subset. Fixed seed. Log loss per step.

MEMORY MEASUREMENT — do this correctly:
torch.mps returns CURRENT allocated memory, not a peak. Start a background sampler
thread that reads torch.mps.current_allocated_memory() every 100ms throughout
training and keeps the maximum. Report that maximum as peak_memory_gb, and record
sample_count and sample_interval_ms alongside it so the method is auditable.
Do NOT take one reading at the end and call it a peak.

Also record: wall-clock duration, tokens per second, trainable parameter count,
adapter file size on disk after saving.

Then: save the adapter, start a FRESH process, reload it, and run one inference.
Reloading in the same process does not prove independence from training state.

=== OUTPUT ===
runs/smoke/<model_slug>/<config_id>/trial.json
runs/smoke/SMOKE_REPORT.md

=== WHAT THE REPORT MUST SHOW ===
Per model and config: did training complete, did loss decrease over 20 steps, peak
memory in GB, duration, tokens/sec, trainable params, adapter size in MB, did the
adapter reload in a fresh process.

State explicitly whether bf16 without quantisation fits comfortably. If C1 and C2
both complete well under 64 GB, record that finding prominently — it means
quantisation is optional for this project, which removes an entire dependency.

=== GATE ===
At least one configuration per model must complete and reload. If none do, stop.
```

---

## Phase 6 — Tool-call output sanity check

```text
[PASTE MASTER CONTEXT]

TASK: Confirm each model can emit parseable tool calls before any fine-tuning.
This establishes the baseline the project measures against.

Create scripts/probe_tool_calls.py.

Build 10 hand-written prompts covering:
  - 3 requiring one obvious tool call
  - 2 where several tools are plausible
  - 2 where a required argument is missing from the user's request
  - 3 where NO tool should be called

Use each model's actual chat template and tool-schema format — record what format
you used. Run all 10 through the base model with greedy decoding, fixed seed.

For each response, record: raw output verbatim, whether it parses as valid JSON or
the model's tool-call format, the tool name if any, the arguments if any.

Do NOT score correctness yet. The evaluator comes later. Record what the model
produced.

=== OUTPUT ===
runs/toolcalls/<model_slug>/baseline_probe.json — all 10, raw outputs included
runs/toolcalls/TOOLCALL_REPORT.md

Report: how many of 10 produced parseable output, how many called a tool at all,
how many correctly abstained on the 3 no-tool cases.

=== NOTE ===
Low scores here are expected and fine. A 2B base model is not good at this. The
point is to establish a starting position and confirm the plumbing works.
```

---

# Results schema

Every script writes JSON with this envelope, so trials stay comparable and auditable:

```json
{
  "schema_version": "1.0",
  "run_id": "uuid",
  "timestamp_utc": "2026-09-17T14:22:03Z",
  "script": "smoke_train.py",
  "git_commit": "abc1234",
  "environment": {
    "platform": "macOS-26.x-arm64",
    "chip": "Apple M1 Max",
    "unified_memory_gb": 64,
    "python": "3.11.x",
    "torch": "2.x.x",
    "backend": "torch_mps",
    "library_versions": {"peft": "0.x", "transformers": "4.x"}
  },
  "model": {
    "id": "<exact hf id>",
    "revision": "<commit sha>",
    "precision": "bf16",
    "quantisation": "none"
  },
  "config": { "...": "the exact configuration tested" },
  "measurements": {
    "peak_memory_gb": 11.3,
    "memory_metric": "unified",
    "memory_method": "sampled",
    "sample_interval_ms": 100,
    "sample_count": 4120,
    "duration_seconds": 412.7,
    "tokens_per_second": 248.1
  },
  "results": { "...": "script-specific" },
  "status": "PASS",
  "failures": [],
  "not_determined": []
}
```

**Three fields do real work.** `memory_metric` prevents a Mac number ever being silently compared with a VRAM number. `memory_method` records that the peak was sampled rather than read once. `not_determined` gives the agent a place to put honest gaps instead of filling them with plausible values.

---

# What a passing run looks like

| Prompt | Pass | Investigate |
|---|---|---|
| 1 Environment | MPS available; per-group rank/alpha confirmed on at least one backend | Neither backend supports per-group → **stop, escalate to supervisor** |
| 2 Acquisition | Both models downloaded with pinned revision SHAs; chat templates recorded | Gemma gated → accept licence, authenticate |
| 3 Architecture | Local/global pattern read from a **named** config field | Field not found → **stop**, the adapter cannot be written |
| 4 Gradients | All valid targets GRADIENT_OK; **negative control correctly fails** | Control passes → check is broken or the checkpoint differs from the literature |
| 5 Smoke train | ≥1 config completes, loss decreases, adapter reloads in a fresh process | Nothing completes → stop |
| 6 Tool calls | ≥7 of 10 produce parseable output | 0 parseable → chat template or schema format is wrong |

## Three findings to watch for

**bf16 fits comfortably.** If Phase 5 shows C1 and C2 well under 64 GB, quantisation becomes optional. That removes bitsandbytes, NF4, paged optimizers and LoftQ from the critical path in one stroke — a large simplification, and worth telling your supervisor.

**k_proj/v_proj appear on all 35 Gemma layers.** Then the KV-sharing claim in your design documents and Tab 5 is wrong and needs correcting. Better found by your own harness than by an examiner.

**The negative control passes when it should fail.** Either the Falcon exclusion does not apply to your checkpoint, or the gradient check is broken. Both are important; the second is urgent.

---

# Order and time

| | Prompt | Time | Blocking? |
|---|---|---|---|
| 1 | Environment + capability probe | 3 h | **Yes — everything** |
| 2 | Model acquisition | 1 h + download | Yes |
| 3 | Architecture verification | 2 h | Yes — blocks the adapters |
| 4 | Gradient verification | 3 h | Yes — blocks the validity layer |
| 5 | Smoke train + profiling | 3 h | No |
| 6 | Tool-call probe | 2 h | No |

Two days if nothing surprises you. **Run 1 to completion before starting 2.** The point of a gate is that you stop at it.

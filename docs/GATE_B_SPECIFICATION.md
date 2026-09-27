# Technical Specification — Gate A Data Split through Gate C Pilot Execution

Four implementation phases. **Execute Phase 7 (Data Split) first** — it is the only thing standing between you and Gate A, and everything after it is contaminated if it is wrong.

Key architectural decisions must be aligned prior to pilot completion.

---

## Updated Architectural Context & Hardware Baseline

This specification summarizes the verified hardware and model baselines established in Stage 1.

```text
PROJECT: ALOE — a framework that inspects a language model's structure,
automatically derives a LoRA search space grouped by architectural role, and
searches it under multiple objectives for agentic tool-calling performance.
BSc Computer Science Final Year Project. Repository: ~/Development/fyp/aloe

HARDWARE: Apple M1 Max, 64 GB unified memory, macOS. Apple Silicon, NOT NVIDIA.

ESTABLISHED FACTS — do not re-derive, do not contradict without evidence:
- Backend is peft on MPS (ADR-010). It supports per-group rank_pattern and
  alpha_pattern. MLX supports uniform rank only and is reserved for baselines.
- bitsandbytes and Unsloth do not work here. Do not import them. 4-bit is
  NOT_SUPPORTED on this machine and bf16 fits comfortably (ADR-011).
- torch.cuda.* does not exist. torch.mps gives CURRENT memory, not peak. Peaks
  must be sampled on a background thread at 100ms and the maximum kept.

MODELS — acquired, pinned, verified:
  google/gemma-4-E2B-it        SHA 3e22461f65e89153144f8adb70e3b8c2cc9845a7
      35 layers, 28 sliding (512-window) + 7 global, from text_config.layer_types
      k_proj/v_proj on layers 0-14 only; text_config.num_kv_shared_layers = 20
      Apache 2.0. bf16 9.57 GB. 713 ms/LoRA step, 24.8 tok/s decode.
  google/gemma-4-E4B-it        42 layers, 35 sliding + 7 global (5:1 ratio)
      k_proj/v_proj on layers 0-23 only. Apache 2.0. bf16 14.89 GB.
      1372 ms/LoRA step, 16.6 tok/s decode.
  tiiuae/Falcon-H1-1.5B-Instruct  SHA 80ebc50d7799a440b96c93bb6686a3924a09b0cb
      24 layers, parallel hybrid: attention + Mamba-2 summed before MLP
      24 conv1d (64,1,4) and 24 out_proj (2048,2048) — conv1d is INERT
      6918 ms/LoRA step. SSM chunk scans fall back to Python on MPS.

VERIFIED GRADIENT BEHAVIOUR (Gate 4, runs/gradients/):
  541 adapter parameters across attention, FFN and SSM-linear: all GRADIENT_OK.
  48 adapter parameters on Falcon-H1 conv1d: all GRADIENT_MISSING.
  The negative control works. Do not weaken or bypass verify_gradients().

RULES THAT OVERRIDE CONVENIENCE:
1. NEVER invent a number, an ID, a config field or a result. Write "NOT DETERMINED"
   and say why. This project has been damaged by fabricated figures before.
2. FAIL LOUD. No bare except. No silent defaults. Missing config field -> raise,
   naming the field.
3. Every claim in a report traces to a command whose output you can show.
4. JSON first, Markdown generated FROM the JSON. Never hand-write the Markdown.
5. Do not modify aloe/architecture/ or aloe/search_space/ without saying so
   explicitly. Those are the research core, covered by ADR-002, and there is a
   test that fails if a training framework is imported into them.
```

---

## Phase 7 — xLAM dataset and the tool-disjoint split

**Run this first.** Gate A is not met without it, and every evaluation number produced before it exists is unusable.

```text
[PASTE UPDATED MASTER CONTEXT]

TASK: Build the dataset foundation. Gate A requires LoRA training on an xLAM pilot
split, and that split must be TOOL-DISJOINT.

WHY TOOL-DISJOINT MATTERS: if an API appears in both training and validation, the
model can score well by memorising that specific tool's signature rather than by
learning to call tools. The measured improvement would then be contamination, not
capability, and every result built on it would be invalid. This is not a nicety.

Create aloe/datasets/xlam.py and scripts/build_xlam_split.py.

=== PART A: acquire and understand ===
1. Acquire the xLAM function-calling dataset (Salesforce/xlam-function-calling-60k).
   It is GATED — if authentication fails, report the exact error and the URL where
   access must be requested. Do not substitute another dataset.
2. Pin the exact revision SHA. Record licence, size, and record count.
3. Map the schema completely: where the user query lives, where the available tool
   list lives, where the expected tool call lives, argument structure, and any
   category or verification metadata. Print one full example verbatim.

=== PART B: the split ===
4. Extract the set of unique tool/API names across the whole dataset. Report the
   count and the distribution of how many examples use each tool.
5. Partition the TOOL NAMES (not the examples) into train and validation sets.
   Then assign examples to splits by which tools they use. An example that uses
   tools from both partitions goes to NEITHER — discard it and count how many.
6. Target a small pilot: roughly 800 training examples and 200 validation examples.
   Exact numbers matter less than disjointness.
7. Stratify where the data allows: single obvious call, multiple plausible tools,
   missing required argument, no tool needed. Report the actual distribution you
   achieved — do not claim stratification you did not get.

=== PART C: prove it ===
8. Write a manifest containing: dataset revision SHA, the exact example IDs in each
   split, the tool-name partition, a SHA-256 of the sorted ID list per split, and
   the discard count.
9. Write a test, tests/test_xlam_split.py, that loads the manifest and asserts the
   intersection of train tools and validation tools is EMPTY. This test must run in
   CI and must fail loudly if the split is ever regenerated incorrectly.

=== OUTPUT ===
data/manifests/xlam_pilot_v1.json
runs/dataset/XLAM_SPLIT_REPORT.md generated from the manifest
tests/test_xlam_split.py

=== GATE ===
The disjointness test must pass. If tool names cannot be reliably extracted from
the schema, STOP and report what the schema actually contains — do not approximate
disjointness with a heuristic like string similarity.
```

---

## Phase 8 — Gate B: the grammar differentiation proof

This is the milestone that matters. It is the moment ALOE exists as a concept rather than a plan.

```text
[PASTE UPDATED MASTER CONTEXT]

TASK: Demonstrate that ALOE derives DIFFERENT search grammars for structurally
different models, with no code change between runs.

This is Gate B and it is the project's central claim. The demonstration must be
reproducible by someone else from the repository alone.

=== PART A: make `aloe inspect` work end to end ===
1. Confirm aloe/pipeline.py implements inspect_to_grammar(model_id, backend).
2. Confirm the CLI writes the grammar JSON and prints a readable summary.
3. Run it on all THREE models:
     aloe inspect google/gemma-4-E2B-it   --out runs/grammars/gemma_e2b.json
     aloe inspect google/gemma-4-E4B-it   --out runs/grammars/gemma_e4b.json
     aloe inspect tiiuae/Falcon-H1-1.5B-Instruct --out runs/grammars/falcon_h1.json

   E4B costs nothing extra — inspection runs on meta tensors, no weights loaded.
   Including it tests something E2B alone cannot: whether the derived grammar
   responds to a change in the LOCAL:GLOBAL RATIO (E2B is 4:1, E4B is 5:1) and not
   merely to a change in model family.

=== PART B: prove the grammars differ, and differ correctly ===
4. Create scripts/prove_differentiation.py producing a comparison table:
   - component types present in each grammar
   - number of target modules per group
   - number of excluded modules per group, with reasons
   - total grammar cardinality
5. Assert and report:
   a. Gemma grammars contain LOCAL_ATTENTION and GLOBAL_ATTENTION; Falcon does not.
   b. Falcon's grammar contains SSM; neither Gemma does.
   c. Falcon's excluded set contains 24 conv1d modules with a reason naming the
      custom Mamba kernel.
   d. Gemma E2B's attention target count reflects k_proj/v_proj existing on layers
      0-14 only — NOT all 35 layers.
   e. E2B and E4B produce different local:global target ratios.

6. CRITICAL — prove no code changed between runs. Record the git commit SHA before
   the first inspect and after the last. They must be identical. State this in the
   report. If any file was modified mid-sequence, the demonstration is void and
   must be re-run.

=== OUTPUT ===
runs/grammars/*.json
runs/grammars/DIFFERENTIATION_REPORT.md
tests/test_differentiation.py — asserts (a) through (e) as automated tests

=== GATE ===
All five assertions pass AND the git SHA is unchanged across the three runs.
If an assertion fails, report which and why. Do NOT adjust the assertion to match
the output — investigate whether the grammar or the assertion is wrong.
```

**Why the git SHA check matters.** The claim is "same code, different models, different grammars". Without a recorded commit either side, that claim is unverifiable — and it is exactly the claim a sceptical examiner will probe. Make it mechanical.

---

## Phase 9 — TrialConfig, executor and trial store

```text
[PASTE UPDATED MASTER CONTEXT]

TASK: Make one grammar configuration executable end to end and reproducibly stored.

Create aloe/execution/trial_config.py, complete aloe/execution/backends/peft_mps.py,
and create aloe/tracking/trial_store.py.

=== PART A: TrialConfig ===
A pydantic model describing ONE point in the search space:
  model_id, model_revision, dataset_manifest_path, seed, backend, precision,
  and per-group settings: {component_type: {enabled, rank, alpha}}

It must be able to round-trip to JSON and back with no loss. Add a method that
converts it into the peft LoraConfig arguments — specifically the rank_pattern and
alpha_pattern dictionaries plus the resolved target_modules list.

=== PART B: executor ===
run_trial(config) must:
  1. Call verify_gradients() BEFORE training. If any enabled group returns
     GRADIENT_MISSING, abort the trial and record it as INVALID_CONFIG with the
     offending modules named. Do not train a configuration that cannot learn.
  2. Train on the xLAM pilot split from Phase 7.
  3. Sample peak memory on a background thread at 100ms throughout.
  4. Save the adapter, then reload it in a FRESH SUBPROCESS and run one inference.
  5. Return measured cost: peak memory, wall-clock, trainable params, adapter size.

=== PART C: trial store ===
Every trial writes a JSON record containing the full envelope from the results
schema, plus: the TrialConfig verbatim, the dataset manifest SHA, the git commit,
all library versions, and the hardware string.

Add a replay function: given a stored trial ID, reconstruct the TrialConfig and
re-run it. Assert the result matches within tolerance. A trial you cannot replay is
not a result.

=== OUTPUT ===
runs/trials/<trial_id>/trial.json
tests/test_trial_roundtrip.py — TrialConfig serialises and deserialises losslessly
tests/test_invalid_config_aborts.py — a config targeting Falcon conv1d is rejected
                                      BEFORE training, not after

=== GATE ===
One trial completes, stores, and replays to the same result. And a deliberately
invalid configuration is rejected by verify_gradients() rather than wasting a
training run.
```

The second test is the one worth caring about. It proves the validity layer **saves compute**, which is a concrete, measurable benefit you can quantify: *"N configurations in the search space would have trained to no effect; ALOE rejected them in one backward pass each."*

---

## Phase 10 — Pilot runs and the first Pareto data

```text
[PASTE UPDATED MASTER CONTEXT]

TASK: Run the September pilot. The question is whether the pipeline works, NOT
whether ALOE wins. Draw no scientific conclusions from this.

Create scripts/run_pilot.py.

=== CONFIGURATIONS ===
For google/gemma-4-E2B-it and tiiuae/Falcon-H1-1.5B-Instruct:
  P1. Global baseline — uniform rank 8, alpha 16, all valid targets
  P2. Typed A — attention groups rank 16 alpha 32, FFN rank 4 alpha 8
  P3. Typed B — attention groups rank 4 alpha 64, FFN rank 16 alpha 32
      (P3 deliberately uses alpha far from the 2r heuristic — that is Gap 2)

Fixed seed. Identical training budget across all configurations. Same dataset split.

=== FALCON BUDGET ===
Falcon-H1 costs 6918 ms per step against Gemma's 713 ms — roughly 10x. Before
running, measure how many steps fit in a 45-minute per-trial budget for each model
and report the two numbers. If Falcon cannot complete a meaningful run in that
budget, reduce its step count and RECORD THE ASYMMETRY EXPLICITLY in the report.
Do not silently give the two models different budgets.

=== OUTPUT ===
runs/pilot/<model>/<config_id>/trial.json
runs/pilot/PILOT_REPORT.md containing, per model:
  - a table of config, tool-use score, peak memory, duration, adapter size
  - a 2D scatter of tool-use score against peak memory, with the non-dominated
    points marked
  - which configurations failed and why

=== WHAT THE REPORT MUST NOT SAY ===
Do not write that typed allocation outperforms the baseline. Six runs at one seed
cannot support that and claiming it would be the exact error the project's
statistical-validity theme exists to prevent. Report what was measured. State
explicitly that no significance claim is made.

=== GATE ===
At least P1 and one typed configuration complete for each model.
```

---

# Two decisions that need you, not an agent

## 1. Name the memory metric — do this before Phase 10 pilot runs

Gate 5 reported *"10.28 GB peak RSS, ~1.31 GB MPS driver allocation"*. Those differ by 8×, and a 9.51 GB model cannot occupy 1.31 GB of anything.

One of them is your Pareto objective. Decide which, add it to ADR-004, and record both in every trial with the objective named. Suggested wording:

> On Apple Silicon, `peak_memory_gb` is defined as peak process resident set size sampled at 100 ms intervals, because unified memory means model weights are counted in RSS rather than in a separate device allocator. `mps_driver_allocated_gb` is recorded alongside it for diagnostic purposes but is not the optimisation objective.

If you leave this ambiguous until you have Pareto fronts, you will have to regenerate them.

## 2. Decide the Falcon budget — before October, not during it

| Model | Per 500-step trial | 30 trials |
|---|---|---|
| Gemma 4 E2B | ~6 min | **~3 h** |
| Gemma 4 E4B | ~11 min | **~6 h** |
| Falcon-H1 | ~58 min | **~29 h** |

Gemma is comfortable. Falcon is a multi-day job before you add seeds.

Three options, in order of preference:

1. **Shorter trials for Falcon** — fewer steps, shorter sequences. Document the asymmetry.
2. **Asymmetric trial counts** — full search on Gemma, reduced on Falcon, enough to show the grammar differs and works.
3. **Rent a CUDA box for Falcon only** — but then metrics are cross-backend and ADR-004 forbids pooling them.

Option 1 or 2. Raise it with your supervisor as a scoping decision rather than discovering it as a scheduling failure.

---

# Order and time

| | Phase | Time | Blocking |
|---|---|---|---|
| 7 | xLAM tool-disjoint split | 4 h | **Yes — Gate A, and everything downstream** |
| — | Name the memory metric | 15 min | Yes, before Phase 10 pilot runs |
| 8 | Gate B differentiation proof | 3 h | **This is the milestone** |
| 9 | TrialConfig, executor, store | 5 h | Yes — Gate C |
| — | Decide Falcon budget | 30 min | Before Phase 10 |
| 10 | Pilot runs | 6 h + compute | No |

**Phase 8 is the key milestone.** Once `aloe inspect` produces three different grammars from one unchanged commit, the project's central claim is demonstrated and everything after it is refinement.

Phase 7 comes first because a contaminated split would invalidate downstream evaluation numbers.

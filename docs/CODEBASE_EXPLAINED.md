# The Codebase, Explained From Scratch

For when you need to defend this in a viva. No jargon without a definition, small numbers before big ones.

---

# 1. What problem does any of this solve?

You want to fine-tune a small language model so it gets better at calling tools. Full fine-tuning means changing all ~2 billion numbers in the model. Too slow, too much memory.

**LoRA** is the shortcut. Instead of changing a weight matrix `W`, you leave `W` frozen and add a small correction:

```
W_new = W + (B × A) × (alpha / rank)
```

`A` and `B` are two small matrices you train. `rank` controls how small. `alpha` controls how loud the correction is.

## The small version

Say one weight matrix is **4 × 4** — sixteen numbers.

```
W (frozen, 4×4 = 16 numbers)
```

With **rank = 1**, LoRA adds:

```
A = 1×4  (4 numbers)
B = 4×1  (4 numbers)
```

You train **8 numbers instead of 16**. `B × A` multiplies back out to a full 4×4 correction, but it only has 8 numbers of freedom behind it.

With **rank = 2** you'd train 16 numbers — no saving at this size. Rank only pays off when `W` is large.

**alpha** scales the result. If `alpha = 2` and `rank = 1`, the correction gets multiplied by `2/1 = 2`. Double the effect. The near-universal convention is `alpha = 2 × rank`, which makes the scale always 2. **Nobody checks whether 2 is right.** That is your Gap 2.

## The real version

Gemma 4 E2B has roughly **2 billion** parameters. One attention projection might be **2048 × 2048** — about 4.2 million numbers.

At `rank = 8`:
```
A = 8 × 2048    = 16,384 numbers
B = 2048 × 8    = 16,384 numbers
total             32,768  ≈ 0.8% of that matrix
```

Across the whole model you train roughly 1–2% of the parameters. That's the saving.

---

# 2. The question ALOE actually asks

A model isn't one matrix. Gemma 4 E2B has **35 layers**, and each layer has several matrices. Which ones get an adapter? What rank each? What alpha each?

Right now everyone either:
- puts the same rank on everything (standard LoRA), or
- computes an **importance score** per matrix and gives the important ones more rank (AdaLoRA, LAARA, IFCLoRA and five others).

**Nobody asks what *kind* of layer it is.**

Gemma 4 alternates between two sorts of attention: **local** (each word only looks at nearby words) and **global** (each word looks at everything). Those do different jobs. ALOE's bet is that they need different rank and alpha — and that you can work out which is which *by reading the model*, automatically, without a human writing the config.

That's it. That's the whole project.

---

# 3. The pipeline, with a toy model

Imagine a tiny model: **2 layers**, and each layer has 3 matrices.

```
layer 0:  q_proj   k_proj   mlp_up
layer 1:  q_proj   k_proj   mlp_up
```

Layer 0 uses local attention, layer 1 uses global. Watch what happens at each step.

### Step 1 — INVENTORY (`inspector.py`)

Walk the model and write down every part. No opinions yet, just a list.

```
name              class    layer   shape      params
layer.0.q_proj    Linear     0    (8, 8)         64
layer.0.k_proj    Linear     0    (8, 8)         64
layer.0.mlp_up    Linear     0    (8, 16)       128
layer.1.q_proj    Linear     1    (8, 8)         64
layer.1.k_proj    Linear     1    (8, 8)         64
layer.1.mlp_up    Linear     1    (8, 16)       128
```

Six rows. This step is identical for **every** model — walking a tree is walking a tree.

### Step 2 — CLASSIFY (`typing.py` + `adapters/`)

Now assign *meaning*. This is the part that's model-specific, so it lives in a per-family file.

```
layer.0.q_proj  →  LOCAL_ATTENTION
layer.0.k_proj  →  LOCAL_ATTENTION
layer.0.mlp_up  →  FFN
layer.1.q_proj  →  GLOBAL_ATTENTION
layer.1.k_proj  →  GLOBAL_ATTENTION
layer.1.mlp_up  →  FFN
```

The adapter reads the model's config to know layer 0 is local and layer 1 is global. **It must not guess.**

### Step 3 — VALIDATE (`constraints.py`)

Ask of each one: *can this actually learn if I attach an adapter?*

```
layer.0.q_proj  →  VALID      "Standard nn.Linear target."
layer.1.k_proj  →  VALID      "Standard nn.Linear target."
...
```

On a Falcon-H1 model you'd also see:

```
layer.0.conv1d  →  EXCLUDED   "Feeds a custom Mamba kernel that bypasses the
                               LoRA forward pass; adapter is silently inert."
```

**This step is your research contribution.** More on why in §5.

### Step 4 — EMIT GRAMMAR (`grammar.py`)

Group by type. Each group gets its own knobs.

```
LOCAL_ATTENTION:
  targets: [layer.0.q_proj, layer.0.k_proj]
  enabled: [yes, no]
  rank:    [4, 8, 16, 32]
  alpha:   [8, 16, 32, 64, 128]

GLOBAL_ATTENTION:
  targets: [layer.1.q_proj, layer.1.k_proj]
  enabled: [yes, no]
  rank:    [4, 8, 16, 32]
  alpha:   [8, 16, 32, 64, 128]

FFN:
  targets: [layer.0.mlp_up, layer.1.mlp_up]
  enabled: [yes, no]
  rank:    [4, 8, 16, 32]
  alpha:   [8, 16, 32, 64, 128]
```

When a group is enabled, there are 4 × 5 = 20 (rank × alpha) choices. When disabled, rank and alpha choices collapse into 1 inactive state. That gives **1 + (4 × 5) = 21 operational states per group**, yielding 21³ − 1 = **9,260 non-trivial combinations** (9,261 with all-off baseline).

Nobody typed that out. It was *derived* from the model. Point at a different model and you get a different grammar with no code change.

### Step 5 — EXECUTE (`execution/backends/`)

Pick one combination, train it, measure what it cost.

```
LOCAL_ATTENTION:  on,  rank 8,  alpha 32
GLOBAL_ATTENTION: on,  rank 16, alpha 64
FFN:              off
→ train → 4.2 GB peak memory, 11 minutes, 38 MB adapter, tool score 0.61
```

### Step 6 — SCORE & STORE (`evaluation/`, `tracking/`)

Write all of it down so you can reproduce it.

### At real scale

Gemma 4 E2B: **35 layers**, roughly **250 adaptable modules**, **3 groups**. Same 21 states per group, same 9,260 non-trivial combinations — because the grammar groups by *type*, not by module. That's the whole reason typing matters: it keeps the search space small enough to search.


Score by importance instead and you'd have 250 separate decisions. Unsearchable.

---

# 4. What each file does

```
aloe/
├── architecture/
│   ├── schema.py       The vocabulary. Defines ComponentType (LOCAL_ATTENTION,
│   │                   SSM, FFN...) and Validity (VALID, EXCLUDED...). Just
│   │                   definitions — no logic. ✅ Done
│   │
│   ├── inspector.py    Step 1. Walks the model, lists every part. ✅ Done, works
│   │
│   ├── typing.py       Step 2 plumbing. Holds the registry of per-family
│   │                   adapters and picks the right one. ✅ Done
│   │
│   ├── constraints.py  Step 3. The validity rules. ⚠️ Structure done, rules
│   │                   half-written
│   │
│   └── adapters/
│       ├── gemma4.py       Step 2 for Gemma. ⚠️ Planned for Phase 2
│       └── falcon_h1.py    Step 2 for Falcon. ⚠️ Planned for Phase 2
│
├── search_space/
│   ├── schema.py       What a grammar looks like. ✅ Done
│   └── grammar.py      Step 4. Turns a description into a grammar. ✅ Done
│
├── execution/
│   └── backends/
│       ├── base.py         The contract every backend must satisfy. ✅ Done
│       ├── cuda_peft.py    For NVIDIA machines. ❌ Not written
│       └── mlx_backend.py  For your Mac. ❌ Not written — this is next
│
├── evaluation/         Step 6. Tool-call scoring + memory measurement. ❌
├── tracking/           Step 6. Saving trial records. ❌
└── cli/main.py         The `aloe inspect` command. ✅ Wired up
```

**Roughly 40% built.** Everything that defines *what ALOE is* exists. Everything that *runs a model* does not.

---

# 5. The one idea you must be able to defend

A viva examiner will ask: *"Hasn't this been done?"*

Partly. **Borobia et al. (arXiv:2604.22127) §3.2** already built an automated pipeline that walks the module tree and classifies modules by type. Steps 1 and 2 are theirs. **Concede that cleanly** — it makes everything else credible.

Here is where they stop. Their pipeline's final stage verifies *"all expected modules received adapters and no forbidden modules were modified."* It checks **did the adapter attach where I asked?**

And they still had to exclude Falcon-H1's `conv1d` and `out_proj` **by hand**, after preliminary experiments showed training instability, because those weights feed a custom Mamba kernel that skips the LoRA path entirely. An adapter there attaches perfectly and never receives a gradient. Their check said everything was fine.

> **Their check: did it attach?**
> **Your check: can it learn?**

That is why `verify_gradients()` exists in `base.py`: one forward-backward pass, then assert a non-zero gradient actually reached each adapter. A target that attaches but never learns is `EXCLUDED`, not `VALID`.

**If you remember one thing from this document, remember that paragraph.**

---

# 6. Why the folders are split where they are

Two rules, both in `docs/DECISIONS.md`:

**`architecture/` and `search_space/` may not import any training library.** No `peft`, no `bitsandbytes`, no `mlx`. There's a test that fails if anyone tries (`tests/test_core_isolation.py`).

Why: the research contribution is "inspect a model, derive a search space". If that code imports a training library, the contribution is welded to one vendor's GPUs, and "works across architectures" quietly becomes "works across architectures, on NVIDIA".

**Adding a new model must mean adding one file in `adapters/` and nothing else.** If supporting Qwen3.5 required editing the grammar generator or the executor, the abstraction has failed — and that failure *is* research question SQ2. It's testable.

---

# 7. Run it today

```bash
cd ~/Development/fyp/aloe
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx,dev]"
pytest -q          # 5 tests should pass
```

The tests pass without any model downloaded — they use a fake 3-module model. That's deliberate: you can verify the logic without waiting on a 2 GB download.

`aloe inspect` won't work yet — it needs `aloe/pipeline.py`, which is defined as task T2 in the engineering specification.

---

# 8. Questions to be able to answer

| Question | Where the answer lives |
|---|---|
| What does ALOE actually do? | §2 and §3 above |
| How is it different from AdaLoRA? | AdaLoRA scores matrices numerically; ALOE groups by architectural role |
| How is it different from Borobia et al.? | §5 — attach vs. learn; and they don't search rank or alpha |
| Why group by type instead of per-module? | §3 real-scale note — 3 groups is searchable, 250 modules isn't |
| Why doesn't the core import PEFT? | §6 |
| What is alpha and why does it matter? | §1 — and nobody searches it, which is Gap 2 |
| What happens if the typing is wrong? | The grammar is silently wrong and no test catches it. Hence: read from config, never guess |

If you can answer those seven without notes, you can defend the codebase.

# Apple Silicon — What Changes and Why

You are now on an M1 Max with 64 GB. Almost every fine-tuning tutorial assumes NVIDIA. This explains what is actually different, what breaks, and what you gain.

---

# 1. Unified memory

## The NVIDIA arrangement

An NVIDIA machine has two separate pools:

```
  CPU  ──────────►  System RAM      32 GB
                        │
                   PCIe bus  ← the bottleneck
                        │
  GPU  ──────────►  VRAM             8 GB   ← the hard wall
```

Your model must fit in **VRAM**. On an 8 GB card, an 8.1 GB model does not run. Not slowly — not at all. And every byte moving between the two pools crosses PCIe, which is slow.

That wall is why QLoRA exists: squeeze the weights to 4 bits so they fit.

## The Apple arrangement

```
  CPU  ┐
       ├──────►  Unified Memory      64 GB
  GPU  ┘                              ↑
                         one pool, both processors
```

One pool. No copying, no PCIe. The GPU reads the same bytes the CPU wrote.

## Small numbers

A model with **1,000 parameters** at 2 bytes each (bf16) = **2 KB**.

| | NVIDIA 8 GB | M1 Max 64 GB |
|---|---|---|
| Model weights | 2 KB in VRAM | 2 KB in unified |
| Copy to GPU | yes, over PCIe | **none needed** |
| Usable ceiling | 8 GB | ~48–56 GB (OS keeps some) |

## Real numbers — Gemma 4 E2B

About 2 billion parameters (5.1 B total, ~2.3 B effective).

| Precision | Size | Fits 8 GB card? | Fits your Mac? |
|---|---|---|---|
| fp32 (4 bytes) | ~8.0 GB | ✗ | ✓ easily |
| bf16 (2 bytes) | ~4.0 GB | tight, with nothing left for training | ✓ easily |
| 4-bit NF4 | ~1.2 GB | ✓ | ✓ trivially |

Training needs roughly **2–3× the weight size** on top, for gradients, optimizer state and activations.

- On an 8 GB card: bf16 needs ~4 GB weights + ~8 GB overhead = **12 GB. Doesn't fit.** You are forced into 4-bit.
- On your 64 GB Mac: bf16 needs ~12 GB total. **You have 64.**

> **You may not need quantisation at all.** That is the headline. Measure bf16 LoRA first. If it fits and trains, you have removed an entire failure surface from the project.

---

# 2. Why bitsandbytes does not work here

`bitsandbytes` is the library that does NF4 4-bit quantisation. It is written as **CUDA kernels** — small programs compiled for NVIDIA hardware specifically. They are not Python. They cannot be reinterpreted for another chip.

Apple GPUs run **Metal**, a different instruction set. A CUDA kernel on Apple Silicon is a Windows `.exe` on a Mac: the file exists, the machine cannot execute it.

Status as of now: mainline bitsandbytes fails on ARM64 with CUDA-availability errors. An MPS backend exists **only as an unmerged pull request** that delegates to a separate `mps-bitsandbytes` package. Not something to build an FYP on.

**Unsloth** is CUDA-only for the same reason.

So the two libraries every QLoRA tutorial uses are both unavailable. This is not a configuration problem you can fix.

---

# 3. Your two options on Mac

### PyTorch with the MPS backend
PyTorch speaks Metal through a backend called **MPS** (Metal Performance Shaders). `device="mps"` instead of `device="cuda"`.

- ✅ Everything you know about PyTorch still applies; HuggingFace `transformers` works
- ✅ `peft` works for plain LoRA (it's pure PyTorch)
- ❌ No `bitsandbytes`, so **no 4-bit**
- ❌ Some operations silently fall back to CPU and get slow

### MLX (Apple's own framework)
Built by Apple specifically for unified memory. `mlx-lm` does LoRA and QLoRA natively.

- ✅ Native 4-bit that actually works
- ✅ Fastest option on this hardware — roughly 250 tokens/sec for a 7B LoRA on an M1 Max
- ❌ Different API from PyTorch — a real learning cost
- ❓ **Unknown: can it target individual modules with per-group rank and alpha?**

That last question is not rhetorical. **It decides your project's shape**, which is why it's task I092 and why you do it first.

---

# 4. The decision tree

```
Can mlx-lm set a different rank per named module group?
│
├── YES → MLX is your main backend. Everything runs locally. Best case.
│
└── NO  → Can PyTorch + peft on MPS do it in bf16, no quantisation?
          │
          ├── YES → Use that. 64 GB means you don't need 4-bit for 2B models.
          │         Slower than MLX but the API is familiar and peft already
          │         supports rank_pattern / alpha_pattern, which is exactly
          │         the per-group mechanism ALOE needs.
          │
          └── NO  → MLX runs baselines locally; typed runs go to a rented
                    CUDA box. Costs money and lead time — which is precisely
                    why you want this answer today, not on 27 September.
```

**My expectation, stated so you can prove me wrong:** the middle branch wins. `peft`'s `rank_pattern` and `alpha_pattern` take regex-keyed dictionaries mapping module patterns to per-module rank and alpha. That is the exact mechanism the grammar needs, it is pure PyTorch, and it should run on MPS. MLX may still be faster, but peft-on-MPS is the lower-risk path to Gate B.

Test it, don't take my word for it.

---

# 5. What changes in the code

Very little, and that is by design.

| Layer | Changes? |
|---|---|
| `architecture/` — inspect, type, validate | **No.** Pure PyTorch module-tree walking. Runs on any chip |
| `search_space/` — grammar | **No.** Plain data |
| `execution/backends/` | **Yes.** This is the only place that knows about hardware |
| `evaluation/resources.py` | **Yes.** Memory is measured differently |

That containment is exactly why ADR-002 forbids the core importing a training framework. If `constraints.py` had imported `bitsandbytes`, the whole project would be dead on this machine.

## Concrete differences

**Device**
```python
device = "cuda"                        # old
device = "mps"                         # new
# better:
device = "mps" if torch.backends.mps.is_available() else "cpu"
```
Any hardcoded `.cuda()` or `device="cuda"` is now a bug.

**Memory measurement**
```python
torch.cuda.max_memory_allocated()      # does not exist on Mac
torch.mps.current_allocated_memory()   # this instead
torch.mps.driver_allocated_memory()    # total driver-side
```
Note the difference: CUDA gives you a **peak** since last reset. MPS gives **current**. To get a peak you must sample during training — poll on a background thread, or read after each step and keep the maximum. Do not report "peak memory" from a single reading at the end; that is not a peak.

**Precision**
- Avinash (arXiv:2509.12229) found **fp16 beats bf16** on an RTX 4060. **That is a CUDA result.** Apple Silicon has native bf16 support and different tradeoffs. Re-measure; do not inherit the finding.

**Optimizer**
- PagedAdamW is a bitsandbytes feature. Not available. Use standard AdamW — and with 64 GB you don't need paging, which is the problem paging solved.

---

# 6. What to record instead of VRAM

Your trial records currently assume VRAM. Change the schema to:

```python
{
  "backend": "mlx" | "cuda_peft" | "torch_mps",
  "chip": "Apple M1 Max",
  "unified_memory_gb": 64,
  "peak_memory_gb": 11.3,          # sampled, not a single reading
  "memory_metric": "unified",       # or "vram" — never leave this implicit
  "os": "macOS 26.x",
  "torch_version": "2.x",
}
```

**Never average a Mac number with an NVIDIA number.** Peak VRAM on a dedicated card and peak unified memory on a shared pool are different physical quantities. Two Pareto fronts, reported separately. Pooling them would be a methodological error an examiner could reasonably fail you on.

---

# 7. What this does to your research claims

**The honest position.** RQ3 says "consumer hardware constraints". An M1 Max 64 GB *is* consumer hardware — you can buy one in a shop. But it is not an 8 GB RTX 4060, and pretending otherwise would be the kind of imprecision that gets caught.

Say it plainly in the write-up:

> Experiments were conducted on an Apple M1 Max with 64 GB unified memory, a consumer workstation configuration. Results are not directly comparable to studies on discrete NVIDIA GPUs with fixed VRAM budgets, and peak unified memory is reported separately from peak VRAM throughout.

**What you gain.** Something genuinely better than what you had. A framework that derives correct search spaces across **two architectures and two execution backends** is a much stronger reusability claim (SQ2) than one that does neither. Falcon-H1 and Gemma prove architecture-independence; MLX and CUDA prove stack-independence.

And it produces a finding you could not otherwise have had: **validity is a function of (architecture, backend), not architecture alone.** A module that is a legal LoRA target under PEFT may be unsupported under MLX. Nobody in your literature has said that, because nobody ran two backends.

The constraint made the project better. Say so.

---

# 8. First three things to do on this machine

```bash
cd ~/Development/fyp/aloe
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx,dev]"
pytest -q                                    # 5 tests, should pass

python3 -c "import torch; print(torch.backends.mps.is_available())"   # expect True
python3 -c "import mlx.core as mx; print(mx.default_device())"        # expect Device(gpu, 0)
```

Then **task I092**: the MLX probe. Everything downstream depends on its answer.

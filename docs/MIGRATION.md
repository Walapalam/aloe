# Machine Migration — M1 Pro 16GB → M1 Max 64GB

Do the push **before** you start moving anything else.

## On the old machine

```bash
cd /Users/walapalam/Development/fyp/aloe
git init
git add .
git commit -m "ALOE Core v0.1 scaffold: schema, inspector, grammar, ADRs"
git branch -M main
```

Create a **private** repo on GitHub named `aloe` — do not tick "add a README",
the repo already has one. Then:

```bash
git remote add origin git@github.com:<your-username>/aloe.git
git push -u origin main
```

If you use HTTPS rather than SSH, GitHub needs a personal access token rather than a
password. SSH is less friction across two machines:

```bash
ssh-keygen -t ed25519 -C "raqeebmr3@gmail.com"
cat ~/.ssh/id_ed25519.pub     # paste into GitHub → Settings → SSH keys
ssh -T git@github.com         # verify
```

## Keep the repo private

This is unpublished FYP work with a competitor actively working the same problem
(Borobia et al. name rank ablation as future work). Private until submission. The git
history is also timestamped evidence of when the work was done — worth having.

## What does NOT go in git

Already covered by `.gitignore`: model weights, adapters, trial outputs, raw datasets,
`.venv`. If `git status` ever shows a `.safetensors` file, stop and fix the ignore
rules rather than committing it.

Dataset **manifests** (hashes, split definitions) do go in. The data itself does not.

## On the new machine

```bash
git clone git@github.com:<your-username>/aloe.git
cd aloe
python3 -m venv .venv && source .venv/bin/activate
make install-mlx
pytest -q
aloe inspect <a small model>
```

## Things that will differ on the M1 Max

| | Note |
|---|---|
| bitsandbytes | Will not install usefully. Expected — use the `mlx` extra, not `cuda`. |
| torch device | `mps`, not `cuda`. Any hardcoded `.cuda()` or `device="cuda"` is a bug. |
| Memory metric | Unified memory, not VRAM. `torch.mps.current_allocated_memory()` and MLX's own metrics — `torch.cuda.max_memory_allocated()` does not exist here. |
| 64GB unified | You may not need 4-bit at all for 2B models. Measure bf16 LoRA before assuming QLoRA is required. |
| Throughput | Roughly 250 tok/s on an M1 Max for a 7B LoRA, per published reports. Verify on your own hardware and log it. |

## First thing to do after the move

Run the MLX probe (task T1 in the engineering specification). Whether mlx-lm supports per-module
rank and alpha decides the shape of the whole grammar, and you want that answer on day
one, not on 27 September.

#!/usr/bin/env python3
"""Deterministic generator for L4-pinned ALOE mobile Colab Pro+ shard notebooks."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "notebooks" / "colab_shards"

FALCON_SHA = "f0d0ea2b56b441e18d475366ea249617afe65883"
LFM3_SHA = "ab9877ce5110f3889d2d2f028a4d3ff559bd48b2"
GEMMA_SHA = "8187255cae076f583dce455acaf5019b20bad1e4"

SHARDS = [
    {
        "filename": "F1_falcon_b2_borobia.ipynb",
        "title": "ALOE Shard F1 — Falcon-H1-1.5B (b2 + Borobia Search & 3-Seed Confirm | NVIDIA L4)",
        "shard_id": "F1_falcon_b2_borobia",
        "model": "tiiuae/Falcon-H1-1.5B-Instruct",
        "pinned_sha": FALCON_SHA,
        "arms": "b2,borobia_attn,borobia_ssm,borobia_ffn",
        "phases": "search,confirm",
        "search_seed": 42,
        "eta": "~9.5h on 1x NVIDIA L4",
    },
    {
        "filename": "F2_falcon_confirm_aloe_b0_b1.ipynb",
        "title": "ALOE Shard F2 — Falcon-H1-1.5B (aloe, b0, b1 3-Seed Confirm | NVIDIA L4)",
        "shard_id": "F2_falcon_confirm_aloe_b0_b1",
        "model": "tiiuae/Falcon-H1-1.5B-Instruct",
        "pinned_sha": FALCON_SHA,
        "arms": "aloe,b0,b1",
        "phases": "confirm",
        "search_seed": 42,
        "eta": "~2.5-3h on 1x NVIDIA L4",
    },
    {
        "filename": "L1_lfm3_aloe.ipynb",
        "title": "ALOE Shard L1 — LiquidAI/LFM3-1.7B (aloe Search & 3-Seed Confirm | NVIDIA L4)",
        "shard_id": "L1_lfm3_aloe",
        "model": "LiquidAI/LFM3-1.7B",
        "pinned_sha": LFM3_SHA,
        "arms": "aloe",
        "phases": "search,confirm",
        "search_seed": 42,
        "eta": "~6.5-8h on 1x NVIDIA L4",
    },
    {
        "filename": "L2_lfm3_b0.ipynb",
        "title": "ALOE Shard L2 — LiquidAI/LFM3-1.7B (b0 Search & 3-Seed Confirm | NVIDIA L4)",
        "shard_id": "L2_lfm3_b0",
        "model": "LiquidAI/LFM3-1.7B",
        "pinned_sha": LFM3_SHA,
        "arms": "b0",
        "phases": "search,confirm",
        "search_seed": 42,
        "eta": "~6.5-8h on 1x NVIDIA L4",
    },
    {
        "filename": "L3_lfm3_b1.ipynb",
        "title": "ALOE Shard L3 — LiquidAI/LFM3-1.7B (b1 Search & 3-Seed Confirm | NVIDIA L4)",
        "shard_id": "L3_lfm3_b1",
        "model": "LiquidAI/LFM3-1.7B",
        "pinned_sha": LFM3_SHA,
        "arms": "b1",
        "phases": "search,confirm",
        "search_seed": 42,
        "eta": "~6.5-8h on 1x NVIDIA L4",
    },
    {
        "filename": "L4_lfm3_b2_borobia.ipynb",
        "title": "ALOE Shard L4 — LiquidAI/LFM3-1.7B (b2 + Borobia Search & 3-Seed Confirm | NVIDIA L4)",
        "shard_id": "L4_lfm3_b2_borobia",
        "model": "LiquidAI/LFM3-1.7B",
        "pinned_sha": LFM3_SHA,
        "arms": "b2,borobia_attn,borobia_ssm,borobia_ffn",
        "phases": "search,confirm",
        "search_seed": 42,
        "eta": "~9.5-11h on 1x NVIDIA L4",
    },
    {
        "filename": "F3_falcon_seed1337_aloe_b1.ipynb",
        "title": "ALOE Optional Shard F3 — Falcon-H1-1.5B Extra Search Seed 1337 (aloe, b1 | NVIDIA L4)",
        "shard_id": "F3_falcon_seed1337_aloe_b1",
        "model": "tiiuae/Falcon-H1-1.5B-Instruct",
        "pinned_sha": FALCON_SHA,
        "arms": "aloe,b1",
        "phases": "search",
        "search_seed": 1337,
        "eta": "~10h on 1x NVIDIA L4 (optional bonus search seed)",
    },
    {
        "filename": "F4_falcon_seed2026_aloe_b1.ipynb",
        "title": "ALOE Optional Shard F4 — Falcon-H1-1.5B Extra Search Seed 2026 (aloe, b1 | NVIDIA L4)",
        "shard_id": "F4_falcon_seed2026_aloe_b1",
        "model": "tiiuae/Falcon-H1-1.5B-Instruct",
        "pinned_sha": FALCON_SHA,
        "arms": "aloe,b1",
        "phases": "search",
        "search_seed": 2026,
        "eta": "~10h on 1x NVIDIA L4 (optional bonus search seed)",
    },
    {
        "filename": "L5_lfm3_seed1337_aloe_b1.ipynb",
        "title": "ALOE Shard L5 — LiquidAI/LFM3-1.7B Zero-Shot + Extra Search Seed 1337 (aloe, b1 | NVIDIA L4)",
        "shard_id": "L5_lfm3_seed1337_aloe_b1",
        "model": "LiquidAI/LFM3-1.7B",
        "pinned_sha": LFM3_SHA,
        "arms": "aloe,b1",
        "phases": "zero_shot,search",
        "search_seed": 1337,
        "eta": "~10h on 1x NVIDIA L4 (Zero-Shot + Paired Seed 1337)",
    },
    {
        "filename": "L6_lfm3_seed2026_aloe_b1.ipynb",
        "title": "ALOE Shard L6 — LiquidAI/LFM3-1.7B Extra Search Seed 2026 (aloe, b1 | NVIDIA L4)",
        "shard_id": "L6_lfm3_seed2026_aloe_b1",
        "model": "LiquidAI/LFM3-1.7B",
        "pinned_sha": LFM3_SHA,
        "arms": "aloe,b1",
        "phases": "search",
        "search_seed": 2026,
        "eta": "~10h on 1x NVIDIA L4 (Paired Seed 2026)",
    },
    {
        "filename": "Z1_falcon_zero_shot.ipynb",
        "title": "ALOE Shard Z1 — Falcon-H1-1.5B Base Zero-Shot Validation Baseline (NVIDIA L4)",
        "shard_id": "Z1_falcon_zero_shot",
        "model": "tiiuae/Falcon-H1-1.5B-Instruct",
        "pinned_sha": FALCON_SHA,
        "arms": "aloe",
        "phases": "zero_shot",
        "search_seed": 42,
        "eta": "~5m on 1x NVIDIA L4",
    },
    {
        "filename": "G1_gemma_cuda_supp_aloe_b0_b1.ipynb",
        "title": "ALOE Supplementary Shard G1 — Gemma-4-E2B CUDA Cross-Check (zero_shot + aloe, b0, b1 | NVIDIA L4)",
        "shard_id": "G1_gemma_cuda_supp_aloe_b0_b1",
        "model": "google/gemma-4-E2B-it",
        "pinned_sha": GEMMA_SHA,
        "arms": "aloe,b0,b1",
        "phases": "zero_shot,search,confirm",
        "search_seed": 42,
        "campaign_prefix": "campaigns_cuda_supplementary",
        "confirm_prefix": "confirm_3seed_cuda_supplementary",
        "allow_supplementary_cuda": True,
        "eta": "~12h on 1x NVIDIA L4 (isolated CUDA supplementary)",
    },
    {
        "filename": "G2_gemma_cuda_supp_b2_borobia.ipynb",
        "title": "ALOE Supplementary Shard G2 — Gemma-4-E2B CUDA Cross-Check (b2, borobia_attn, borobia_ffn | NVIDIA L4)",
        "shard_id": "G2_gemma_cuda_supp_b2_borobia",
        "model": "google/gemma-4-E2B-it",
        "pinned_sha": GEMMA_SHA,
        "arms": "b2,borobia_attn,borobia_ffn",
        "phases": "search,confirm",
        "search_seed": 42,
        "campaign_prefix": "campaigns_cuda_supplementary",
        "confirm_prefix": "confirm_3seed_cuda_supplementary",
        "allow_supplementary_cuda": True,
        "eta": "~9h on 1x NVIDIA L4 (isolated CUDA supplementary)",
    },
]


def generate_all(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in SHARDS:
        title = spec["title"]
        shard_id = spec["shard_id"]
        pinned_sha = str(spec["pinned_sha"])
        arms = spec["arms"]
        phases = spec["phases"]
        eta = spec["eta"]
        model = spec["model"]
        search_seed = spec["search_seed"]
        campaign_prefix = spec.get("campaign_prefix", "")
        confirm_prefix = spec.get("confirm_prefix", "")
        allow_supp_cuda = bool(spec.get("allow_supplementary_cuda", False))
        extra_cmd_lines: list[str] = []
        if campaign_prefix:
            extra_cmd_lines.append(f'        "--campaign-prefix", "{campaign_prefix}",\n')
        if confirm_prefix:
            extra_cmd_lines.append(f'        "--confirm-prefix", "{confirm_prefix}",\n')
        if allow_supp_cuda:
            extra_cmd_lines.append('        "--allow-supplementary-cuda",\n')

        md_lines = [
            f"# {title}\n",
            "\n",
            "**Mobile 1-Tap Instructions (~30 seconds):**\n",
            "1. Ensure `HF_TOKEN` is saved once in Colab **Secrets** (key icon on the left sidebar, *Notebook access* toggled ON).\n",
            "2. Tap the **Play** button on the single cell below and tap **Grant access**.\n",
            "3. Once output shows `[SHARD LIVE] ... Starting trials...` (~25s), **close your browser / lock your phone**.\n",
            f"- **Pinned Code SHA:** `{pinned_sha[:12]}` | **Arms:** `{arms}` | **Phases:** `{phases}` | **ETA:** `{eta}`\n",
            "- Auto-uploads every completed trial to private `Walapalam/aloe-ipd-v1-runs` and **auto-disconnects the L4 GPU** from the notebook kernel when finished.\n",
        ]

        code_lines = [
            f"# @title Run {shard_id} ({eta})\n",
            'FORCE_STEAL_LEASE = False  # @param {type:"boolean"}\n',
            "import getpass, os, subprocess, sys, time, torch\n",
            "\n",
            "def _unassign_colab_kernel(delay_sec: float) -> None:\n",
            "    try:\n",
            "        from google.colab import runtime\n",
            '        print(f"[COLAB] Disconnecting L4 runtime in {delay_sec:.0f}s to conserve compute units...", flush=True)\n',
            "        time.sleep(delay_sec)\n",
            "        runtime.unassign()\n",
            "    except Exception:\n",
            "        pass\n",
            "\n",
            "# 1. Fail-closed GPU check (<1s before installing or burning compute units)\n",
            "assert torch.cuda.is_available(), (\n",
            '    "No GPU attached! Tap Runtime -> Change runtime type -> L4 GPU -> Save, then tap Play again."\n',
            ")\n",
            "gpu_name = torch.cuda.get_device_name(0)\n",
            'assert gpu_name == "NVIDIA L4", (\n',
            "    f\"ADR-019 requires exact GPU 'NVIDIA L4', got '{gpu_name}'. \"\n",
            '    "Tap Runtime -> Change runtime type -> L4 GPU -> Save, then tap Play again."\n',
            ")\n",
            'print(f"[1/4] GPU verified: {gpu_name} | PyTorch {torch.__version__}", flush=True)\n',
            "\n",
            "# 2. Read HF_TOKEN from Colab Secrets (or getpass fallback)\n",
            'hf_token = ""\n',
            "try:\n",
            "    from google.colab import userdata\n",
            '    hf_token = (userdata.get("HF_TOKEN") or "").strip()\n',
            "except Exception:\n",
            "    pass\n",
            "if not hf_token:\n",
            '    hf_token = getpass.getpass("Paste Hugging Face write token (hf_...): ").strip()\n',
            'assert hf_token.startswith("hf_"), "Valid HF_TOKEN starting with hf_ is required."\n',
            'os.environ["HF_TOKEN"] = hf_token\n',
            'os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token\n',
            'print("[2/4] HF_TOKEN loaded securely.", flush=True)\n',
            "\n",
            "rc = 1\n",
            "try:\n",
            "    # 3. Install frozen user-space dependencies (preserving Colab pre-installed torch+cu130)\n",
            "    subprocess.run([\n",
            '        sys.executable, "-m", "pip", "install", "-q",\n',
            '        "transformers==5.24.0", "peft==0.22.0", "botorch==0.18.1",\n',
            '        "accelerate>=1.0.0", "datasets>=3.0.0", "huggingface_hub>=0.26.0",\n',
            '        "pydantic>=2.9.0", "rich>=13.9.0", "typer>=0.12.5"\n',
            "    ], check=True)\n",
            '    print("[3/4] Dependencies ready. Preparing isolated shard worktree...", flush=True)\n',
            "\n",
            "    REPO_URL = \"https://github.com/Walapalam/aloe.git\"\n",
            f'    ORCH_DIR = "/content/aloe_orch_{shard_id}"\n',
            f'    WORK_DIR = "/content/aloe_work_{shard_id}"\n',
            f'    PINNED_SHA = "{pinned_sha}"\n',
            "\n",
            "    if not os.path.exists(ORCH_DIR):\n",
            '        subprocess.run(["git", "clone", "-q", "-b", "feat/ipd-protocol-v1", REPO_URL, ORCH_DIR], check=True)\n',
            "    else:\n",
            '        subprocess.run(["git", "-C", ORCH_DIR, "fetch", "-q", "origin", "feat/ipd-protocol-v1"], check=True)\n',
            '        subprocess.run(["git", "-C", ORCH_DIR, "reset", "--hard", "origin/feat/ipd-protocol-v1"], check=True)\n',
            "\n",
            "    if not os.path.exists(WORK_DIR):\n",
            '        subprocess.run(["git", "clone", "-q", "-b", "feat/ipd-protocol-v1", REPO_URL, WORK_DIR], check=True)\n',
            "    else:\n",
            '        subprocess.run(["git", "-C", WORK_DIR, "fetch", "-q", "origin", "feat/ipd-protocol-v1"], check=True)\n',
            "    # Reset tracked files ONLY (never git clean -fdq so mid-trial checkpoints survive local re-runs)\n",
            '    subprocess.run(["git", "-C", WORK_DIR, "checkout", "-q", "--detach", PINNED_SHA], check=True)\n',
            '    subprocess.run(["git", "-C", WORK_DIR, "reset", "--hard", PINNED_SHA], check=True)\n',
            "\n",
            '    print("[4/4] Launching atomic shard runner (safe to close phone once [SHARD LIVE] prints)...", flush=True)\n',
            "    cmd = [\n",
            '        sys.executable, "-u", f"{ORCH_DIR}/scripts/run_colab_shard.py",\n',
            f'        "--shard-id", "{shard_id}",\n',
            f'        "--model", "{model}",\n',
            '        "--work-repo", WORK_DIR,\n',
            '        "--expected-git-sha", PINNED_SHA,\n',
            f'        "--arms", "{arms}",\n',
            f'        "--phases", "{phases}",\n',
            f'        "--search-seed", "{search_seed}",\n',
            *extra_cmd_lines,
            '        "--pinned-gpu", "L4",\n',
            '        "--expected-exact-gpu", "NVIDIA L4",\n',
            '        "--hf-repo", "Walapalam/aloe-ipd-v1-runs",\n',
            "    ]\n",
            "    if FORCE_STEAL_LEASE:\n",
            '        cmd.append("--force-steal-lease")\n',
            "    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
            "    assert proc.stdout is not None\n",
            "    for line in proc.stdout:\n",
            '        print(line, end="", flush=True)\n',
            "    rc = proc.wait()\n",
            "    if rc != 0:\n",
            "        raise RuntimeError(f\"Shard runner exited with status {rc}\")\n",
            "finally:\n",
            "    _unassign_colab_kernel(3.0 if rc == 0 else 25.0)\n",
        ]

        nb = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "name": title,
                    "provenance": [],
                    "gpuType": "L4",
                    "machine_shape": "hm",
                    "background_execution": "on",
                },
                "accelerator": "GPU",
                "kernelspec": {
                    "display_name": "Python 3",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                },
            },
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {"id": f"md_{shard_id}"},
                    "source": md_lines,
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {"id": f"run_{shard_id}", "cellView": "form"},
                    "outputs": [],
                    "source": code_lines,
                },
            ],
        }
        p = out_dir / str(spec["filename"])
        p.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
        written.append(p)
        print(f"Generated {p.name}")
    return written


if __name__ == "__main__":
    generate_all()

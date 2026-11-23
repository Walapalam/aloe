#!/usr/bin/env python3
"""Local Hugging Face Shard Sync, Summary Rebuilder, & Cryptographic Verifier (`sync_and_verify_hf_shards.py`).

Downloads completed per-arm Colab shards from `Walapalam/aloe-ipd-v1-runs`, verifies every
completed arm via `verify_campaign_dir`, `_verify_completed_trial_receipt`, and exact `git_commit`
pins (`f0d0ea2` for Falcon-H1, `8187255` for LFM3-1.7B / Gemma 4-E2B), deterministically
reconstructs `model_all_arms_summary.json` and `hypervolume_3d_report.json` without cross-VM races,
and emits a unified progress report across `campaigns/` and `campaigns_multiseed/`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloe.execution.trial_config import TrialConfig  # noqa: E402
from aloe.tracking.trial_store import atomic_write_text  # noqa: E402
from scripts.compute_3d_hypervolume import evaluate_model_campaigns_3d_hv  # noqa: E402
from scripts.run_colab_a100_campaign import REQUIRED_CONFIRM_SEEDS, _verify_completed_trial_receipt  # noqa: E402
from scripts.run_colab_shard import verify_arm_git_commits, verify_winner_adapter  # noqa: E402
from scripts.verify_campaign import verify_campaign_dir, verify_manifest  # noqa: E402

EXPECTED_CUDA_MODELS = {
    "tiiuae--Falcon-H1-1.5B-Instruct": ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ssm", "borobia_ffn"),
    "LiquidAI--LFM3-1.7B": ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ssm", "borobia_ffn"),
}

EXPECTED_GIT_PINS = {
    "tiiuae--Falcon-H1-1.5B-Instruct": "f0d0ea2b56b441e18d475366ea249617afe65883",
    "LiquidAI--LFM3-1.7B": "ab9877ce5110f3889d2d2f028a4d3ff559bd48b2",
    "google--gemma-4-E2B-it": "8187255cae076f583dce455acaf5019b20bad1e4",
}


def _resolve_hf_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    env_file = Path.home() / ".env"
    if env_file.is_file():
        m = re.search(r"(hf_[A-Za-z0-9]{20,})", env_file.read_text(errors="ignore"))
        if m:
            return m.group(1)
    raise RuntimeError("HF_TOKEN not found in environment or ~/.env")


def _audit_campaigns_tree(
    campaigns_root: Path,
    *,
    confirm_root: Path | None,
    manifest_sha: str,
    split_id_sets: dict[str, set[str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not campaigns_root.is_dir():
        return out

    for model_dir in sorted(p for p in campaigns_root.iterdir() if p.is_dir()):
        slug = model_dir.name
        expected_git = EXPECTED_GIT_PINS.get(slug)
        hw_lock = model_dir / "hardware_lock.json"
        locked_gpu = None
        if hw_lock.is_file():
            try:
                locked_gpu = json.loads(hw_lock.read_text(encoding="utf-8")).get("locked_gpu_name")
            except Exception:
                locked_gpu = None

        merged_summaries: dict[str, Any] = {}
        arms_status: dict[str, Any] = {}
        for arm_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and (p / "campaign_ledger.json").is_file()):
            arm = arm_dir.name
            try:
                ldata = json.loads((arm_dir / "campaign_ledger.json").read_text(encoding="utf-8"))
                done_n = sum(1 for e in ldata.get("entries", []) if e.get("status") == "DONE")
                target_n = int(ldata.get("target_trials", 20))
                sfile = arm_dir / "campaign_summary.json"
                verified_search = False
                confirm_verified_seeds: list[int] = []
                err_msg: str | None = None

                if expected_git:
                    verify_arm_git_commits(arm_dir, expected_git)

                if done_n == target_n and sfile.is_file():
                    verify_campaign_dir(
                        arm_dir,
                        expected_manifest_sha=manifest_sha,
                        expected_split_ids=split_id_sets,
                        allow_smoke_budget=False,
                    )
                    sdata = json.loads(sfile.read_text(encoding="utf-8"))
                    verify_winner_adapter(arm_dir, str(sdata["selected_winner_trial_id"]))
                    merged_summaries[arm] = sdata
                    verified_search = True

                    if confirm_root is not None:
                        wcfg = TrialConfig.model_validate(sdata["selected_winner_config"]).model_dump(mode="json")
                        exp_gpu = str(locked_gpu or ldata.get("locked_gpu_name") or "")
                        c_arm_dir = confirm_root / slug / arm
                        if expected_git and c_arm_dir.is_dir():
                            verify_arm_git_commits(c_arm_dir, expected_git)
                        for c_seed in REQUIRED_CONFIRM_SEEDS:
                            tid = f"{arm}_winner_s{c_seed}_validation"
                            c_trial = c_arm_dir / "trials" / tid / "trial.json"
                            if exp_gpu and _verify_completed_trial_receipt(
                                c_trial,
                                expected_seed=c_seed,
                                expected_split="validation",
                                expected_split_ids=split_id_sets["validation"],
                                expected_manifest_sha=manifest_sha,
                                expected_gpu_name=exp_gpu,
                                expected_winner_cfg=wcfg,
                            ):
                                confirm_verified_seeds.append(c_seed)
            except Exception as exc:
                done_n = -1
                target_n = -1
                verified_search = False
                confirm_verified_seeds = []
                err_msg = str(exc)

            entry_stat: dict[str, Any] = {
                "done_trials": done_n,
                "target_trials": target_n,
                "search_verified": verified_search,
                "confirm_verified_seeds": confirm_verified_seeds,
            }
            if err_msg:
                entry_stat["verification_error"] = err_msg
            arms_status[arm] = entry_stat

        if merged_summaries:
            atomic_write_text(
                model_dir / "model_all_arms_summary.json",
                json.dumps(merged_summaries, indent=2) + "\n",
            )
        hv_rep = None
        if any(v.get("done_trials", 0) > 0 for v in arms_status.values()):
            try:
                hv_rep = evaluate_model_campaigns_3d_hv(model_dir)
                atomic_write_text(
                    model_dir / "hypervolume_3d_report.json",
                    json.dumps(hv_rep, indent=2) + "\n",
                )
            except Exception as hv_exc:
                hv_rep = {"error": str(hv_exc)}

        out[slug] = {
            "locked_gpu_name": locked_gpu,
            "expected_git_sha": expected_git,
            "arms": arms_status,
            "contribution_disentanglement": (hv_rep or {}).get("contribution_disentanglement"),
        }
    return out


def sync_and_audit_once(
    *,
    hf_repo: str,
    dest_dir: Path,
    manifest_path: Path,
    raw_data_path: Path | None,
    hf_token: str,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        local_dir=str(dest_dir),
        allow_patterns=["leases/**", "campaigns/**", "campaigns_multiseed/**", "confirm_3seed/**"],
        ignore_patterns=["**/checkpoint-*/*"],
        token=hf_token,
    )

    m_audit = verify_manifest(
        manifest_path,
        raw_data_path=raw_data_path if (raw_data_path and raw_data_path.is_file()) else None,
        enforce_frozen_v1=True,
        require_raw_data=bool(raw_data_path and raw_data_path.is_file()),
    )
    manifest_sha = str(m_audit["manifest_sha256"])
    split_id_sets: dict[str, set[str]] = m_audit["split_id_sets"]

    leases_report: dict[str, Any] = {}
    leases_dir = dest_dir / "leases"
    if leases_dir.is_dir():
        for lp in sorted(leases_dir.glob("*.json")):
            try:
                leases_report[lp.stem] = json.loads(lp.read_text(encoding="utf-8"))
            except Exception:
                pass

    models_report = _audit_campaigns_tree(
        dest_dir / "campaigns",
        confirm_root=dest_dir / "confirm_3seed",
        manifest_sha=manifest_sha,
        split_id_sets=split_id_sets,
    )

    multiseed_report: dict[str, Any] = {}
    ms_root = dest_dir / "campaigns_multiseed"
    if ms_root.is_dir():
        for seed_dir in sorted(p for p in ms_root.iterdir() if p.is_dir()):
            multiseed_report[seed_dir.name] = _audit_campaigns_tree(
                seed_dir,
                confirm_root=None,
                manifest_sha=manifest_sha,
                split_id_sets=split_id_sets,
            )

    cuda_all_complete = True
    for exp_slug, exp_arms in EXPECTED_CUDA_MODELS.items():
        mrep = models_report.get(exp_slug, {}).get("arms", {})
        for a in exp_arms:
            astat = mrep.get(a)
            if not astat or not astat["search_verified"] or astat["confirm_verified_seeds"] != REQUIRED_CONFIRM_SEEDS:
                cuda_all_complete = False

    report = {
        "synced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_all_complete": cuda_all_complete,
        "leases": leases_report,
        "models": models_report,
        "multiseed": multiseed_report,
    }
    atomic_write_text(dest_dir / "sync_verification_status.json", json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync & verify per-arm HF shards locally")
    parser.add_argument("--hf-repo", default="Walapalam/aloe-ipd-v1-runs")
    parser.add_argument("--dest-dir", default=str(REPO_ROOT.parent / "runs" / "colab_l4_ipd_v1"))
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifests" / "xlam_ipd_v1.json"))
    parser.add_argument("--raw-data", default=str(REPO_ROOT / "data" / "raw" / "xlam" / "xlam_function_calling_60k.json"))
    parser.add_argument("--watch-interval", type=int, default=0, help="If >0, poll every N seconds")
    parser.add_argument("--stop-when-complete", action="store_true")
    args = parser.parse_args()

    token = _resolve_hf_token()
    dest = Path(args.dest_dir).resolve()
    manifest_p = Path(args.manifest).resolve()
    raw_p = Path(args.raw_data).resolve() if args.raw_data else None

    while True:
        try:
            rep = sync_and_audit_once(
                hf_repo=args.hf_repo,
                dest_dir=dest,
                manifest_path=manifest_p,
                raw_data_path=raw_p,
                hf_token=token,
            )
            print(json.dumps(rep, indent=2), flush=True)
            if args.watch_interval <= 0 or (args.stop_when_complete and rep["cuda_all_complete"]):
                return 0
        except Exception as loop_exc:
            if args.watch_interval <= 0:
                raise
            print(f"[SYNC WATCH WARN] Transient sync error ({loop_exc}); retrying in {args.watch_interval}s...", flush=True)
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    raise SystemExit(main())

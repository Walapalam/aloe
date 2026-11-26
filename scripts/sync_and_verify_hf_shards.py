#!/usr/bin/env python3
"""Local Hugging Face Shard Sync, Summary Rebuilder, & Strict Integrity Verifier (`sync_and_verify_hf_shards.py`).

Downloads completed per-arm shards from `Walapalam/aloe-ipd-v1-runs`, verifies every search,
3-seed confirmation, zero-shot validation, and (post-unseal) 63-receipt final-test run against
exact `git_commit` pins (`f0d0ea2b56b4` for Falcon-H1, `ab9877ce5110` for LFM3-1.7B,
`8187255cae07` for Gemma-4-E2B), deterministically reconstructs `model_all_arms_summary.json`
and shared-bound `hypervolume_3d_report.json` reports, and emits `sync_verification_status.json`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloe.execution.trial_config import TrialConfig  # noqa: E402
from aloe.tracking.trial_store import atomic_write_text  # noqa: E402
from scripts.compute_3d_hypervolume import (  # noqa: E402
    compute_union_raw_max_bounds,
    evaluate_model_campaigns_3d_hv,
)
from scripts.final_test_audit import (  # noqa: E402
    CANONICAL_PROTOCOL_MODELS,
    has_any_local_test_or_attempt_artifacts,
    load_ceremony_pins,
    verify_complete_63_final_test_gate,
    verify_strict_trial_receipt,
    verify_strict_zero_shot_receipt,
)
from scripts.run_colab_a100_campaign import REQUIRED_CONFIRM_SEEDS  # noqa: E402
from scripts.run_colab_shard import verify_arm_git_commits, verify_winner_adapter  # noqa: E402
from scripts.verify_campaign import verify_campaign_dir, verify_manifest  # noqa: E402

EXPECTED_CUDA_MODELS = {
    "tiiuae--Falcon-H1-1.5B-Instruct": ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ssm", "borobia_ffn"),
    "LiquidAI--LFM3-1.7B": ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ssm", "borobia_ffn"),
}

EXPECTED_GIT_PINS = {
    slug: spec["git_commit"] for slug, spec in CANONICAL_PROTOCOL_MODELS.items()
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
        spec = CANONICAL_PROTOCOL_MODELS.get(slug)
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

                    if confirm_root is not None and spec is not None:
                        wcfg = TrialConfig.model_validate(sdata["selected_winner_config"])
                        exp_gpu = str(locked_gpu or ldata.get("locked_gpu_name") or "")
                        exp_mem = str(ldata.get("locked_memory_metric_name") or spec["locked_memory_metric_name"])
                        c_arm_dir = confirm_root / slug / arm
                        if expected_git and c_arm_dir.is_dir():
                            verify_arm_git_commits(c_arm_dir, expected_git)
                        for c_seed in REQUIRED_CONFIRM_SEEDS:
                            tid = f"{arm}_winner_s{c_seed}_validation"
                            c_trial = c_arm_dir / "trials" / tid / "trial.json"
                            if exp_gpu and verify_strict_trial_receipt(
                                c_trial,
                                expected_seed=c_seed,
                                expected_split="validation",
                                expected_split_ids=split_id_sets["validation"],
                                expected_manifest_sha=manifest_sha,
                                expected_git_sha=spec["git_commit"],
                                expected_model_id=spec["model_id"],
                                expected_model_revision=spec["model_revision"],
                                expected_gpu_name=exp_gpu,
                                expected_memory_metric=exp_mem,
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
                hv_rep = evaluate_model_campaigns_3d_hv(
                    model_dir,
                    normalization_scope="canonical_model_all_arms_s42",
                )
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


def _compute_shared_multiseed_hv_reports(dest_dir: Path) -> dict[str, Any]:
    """Compute invariant shared-bound 3-seed `ALOE` vs `B1` HV across `[42, 1337, 2026]` per model."""
    out: dict[str, Any] = {}
    for slug in ("tiiuae--Falcon-H1-1.5B-Instruct", "LiquidAI--LFM3-1.7B"):
        seed_dirs = {
            42: dest_dir / "campaigns" / slug,
            1337: dest_dir / "campaigns_multiseed" / "seed_1337" / slug,
            2026: dest_dir / "campaigns_multiseed" / "seed_2026" / slug,
        }
        if not all(d.is_dir() for d in seed_dirs.values()):
            continue

        model_summary: dict[str, Any] = {}
        for scope_name, filt in (
            ("paired_aloe_b1_union_3seed", {"aloe", "b1"}),
            ("all_arms_union_3seed", None),
        ):
            bounds = compute_union_raw_max_bounds(list(seed_dirs.values()), arms_filter=filt)
            per_seed: dict[str, Any] = {}
            aloe_hvs: list[float] = []
            b1_hvs: list[float] = []
            aloe_raw_hvs: list[float] = []
            b1_raw_hvs: list[float] = []

            for s_val, s_dir in seed_dirs.items():
                rep = evaluate_model_campaigns_3d_hv(
                    s_dir,
                    shared_bounds=bounds,
                    normalization_scope=scope_name,
                    arms_filter={"aloe", "b1"},
                )
                if s_val != 42 and scope_name == "paired_aloe_b1_union_3seed":
                    atomic_write_text(
                        s_dir / "hypervolume_3d_report_shared_bounds.json",
                        json.dumps(rep, indent=2) + "\n",
                    )
                a_hv = float(rep["arms"]["aloe"]["normalized_3d_hv"])
                b_hv = float(rep["arms"]["b1"]["normalized_3d_hv"])
                a_raw = float(rep["arms"]["aloe"]["raw_3d_hv"])
                b_raw = float(rep["arms"]["b1"]["raw_3d_hv"])
                aloe_hvs.append(a_hv)
                b1_hvs.append(b_hv)
                aloe_raw_hvs.append(a_raw)
                b1_raw_hvs.append(b_raw)
                per_seed[f"seed_{s_val}"] = {
                    "aloe_norm_3d_hv": a_hv,
                    "b1_norm_3d_hv": b_hv,
                    "aloe_raw_3d_hv": a_raw,
                    "b1_raw_3d_hv": b_raw,
                    "rel_hv_delta_pct": round(100.0 * (a_hv - b_hv) / b_hv, 2) if b_hv > 0 else None,
                    "median_em_delta": rep["contribution_disentanglement"]["C2_median_em_delta"],
                    "aloe_won": a_hv > b_hv,
                }

            mean_a = statistics.mean(aloe_hvs)
            mean_b = statistics.mean(b1_hvs)
            model_summary[scope_name] = {
                "shared_raw_bounds": {
                    "exact_match": [round(bounds[0][0], 6), round(bounds[1][0], 6)],
                    "trainable_params": [int(-bounds[1][1]), int(-bounds[0][1])],
                    "search_memory_mb": [round(-bounds[1][2], 3), round(-bounds[0][2], 3)],
                },
                "per_seed": per_seed,
                "aloe_mean_norm_hv": round(mean_a, 6),
                "aloe_sd_norm_hv": round(statistics.stdev(aloe_hvs), 6),
                "b1_mean_norm_hv": round(mean_b, 6),
                "b1_sd_norm_hv": round(statistics.stdev(b1_hvs), 6),
                "mean_rel_hv_delta_pct": round(100.0 * (mean_a - mean_b) / mean_b, 2) if mean_b > 0 else None,
                "aloe_seed_wins": sum(1 for a, b in zip(aloe_hvs, b1_hvs) if a > b),
                "aloe_mean_raw_hv": round(statistics.mean(aloe_raw_hvs), 3),
                "b1_mean_raw_hv": round(statistics.mean(b1_raw_hvs), 3),
            }
        out[slug] = model_summary
        atomic_write_text(
            dest_dir / "campaigns" / slug / "multiseed_shared_hv_report.json",
            json.dumps(model_summary, indent=2) + "\n",
        )
    return out


def sync_and_audit_once(
    *,
    hf_repo: str,
    dest_dir: Path,
    manifest_path: Path,
    raw_data_path: Path | None,
    hf_token: str,
    skip_download: bool = False,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    if not skip_download:
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(dest_dir),
            allow_patterns=[
                "leases/**",
                "campaigns/**",
                "campaigns_multiseed/**",
                "confirm_3seed/**",
                "zero_shot_baselines/**",
                "zero_shot_baselines_cuda_supplementary/**",
                "final_sealed_test/**",
                "zero_shot_test/**",
                "final_test_attempts/**",
                "campaigns_cuda_supplementary/**",
                "confirm_3seed_cuda_supplementary/**",
            ],
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

    multiseed_shared_hv = _compute_shared_multiseed_hv_reports(dest_dir)

    zero_shot_report: dict[str, Any] = {}
    zs_root = dest_dir / "zero_shot_baselines"
    for slug, spec in CANONICAL_PROTOCOL_MODELS.items():
        zf = zs_root / slug / "zero_shot_validation.json"
        if zf.is_file():
            try:
                verified_rec = verify_strict_zero_shot_receipt(
                    zf,
                    expected_split="validation",
                    expected_split_ids=split_id_sets["validation"],
                    expected_manifest_sha=manifest_sha,
                    expected_git_sha=spec["git_commit"],
                    expected_model_id=spec["model_id"],
                    expected_model_revision=spec["model_revision"],
                    expected_gpu_name=spec["locked_gpu_name"],
                    expected_memory_metric=spec["locked_memory_metric_name"],
                    raise_on_error=True,
                )
                zero_shot_report[slug] = {
                    **(verified_rec or {}),
                    "strict_jsonl_sha256_verified": True,
                }
            except Exception as exc:
                zero_shot_report[slug] = {
                    "strict_jsonl_sha256_verified": False,
                    "verification_error": str(exc),
                }

    ceremony_pins = load_ceremony_pins()
    final_test_status = verify_complete_63_final_test_gate(
        dest_dir,
        manifest_sha=manifest_sha,
        split_id_sets=split_id_sets,
        expected_seal_shas=ceremony_pins.get("seal_sha256_by_model", {}),
        require_all_63=False,
    )
    local_test_artifacts = has_any_local_test_or_attempt_artifacts(dest_dir)

    cuda_supp_report = _audit_campaigns_tree(
        dest_dir / "campaigns_cuda_supplementary",
        confirm_root=dest_dir / "confirm_3seed_cuda_supplementary",
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

    pre_unseal_471_verified = True
    for slug, spec in CANONICAL_PROTOCOL_MODELS.items():
        mrep = models_report.get(slug, {}).get("arms", {})
        for a in spec["arms"]:
            astat = mrep.get(a)
            if not astat or not astat["search_verified"] or astat["confirm_verified_seeds"] != REQUIRED_CONFIRM_SEEDS:
                pre_unseal_471_verified = False
        if not zero_shot_report.get(slug, {}).get("strict_jsonl_sha256_verified"):
            pre_unseal_471_verified = False
    for ms_slug in EXPECTED_CUDA_MODELS:
        for seed_name in ("seed_1337", "seed_2026"):
            ms_arms = multiseed_report.get(seed_name, {}).get(ms_slug, {}).get("arms", {})
            for ms_arm in ("aloe", "b1"):
                ms_stat = ms_arms.get(ms_arm)
                if (
                    not ms_stat
                    or not ms_stat["search_verified"]
                    or ms_stat["done_trials"] != 20
                    or ms_stat["target_trials"] != 20
                ):
                    pre_unseal_471_verified = False

    pre_unseal_ceremony_ready = bool(pre_unseal_471_verified and not local_test_artifacts)
    report = {
        "synced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_all_complete": cuda_all_complete,
        "pre_unseal_all_3_models_verified": pre_unseal_471_verified,
        "pre_unseal_471_receipts_verified": pre_unseal_471_verified,
        "pre_unseal_ceremony_ready": pre_unseal_ceremony_ready,
        "any_local_test_or_attempt_artifacts_present": bool(local_test_artifacts),
        "local_test_or_attempt_artifact_count": len(local_test_artifacts),
        "final_test_63_gate": final_test_status,
        "leases": leases_report,
        "models": models_report,
        "multiseed": multiseed_report,
        "multiseed_shared_bounds_hv": multiseed_shared_hv,
        "zero_shot": zero_shot_report,
        "cuda_supplementary": cuda_supp_report,
    }
    atomic_write_text(dest_dir / "sync_verification_status.json", json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync & verify per-arm HF shards locally")
    parser.add_argument("--hf-repo", default="Walapalam/aloe-ipd-v1-runs")
    parser.add_argument("--dest-dir", default=str(REPO_ROOT.parent / "runs" / "colab_l4_ipd_v1"))
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifests" / "xlam_ipd_v1.json"))
    parser.add_argument("--raw-data", default=str(REPO_ROOT / "data" / "raw" / "xlam" / "xlam_function_calling_60k.json"))
    parser.add_argument("--skip-download", action="store_true", help="Audit local dest-dir without network download")
    parser.add_argument("--watch-interval", type=int, default=0, help="If >0, poll every N seconds")
    parser.add_argument("--stop-when-complete", action="store_true")
    args = parser.parse_args()

    token = "" if args.skip_download else _resolve_hf_token()
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
                skip_download=args.skip_download,
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

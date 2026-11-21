#!/usr/bin/env python3
"""Colab Pro+ NVIDIA A100 Campaign Orchestrator (`run_colab_a100_campaign.py`).

Enforces end-to-end ADR-012 / ADR-014 / ADR-017 / ADR-018 protocol invariants:
1. Entry-point `verify_manifest(..., enforce_frozen_v1=True, require_raw_data=True)` +
   clean Git tree (`--forbid-dirty-git`).
2. Single model-wide expiring lease (`campaigns/<slug>/.campaign_lease.json`, 5-minute TTL
   kept alive continuously by a 30-second daemon heartbeat thread while the session runs,
   auto-expiring within 5 minutes if Colab preempts the VM!).
3. Model-wide `hardware_lock.json` written inside the lease, locking exact GPU SKU
   (`NVIDIA A100-SXM4-40GB` vs `80GB`) and `cuda_max_memory_allocated` across ALL arms/phases.
4. Strict phase gates:
   - `search`     : fixed `--trials 20 --max-steps 125` across requested arms.
   - `confirm`    : verifies each arm's `campaign_summary.json` (`protocol_tag == "protocol-v1"`
                    + `verify_campaign_dir`) and runs `--seeds 42,1337,2026` on `validation`.
   - `final-test` : requires `--confirm-sealed-test`, verified 3-seed confirmation receipts
                    (exact full winner config hash, GPU SKU, manifest SHA, 200 frozen validation
                    IDs, and predictions SHA-256), and writes/verifies `test_unsealed.json`
                    passed via `--seal-file` to `worker_entry` (`run_trial` enforces seal!).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_CONFIRM_SEEDS = [42, 1337, 2026]
LEASE_TTL_SECONDS = 300
LEASE_HEARTBEAT_INTERVAL_SECONDS = 30.0


def compute_full_winner_sha256(wcfg: Any) -> str:
    """Hash complete TrialConfig excluding only mutable `trial_id`, `seed`, and local filesystem path strings."""
    dump = wcfg.model_dump(mode="json") if hasattr(wcfg, "model_dump") else dict(wcfg)
    for k in ("trial_id", "seed", "dataset_manifest_path", "dataset_raw_path"):
        dump.pop(k, None)
    canonical = json.dumps(dump, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _abi_wheel_subdir(base_wheel_dir: Path) -> Path:
    import torch

    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_tag = torch.__version__.split("+")[0]
    cu_tag = (torch.version.cuda or "cpu").replace(".", "")
    return base_wheel_dir / f"{py_tag}_torch{torch_tag}_cu{cu_tag}"


def ensure_drive_wheels(base_wheel_dir: Path, *, need_mamba: bool, need_causal_conv: bool) -> None:
    """Install or build/cache ABI-matched CUDA wheels into Google Drive."""
    if not (need_mamba or need_causal_conv):
        return

    pkgs: list[str] = []
    specs: list[str] = []
    if need_causal_conv or need_mamba:
        pkgs.append("causal-conv1d")
        specs.append("causal-conv1d>=1.4.0,<2.0.0")
    if need_mamba:
        pkgs.append("mamba-ssm")
        specs.append("mamba-ssm>=2.2.2,<3.0.0")

    def _imports_ok() -> bool:
        try:
            import causal_conv1d  # type: ignore  # noqa: F401
            if need_mamba:
                import mamba_ssm  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False

    if _imports_ok():
        print("[WHEELS] Fast CUDA hybrid kernels already active.")
        return

    wheel_dir = _abi_wheel_subdir(base_wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    cached_wheels = list(wheel_dir.glob("*.whl"))
    if cached_wheels:
        print(f"[WHEELS] Installing {len(cached_wheels)} ABI-matched wheels from {wheel_dir}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheel_dir), *pkgs],
            check=False,
        )
        if _imports_ok():
            print("[WHEELS] Installed cached Drive wheels successfully.")
            return

    print(f"[WHEELS] Building & caching {pkgs} into {wheel_dir} (one-time per PyTorch/CUDA ABI)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "-w", str(wheel_dir), "packaging", "ninja", *specs],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--find-links", str(wheel_dir), *pkgs],
        check=True,
    )


def _refresh_lease(lease_path: Path, token: str, phase: str, ttl_seconds: float = LEASE_TTL_SECONDS) -> None:
    from aloe.tracking.trial_store import atomic_write_text

    now = time.time()
    atomic_write_text(
        lease_path,
        json.dumps(
            {
                "token": token,
                "phase": phase,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "updated_epoch": now,
                "expires_epoch": now + ttl_seconds,
            },
            indent=2,
        ),
    )


@contextmanager
def acquire_model_lease(
    lease_path: Path,
    *,
    phase: str,
    force_unlock: bool = False,
    ttl_seconds: float = LEASE_TTL_SECONDS,
    heartbeat_interval_seconds: float = LEASE_HEARTBEAT_INTERVAL_SECONDS,
) -> Iterator[str]:
    """Model-wide expiring lease with background daemon heartbeat so multi-hour runs never expire mid-arm."""
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    if force_unlock:
        lease_path.unlink(missing_ok=True)

    now = time.time()
    if lease_path.is_file():
        try:
            cur = json.loads(lease_path.read_text(encoding="utf-8"))
            if float(cur.get("expires_epoch", 0.0)) > now:
                raise RuntimeError(
                    f"Active model campaign lease at {lease_path} (phase={cur.get('phase')}, "
                    f"host={cur.get('host')}, expires in {int(float(cur['expires_epoch']) - now)}s). "
                    "Wait for expiry or pass --force-unlock only after confirming no other session is live."
                )
        except RuntimeError:
            raise
        except Exception:
            pass

    token = f"{socket.gethostname()}_{os.getpid()}_{int(now * 1000)}"
    _refresh_lease(lease_path, token, phase, ttl_seconds=ttl_seconds)

    stop_evt = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop_evt.wait(heartbeat_interval_seconds):
            try:
                if lease_path.is_file():
                    cur_hb = json.loads(lease_path.read_text(encoding="utf-8"))
                    if cur_hb.get("token") != token:
                        stop_evt.set()
                        return
                _refresh_lease(lease_path, token, phase, ttl_seconds=ttl_seconds)
            except Exception:
                pass

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()
    try:
        yield token
    finally:
        stop_evt.set()
        hb_thread.join(timeout=2.0)
        if lease_path.is_file():
            try:
                cur = json.loads(lease_path.read_text(encoding="utf-8"))
                if cur.get("token") == token:
                    lease_path.unlink(missing_ok=True)
            except Exception:
                pass


def _verify_completed_trial_receipt(
    trial_json_path: Path,
    *,
    expected_seed: int,
    expected_split: str,
    expected_split_ids: set[str],
    expected_manifest_sha: str,
    expected_gpu_name: str,
    expected_winner_cfg: dict[str, Any],
) -> bool:
    from scripts.verify_campaign import sha256_file

    if not trial_json_path.is_file():
        return False
    rec = json.loads(trial_json_path.read_text(encoding="utf-8"))
    tcfg = rec.get("trial_config") or {}
    metrics = rec.get("metrics") or {}
    obs_gpu = rec.get("gpu_name") or rec.get("hardware")
    if (
        rec.get("status") != "PASS"
        or obs_gpu != expected_gpu_name
        or rec.get("dataset_manifest_sha256") != expected_manifest_sha
        or int(tcfg.get("seed", -1)) != expected_seed
        or int(metrics.get("eval_samples") or 0) != len(expected_split_ids)
        or metrics.get("ast_exact_match_accuracy") is None
        or metrics.get("evaluator_version") != "2.0.0-strict"
    ):
        return False

    for field in (
        "model_id",
        "model_revision",
        "backend",
        "group_settings",
        "explicit_target_modules",
        "training",
        "peft_method",
        "adalora_init_r",
        "adalora_target_r",
        "adalora_tinit",
        "adalora_tfinal",
        "adalora_delta_t",
    ):
        if tcfg.get(field) != expected_winner_cfg.get(field):
            return False

    pred_file = trial_json_path.parent / f"predictions_{expected_split}.jsonl"
    expected_sha = metrics.get("predictions_sha256")
    if not pred_file.is_file() or not expected_sha or expected_sha == "UNKNOWN":
        return False
    if sha256_file(pred_file) != expected_sha:
        return False

    parsed = [json.loads(ln) for ln in pred_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(parsed) != len(expected_split_ids):
        return False
    rec_ids = {str(r.get("sample_id")) for r in parsed}
    if rec_ids != expected_split_ids:
        return False
    expected_tid = trial_json_path.parent.name
    expected_arm = str(expected_winner_cfg.get("arm", ""))
    return all(
        r.get("split") == expected_split
        and r.get("evaluator_version") == "2.0.0-strict"
        and r.get("trial_id") == expected_tid
        and (not expected_arm or r.get("arm") == expected_arm)
        for r in parsed
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run/resume ALOE campaigns on Colab Pro+ A100")
    parser.add_argument("--model", required=True, help="Hybrid target model ID (Falcon-H1 or LFM3)")
    parser.add_argument(
        "--phase",
        choices=("search", "confirm", "final-test"),
        default="search",
        help="Execution phase: 20-trial search, 3-seed confirmation, or sealed 400-item test",
    )
    parser.add_argument(
        "--arms",
        default="aloe,b0,b1,b2,borobia_attn,borobia_ssm,borobia_ffn",
        help="Comma-separated arms to execute sequentially",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=125)
    parser.add_argument("--seeds", default="42,1337,2026", help="Comma-separated seeds for confirm/final-test")
    parser.add_argument("--pinned-gpu", default="A100", help="Substring required in CUDA GPU name")
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/aloe_ipd_v1",
        help="Persistent Google Drive folder for atomic resume & artifact backup",
    )
    parser.add_argument("--manifest", default="data/manifests/xlam_ipd_v1.json")
    parser.add_argument("--raw-data", default="data/raw/xlam/xlam_function_calling_60k.json")
    parser.add_argument("--confirm-sealed-test", action="store_true", help="Required token to unlock --phase final-test")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly reset FAILED trials to PENDING")
    parser.add_argument("--force-unlock", action="store_true")
    args = parser.parse_args()

    from aloe.evaluation.resources import enforce_hardware_contract
    from aloe.execution.campaign_runner import CampaignRunner, compute_exact_feasible_pool
    from aloe.execution.trial_config import TrainingParameters, TrialConfig
    from aloe.pipeline import inspect_to_grammar
    from aloe.tracking.trial_store import atomic_write_text, get_git_commit
    from scripts.verify_campaign import verify_campaign_dir, verify_manifest

    model_lower = args.model.lower()
    if "gemma" in model_lower:
        raise RuntimeError(
            "ADR-012 hardware-tier boundary: Gemma 4-E2B is assigned to local Apple Silicon MPS "
            "(`process_rss_sampled_100ms`), not Colab CUDA A100."
        )

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.phase in ("confirm", "final-test") and seed_list != REQUIRED_CONFIRM_SEEDS:
        raise ValueError(
            f"Protocol-v1 {args.phase} requires --seeds 42,1337,2026 in exact order; got {seed_list}"
        )
    if args.phase == "search" and (args.trials != 20 or args.max_steps != 125):
        raise ValueError(
            f"Official protocol-v1 search is fixed at --trials 20 --max-steps 125; got trials={args.trials}, max_steps={args.max_steps}"
        )
    if args.phase == "final-test" and not args.confirm_sealed_test:
        raise PermissionError(
            "Refusing --phase final-test without explicit --confirm-sealed-test after locking 3-seed winners."
        )

    manifest_audit = verify_manifest(
        Path(args.manifest),
        raw_data_path=Path(args.raw_data),
        enforce_frozen_v1=True,
        require_raw_data=True,
    )
    if not manifest_audit["raw_xlam_independently_verified"]:
        raise RuntimeError("Official protocol-v1 requires verified raw xLAM SHA-256 and tool-disjointness.")
    manifest_sha = str(manifest_audit["manifest_sha256"])
    split_id_sets: dict[str, set[str]] = manifest_audit["split_id_sets"]
    git_sha = get_git_commit(REPO_ROOT, forbid_dirty=True)

    drive_root = Path(args.drive_root)
    need_mamba = "falcon-h1" in model_lower
    need_causal_conv = need_mamba or ("lfm3" in model_lower)
    ensure_drive_wheels(
        drive_root / "wheels_cache",
        need_mamba=need_mamba,
        need_causal_conv=need_causal_conv,
    )

    desc, grammar = inspect_to_grammar(args.model, backend="auto")
    slug = desc.model_id.replace("/", "--")
    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    active_arms = [
        a for a in arm_list
        if not (a == "borobia_ssm" and not any(g.component_type.value == "ssm" for g in grammar.groups))
    ]
    parity_info = compute_exact_feasible_pool(desc, grammar, parity_tol=0.05)
    baseline_params = int(parity_info["baseline_params"])

    lease_file = drive_root / "campaigns" / slug / ".campaign_lease.json"
    hw_lock_file = drive_root / "campaigns" / slug / "hardware_lock.json"

    with acquire_model_lease(lease_file, phase=args.phase, force_unlock=args.force_unlock):
        prelocked_gpu: str | None = None
        if hw_lock_file.is_file():
            lock_meta = json.loads(hw_lock_file.read_text(encoding="utf-8"))
            if lock_meta.get("model_id") != desc.model_id:
                raise RuntimeError(f"Hardware lock model mismatch in {hw_lock_file}")
            prelocked_gpu = lock_meta.get("locked_gpu_name")

        hw = enforce_hardware_contract(
            args.model,
            "cuda",
            pinned_gpu=args.pinned_gpu,
            exact_gpu_name=prelocked_gpu,
            official_campaign=True,
        )
        exact_gpu_name = str(hw["gpu_name"])
        if not hw_lock_file.is_file():
            atomic_write_text(
                hw_lock_file,
                json.dumps(
                    {
                        "model_id": desc.model_id,
                        "locked_gpu_name": exact_gpu_name,
                        "locked_memory_metric_name": "cuda_max_memory_allocated",
                    },
                    indent=2,
                ),
            )
        print(f"[HARDWARE LOCKED] {hw['hardware_label']} | model_lock='{exact_gpu_name}' | git={git_sha}")

        if args.phase == "search":
            out_summary = drive_root / "campaigns" / slug / "model_all_arms_summary.json"
            merged_results: dict[str, object] = {}
            if out_summary.is_file():
                try:
                    merged_results = json.loads(out_summary.read_text(encoding="utf-8"))
                except Exception:
                    merged_results = {}

            for arm in active_arms:
                local_dir = REPO_ROOT / "runs" / "ipd_campaigns" / slug / arm
                mirror_dir = drive_root / "campaigns" / slug / arm
                print(f"\n=== [SEARCH] RUNNING / RESUMING {desc.model_id} :: arm={arm} ===")
                runner = CampaignRunner(
                    arch=desc,
                    grammar=grammar,
                    campaign_dir=local_dir,
                    arm=arm,
                    target_trials=args.trials,
                    manifest_path=args.manifest,
                    raw_path=args.raw_data,
                    base_training=TrainingParameters(max_steps=args.max_steps),
                    pinned_gpu=args.pinned_gpu,
                    exact_gpu_name=exact_gpu_name,
                    hardware_lock_path=hw_lock_file,
                    official_campaign=True,
                    forbid_dirty_git=True,
                    use_subprocess_worker=True,
                    mirror_dir=mirror_dir,
                )
                summary = runner.run_campaign(retry_failed=args.retry_failed)
                merged_results[arm] = summary
                atomic_write_text(out_summary, json.dumps(merged_results, indent=2))
                print(
                    f"[DONE] arm={arm} winner={summary['selected_winner_trial_id']} "
                    f"val_acc={summary['selected_winner_metrics']['tool_accuracy']:.2%}"
                )
            return 0

        # Load & independently audit locked search winners for all active arms
        winners_by_arm: dict[str, TrialConfig] = {}
        winner_sigs_by_arm: dict[str, str] = {}
        for arm in active_arms:
            arm_mirror_dir = drive_root / "campaigns" / slug / arm
            search_summary_file = arm_mirror_dir / "campaign_summary.json"
            if not search_summary_file.is_file():
                raise FileNotFoundError(f"Locked search summary missing for arm '{arm}': {search_summary_file}")
            sdata = json.loads(search_summary_file.read_text(encoding="utf-8"))
            if sdata.get("protocol_tag") != "protocol-v1":
                raise PermissionError(
                    f"Search summary for arm '{arm}' has protocol_tag='{sdata.get('protocol_tag')}', "
                    "not 'protocol-v1'; refusing confirmation/final evaluation on smoke summary."
                )
            verify_campaign_dir(
                arm_mirror_dir,
                expected_manifest_sha=manifest_sha,
                expected_split_ids=split_id_sets,
                allow_smoke_budget=False,
            )
            wcfg = TrialConfig.model_validate(sdata["selected_winner_config"])
            winners_by_arm[arm] = wcfg
            winner_sigs_by_arm[arm] = compute_full_winner_sha256(wcfg)

        unseal_path = drive_root / "final_sealed_test" / slug / "test_unsealed.json"
        if args.phase == "final-test":
            for arm, wcfg in winners_by_arm.items():
                locked_dump = wcfg.model_dump(mode="json")
                for c_seed in REQUIRED_CONFIRM_SEEDS:
                    c_tid = f"{arm}_winner_s{c_seed}_validation"
                    c_receipt = drive_root / "confirm_3seed" / slug / arm / "trials" / c_tid / "trial.json"
                    if not _verify_completed_trial_receipt(
                        c_receipt,
                        expected_seed=c_seed,
                        expected_split="validation",
                        expected_split_ids=split_id_sets["validation"],
                        expected_manifest_sha=manifest_sha,
                        expected_gpu_name=exact_gpu_name,
                        expected_winner_cfg=locked_dump,
                    ):
                        raise PermissionError(
                            f"Sealed test remains locked: missing or invalid 3-seed confirmation receipt {c_receipt}"
                        )

            merged_seal_sigs: dict[str, str] = {}
            if unseal_path.is_file():
                existing_seal = json.loads(unseal_path.read_text(encoding="utf-8"))
                merged_seal_sigs = dict(existing_seal.get("winner_signatures") or {})
                for arm, w_hash in winner_sigs_by_arm.items():
                    if arm in merged_seal_sigs and merged_seal_sigs[arm] != w_hash:
                        raise PermissionError(
                            f"Sealed test split was already unsealed for arm '{arm}' with hash "
                            f"{merged_seal_sigs[arm][:12]} != {w_hash[:12]}; refusing altered winner re-evaluation!"
                        )
            merged_seal_sigs.update(winner_sigs_by_arm)
            atomic_write_text(
                unseal_path,
                json.dumps(
                    {
                        "model_id": desc.model_id,
                        "unsealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "git_commit": git_sha,
                        "manifest_sha256": manifest_sha,
                        "locked_gpu_name": exact_gpu_name,
                        "winner_signatures": merged_seal_sigs,
                    },
                    indent=2,
                ),
            )

        eval_split = "validation" if args.phase == "confirm" else "test"
        expected_ids_for_phase = split_id_sets[eval_split]
        phase_dir_name = "confirm_3seed" if args.phase == "confirm" else "final_sealed_test"

        for arm, winner_cfg in winners_by_arm.items():
            locked_dump = winner_cfg.model_dump(mode="json")
            local_trials_dir = REPO_ROOT / "runs" / phase_dir_name / slug / arm / "trials"
            mirror_trials_dir = drive_root / phase_dir_name / slug / arm / "trials"
            local_trials_dir.mkdir(parents=True, exist_ok=True)

            for s_val in seed_list:
                tid = f"{arm}_winner_s{s_val}_{eval_split}"
                done_mirror = mirror_trials_dir / tid / "trial.json"
                if _verify_completed_trial_receipt(
                    done_mirror,
                    expected_seed=s_val,
                    expected_split=eval_split,
                    expected_split_ids=expected_ids_for_phase,
                    expected_manifest_sha=manifest_sha,
                    expected_gpu_name=exact_gpu_name,
                    expected_winner_cfg=locked_dump,
                ):
                    print(f"[VERIFIED DONE] {tid}")
                    continue

                seeded_cfg = winner_cfg.model_copy(
                    update={
                        "trial_id": tid,
                        "arm": arm,
                        "seed": s_val,
                        "dataset_manifest_path": args.manifest,
                        "dataset_raw_path": args.raw_data,
                    }
                )
                cfg_path = local_trials_dir / tid / "worker_trial_config.json"
                seeded_cfg.to_file(cfg_path)
                cmd = [
                    sys.executable,
                    "-m",
                    "aloe.execution.worker_entry",
                    "--config",
                    str(cfg_path),
                    "--trials-dir",
                    str(local_trials_dir),
                    "--eval-split",
                    eval_split,
                    "--parity-tol",
                    "0.05",
                    "--baseline-params",
                    str(baseline_params),
                    "--pinned-gpu",
                    args.pinned_gpu,
                    "--exact-gpu-name",
                    exact_gpu_name,
                    "--official-campaign",
                    "--forbid-dirty-git",
                    "--mirror-dir",
                    str(mirror_trials_dir),
                ]
                if seeded_cfg.peft_method == "lora":
                    cmd.append("--enforce-parity")
                if eval_split == "test":
                    cmd.extend(["--final", "--seal-file", str(unseal_path)])

                print(f"=== [{args.phase.upper()}] {desc.model_id} :: {tid} ===")
                subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

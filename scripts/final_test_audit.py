#!/usr/bin/env python3
"""Strict Outer Seal, Trial Receipt, Zero-Shot JSONL, Attempt-Log, & 63-Receipt Auditor (`final_test_audit.py`).

Enforces end-to-end Protocol-v1 (`IPD-PROTOCOL-V1`) pre-unseal and post-unseal invariants without
modifying the locked model runtime commits (`8187255cae07`, `f0d0ea2b56b4`, `ab9877ce5110`):
1. Read-only `test_unsealed.json` verification (`verify_immutable_model_seal`):
   - Requires an explicit 64-hex `expected_seal_sha256` (workers NEVER create, merge, or rewrite seals).
   - Binds `model_id`, `model_revision`, `git_commit`, `manifest_sha256`, `locked_gpu_name`,
     `locked_memory_metric_name`, `allowed_seeds == [42, 1337, 2026]`, `max_steps == 125`,
     `max_new_tokens == 192`, the COMPLETE frozen arm set (`6` Gemma, `7` Falcon, `7` LFM3),
     and all 8 frozen orchestration/statistical artifact SHA-256 digests.
2. Full-hash trial receipt verification (`verify_strict_trial_receipt`):
   - Compares `compute_full_winner_sha256(tcfg)` against the locked winner config hash.
   - Enforces `status == "PASS"`, `optimizer_steps_completed == 125`, `fresh_process_reload is True`,
     `git_commit`, `model_revision`, `gpu_name`, `memory_metric_name`, `dataset_manifest_sha256`,
     positive finite `trainable_params` and `search_memory_mb`, LoRA `is_budget_matched_5pct is True`
     (or AdaLoRA positive `adalora_init_params` + `adalora_final_active_params`), JSON boolean
     `exact_match`, and finite recomputed EM matching `ast_exact_match_accuracy` and `tool_accuracy`.
3. Strict zero-shot receipt & JSONL verification (`verify_strict_zero_shot_receipt`):
   - Verifies `zero_shot_<split>.json` + `predictions_<split>.jsonl` SHA-256, exact split IDs,
     `trainable_parameters == 0`, memory metric & finite memory, boolean `exact_match`, and finite EM.
4. Global 63-receipt & attempt-log final-test gate (`verify_complete_63_final_test_gate`):
   - Requires all `20` frozen winners x `3` seeds (`60` receipts) + `3` hardware-matched
     `zero_shot_test` receipts = `63` verified receipts, plus append-only `final_test_attempts/`
     audit (`completed == 1` per run, disclosing any `launched > 1` infrastructure retries).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloe.evaluation.tool_calling import EVALUATOR_VERSION  # noqa: E402
from aloe.execution.trial_config import TrialConfig  # noqa: E402
from scripts.run_colab_a100_campaign import REQUIRED_CONFIRM_SEEDS, compute_full_winner_sha256  # noqa: E402
from scripts.verify_campaign import sha256_file, verify_campaign_dir  # noqa: E402

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

WORKER_SHIM_SOURCE = (
    "from __future__ import annotations\n"
    "import copy\n"
    "try:\n"
    "    from aloe.execution.backends.peft_mps import PeftMPSBackend\n"
    "    _orig_vg = PeftMPSBackend.verify_gradients\n"
    "    if not getattr(_orig_vg, '_aloe_bf16_probe_capped', False):\n"
    "        def _safe_vg(self, *args, **kwargs):\n"
    "            if not kwargs.get('init_lora_weights', False) and kwargs.get('peft_config_override') is not None:\n"
    "                pc = copy.deepcopy(kwargs['peft_config_override'])\n"
    "                if hasattr(pc, 'lora_alpha') and pc.lora_alpha is not None and pc.lora_alpha > 2:\n"
    "                    pc.lora_alpha = 2\n"
    "                if getattr(pc, 'alpha_pattern', None):\n"
    "                    pc.alpha_pattern = {k: min(v, 2) for k, v in pc.alpha_pattern.items()}\n"
    "                kwargs = dict(kwargs)\n"
    "                kwargs['peft_config_override'] = pc\n"
    "                if 'lora_alpha' in kwargs and kwargs['lora_alpha'] is not None and kwargs['lora_alpha'] > 2:\n"
    "                    kwargs['lora_alpha'] = 2\n"
    "            return _orig_vg(self, *args, **kwargs)\n"
    "        _safe_vg._aloe_bf16_probe_capped = True\n"
    "        PeftMPSBackend.verify_gradients = _safe_vg\n"
    "except Exception:\n"
    "    pass\n"
)

FROZEN_SEAL_HASH_KEYS = (
    "colab_bootstrap_sha256",
    "mps_bootstrap_sha256",
    "bootstrap_sha256",
    "shim_sha256",
    "analysis_plan_sha256",
    "audit_script_sha256",
    "evaluator_script_sha256",
    "bootstrap_engine_sha256",
)

MULTISEED_CUDA_SLUGS = (
    "tiiuae--Falcon-H1-1.5B-Instruct",
    "LiquidAI--LFM3-1.7B",
)
MULTISEED_SEARCH_SEEDS = (1337, 2026)
MULTISEED_PAIRED_ARMS = ("aloe", "b1")

CANONICAL_PROTOCOL_MODELS: dict[str, dict[str, Any]] = {
    "google--gemma-4-E2B-it": {
        "model_id": "google/gemma-4-E2B-it",
        "model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "git_commit": "8187255cae076f583dce455acaf5019b20bad1e4",
        "locked_gpu_name": "Apple M1 Max",
        "locked_memory_metric_name": "process_rss_sampled_100ms",
        "arms": ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ffn"),
        "baseline_params": 12079104,
    },
    "tiiuae--Falcon-H1-1.5B-Instruct": {
        "model_id": "tiiuae/Falcon-H1-1.5B-Instruct",
        "model_revision": "80ebc50d7799a440b96c93bb6686a3924a09b0cb",
        "git_commit": "f0d0ea2b56b441e18d475366ea249617afe65883",
        "locked_gpu_name": "NVIDIA L4",
        "locked_memory_metric_name": "cuda_max_memory_allocated",
        "arms": ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ssm", "borobia_ffn"),
        "baseline_params": 7578624,
    },
    "LiquidAI--LFM3-1.7B": {
        "model_id": "LiquidAI/LFM3-1.7B",
        "model_revision": "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "git_commit": "ab9877ce5110f3889d2d2f028a4d3ff559bd48b2",
        "locked_gpu_name": "NVIDIA L4",
        "locked_memory_metric_name": "cuda_max_memory_allocated",
        "arms": ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ssm", "borobia_ffn"),
        "baseline_params": 8052736,
    },
}


def compute_frozen_code_digests(
    repo_root: Path = REPO_ROOT,
    *,
    analysis_plan_bytes: bytes | None = None,
    shim_path: Path | None = None,
) -> dict[str, str]:
    """Compute the 8 frozen orchestration, shim, plan, auditor, and statistical engine SHA-256 digests."""
    colab_sha = sha256_file(repo_root / "scripts" / "run_colab_shard.py")
    mps_sha = sha256_file(repo_root / "scripts" / "run_local_mps_final.py")
    combo_sha = hashlib.sha256(f"colab:{colab_sha}|mps:{mps_sha}".encode("utf-8")).hexdigest()
    shim_sha = (
        sha256_file(shim_path)
        if shim_path is not None
        else hashlib.sha256(WORKER_SHIM_SOURCE.encode("utf-8")).hexdigest()
    )
    plan_sha = (
        hashlib.sha256(analysis_plan_bytes).hexdigest()
        if analysis_plan_bytes is not None
        else sha256_file(repo_root / "docs" / "PRE_UNSEAL_ANALYSIS_PLAN.md")
    )
    audit_sha = sha256_file(repo_root / "scripts" / "final_test_audit.py")
    eval_sha = sha256_file(repo_root / "scripts" / "evaluate_final_test_suite.py")
    boot_engine_sha = sha256_file(repo_root / "aloe" / "evaluation" / "bootstrap.py")
    return {
        "colab_bootstrap_sha256": colab_sha,
        "mps_bootstrap_sha256": mps_sha,
        "bootstrap_sha256": combo_sha,
        "shim_sha256": shim_sha,
        "analysis_plan_sha256": plan_sha,
        "audit_script_sha256": audit_sha,
        "evaluator_script_sha256": eval_sha,
        "bootstrap_engine_sha256": boot_engine_sha,
    }


def load_ceremony_pins(pins_path: Path | None = None) -> dict[str, Any]:
    target = pins_path or (REPO_ROOT / "docs" / "CEREMONY_PINS.json")
    if not target.is_file():
        return {"hf_seal_commit_oid": "", "seal_sha256_by_model": {}}
    return json.loads(target.read_text(encoding="utf-8"))


def _is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _is_pos_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val > 0


def verify_immutable_model_seal(
    seal_path: Path,
    *,
    slug: str,
    manifest_sha: str,
    campaigns_model_dir: Path,
    expected_split_ids: dict[str, set[str]],
    expected_seal_sha256: str,
    verify_code_digests: bool = True,
    repo_root: Path = REPO_ROOT,
    shim_path: Path | None = None,
    frozen_plan_path: Path | None = None,
) -> dict[str, Any]:
    """Verify an immutable pre-committed `test_unsealed.json` against the complete frozen arm set & code digests.

    Raises `PermissionError` on missing/empty `expected_seal_sha256`, missing seal, subset-arm seal,
    post-unseal arm addition, metadata mismatch, winner-hash mismatch, or frozen code/plan digest mismatch.
    Never modifies `seal_path`.
    """
    if slug not in CANONICAL_PROTOCOL_MODELS:
        raise PermissionError(f"Non-canonical model slug '{slug}' cannot unseal official test split.")
    spec = CANONICAL_PROTOCOL_MODELS[slug]

    if not expected_seal_sha256 or not HEX64_RE.fullmatch(expected_seal_sha256):
        raise PermissionError(
            f"Approved 64-char ceremony expected_seal_sha256 is required for {slug}; got {expected_seal_sha256!r}"
        )
    if not seal_path.is_file():
        raise PermissionError(
            f"Missing pre-committed read-only seal {seal_path}; workers are forbidden from self-issuing seals."
        )
    actual_seal_sha = sha256_file(seal_path)
    if actual_seal_sha != expected_seal_sha256:
        raise PermissionError(
            f"Seal SHA-256 mismatch at {seal_path}: got {actual_seal_sha}, expected {expected_seal_sha256}"
        )

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    required_fields = {
        "protocol_tag": "protocol-v1",
        "model_id": spec["model_id"],
        "model_revision": spec["model_revision"],
        "git_commit": spec["git_commit"],
        "manifest_sha256": manifest_sha,
        "locked_gpu_name": spec["locked_gpu_name"],
        "locked_memory_metric_name": spec["locked_memory_metric_name"],
        "allowed_seeds": REQUIRED_CONFIRM_SEEDS,
        "max_steps": 125,
        "max_new_tokens": 192,
    }
    for k, expected_val in required_fields.items():
        if seal.get(k) != expected_val:
            raise PermissionError(
                f"Seal metadata mismatch on '{k}' in {seal_path}: got {seal.get(k)!r}, expected {expected_val!r}"
            )

    for hk in FROZEN_SEAL_HASH_KEYS:
        hv = seal.get(hk)
        if not isinstance(hv, str) or not HEX64_RE.fullmatch(hv):
            raise PermissionError(f"Seal missing valid 64-hex frozen digest '{hk}' in {seal_path}: {hv!r}")

    expected_combo = hashlib.sha256(
        f"colab:{seal['colab_bootstrap_sha256']}|mps:{seal['mps_bootstrap_sha256']}".encode("utf-8")
    ).hexdigest()
    if seal["bootstrap_sha256"] != expected_combo:
        raise PermissionError(f"Seal bootstrap_sha256 inconsistent with colab/mps digests in {seal_path}")

    if verify_code_digests:
        live_digests = compute_frozen_code_digests(repo_root, shim_path=shim_path)
        for hk, live_val in live_digests.items():
            if seal.get(hk) != live_val:
                raise PermissionError(
                    f"Frozen artifact SHA-256 mismatch on '{hk}' for {slug}: "
                    f"seal={seal.get(hk)!r} != live={live_val!r}"
                )
    if frozen_plan_path is not None:
        if not frozen_plan_path.is_file():
            raise PermissionError(f"Missing published frozen analysis plan: {frozen_plan_path}")
        if sha256_file(frozen_plan_path) != seal["analysis_plan_sha256"]:
            raise PermissionError(
                f"Published analysis plan SHA-256 mismatch at {frozen_plan_path} against {seal_path}"
            )

    expected_arms = set(spec["arms"])
    seal_sigs = seal.get("winner_signatures")
    if not isinstance(seal_sigs, dict) or set(seal_sigs.keys()) != expected_arms:
        raise PermissionError(
            f"Seal winner_signatures in {seal_path} must match complete frozen arm set {sorted(expected_arms)}; "
            f"got {sorted(seal_sigs.keys()) if isinstance(seal_sigs, dict) else type(seal_sigs).__name__}"
        )

    winners_by_arm: dict[str, TrialConfig] = {}
    recomputed_sigs: dict[str, str] = {}
    for arm in spec["arms"]:
        arm_dir = campaigns_model_dir / arm
        sfile = arm_dir / "campaign_summary.json"
        if not sfile.is_file():
            raise PermissionError(f"Missing locked campaign_summary.json for {slug}/{arm}: {sfile}")
        verify_campaign_dir(
            arm_dir,
            expected_manifest_sha=manifest_sha,
            expected_split_ids=expected_split_ids,
            allow_smoke_budget=False,
        )
        sdata = json.loads(sfile.read_text(encoding="utf-8"))
        if sdata.get("protocol_tag") != "protocol-v1":
            raise PermissionError(f"Invalid protocol_tag in {sfile}: {sdata.get('protocol_tag')!r}")
        wcfg = TrialConfig.model_validate(sdata["selected_winner_config"])
        winners_by_arm[arm] = wcfg
        recomputed_sigs[arm] = compute_full_winner_sha256(wcfg)

    if seal_sigs != recomputed_sigs:
        raise PermissionError(
            f"Seal winner_signatures mismatch against locked campaign_summary.json in {seal_path}!"
        )

    return {
        "seal_sha256": actual_seal_sha,
        "seal": seal,
        "winners_by_arm": winners_by_arm,
        "winner_signatures": recomputed_sigs,
    }


def verify_strict_trial_receipt(
    trial_json_path: Path,
    *,
    expected_seed: int,
    expected_split: str,
    expected_split_ids: set[str],
    expected_manifest_sha: str,
    expected_git_sha: str,
    expected_model_id: str,
    expected_model_revision: str,
    expected_gpu_name: str,
    expected_memory_metric: str,
    expected_winner_cfg: dict[str, Any] | TrialConfig,
    raise_on_error: bool = False,
) -> bool:
    """Strictly verify a confirmation or final-test `trial.json` receipt + `predictions_<split>.jsonl`."""
    def _fail(msg: str) -> bool:
        if raise_on_error:
            raise ValueError(f"[{trial_json_path}] {msg}")
        return False

    if not trial_json_path.is_file():
        return _fail("trial.json file missing")

    rec = json.loads(trial_json_path.read_text(encoding="utf-8"))
    tcfg = rec.get("trial_config") or {}
    metrics = rec.get("metrics") or {}
    obs_gpu = rec.get("gpu_name") or rec.get("hardware")

    if rec.get("status") != "PASS":
        return _fail(f"status={rec.get('status')!r} != 'PASS'")
    if rec.get("git_commit") != expected_git_sha:
        return _fail(f"git_commit={rec.get('git_commit')!r} != {expected_git_sha!r}")
    if rec.get("model_id") != expected_model_id:
        return _fail(f"model_id={rec.get('model_id')!r} != {expected_model_id!r}")
    if rec.get("model_revision") != expected_model_revision:
        return _fail(f"model_revision={rec.get('model_revision')!r} != {expected_model_revision!r}")
    if obs_gpu != expected_gpu_name:
        return _fail(f"gpu_name={obs_gpu!r} != {expected_gpu_name!r}")
    if metrics.get("memory_metric_name") != expected_memory_metric:
        return _fail(f"memory_metric_name={metrics.get('memory_metric_name')!r} != {expected_memory_metric!r}")
    if rec.get("dataset_manifest_sha256") != expected_manifest_sha:
        return _fail("dataset_manifest_sha256 mismatch")
    if tcfg.get("seed") != expected_seed or isinstance(tcfg.get("seed"), bool):
        return _fail(f"seed={tcfg.get('seed')!r} != {expected_seed}")
    if metrics.get("optimizer_steps_completed") != 125 or isinstance(metrics.get("optimizer_steps_completed"), bool):
        return _fail(f"optimizer_steps_completed={metrics.get('optimizer_steps_completed')!r} != 125")
    if metrics.get("fresh_process_reload") is not True:
        return _fail(f"fresh_process_reload={metrics.get('fresh_process_reload')!r} != True")
    if metrics.get("eval_samples") != len(expected_split_ids) or isinstance(metrics.get("eval_samples"), bool):
        return _fail(f"eval_samples={metrics.get('eval_samples')!r} != {len(expected_split_ids)}")
    if metrics.get("evaluator_version") != EVALUATOR_VERSION:
        return _fail(f"evaluator_version={metrics.get('evaluator_version')!r} != {EVALUATOR_VERSION!r}")

    t_params = metrics.get("trainable_params")
    if not (_is_finite_number(t_params) and int(t_params) > 0 and float(t_params) == float(int(t_params))):
        return _fail(f"invalid trainable_params={t_params!r}")

    if compute_full_winner_sha256(tcfg) != compute_full_winner_sha256(expected_winner_cfg):
        return _fail("full winner configuration SHA-256 mismatch")

    peft_method = str(tcfg.get("peft_method", "lora"))
    if peft_method == "lora":
        if metrics.get("is_budget_matched_5pct") is not True:
            return _fail(f"LoRA is_budget_matched_5pct={metrics.get('is_budget_matched_5pct')!r} != True")
    elif peft_method == "adalora":
        if not (_is_pos_int(metrics.get("adalora_init_params")) and _is_pos_int(metrics.get("adalora_final_active_params"))):
            return _fail("AdaLoRA receipt missing positive int adalora_init_params / adalora_final_active_params")
    else:
        return _fail(f"unsupported peft_method={peft_method!r}")

    mem_raw = metrics.get("search_memory_mb")
    if not (_is_finite_number(mem_raw) and float(mem_raw) > 0.0):
        return _fail(f"invalid search_memory_mb={mem_raw!r}")

    pred_file = trial_json_path.parent / f"predictions_{expected_split}.jsonl"
    expected_pred_sha = metrics.get("predictions_sha256")
    if not pred_file.is_file() or not expected_pred_sha or expected_pred_sha == "UNKNOWN":
        return _fail("missing predictions JSONL or predictions_sha256")
    if sha256_file(pred_file) != expected_pred_sha:
        return _fail("predictions JSONL SHA-256 mismatch")

    rows = [json.loads(ln) for ln in pred_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(rows) != len(expected_split_ids) or len(rows) == 0:
        return _fail(f"len(rows)={len(rows)} != {len(expected_split_ids)}")
    rec_ids = {str(r.get("sample_id")) for r in rows}
    if rec_ids != expected_split_ids:
        return _fail("prediction sample_id set mismatch against frozen manifest split")

    expected_tid = trial_json_path.parent.name
    w_dump = expected_winner_cfg.model_dump(mode="json") if hasattr(expected_winner_cfg, "model_dump") else dict(expected_winner_cfg)
    expected_arm = str(w_dump.get("arm", ""))

    for r in rows:
        if type(r.get("exact_match")) is not bool:
            return _fail(f"exact_match must be a JSON boolean in sample_id={r.get('sample_id')!r}")
        if (
            r.get("split") != expected_split
            or r.get("evaluator_version") != EVALUATOR_VERSION
            or r.get("trial_id") != expected_tid
            or r.get("model_id") != expected_model_id
            or r.get("model_revision") != expected_model_revision
            or r.get("seed") != expected_seed
            or isinstance(r.get("seed"), bool)
            or (expected_arm and r.get("arm") != expected_arm)
        ):
            return _fail(f"row-level metadata mismatch in sample_id={r.get('sample_id')!r}")

    recomp_em = sum(1 for r in rows if r["exact_match"]) / float(len(rows))
    rep_em = metrics.get("ast_exact_match_accuracy")
    rep_tool = metrics.get("tool_accuracy")
    if not all(
        _is_finite_number(v) and 0.0 <= float(v) <= 1.0 and abs(recomp_em - float(v)) <= 1e-6
        for v in (rep_em, rep_tool)
    ):
        return _fail(f"invalid or inconsistent reported accuracy ({rep_em!r} / {rep_tool!r}) vs recomputed {recomp_em:.6f}")

    return True


def verify_strict_zero_shot_receipt(
    receipt_path: Path,
    *,
    expected_split: str,
    expected_split_ids: set[str],
    expected_manifest_sha: str,
    expected_git_sha: str,
    expected_model_id: str,
    expected_model_revision: str,
    expected_gpu_name: str,
    expected_memory_metric: str | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any] | None:
    """Strictly verify `zero_shot_<split>.json` and its `predictions_<split>.jsonl`."""
    def _fail(msg: str) -> None:
        if raise_on_error:
            raise ValueError(f"[{receipt_path}] {msg}")
        return None

    if not receipt_path.is_file():
        return _fail("zero-shot receipt file missing")
    rec = json.loads(receipt_path.read_text(encoding="utf-8"))
    if rec.get("protocol_tag") != "protocol-v1":
        return _fail(f"protocol_tag={rec.get('protocol_tag')!r} != 'protocol-v1'")
    if rec.get("arm") != "zero_shot":
        return _fail(f"arm={rec.get('arm')!r} != 'zero_shot'")
    if rec.get("split") != expected_split:
        return _fail(f"split={rec.get('split')!r} != {expected_split!r}")
    if rec.get("model_id") != expected_model_id:
        return _fail(f"model_id={rec.get('model_id')!r} != {expected_model_id!r}")
    if rec.get("model_revision") != expected_model_revision:
        return _fail(f"model_revision={rec.get('model_revision')!r} != {expected_model_revision!r}")
    if rec.get("git_commit") != expected_git_sha:
        return _fail(f"git_commit={rec.get('git_commit')!r} != {expected_git_sha!r}")
    if rec.get("gpu_name") != expected_gpu_name:
        return _fail(f"gpu_name={rec.get('gpu_name')!r} != {expected_gpu_name!r}")
    if rec.get("manifest_sha256") != expected_manifest_sha:
        return _fail("manifest_sha256 mismatch")
    if rec.get("trainable_parameters") != 0 or isinstance(rec.get("trainable_parameters"), bool):
        return _fail(f"trainable_parameters={rec.get('trainable_parameters')!r} != 0")
    if rec.get("sample_count") != len(expected_split_ids) or isinstance(rec.get("sample_count"), bool):
        return _fail(f"sample_count={rec.get('sample_count')!r} != {len(expected_split_ids)}")
    if rec.get("evaluator_version") != EVALUATOR_VERSION:
        return _fail(f"evaluator_version={rec.get('evaluator_version')!r} != {EVALUATOR_VERSION!r}")

    slug = expected_model_id.replace("/", "--")
    canon_mem_metric = expected_memory_metric or CANONICAL_PROTOCOL_MODELS.get(slug, {}).get("locked_memory_metric_name")
    if expected_split == "test" or "memory_metric_name" in rec or expected_gpu_name == "Apple M1 Max":
        if canon_mem_metric and rec.get("memory_metric_name") != canon_mem_metric:
            return _fail(f"memory_metric_name={rec.get('memory_metric_name')!r} != {canon_mem_metric!r}")
        if not (_is_finite_number(rec.get("search_memory_mb")) and float(rec["search_memory_mb"]) > 0.0):
            return _fail(f"invalid search_memory_mb={rec.get('search_memory_mb')!r}")
    else:
        # Legacy CUDA validation zero-shot receipts recorded peak_cuda_allocated_mb before memory_metric_name was added
        if expected_gpu_name != "NVIDIA L4" or not (
            _is_finite_number(rec.get("peak_cuda_allocated_mb")) and float(rec["peak_cuda_allocated_mb"]) > 0.0
        ):
            return _fail(f"invalid peak_cuda_allocated_mb={rec.get('peak_cuda_allocated_mb')!r}")

    pred_path = receipt_path.parent / f"predictions_{expected_split}.jsonl"
    exp_sha = rec.get("predictions_sha256")
    if not pred_path.is_file() or not exp_sha or sha256_file(pred_path) != exp_sha:
        return _fail("predictions JSONL missing or SHA-256 mismatch")

    rows = [json.loads(ln) for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(rows) != len(expected_split_ids) or len(rows) == 0:
        return _fail(f"len(rows)={len(rows)} != {len(expected_split_ids)}")
    if {str(r.get("sample_id")) for r in rows} != expected_split_ids:
        return _fail("zero-shot prediction sample_id set mismatch")

    expected_tid = f"zero_shot_{slug}_{expected_split}"
    for r in rows:
        if type(r.get("exact_match")) is not bool:
            return _fail(f"exact_match must be a JSON boolean at sample_id={r.get('sample_id')!r}")
        if (
            r.get("split") != expected_split
            or r.get("arm") != "zero_shot"
            or r.get("trial_id") != expected_tid
            or r.get("model_id") != expected_model_id
            or r.get("model_revision") != expected_model_revision
            or r.get("seed") != 42
            or isinstance(r.get("seed"), bool)
            or r.get("evaluator_version") != EVALUATOR_VERSION
        ):
            return _fail(f"row metadata mismatch at sample_id={r.get('sample_id')!r}")

    recomp_em = sum(1 for r in rows if r["exact_match"]) / float(len(rows))
    rep_em = rec.get("exact_match_accuracy")
    if not (_is_finite_number(rep_em) and 0.0 <= float(rep_em) <= 1.0 and abs(recomp_em - float(rep_em)) <= 1e-6):
        return _fail(f"invalid or inconsistent reported EM ({rep_em!r}) vs recomputed ({recomp_em:.6f})")

    return rec


def audit_final_test_attempts(
    dest_dir: Path,
    *,
    expected_seal_shas: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Audit append-only `final_test_attempts/<slug>/*.json` across all 63 canonical final-test run IDs."""
    expected_runs: dict[str, set[str]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    records_by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for slug, spec in CANONICAL_PROTOCOL_MODELS.items():
        rset = {f"zero_shot_{slug}_test"}
        for arm in spec["arms"]:
            for s_val in REQUIRED_CONFIRM_SEEDS:
                rset.add(f"{arm}_winner_s{s_val}_test")
        expected_runs[slug] = rset
        counts[slug] = {rid: {"LAUNCHED": 0, "COMPLETED": 0, "FAILED": 0} for rid in sorted(rset)}
        records_by_run[slug] = {rid: [] for rid in sorted(rset)}

    unexpected_records: list[str] = []
    duplicate_completed: list[str] = []
    chronology_or_linkage_errors: list[str] = []
    retried_runs: list[dict[str, Any]] = []
    missing_completed: list[str] = []
    total_attempt_files = 0

    attempts_root = dest_dir / "final_test_attempts"
    if attempts_root.is_dir():
        for slug_dir in sorted(p for p in attempts_root.iterdir() if p.is_dir()):
            slug = slug_dir.name
            if slug not in expected_runs:
                unexpected_records.append(f"non-canonical attempt slug dir: {slug}")
                continue
            exp_seal = str((expected_seal_shas or {}).get(slug, ""))
            for jf in sorted(slug_dir.glob("*.json")):
                total_attempt_files += 1
                try:
                    adata = json.loads(jf.read_text(encoding="utf-8"))
                except Exception as exc:
                    unexpected_records.append(f"corrupt attempt file {jf.name}: {exc}")
                    continue
                rid = str(adata.get("run_id", ""))
                stg = str(adata.get("stage", ""))
                aid = str(adata.get("attempt_id", "")).strip()
                utc = str(adata.get("utc", "")).strip()
                rec_seal = str(adata.get("seal_sha256", "")).strip()
                if (
                    adata.get("slug") != slug
                    or rid not in expected_runs[slug]
                    or stg not in ("LAUNCHED", "COMPLETED", "FAILED")
                    or not aid
                    or not utc
                ):
                    unexpected_records.append(
                        f"invalid attempt record {slug}/{jf.name}: run_id={rid!r} stage={stg!r} attempt_id={aid!r}"
                    )
                    continue
                if exp_seal and rec_seal != exp_seal:
                    unexpected_records.append(
                        f"attempt seal_sha256 mismatch in {slug}/{jf.name}: got {rec_seal!r}, expected {exp_seal!r}"
                    )
                    continue
                counts[slug][rid][stg] += 1
                seq_val = int(adata.get("seq", 1 if stg == "LAUNCHED" else 2))
                records_by_run[slug][rid].append(
                    {
                        "attempt_id": aid,
                        "stage": stg,
                        "utc": utc,
                        "seq": seq_val,
                        "filename": jf.name,
                    }
                )

    for slug, rmap in counts.items():
        for rid, cmap in rmap.items():
            if cmap["COMPLETED"] > 1:
                duplicate_completed.append(f"{slug}/{rid} has {cmap['COMPLETED']} COMPLETED records")
            if cmap["COMPLETED"] != 1 or cmap["LAUNCHED"] < 1:
                missing_completed.append(
                    f"{slug}/{rid} attempt counts launched={cmap['LAUNCHED']} completed={cmap['COMPLETED']} failed={cmap['FAILED']}"
                )
            if cmap["LAUNCHED"] > 1:
                retried_runs.append({"slug": slug, "run_id": rid, **cmap})

            rlist = sorted(records_by_run[slug][rid], key=lambda x: (x["utc"], x["seq"], x["filename"]))
            if rlist:
                by_aid: dict[str, list[str]] = {}
                seen_completed = False
                for item in rlist:
                    if seen_completed and item["stage"] == "LAUNCHED":
                        chronology_or_linkage_errors.append(
                            f"{slug}/{rid} recorded LAUNCHED ({item['attempt_id']}) after COMPLETED"
                        )
                    if item["stage"] == "COMPLETED":
                        seen_completed = True
                    by_aid.setdefault(item["attempt_id"], []).append(item["stage"])

                for aid, stages in by_aid.items():
                    if stages[0] != "LAUNCHED" or stages.count("LAUNCHED") != 1 or len(stages) != 2 or stages[1] not in ("COMPLETED", "FAILED"):
                        chronology_or_linkage_errors.append(
                            f"{slug}/{rid} unlinked or unclosed attempt_id={aid!r} stages={stages}"
                        )

    return {
        "total_attempt_files": total_attempt_files,
        "attempt_counts_by_model": counts,
        "retried_runs": retried_runs,
        "duplicate_completed": duplicate_completed,
        "chronology_or_linkage_errors": chronology_or_linkage_errors,
        "unexpected_records": unexpected_records,
        "missing_completed": missing_completed,
    }


def verify_complete_63_final_test_gate(
    dest_dir: Path,
    *,
    manifest_sha: str,
    split_id_sets: dict[str, set[str]],
    expected_seal_shas: dict[str, str] | None = None,
    require_all_63: bool = True,
    require_attempt_logs: bool = True,
    verify_code_digests: bool = True,
) -> dict[str, Any]:
    """Audit `final_sealed_test/` + `zero_shot_test/` + `final_test_attempts/` across all 3 models (63 runs)."""
    test_ids = split_id_sets["test"]
    pins = expected_seal_shas if expected_seal_shas is not None else load_ceremony_pins().get("seal_sha256_by_model", {})
    frozen_plan = dest_dir / "final_sealed_test" / "PRE_UNSEAL_ANALYSIS_PLAN.md"

    models_out: dict[str, Any] = {}
    verified_winner_runs = 0
    verified_zero_shot_runs = 0
    errors: list[str] = []

    for slug, spec in CANONICAL_PROTOCOL_MODELS.items():
        seal_path = dest_dir / "final_sealed_test" / slug / "test_unsealed.json"
        campaigns_dir = dest_dir / "campaigns" / slug
        model_entry: dict[str, Any] = {
            "seal_present": seal_path.is_file(),
            "seal_verified": False,
            "zero_shot_test_verified": False,
            "arms": {},
        }

        winners_by_arm: dict[str, TrialConfig] = {}
        if seal_path.is_file():
            try:
                seal_info = verify_immutable_model_seal(
                    seal_path,
                    slug=slug,
                    manifest_sha=manifest_sha,
                    campaigns_model_dir=campaigns_dir,
                    expected_split_ids=split_id_sets,
                    expected_seal_sha256=str((pins or {}).get(slug, "")),
                    verify_code_digests=verify_code_digests,
                    frozen_plan_path=frozen_plan if frozen_plan.is_file() or require_all_63 else None,
                )
                model_entry["seal_verified"] = True
                model_entry["seal_sha256"] = seal_info["seal_sha256"]
                winners_by_arm = seal_info["winners_by_arm"]
            except Exception as exc:
                errors.append(f"{slug} seal error: {exc}")
        else:
            errors.append(f"{slug} missing seal: {seal_path}")

        zs_test_path = dest_dir / "zero_shot_test" / slug / "zero_shot_test.json"
        if model_entry["seal_verified"] and zs_test_path.is_file():
            try:
                zs_rec = verify_strict_zero_shot_receipt(
                    zs_test_path,
                    expected_split="test",
                    expected_split_ids=test_ids,
                    expected_manifest_sha=manifest_sha,
                    expected_git_sha=spec["git_commit"],
                    expected_model_id=spec["model_id"],
                    expected_model_revision=spec["model_revision"],
                    expected_gpu_name=spec["locked_gpu_name"],
                    expected_memory_metric=spec["locked_memory_metric_name"],
                    raise_on_error=True,
                )
                if zs_rec is not None:
                    model_entry["zero_shot_test_verified"] = True
                    model_entry["zero_shot_exact_match"] = zs_rec["exact_match_accuracy"]
                    verified_zero_shot_runs += 1
            except Exception as exc:
                errors.append(f"{slug} zero_shot_test error: {exc}")
        else:
            errors.append(f"{slug} zero_shot_test missing or seal unverified")

        for arm in spec["arms"]:
            verified_seeds: list[int] = []
            if model_entry["seal_verified"] and arm in winners_by_arm:
                wcfg = winners_by_arm[arm]
                for s_val in REQUIRED_CONFIRM_SEEDS:
                    tid = f"{arm}_winner_s{s_val}_test"
                    tpath = dest_dir / "final_sealed_test" / slug / arm / "trials" / tid / "trial.json"
                    if tpath.is_file():
                        try:
                            ok = verify_strict_trial_receipt(
                                tpath,
                                expected_seed=s_val,
                                expected_split="test",
                                expected_split_ids=test_ids,
                                expected_manifest_sha=manifest_sha,
                                expected_git_sha=spec["git_commit"],
                                expected_model_id=spec["model_id"],
                                expected_model_revision=spec["model_revision"],
                                expected_gpu_name=spec["locked_gpu_name"],
                                expected_memory_metric=spec["locked_memory_metric_name"],
                                expected_winner_cfg=wcfg,
                                raise_on_error=True,
                            )
                            if ok:
                                verified_seeds.append(s_val)
                                verified_winner_runs += 1
                        except Exception as exc:
                            errors.append(f"{slug}/{arm}/s{s_val} test receipt error: {exc}")
                    else:
                        errors.append(f"{slug}/{arm}/s{s_val} test receipt missing")
            model_entry["arms"][arm] = {"test_verified_seeds": verified_seeds}
        models_out[slug] = model_entry

    attempt_audit = audit_final_test_attempts(dest_dir, expected_seal_shas=pins if require_all_63 else None)
    if attempt_audit["duplicate_completed"]:
        errors.extend(attempt_audit["duplicate_completed"])
    if attempt_audit["unexpected_records"]:
        errors.extend(attempt_audit["unexpected_records"])
    if attempt_audit["chronology_or_linkage_errors"]:
        errors.extend(attempt_audit["chronology_or_linkage_errors"])
    if require_attempt_logs and (require_all_63 or attempt_audit["total_attempt_files"] > 0):
        if require_all_63 and attempt_audit["missing_completed"]:
            errors.extend(attempt_audit["missing_completed"])

    complete_63 = (verified_winner_runs == 60) and (verified_zero_shot_runs == 3) and not errors
    summary = {
        "verified_winner_test_runs": verified_winner_runs,
        "expected_winner_test_runs": 60,
        "verified_zero_shot_test_runs": verified_zero_shot_runs,
        "expected_zero_shot_test_runs": 3,
        "total_verified_test_receipts": verified_winner_runs + verified_zero_shot_runs,
        "expected_total_test_receipts": 63,
        "all_63_verified": complete_63,
        "attempt_audit": {
            "total_attempt_files": attempt_audit["total_attempt_files"],
            "retried_runs": attempt_audit["retried_runs"],
            "duplicate_completed": attempt_audit["duplicate_completed"],
            "chronology_or_linkage_errors": attempt_audit["chronology_or_linkage_errors"],
            "unexpected_records": attempt_audit["unexpected_records"],
            "attempt_counts_by_model": attempt_audit["attempt_counts_by_model"],
        },
        "models": models_out,
        "errors": errors,
    }
    if require_all_63 and not complete_63:
        raise PermissionError(
            f"63-receipt final-test gate FAILED ({verified_winner_runs}/60 winner runs, "
            f"{verified_zero_shot_runs}/3 zero-shot runs): {errors[:5]}"
        )
    return summary


def has_any_local_test_or_attempt_artifacts(dest_dir: Path) -> list[str]:
    """Return relative paths of any local files under `final_sealed_test/`, `zero_shot_test/`, or `final_test_attempts/`."""
    found: list[str] = []
    for sub in ("final_sealed_test", "zero_shot_test", "final_test_attempts"):
        root = dest_dir / sub
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    found.append(str(p.relative_to(dest_dir)))
    return found


def build_pre_unseal_ceremony_payloads(
    dest_dir: Path,
    *,
    manifest_sha: str,
    split_id_sets: dict[str, set[str]],
    frozen_digests: dict[str, str],
    unsealed_utc: str,
) -> dict[str, dict[str, Any]]:
    """Construct immutable `test_unsealed.json` payloads ONLY after all 471 pre-unseal receipts pass and zero test files exist."""
    preexisting = has_any_local_test_or_attempt_artifacts(dest_dir)
    if preexisting:
        raise PermissionError(
            f"Cannot issue fresh pre-unseal ceremony: prior test/seal/attempt files exist locally: {preexisting[:5]}"
        )

    for hk in FROZEN_SEAL_HASH_KEYS:
        hv = frozen_digests.get(hk)
        if not isinstance(hv, str) or not HEX64_RE.fullmatch(hv):
            raise PermissionError(f"Invalid or missing frozen digest '{hk}': {hv!r}")

    val_ids = split_id_sets["validation"]
    payloads: dict[str, dict[str, Any]] = {}

    for slug, spec in CANONICAL_PROTOCOL_MODELS.items():
        zs_val_path = dest_dir / "zero_shot_baselines" / slug / "zero_shot_validation.json"
        verify_strict_zero_shot_receipt(
            zs_val_path,
            expected_split="validation",
            expected_split_ids=val_ids,
            expected_manifest_sha=manifest_sha,
            expected_git_sha=spec["git_commit"],
            expected_model_id=spec["model_id"],
            expected_model_revision=spec["model_revision"],
            expected_gpu_name=spec["locked_gpu_name"],
            expected_memory_metric=spec["locked_memory_metric_name"],
            raise_on_error=True,
        )

        # Enforce symmetric multi-seed search verification (160 trials across Falcon-H1 & LFM3)
        if slug in MULTISEED_CUDA_SLUGS:
            for ms_seed in MULTISEED_SEARCH_SEEDS:
                for ms_arm in MULTISEED_PAIRED_ARMS:
                    ms_dir = dest_dir / "campaigns_multiseed" / f"seed_{ms_seed}" / slug / ms_arm
                    if not ms_dir.is_dir():
                        raise PermissionError(f"Missing symmetric multi-seed search dir: {ms_dir}")
                    verify_campaign_dir(
                        ms_dir,
                        expected_manifest_sha=manifest_sha,
                        expected_split_ids=split_id_sets,
                        allow_smoke_budget=False,
                    )
                    for tj in sorted(ms_dir.glob("trials/*/trial.json")):
                        trec = json.loads(tj.read_text(encoding="utf-8"))
                        if trec.get("git_commit") != spec["git_commit"]:
                            raise PermissionError(f"Multi-seed git_commit mismatch in {tj}")

        winner_sigs: dict[str, str] = {}
        for arm in spec["arms"]:
            arm_dir = dest_dir / "campaigns" / slug / arm
            verify_campaign_dir(
                arm_dir,
                expected_manifest_sha=manifest_sha,
                expected_split_ids=split_id_sets,
                allow_smoke_budget=False,
            )
            sdata = json.loads((arm_dir / "campaign_summary.json").read_text(encoding="utf-8"))
            if sdata.get("protocol_tag") != "protocol-v1":
                raise PermissionError(f"Non-protocol-v1 summary in {arm_dir}")
            wcfg = TrialConfig.model_validate(sdata["selected_winner_config"])
            winner_sigs[arm] = compute_full_winner_sha256(wcfg)

            for c_seed in REQUIRED_CONFIRM_SEEDS:
                tid = f"{arm}_winner_s{c_seed}_validation"
                c_path = dest_dir / "confirm_3seed" / slug / arm / "trials" / tid / "trial.json"
                verify_strict_trial_receipt(
                    c_path,
                    expected_seed=c_seed,
                    expected_split="validation",
                    expected_split_ids=val_ids,
                    expected_manifest_sha=manifest_sha,
                    expected_git_sha=spec["git_commit"],
                    expected_model_id=spec["model_id"],
                    expected_model_revision=spec["model_revision"],
                    expected_gpu_name=spec["locked_gpu_name"],
                    expected_memory_metric=spec["locked_memory_metric_name"],
                    expected_winner_cfg=wcfg,
                    raise_on_error=True,
                )

        payloads[slug] = {
            "protocol_tag": "protocol-v1",
            "model_id": spec["model_id"],
            "model_revision": spec["model_revision"],
            "git_commit": spec["git_commit"],
            "manifest_sha256": manifest_sha,
            "locked_gpu_name": spec["locked_gpu_name"],
            "locked_memory_metric_name": spec["locked_memory_metric_name"],
            "allowed_seeds": list(REQUIRED_CONFIRM_SEEDS),
            "max_steps": 125,
            "max_new_tokens": 192,
            **{hk: frozen_digests[hk] for hk in FROZEN_SEAL_HASH_KEYS},
            "unsealed_utc": unsealed_utc,
            "winner_signatures": winner_sigs,
        }
    return payloads

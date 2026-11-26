"""Unit & Integration Tests for Pre-Unseal / Final-Test Orchestration, Attempt Audit, & 63-Receipt Gate.

Covers all mandatory negative and positive invariants on synthetic fixtures ONLY
(NEVER accesses or evaluates the real sealed `splits["test"]`):
1. Missing `test_unsealed.json`, missing/empty `expected_seal_sha256`, or mismatched seal SHA rejected.
2. Subset-arm seal, extra-arm seal, metadata mismatch, or winner-signature mismatch rejected.
3. Frozen code/plan/shim/auditor/evaluator/bootstrap SHA-256 mismatch rejected in both `final_test_audit`
   and `run_colab_shard._verify_immutable_seal_standalone`.
4. Synthetic `zero_shot_test` or `final_test_attempts` file blocks fresh `build_pre_unseal_ceremony_payloads`.
5. Missing or incomplete `seed_1337` / `seed_2026` `ALOE` / `B1` multiseed search blocks `build_pre_unseal_ceremony_payloads`.
6. `NaN` EM / `NaN` `tool_accuracy`, integer `exact_match: 1` (non-bool), `trainable_params <= 0`,
   missing AdaLoRA params, and weakly-valid-but-strictly-invalid cached CUDA zero-shot receipts rejected.
7. `verify_complete_63_final_test_gate` audits `final_test_attempts/`: flags `launched == 2, completed == 1`
   in `retried_runs`, and rejects `completed > 1` or missing `COMPLETED` records.
8. `evaluate_63_receipt_suite` rejects non-canonical `bootstrap_resamples != 10_000` or `rng_seed != 42`
   in official mode, fails closed if BCa falls back to `"percentile"`, and emits exact `composite_claim`.
9. `compute_union_raw_max_bounds` + `evaluate_model_campaigns_3d_hv(..., shared_bounds=...)` enforces
   invariant multi-seed normalization and rejects mixed GPU SKUs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import pytest

from aloe.evaluation.bootstrap import PairedBootstrapResult
from aloe.evaluation.tool_calling import EVALUATOR_VERSION
from aloe.execution.trial_config import GroupTrialSetting, TrialConfig
from scripts.compute_3d_hypervolume import (
    compute_union_raw_max_bounds,
    evaluate_model_campaigns_3d_hv,
)
from scripts.evaluate_final_test_suite import evaluate_63_receipt_suite
from scripts.final_test_audit import (
    CANONICAL_PROTOCOL_MODELS,
    MULTISEED_CUDA_SLUGS,
    MULTISEED_PAIRED_ARMS,
    MULTISEED_SEARCH_SEEDS,
    WORKER_SHIM_SOURCE,
    build_pre_unseal_ceremony_payloads,
    compute_frozen_code_digests,
    verify_complete_63_final_test_gate,
    verify_immutable_model_seal,
    verify_strict_trial_receipt,
    verify_strict_zero_shot_receipt,
)
from scripts.run_colab_a100_campaign import REQUIRED_CONFIRM_SEEDS, compute_full_winner_sha256
from scripts.run_colab_shard import (
    _COLAB_WORKER_SHIM_SOURCE,
    _find_open_launched_attempt_id as _colab_find_open_aid,
    _has_completed_attempt_local as _colab_has_completed,
    _log_final_test_attempt,
    _verify_immutable_seal_standalone,
    _verify_strict_receipt_standalone,
    _verify_strict_zero_shot_standalone,
)
from scripts.run_local_mps_final import (
    _find_open_launched_attempt_id as _mps_find_open_aid,
    _has_completed_attempt_local as _mps_has_completed,
    _log_hf_attempt,
    _replay_local_attempts_to_hf as _mps_replay_attempts,
)
from scripts.sync_and_verify_hf_shards import sync_and_audit_once


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_mock_winner_cfg(model_id: str, arm: str) -> TrialConfig:
    is_b2 = arm == "b2"
    return TrialConfig(
        trial_id=f"winner_{arm}",
        arm=arm,
        peft_method="adalora" if is_b2 else "lora",
        model_id=model_id,
        group_settings={
            "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16),
            "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
        },
    )


def _write_mock_predictions(
    path: Path,
    *,
    split: str,
    ids: list[str],
    model_id: str,
    model_rev: str,
    arm: str,
    trial_id: str,
    seed: int,
    correct_mask: list[bool] | None = None,
    n_correct: int = 0,
) -> str:
    lines: list[str] = []
    for idx, sid in enumerate(ids):
        is_ok = bool(correct_mask[idx]) if correct_mask is not None else (idx < n_correct)
        rec = {
            "sample_id": sid,
            "split": split,
            "model_id": model_id,
            "model_revision": model_rev,
            "arm": arm,
            "trial_id": trial_id,
            "seed": seed,
            "prompt_sha256": "0" * 64,
            "raw_output": '[{"name": "mock_tool", "arguments": {}}]',
            "parse_success": True,
            "tool_match": is_ok,
            "arg_names_match": is_ok,
            "arg_values_match": is_ok,
            "exact_match": is_ok,
            "evaluator_version": EVALUATOR_VERSION,
        }
        lines.append(json.dumps(rec))
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256_bytes(raw)


def _populate_mock_campaign_arm(
    arm_dir: Path,
    *,
    spec: dict[str, Any],
    arm: str,
    manifest_sha: str,
    val_ids: list[str],
) -> TrialConfig:
    target_n = 1 if arm.startswith("borobia_") else 20
    entries = []
    trials_dir = arm_dir / "trials"
    wcfg = _make_mock_winner_cfg(spec["model_id"], arm)
    is_b2 = arm == "b2"

    for i in range(target_n):
        tid = f"trial_{arm}_{i:03d}"
        cfg_i = wcfg.model_copy(
            update={
                "trial_id": tid,
                "group_settings": {
                    "ffn": GroupTrialSetting(enabled=True, rank=8, alpha=16 + i),
                    "global_attention": GroupTrialSetting(enabled=True, rank=8, alpha=16),
                },
            }
        )
        tdir = trials_dir / tid
        pred_sha = _write_mock_predictions(
            tdir / "predictions_validation.jsonl",
            split="validation",
            ids=val_ids,
            model_id=spec["model_id"],
            model_rev=spec["model_revision"],
            arm=arm,
            trial_id=tid,
            seed=42,
            n_correct=len(val_ids) - 1,
        )
        em = (len(val_ids) - 1) / len(val_ids)
        trec = {
            "status": "PASS",
            "git_commit": spec["git_commit"],
            "dataset_manifest_sha256": manifest_sha,
            "model_id": spec["model_id"],
            "model_revision": spec["model_revision"],
            "gpu_name": spec["locked_gpu_name"],
            "trial_config": cfg_i.model_dump(mode="json"),
            "metrics": {
                "ast_exact_match_accuracy": em,
                "tool_accuracy": em,
                "search_memory_mb": 1500.0 + i,
                "trainable_params": spec["baseline_params"],
                "adalora_init_params": spec["baseline_params"] * 2 if is_b2 else None,
                "adalora_final_active_params": int(spec["baseline_params"] * 1.10) if is_b2 else None,
                "memory_metric_name": spec["locked_memory_metric_name"],
                "optimizer_steps_completed": 125,
                "fresh_process_reload": True,
                "is_budget_matched_5pct": True,
                "predictions_sha256": pred_sha,
                "eval_samples": len(val_ids),
                "evaluator_version": EVALUATOR_VERSION,
            },
        }
        (tdir / "trial.json").write_text(json.dumps(trec), encoding="utf-8")
        entries.append(
            {
                "trial_id": tid,
                "status": "DONE",
                "config_signature": f"{arm}_sig_{i}",
                "metrics": {
                    "exact_match_accuracy": em,
                    "tool_accuracy": em,
                    "search_memory_mb": 1500.0 + i,
                    "trainable_parameters": float(spec["baseline_params"]),
                },
            }
        )

    winner_tid = f"trial_{arm}_000"
    winner_cfg = wcfg.model_copy(update={"trial_id": winner_tid})
    t0_json = json.loads((trials_dir / winner_tid / "trial.json").read_text(encoding="utf-8"))
    t0_json["trial_config"] = winner_cfg.model_dump(mode="json")
    (trials_dir / winner_tid / "trial.json").write_text(json.dumps(t0_json), encoding="utf-8")
    (trials_dir / winner_tid / "adapter").mkdir(parents=True, exist_ok=True)
    (trials_dir / winner_tid / "adapter" / "adapter_model.safetensors").write_bytes(b"mock_weights")
    (trials_dir / winner_tid / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")

    ledger = {
        "campaign_id": f"mock_{arm}_s42",
        "model_id": spec["model_id"],
        "arm": arm,
        "target_trials": target_n,
        "manifest_sha256": manifest_sha,
        "locked_gpu_name": spec["locked_gpu_name"],
        "locked_memory_metric_name": spec["locked_memory_metric_name"],
        "entries": entries,
    }
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "campaign_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    em = (len(val_ids) - 1) / len(val_ids)
    summary = {
        "protocol_tag": "protocol-v1",
        "model_id": spec["model_id"],
        "arm": arm,
        "completed_trials": target_n,
        "target_trials": target_n,
        "pareto_trial_ids": [winner_tid],
        "selected_winner_trial_id": winner_tid,
        "selected_winner_metrics": {
            "exact_match_accuracy": em,
            "tool_accuracy": em,
            "trainable_parameters": float(spec["baseline_params"]),
            "search_memory_mb": 1500.0,
        },
        "selected_winner_config": winner_cfg.model_dump(mode="json"),
    }
    (arm_dir / "campaign_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return winner_cfg


def _write_attempt(
    dest_dir: Path,
    slug: str,
    run_id: str,
    stage: str,
    idx: int = 0,
    *,
    attempt_id: str = "aid000000001",
    seal_sha256: str = "",
    utc: str | None = None,
) -> None:
    seq = 1 if stage == "LAUNCHED" else 2
    eff_utc = utc or f"2026-11-26T00:{idx:02d}:00Z"
    p = dest_dir / "final_test_attempts" / slug / f"{run_id}__{seq:02d}_{stage}__{idx:03d}_{attempt_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "protocol_tag": "protocol-v1",
                "attempt_id": attempt_id,
                "seq": seq,
                "seal_sha256": seal_sha256,
                "slug": slug,
                "run_id": run_id,
                "stage": stage,
                "utc": eff_utc,
            }
        ),
        encoding="utf-8",
    )


def test_immutable_seal_enforces_pin_complete_arms_and_frozen_code_digests(tmp_path: Path) -> None:
    slug = "google--gemma-4-E2B-it"
    spec = CANONICAL_PROTOCOL_MODELS[slug]
    manifest_sha = "f" * 64
    val_ids = ["v1", "v2", "v3", "v4"]
    split_sets = {"validation": set(val_ids), "test": {"t1", "t2"}}

    c_dir = tmp_path / "campaigns" / slug
    winners: dict[str, TrialConfig] = {}
    for arm in spec["arms"]:
        winners[arm] = _populate_mock_campaign_arm(
            c_dir / arm, spec=spec, arm=arm, manifest_sha=manifest_sha, val_ids=val_ids
        )

    digests = compute_frozen_code_digests()
    valid_sigs = {arm: compute_full_winner_sha256(cfg) for arm, cfg in winners.items()}
    base_seal = {
        "protocol_tag": "protocol-v1",
        "model_id": spec["model_id"],
        "model_revision": spec["model_revision"],
        "git_commit": spec["git_commit"],
        "manifest_sha256": manifest_sha,
        "locked_gpu_name": spec["locked_gpu_name"],
        "locked_memory_metric_name": spec["locked_memory_metric_name"],
        "allowed_seeds": [42, 1337, 2026],
        "max_steps": 125,
        "max_new_tokens": 192,
        **digests,
        "winner_signatures": valid_sigs,
    }
    seal_path = tmp_path / "final_sealed_test" / slug / "test_unsealed.json"
    seal_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Missing expected_seal_sha256 rejected BEFORE reading seal
    with pytest.raises(PermissionError, match="expected_seal_sha256 is required"):
        verify_immutable_model_seal(
            seal_path,
            slug=slug,
            manifest_sha=manifest_sha,
            campaigns_model_dir=c_dir,
            expected_split_ids=split_sets,
            expected_seal_sha256="",
        )

    # 2. Missing seal file rejected even when 64-hex pin is provided
    with pytest.raises(PermissionError, match="Missing pre-committed read-only seal"):
        verify_immutable_model_seal(
            seal_path,
            slug=slug,
            manifest_sha=manifest_sha,
            campaigns_model_dir=c_dir,
            expected_split_ids=split_sets,
            expected_seal_sha256="a" * 64,
        )

    def _write_and_sha(payload: dict[str, Any]) -> str:
        b = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        seal_path.write_bytes(b)
        return _sha256_bytes(b)

    # 3. Subset-arm and extra-arm seals rejected
    sub_sha = _write_and_sha({**base_seal, "winner_signatures": {"aloe": valid_sigs["aloe"]}})
    with pytest.raises(PermissionError, match="complete frozen arm set"):
        verify_immutable_model_seal(
            seal_path,
            slug=slug,
            manifest_sha=manifest_sha,
            campaigns_model_dir=c_dir,
            expected_split_ids=split_sets,
            expected_seal_sha256=sub_sha,
        )

    ext_sha = _write_and_sha({**base_seal, "winner_signatures": {**valid_sigs, "extra_arm": "0" * 64}})
    with pytest.raises(PermissionError, match="complete frozen arm set"):
        verify_immutable_model_seal(
            seal_path,
            slug=slug,
            manifest_sha=manifest_sha,
            campaigns_model_dir=c_dir,
            expected_split_ids=split_sets,
            expected_seal_sha256=ext_sha,
        )

    # 4. Tampered analysis_plan_sha256, shim_sha256, or colab_bootstrap_sha256 rejected!
    for tamper_key in ("analysis_plan_sha256", "shim_sha256", "evaluator_script_sha256"):
        bad_payload = {**base_seal, tamper_key: "b" * 64}
        bad_sha = _write_and_sha(bad_payload)
        with pytest.raises(PermissionError, match=tamper_key):
            verify_immutable_model_seal(
                seal_path,
                slug=slug,
                manifest_sha=manifest_sha,
                campaigns_model_dir=c_dir,
                expected_split_ids=split_sets,
                expected_seal_sha256=bad_sha,
                verify_code_digests=True,
            )
        with pytest.raises(PermissionError, match=tamper_key):
            _verify_immutable_seal_standalone(
                seal_path,
                expected_model_id=spec["model_id"],
                expected_model_revision=spec["model_revision"],
                expected_git_sha=spec["git_commit"],
                expected_manifest_sha=manifest_sha,
                expected_gpu_name=spec["locked_gpu_name"],
                expected_memory_metric=spec["locked_memory_metric_name"],
                expected_complete_arms=spec["arms"],
                campaigns_model_dir=c_dir,
                expected_split_ids=split_sets,
                expected_seal_sha256=bad_sha,
                verify_code_digests=True,
            )

    # 5. Valid complete seal passes both verifiers
    good_sha = _write_and_sha(base_seal)
    verified = verify_immutable_model_seal(
        seal_path,
        slug=slug,
        manifest_sha=manifest_sha,
        campaigns_model_dir=c_dir,
        expected_split_ids=split_sets,
        expected_seal_sha256=good_sha,
        verify_code_digests=True,
    )
    assert set(verified["winners_by_arm"].keys()) == set(spec["arms"])


def test_ceremony_builder_blocks_prior_zero_shot_test_and_missing_multiseed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_sha = "d" * 64
    val_ids = ["v1", "v2", "v3", "v4"]
    split_sets = {"validation": set(val_ids), "test": {"t1", "t2"}}
    digests = compute_frozen_code_digests()
    dummy_manifest = tmp_path / "dummy_manifest.json"
    dummy_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.sync_and_verify_hf_shards.verify_manifest",
        lambda *_a, **_kw: {"manifest_sha256": manifest_sha, "split_id_sets": split_sets},
    )

    for slug, spec in CANONICAL_PROTOCOL_MODELS.items():
        for arm in spec["arms"]:
            wcfg = _populate_mock_campaign_arm(
                tmp_path / "campaigns" / slug / arm,
                spec=spec,
                arm=arm,
                manifest_sha=manifest_sha,
                val_ids=val_ids,
            )
            is_b2 = arm == "b2"
            for c_seed in REQUIRED_CONFIRM_SEEDS:
                tid = f"{arm}_winner_s{c_seed}_validation"
                cdir = tmp_path / "confirm_3seed" / slug / arm / "trials" / tid
                p_sha = _write_mock_predictions(
                    cdir / "predictions_validation.jsonl",
                    split="validation",
                    ids=val_ids,
                    model_id=spec["model_id"],
                    model_rev=spec["model_revision"],
                    arm=arm,
                    trial_id=tid,
                    seed=c_seed,
                    n_correct=3,
                )
                (cdir / "trial.json").write_text(
                    json.dumps(
                        {
                            "status": "PASS",
                            "git_commit": spec["git_commit"],
                            "dataset_manifest_sha256": manifest_sha,
                            "model_id": spec["model_id"],
                            "model_revision": spec["model_revision"],
                            "gpu_name": spec["locked_gpu_name"],
                            "trial_config": wcfg.model_copy(update={"trial_id": tid, "seed": c_seed}).model_dump(mode="json"),
                            "metrics": {
                                "ast_exact_match_accuracy": 0.75,
                                "tool_accuracy": 0.75,
                                "search_memory_mb": 1550.0,
                                "trainable_params": spec["baseline_params"],
                                "adalora_init_params": spec["baseline_params"] * 2 if is_b2 else None,
                                "adalora_final_active_params": int(spec["baseline_params"] * 1.1) if is_b2 else None,
                                "memory_metric_name": spec["locked_memory_metric_name"],
                                "optimizer_steps_completed": 125,
                                "fresh_process_reload": True,
                                "is_budget_matched_5pct": True,
                                "eval_samples": len(val_ids),
                                "evaluator_version": EVALUATOR_VERSION,
                                "predictions_sha256": p_sha,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

        zs_val_dir = tmp_path / "zero_shot_baselines" / slug
        zs_sha = _write_mock_predictions(
            zs_val_dir / "predictions_validation.jsonl",
            split="validation",
            ids=val_ids,
            model_id=spec["model_id"],
            model_rev=spec["model_revision"],
            arm="zero_shot",
            trial_id=f"zero_shot_{slug}_validation",
            seed=42,
            n_correct=2,
        )
        (zs_val_dir / "zero_shot_validation.json").write_text(
            json.dumps(
                {
                    "protocol_tag": "protocol-v1",
                    "mode": "zero_shot_base_validation",
                    "model_id": spec["model_id"],
                    "model_revision": spec["model_revision"],
                    "arm": "zero_shot",
                    "split": "validation",
                    "sample_count": len(val_ids),
                    "exact_match_accuracy": 0.5,
                    "parse_success_rate": 1.0,
                    "trainable_parameters": 0,
                    "memory_metric_name": spec["locked_memory_metric_name"],
                    "search_memory_mb": 1200.0,
                    "predictions_sha256": zs_sha,
                    "evaluator_version": EVALUATOR_VERSION,
                    "manifest_sha256": manifest_sha,
                    "git_commit": spec["git_commit"],
                    "gpu_name": spec["locked_gpu_name"],
                }
            ),
            encoding="utf-8",
        )

    # 1. Missing symmetric multi-seed search directories block BOTH readiness AND ceremony (Hermes 1A)
    pre_ms_rep = sync_and_audit_once(
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        dest_dir=tmp_path,
        manifest_path=dummy_manifest,
        raw_data_path=None,
        hf_token="",
        skip_download=True,
    )
    assert pre_ms_rep["pre_unseal_all_3_models_verified"] is False
    assert pre_ms_rep["pre_unseal_471_receipts_verified"] is False

    with pytest.raises(PermissionError, match="Missing symmetric multi-seed search dir"):
        build_pre_unseal_ceremony_payloads(
            tmp_path,
            manifest_sha=manifest_sha,
            split_id_sets=split_sets,
            frozen_digests=digests,
            unsealed_utc="2026-11-26T00:00:00Z",
        )

    # Populate all 4x2 symmetric multi-seed directories
    for ms_slug in MULTISEED_CUDA_SLUGS:
        ms_spec = CANONICAL_PROTOCOL_MODELS[ms_slug]
        for ms_seed in MULTISEED_SEARCH_SEEDS:
            for ms_arm in MULTISEED_PAIRED_ARMS:
                _populate_mock_campaign_arm(
                    tmp_path / "campaigns_multiseed" / f"seed_{ms_seed}" / ms_slug / ms_arm,
                    spec=ms_spec,
                    arm=ms_arm,
                    manifest_sha=manifest_sha,
                    val_ids=val_ids,
                )

    post_ms_rep = sync_and_audit_once(
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        dest_dir=tmp_path,
        manifest_path=dummy_manifest,
        raw_data_path=None,
        hf_token="",
        skip_download=True,
    )
    assert post_ms_rep["pre_unseal_all_3_models_verified"] is True
    assert post_ms_rep["pre_unseal_471_receipts_verified"] is True
    assert post_ms_rep["any_local_test_or_attempt_artifacts_present"] is False

    payloads = build_pre_unseal_ceremony_payloads(
        tmp_path,
        manifest_sha=manifest_sha,
        split_id_sets=split_sets,
        frozen_digests=digests,
        unsealed_utc="2026-11-26T00:00:00Z",
    )
    assert {s: len(p["winner_signatures"]) for s, p in payloads.items()} == {
        "google--gemma-4-E2B-it": 6,
        "tiiuae--Falcon-H1-1.5B-Instruct": 7,
        "LiquidAI--LFM3-1.7B": 7,
    }

    # 2. A synthetic zero_shot_test file blocks a fresh ceremony (Hermes 1B / Codex #2)
    leaked_zs = tmp_path / "zero_shot_test" / "google--gemma-4-E2B-it" / "zero_shot_test.json"
    leaked_zs.parent.mkdir(parents=True, exist_ok=True)
    leaked_zs.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="prior test/seal/attempt files exist"):
        build_pre_unseal_ceremony_payloads(
            tmp_path,
            manifest_sha=manifest_sha,
            split_id_sets=split_sets,
            frozen_digests=digests,
            unsealed_utc="2026-11-26T00:00:00Z",
        )


def test_numerical_nan_boolean_and_strict_zero_shot_guards(tmp_path: Path) -> None:
    slug = "tiiuae--Falcon-H1-1.5B-Instruct"
    spec = CANONICAL_PROTOCOL_MODELS[slug]
    manifest_sha = "c" * 64
    ids = ["t1", "t2", "t3", "t4"]
    wcfg = _make_mock_winner_cfg(spec["model_id"], "aloe")
    tid = "aloe_winner_s42_test"
    tdir = tmp_path / "trials" / tid
    p_sha = _write_mock_predictions(
        tdir / "predictions_test.jsonl",
        split="test",
        ids=ids,
        model_id=spec["model_id"],
        model_rev=spec["model_revision"],
        arm="aloe",
        trial_id=tid,
        seed=42,
        n_correct=3,
    )
    tjson = tdir / "trial.json"
    base_rec = {
        "status": "PASS",
        "git_commit": spec["git_commit"],
        "dataset_manifest_sha256": manifest_sha,
        "model_id": spec["model_id"],
        "model_revision": spec["model_revision"],
        "gpu_name": spec["locked_gpu_name"],
        "trial_config": wcfg.model_copy(update={"trial_id": tid, "seed": 42}).model_dump(mode="json"),
        "metrics": {
            "ast_exact_match_accuracy": 0.75,
            "tool_accuracy": 0.75,
            "search_memory_mb": 12000.0,
            "trainable_params": spec["baseline_params"],
            "memory_metric_name": spec["locked_memory_metric_name"],
            "optimizer_steps_completed": 125,
            "fresh_process_reload": True,
            "is_budget_matched_5pct": True,
            "eval_samples": len(ids),
            "evaluator_version": EVALUATOR_VERSION,
            "predictions_sha256": p_sha,
        },
    }
    tjson.write_text(json.dumps(base_rec), encoding="utf-8")
    kw = dict(
        expected_seed=42,
        expected_split="test",
        expected_split_ids=set(ids),
        expected_manifest_sha=manifest_sha,
        expected_git_sha=spec["git_commit"],
        expected_model_id=spec["model_id"],
        expected_model_revision=spec["model_revision"],
        expected_gpu_name=spec["locked_gpu_name"],
        expected_memory_metric=spec["locked_memory_metric_name"],
        expected_winner_cfg=wcfg,
    )
    assert verify_strict_trial_receipt(tjson, **kw) is True
    assert _verify_strict_receipt_standalone(tjson, **kw) is True

    # NaN accuracy, negative trainable_params, or non-bool exact_match MUST fail closed in both verifiers!
    for mut in (
        {"ast_exact_match_accuracy": float("nan")},
        {"tool_accuracy": float("nan")},
        {"trainable_params": -1},
        {"search_memory_mb": 0.0},
    ):
        bad = {**base_rec, "metrics": {**base_rec["metrics"], **mut}}
        tjson.write_text(json.dumps(bad, allow_nan=True), encoding="utf-8")
        assert verify_strict_trial_receipt(tjson, **kw) is False
        assert _verify_strict_receipt_standalone(tjson, **kw) is False

    # Weakly valid CUDA zero-shot receipt (e.g. wrong model_revision or trainable_parameters=1 or missing memory_metric_name on test) rejected!
    zs_dir = tmp_path / "zs_test"
    zs_p_sha = _write_mock_predictions(
        zs_dir / "predictions_test.jsonl",
        split="test",
        ids=ids,
        model_id=spec["model_id"],
        model_rev=spec["model_revision"],
        arm="zero_shot",
        trial_id=f"zero_shot_{slug}_test",
        seed=42,
        n_correct=2,
    )
    zs_json = zs_dir / "zero_shot_test.json"
    weak_zs = {
        "protocol_tag": "protocol-v1",
        "arm": "zero_shot",
        "split": "test",
        "model_id": spec["model_id"],
        "model_revision": "WRONG_REV",
        "git_commit": spec["git_commit"],
        "gpu_name": spec["locked_gpu_name"],
        "manifest_sha256": manifest_sha,
        "trainable_parameters": 0,
        "sample_count": len(ids),
        "exact_match_accuracy": 0.5,
        "memory_metric_name": "cuda_max_memory_allocated",
        "search_memory_mb": 3000.0,
        "predictions_sha256": zs_p_sha,
        "evaluator_version": EVALUATOR_VERSION,
    }
    zs_json.write_text(json.dumps(weak_zs), encoding="utf-8")
    assert _verify_strict_zero_shot_standalone(
        zs_json,
        expected_split="test",
        expected_split_ids=set(ids),
        expected_manifest_sha=manifest_sha,
        expected_git_sha=spec["git_commit"],
        expected_model_id=spec["model_id"],
        expected_model_revision=spec["model_revision"],
        expected_gpu_name=spec["locked_gpu_name"],
    ) is None


def test_global_63_gate_attempt_audit_and_evaluator_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_ids = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
    val_ids = ["v1", "v2", "v3", "v4"]
    manifest_obj = {
        "protocol_version": "IPD-PROTOCOL-V1",
        "splits": {
            "train": {"ids": ["tr1", "tr2"]},
            "validation": {"ids": val_ids},
            "test": {"ids": test_ids},
        },
    }
    manifest_path = tmp_path / "mock_manifest.json"
    manifest_bytes = (json.dumps(manifest_obj) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = _sha256_bytes(manifest_bytes)
    split_sets = {"validation": set(val_ids), "test": set(test_ids)}

    monkeypatch.setattr(
        "scripts.evaluate_final_test_suite.verify_manifest",
        lambda *_a, **_kw: {"manifest_sha256": manifest_sha, "split_id_sets": split_sets},
    )

    digests = compute_frozen_code_digests()
    plan_copy = tmp_path / "final_sealed_test" / "PRE_UNSEAL_ANALYSIS_PLAN.md"
    plan_copy.parent.mkdir(parents=True, exist_ok=True)
    plan_copy.write_bytes((Path(__file__).resolve().parent.parent / "docs" / "PRE_UNSEAL_ANALYSIS_PLAN.md").read_bytes())

    seal_shas: dict[str, str] = {}
    for slug, spec in CANONICAL_PROTOCOL_MODELS.items():
        c_dir = tmp_path / "campaigns" / slug
        winners: dict[str, TrialConfig] = {}
        for arm in spec["arms"]:
            winners[arm] = _populate_mock_campaign_arm(
                c_dir / arm, spec=spec, arm=arm, manifest_sha=manifest_sha, val_ids=val_ids
            )
        seal_path = tmp_path / "final_sealed_test" / slug / "test_unsealed.json"
        seal_path.parent.mkdir(parents=True, exist_ok=True)
        seal_bytes = (
            json.dumps(
                {
                    "protocol_tag": "protocol-v1",
                    "model_id": spec["model_id"],
                    "model_revision": spec["model_revision"],
                    "git_commit": spec["git_commit"],
                    "manifest_sha256": manifest_sha,
                    "locked_gpu_name": spec["locked_gpu_name"],
                    "locked_memory_metric_name": spec["locked_memory_metric_name"],
                    "allowed_seeds": [42, 1337, 2026],
                    "max_steps": 125,
                    "max_new_tokens": 192,
                    **digests,
                    "winner_signatures": {a: compute_full_winner_sha256(w) for a, w in winners.items()},
                },
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        seal_path.write_bytes(seal_bytes)
        seal_shas[slug] = _sha256_bytes(seal_bytes)

        zs_dir = tmp_path / "zero_shot_test" / slug
        zs_mask = [i < 3 for i in range(len(test_ids))]
        zs_sha = _write_mock_predictions(
            zs_dir / "predictions_test.jsonl",
            split="test",
            ids=test_ids,
            model_id=spec["model_id"],
            model_rev=spec["model_revision"],
            arm="zero_shot",
            trial_id=f"zero_shot_{slug}_test",
            seed=42,
            correct_mask=zs_mask,
        )
        (zs_dir / "zero_shot_test.json").write_text(
            json.dumps(
                {
                    "protocol_tag": "protocol-v1",
                    "mode": "zero_shot_base_test",
                    "model_id": spec["model_id"],
                    "model_revision": spec["model_revision"],
                    "arm": "zero_shot",
                    "split": "test",
                    "sample_count": len(test_ids),
                    "exact_match_accuracy": sum(zs_mask) / len(test_ids),
                    "parse_success_rate": 1.0,
                    "trainable_parameters": 0,
                    "memory_metric_name": spec["locked_memory_metric_name"],
                    "search_memory_mb": 1200.0,
                    "predictions_sha256": zs_sha,
                    "evaluator_version": EVALUATOR_VERSION,
                    "manifest_sha256": manifest_sha,
                    "git_commit": spec["git_commit"],
                    "gpu_name": spec["locked_gpu_name"],
                }
            ),
            encoding="utf-8",
        )
        _write_attempt(tmp_path, slug, f"zero_shot_{slug}_test", "LAUNCHED", 1, attempt_id="aid_zs_1", seal_sha256=seal_shas[slug])
        _write_attempt(tmp_path, slug, f"zero_shot_{slug}_test", "COMPLETED", 2, attempt_id="aid_zs_1", seal_sha256=seal_shas[slug])

        for arm_idx, (arm, wcfg) in enumerate(winners.items()):
            for s_idx, s_val in enumerate(REQUIRED_CONFIRM_SEEDS):
                tid = f"{arm}_winner_s{s_val}_test"
                tdir = tmp_path / "final_sealed_test" / slug / arm / "trials" / tid
                # Give non-constant item differences so BCa jackknife has non-zero variance!
                if arm == "aloe":
                    mask = [i < 7 for i in range(len(test_ids))]
                elif arm == "b2":
                    mask = [i < 2 for i in range(len(test_ids))]
                else:
                    mask = [(i + arm_idx + s_idx) % 8 < 6 for i in range(len(test_ids))]
                p_sha = _write_mock_predictions(
                    tdir / "predictions_test.jsonl",
                    split="test",
                    ids=test_ids,
                    model_id=spec["model_id"],
                    model_rev=spec["model_revision"],
                    arm=arm,
                    trial_id=tid,
                    seed=s_val,
                    correct_mask=mask,
                )
                em_v = sum(mask) / len(test_ids)
                seeded = wcfg.model_copy(update={"trial_id": tid, "seed": s_val})
                is_b2 = arm == "b2"
                params_val = spec["baseline_params"] - 10000 if arm == "aloe" else spec["baseline_params"]
                mem_val = 1400.0 if arm == "aloe" else 1600.0
                (tdir / "trial.json").write_text(
                    json.dumps(
                        {
                            "status": "PASS",
                            "git_commit": spec["git_commit"],
                            "dataset_manifest_sha256": manifest_sha,
                            "model_id": spec["model_id"],
                            "model_revision": spec["model_revision"],
                            "gpu_name": spec["locked_gpu_name"],
                            "trial_config": seeded.model_dump(mode="json"),
                            "metrics": {
                                "ast_exact_match_accuracy": em_v,
                                "tool_accuracy": em_v,
                                "search_memory_mb": mem_val,
                                "trainable_params": params_val,
                                "adalora_init_params": spec["baseline_params"] * 2 if is_b2 else None,
                                "adalora_final_active_params": int(spec["baseline_params"] * 1.12) if is_b2 else None,
                                "memory_metric_name": spec["locked_memory_metric_name"],
                                "optimizer_steps_completed": 125,
                                "fresh_process_reload": True,
                                "is_budget_matched_5pct": True,
                                "eval_samples": len(test_ids),
                                "evaluator_version": EVALUATOR_VERSION,
                                "predictions_sha256": p_sha,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                if slug == "google--gemma-4-E2B-it" and tid == "aloe_winner_s42_test":
                    # Simulate a linked preemption retry: attempt 1 LAUNCHED -> FAILED, attempt 2 LAUNCHED -> COMPLETED
                    _write_attempt(tmp_path, slug, tid, "LAUNCHED", 1, attempt_id="aid_try1", seal_sha256=seal_shas[slug], utc="2026-11-26T00:01:00Z")
                    _write_attempt(tmp_path, slug, tid, "FAILED", 2, attempt_id="aid_try1", seal_sha256=seal_shas[slug], utc="2026-11-26T00:02:00Z")
                    _write_attempt(tmp_path, slug, tid, "LAUNCHED", 3, attempt_id="aid_try2", seal_sha256=seal_shas[slug], utc="2026-11-26T00:03:00Z")
                    _write_attempt(tmp_path, slug, tid, "COMPLETED", 4, attempt_id="aid_try2", seal_sha256=seal_shas[slug], utc="2026-11-26T00:04:00Z")
                else:
                    _write_attempt(tmp_path, slug, tid, "LAUNCHED", 1, attempt_id="aid_ok_1", seal_sha256=seal_shas[slug])
                    _write_attempt(tmp_path, slug, tid, "COMPLETED", 2, attempt_id="aid_ok_1", seal_sha256=seal_shas[slug])

    gate = verify_complete_63_final_test_gate(
        tmp_path,
        manifest_sha=manifest_sha,
        split_id_sets=split_sets,
        expected_seal_shas=seal_shas,
        require_all_63=True,
        require_attempt_logs=True,
    )
    assert gate["all_63_verified"] is True
    assert len(gate["attempt_audit"]["retried_runs"]) == 1

    # Non-canonical bootstrap_resamples or rng_seed rejected in official mode (Codex #5)
    with pytest.raises(PermissionError, match="bootstrap_resamples=10000"):
        evaluate_63_receipt_suite(
            tmp_path,
            manifest_path=manifest_path,
            raw_data_path=None,
            dry_run_validation=False,
            bootstrap_resamples=500,
            rng_seed=42,
            expected_seal_shas=seal_shas,
        )

    # Official evaluation succeeds with locked B=10_000, seed=42 and emits composite_claim
    official_res = evaluate_63_receipt_suite(
        tmp_path,
        manifest_path=manifest_path,
        raw_data_path=None,
        dry_run_validation=False,
        bootstrap_resamples=10_000,
        rng_seed=42,
        expected_seal_shas=seal_shas,
    )
    gemma_comps = official_res["models"]["google--gemma-4-E2B-it"]["paired_comparisons_aloe_vs"]
    for pc in ("b0", "b1", "b2"):
        assert gemma_comps[pc]["composite_claim"] in (
            "SUPERIOR_EM",
            "NON_INFERIOR_EFFICIENCY_WIN",
            "INCONCLUSIVE_OR_INFERIOR",
        )
        assert gemma_comps[pc]["bca_ci_method_actual"] == "bca"

    # BCa fallback to percentile MUST fail closed (Hermes #5)
    monkeypatch.setattr(
        "scripts.evaluate_final_test_suite.paired_bootstrap",
        lambda *_a, **_kw: PairedBootstrapResult(
            mean_diff=0.1,
            ci_low=0.01,
            ci_high=0.2,
            ci_method="percentile",
            p_value=0.01,
            p_value_bonferroni=None,
            n_comparisons=1,
            cohens_d=0.5,
            n_queries=8,
            n_seeds=3,
            B=10_000,
            rng_seed=42,
            superior=True,
        ),
    )
    with pytest.raises(RuntimeError, match="Pre-registered BCa CI failed"):
        evaluate_63_receipt_suite(
            tmp_path,
            manifest_path=manifest_path,
            raw_data_path=None,
            dry_run_validation=False,
            bootstrap_resamples=10_000,
            rng_seed=42,
            expected_seal_shas=seal_shas,
        )

    # Post-completion LAUNCHED (`LAUNCHED -> COMPLETED -> LAUNCHED`) MUST fail the 63-receipt gate closed (Codex #4)
    _write_attempt(
        tmp_path,
        "google--gemma-4-E2B-it",
        "b0_winner_s42_test",
        "LAUNCHED",
        50,
        attempt_id="aid_after_done",
        seal_sha256=seal_shas["google--gemma-4-E2B-it"],
        utc="2026-11-26T00:50:00Z",
    )
    with pytest.raises(PermissionError, match="LAUNCHED .* after COMPLETED"):
        verify_complete_63_final_test_gate(
            tmp_path,
            manifest_sha=manifest_sha,
            split_id_sets=split_sets,
            expected_seal_shas=seal_shas,
            require_all_63=True,
            require_attempt_logs=True,
        )

    # Duplicate COMPLETED attempt record (>1) MUST fail the 63-receipt gate closed (Opus G3)
    _write_attempt(
        tmp_path,
        "google--gemma-4-E2B-it",
        "aloe_winner_s42_test",
        "COMPLETED",
        99,
        attempt_id="aid_try2",
        seal_sha256=seal_shas["google--gemma-4-E2B-it"],
        utc="2026-11-26T00:59:00Z",
    )
    with pytest.raises(PermissionError, match="COMPLETED records"):
        verify_complete_63_final_test_gate(
            tmp_path,
            manifest_sha=manifest_sha,
            split_id_sets=split_sets,
            expected_seal_shas=seal_shas,
            require_all_63=True,
            require_attempt_logs=True,
        )


def test_colab_split_orch_vs_work_repo_import_and_missing_artifact_guard(tmp_path: Path) -> None:
    """Reproduce exact Colab layout where WORK_DIR lacks `final_test_audit.py` while ORCH_DIR has it."""
    import os
    import shutil
    import subprocess
    import sys

    assert _COLAB_WORKER_SHIM_SOURCE == WORKER_SHIM_SOURCE

    real_root = Path(__file__).resolve().parent.parent
    orch_dir = tmp_path / "orch"
    work_dir = tmp_path / "work"
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir(parents=True)
    (shim_dir / "sitecustomize.py").write_text(_COLAB_WORKER_SHIM_SOURCE, encoding="utf-8")

    for rel in (
        "scripts/run_colab_shard.py",
        "scripts/run_local_mps_final.py",
        "scripts/final_test_audit.py",
        "scripts/evaluate_final_test_suite.py",
        "aloe/evaluation/bootstrap.py",
        "docs/PRE_UNSEAL_ANALYSIS_PLAN.md",
    ):
        dst = orch_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real_root / rel, dst)

    # Simulate pinned WORK_DIR containing runtime code + old scripts WITHOUT final_test_audit.py!
    for sub in ("aloe", "scripts"):
        dst_sub = work_dir / sub
        shutil.copytree(real_root / sub, dst_sub)
    for newer_only in ("final_test_audit.py", "evaluate_final_test_suite.py", "prepare_unseal_ceremony.py", "run_local_mps_final.py"):
        f_rm = work_dir / "scripts" / newer_only
        if f_rm.exists():
            f_rm.unlink()

    slug = "tiiuae--Falcon-H1-1.5B-Instruct"
    spec = CANONICAL_PROTOCOL_MODELS[slug]
    manifest_sha = "b" * 64
    val_ids = ["v1", "v2", "v3", "v4"]
    split_sets = {"validation": val_ids, "test": ["t1", "t2"]}
    c_dir = tmp_path / "campaigns" / slug
    winners = {
        arm: _populate_mock_campaign_arm(c_dir / arm, spec=spec, arm=arm, manifest_sha=manifest_sha, val_ids=val_ids)
        for arm in spec["arms"]
    }
    digests = compute_frozen_code_digests(orch_dir, shim_path=shim_dir / "sitecustomize.py")
    seal_path = tmp_path / "test_unsealed.json"
    seal_bytes = (
        json.dumps(
            {
                "protocol_tag": "protocol-v1",
                "model_id": spec["model_id"],
                "model_revision": spec["model_revision"],
                "git_commit": spec["git_commit"],
                "manifest_sha256": manifest_sha,
                "locked_gpu_name": spec["locked_gpu_name"],
                "locked_memory_metric_name": spec["locked_memory_metric_name"],
                "allowed_seeds": [42, 1337, 2026],
                "max_steps": 125,
                "max_new_tokens": 192,
                **digests,
                "winner_signatures": {a: compute_full_winner_sha256(w) for a, w in winners.items()},
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    seal_path.write_bytes(seal_bytes)
    seal_sha = _sha256_bytes(seal_bytes)

    probe_code = """
import importlib.util, json, sys
from pathlib import Path
cfg = json.loads(sys.argv[1])
# Exact Colab sys.path setup: shim_dir, work_dir, then orch_dir/scripts (from direct script execution)
sys.path[:] = [cfg["shim_dir"], cfg["work_dir"], str(Path(cfg["orch_script"]).parent)] + [
    p for p in sys.path if "Development/fyp" not in p
]
spec_mod = importlib.util.spec_from_file_location("run_colab_shard_main", cfg["orch_script"])
mod = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(mod)
res = mod._verify_immutable_seal_standalone(
    Path(cfg["seal_path"]),
    expected_model_id=cfg["model_id"],
    expected_model_revision=cfg["model_revision"],
    expected_git_sha=cfg["git_commit"],
    expected_manifest_sha=cfg["manifest_sha"],
    expected_gpu_name=cfg["gpu_name"],
    expected_memory_metric=cfg["mem_metric"],
    expected_complete_arms=tuple(cfg["arms"]),
    campaigns_model_dir=Path(cfg["c_dir"]),
    expected_split_ids={"validation": set(cfg["val_ids"]), "test": set(cfg["test_ids"])},
    expected_seal_sha256=cfg["seal_sha"],
    shim_path=Path(cfg["shim_dir"]) / "sitecustomize.py",
    frozen_plan_path=Path(cfg["orch_dir"]) / "docs" / "PRE_UNSEAL_ANALYSIS_PLAN.md",
    verify_code_digests=True,
)
assert len(res["winners_by_arm"]) == 7
"""
    probe_arg = json.dumps(
        {
            "shim_dir": str(shim_dir),
            "work_dir": str(work_dir),
            "orch_dir": str(orch_dir),
            "orch_script": str(orch_dir / "scripts" / "run_colab_shard.py"),
            "seal_path": str(seal_path),
            "model_id": spec["model_id"],
            "model_revision": spec["model_revision"],
            "git_commit": spec["git_commit"],
            "manifest_sha": manifest_sha,
            "gpu_name": spec["locked_gpu_name"],
            "mem_metric": spec["locked_memory_metric_name"],
            "arms": list(spec["arms"]),
            "c_dir": str(c_dir),
            "val_ids": val_ids,
            "test_ids": split_sets["test"],
            "seal_sha": seal_sha,
        }
    )
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    clean_env["PYTHONPATH"] = f"{shim_dir}:{work_dir}"
    subprocess.run([sys.executable, "-c", probe_code, probe_arg], cwd=str(work_dir), env=clean_env, check=True)

    # Deleting any frozen artifact in ORCH_DIR MUST fail closed instead of silently skipping!
    (orch_dir / "scripts" / "evaluate_final_test_suite.py").unlink()
    proc_fail = subprocess.run(
        [sys.executable, "-c", probe_code, probe_arg],
        cwd=str(work_dir),
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert proc_fail.returncode != 0
    assert "Missing frozen artifact" in proc_fail.stderr


def test_remote_first_attempt_logging_and_restart_recovery(tmp_path: Path) -> None:
    """Prove failed remote commit leaves NO phantom local COMPLETED and restart closes open LAUNCHED idempotently."""
    class _MockInfo:
        sha = "head_0"

    class _MockApi:
        def __init__(self) -> None:
            self.fail_commit = False
            self.committed_paths: list[str] = []

        def dataset_info(self, repo_id: str) -> _MockInfo:
            return _MockInfo()

        def create_commit(self, *, operations: list[Any], **kwargs: Any) -> None:
            if self.fail_commit:
                raise RuntimeError("Simulated HF 503 during COMPLETED commit")
            for op in operations:
                self.committed_paths.append(op.path_in_repo)

    api = _MockApi()
    persist = tmp_path / "persist"
    run_id = "aloe_winner_s42_test"
    seal_sha = "1" * 64

    # 1. LAUNCHED succeeds remotely + locally
    _log_hf_attempt(
        api,
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        persist_dir=persist,
        run_id=run_id,
        stage="LAUNCHED",
        attempt_id="aid_crash_test",
        seal_sha256=seal_sha,
        seq=1,
    )
    assert _mps_find_open_aid(persist, run_id) == "aid_crash_test"
    assert _mps_has_completed(persist, run_id) is False

    # 2. COMPLETED fails remotely -> NO local COMPLETED file may exist!
    api.fail_commit = True
    with pytest.raises(RuntimeError, match="Simulated HF 503"):
        _log_hf_attempt(
            api,
            hf_repo="Walapalam/aloe-ipd-v1-runs",
            persist_dir=persist,
            run_id=run_id,
            stage="COMPLETED",
            attempt_id="aid_crash_test",
            seal_sha256=seal_sha,
            seq=2,
        )
    assert _mps_has_completed(persist, run_id) is False
    assert _mps_find_open_aid(persist, run_id) == "aid_crash_test"

    # 3. Restart replays existing attempt files and closes the exact open LAUNCHED attempt_id
    api.fail_commit = False
    _mps_replay_attempts(api, hf_repo="Walapalam/aloe-ipd-v1-runs", persist_dir=persist)
    open_aid = _mps_find_open_aid(persist, run_id)
    assert open_aid == "aid_crash_test"
    _log_hf_attempt(
        api,
        hf_repo="Walapalam/aloe-ipd-v1-runs",
        persist_dir=persist,
        run_id=run_id,
        stage="COMPLETED",
        attempt_id=open_aid,
        seal_sha256=seal_sha,
        seq=2,
    )
    assert _mps_has_completed(persist, run_id) is True
    assert _mps_find_open_aid(persist, run_id) is None

    # Same remote-first guarantee on Colab syncer path
    class _FailingSyncer:
        shard_id = "T1"
        import threading as _th
        _lock = _th.Lock()

        def _cas_commit_owned(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Simulated Colab CAS failure")

    colab_store = tmp_path / "colab_store"
    with pytest.raises(RuntimeError, match="Simulated Colab CAS failure"):
        _log_final_test_attempt(
            _FailingSyncer(),  # type: ignore[arg-type]
            slug="LiquidAI--LFM3-1.7B",
            run_id=run_id,
            stage="COMPLETED",
            attempt_id="aid_colab_1",
            seal_sha256=seal_sha,
            seq=2,
            local_store=colab_store,
        )
    assert _colab_has_completed(colab_store, "LiquidAI--LFM3-1.7B", run_id) is False
    assert _colab_find_open_aid(colab_store, "LiquidAI--LFM3-1.7B", run_id) is None


def test_shared_bounds_hypervolume_invariance_and_gpu_isolation(tmp_path: Path) -> None:
    s42_dir = tmp_path / "s42"
    s1337_dir = tmp_path / "s1337"
    for d, acc_a, acc_b in ((s42_dir, 0.80, 0.78), (s1337_dir, 0.84, 0.79)):
        for arm, acc in (("aloe", acc_a), ("b1", acc_b)):
            adir = d / arm
            adir.mkdir(parents=True, exist_ok=True)
            (adir / "campaign_ledger.json").write_text(
                json.dumps(
                    {
                        "model_id": "LiquidAI/LFM3-1.7B",
                        "arm": arm,
                        "locked_gpu_name": "NVIDIA L4",
                        "locked_memory_metric_name": "cuda_max_memory_allocated",
                        "entries": [
                            {
                                "trial_id": "t0",
                                "status": "DONE",
                                "metrics": {
                                    "exact_match_accuracy": acc,
                                    "tool_accuracy": acc,
                                    "trainable_parameters": 8000000.0,
                                    "search_memory_mb": 12000.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

    bounds = compute_union_raw_max_bounds([s42_dir, s1337_dir])
    r42 = evaluate_model_campaigns_3d_hv(s42_dir, shared_bounds=bounds, normalization_scope="union_test")
    r1337 = evaluate_model_campaigns_3d_hv(s1337_dir, shared_bounds=bounds, normalization_scope="union_test")
    assert r42["shared_raw_bounds"] == r1337["shared_raw_bounds"]
    assert r1337["arms"]["aloe"]["normalized_3d_hv"] > r42["arms"]["aloe"]["normalized_3d_hv"]

    bad_dir = tmp_path / "s_bad" / "aloe"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "campaign_ledger.json").write_text(
        json.dumps(
            {
                "model_id": "LiquidAI/LFM3-1.7B",
                "arm": "aloe",
                "locked_gpu_name": "Apple M1 Max",
                "locked_memory_metric_name": "process_rss_sampled_100ms",
                "entries": [
                    {
                        "trial_id": "t0",
                        "status": "DONE",
                        "metrics": {
                            "exact_match_accuracy": 0.8,
                            "tool_accuracy": 0.8,
                            "trainable_parameters": 8000000.0,
                            "search_memory_mb": 2000.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="mixed GPU SKUs"):
        compute_union_raw_max_bounds([s42_dir, tmp_path / "s_bad"])

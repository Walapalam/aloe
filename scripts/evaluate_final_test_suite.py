#!/usr/bin/env python3
"""Pre-Registered 63-Receipt Paired Statistical Evaluator (`evaluate_final_test_suite.py`).

Executes the frozen `PRE_UNSEAL_ANALYSIS_PLAN.md` methodology over the 63 verified receipts
(60 winnerxseed runs across `[42, 1337, 2026]` + 3 hardware-matched deterministic zero-shot runs):
- Locks `bootstrap_resamples == 10_000` and `rng_seed == 42` for official `splits["test"]` runs.
- Enforces the 63-receipt gate, append-only attempt-log audit, explicit `expected_seal_shas`,
  and frozen `PRE_UNSEAL_ANALYSIS_PLAN.md` + evaluator/auditor/bootstrap SHA-256 digests.
- Fails closed if `paired_bootstrap(..., ci_method="bca")` ever falls back from `"bca"`.
- Emits the exact pre-registered `composite_claim` (`"SUPERIOR_EM"` /
  `"NON_INFERIOR_EFFICIENCY_WIN"` / `"INCONCLUSIVE_OR_INFERIOR"`) for `ALOE vs {B0, B1, B2}`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloe.evaluation.bootstrap import holm_bonferroni, mcnemar_exact, paired_bootstrap  # noqa: E402
from aloe.evaluation.prediction_store import PredictionStore, build_paired_outcome_matrix  # noqa: E402
from aloe.execution.trial_config import TrialConfig  # noqa: E402
from aloe.tracking.trial_store import atomic_write_text  # noqa: E402
from scripts.final_test_audit import (  # noqa: E402
    CANONICAL_PROTOCOL_MODELS,
    compute_frozen_code_digests,
    load_ceremony_pins,
    verify_complete_63_final_test_gate,
    verify_strict_trial_receipt,
    verify_strict_zero_shot_receipt,
)
from scripts.run_colab_a100_campaign import REQUIRED_CONFIRM_SEEDS  # noqa: E402
from scripts.verify_campaign import sha256_file, verify_manifest  # noqa: E402

PRIMARY_COMPARATORS = ("b0", "b1", "b2")
NON_INFERIORITY_MARGIN = -0.02  # -2.0 percentage points on Exact Match


def _per_seed_discordant_counts(
    a_mat: dict[str, list[int]],
    b_mat: dict[str, list[int]],
    seeds: list[int],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    qids = sorted(a_mat.keys())
    for idx, s_val in enumerate(seeds):
        n11 = n10 = n01 = n00 = 0
        for q in qids:
            va = bool(a_mat[q][idx])
            vb = bool(b_mat[q][idx])
            if va and vb:
                n11 += 1
            elif va and not vb:
                n10 += 1
            elif not va and vb:
                n01 += 1
            else:
                n00 += 1
        out[f"seed_{s_val}"] = {
            "both_correct": n11,
            "aloe_only": n10,
            "comparator_only": n01,
            "both_wrong": n00,
        }
    return out


def _compute_resource_gains(
    aloe_params: int,
    aloe_mem_raw: float,
    comp: str,
    comp_nom_params: int,
    comp_act_params_raw: float,
    comp_mem_raw: float,
) -> dict[str, Any]:
    if comp == "zero_shot":
        return {
            "applicable": False,
            "fewer_nominal_params": False,
            "fewer_active_params": False,
            "strict_param_gain": False,
            "strict_memory_gain": False,
            "any_strict_resource_gain": False,
        }
    fewer_nom = bool(aloe_params < comp_nom_params)
    fewer_act = bool(aloe_params < comp_act_params_raw)
    strict_param = bool(fewer_nom and fewer_act)
    strict_mem = bool(aloe_mem_raw < comp_mem_raw)
    return {
        "applicable": True,
        "fewer_nominal_params": fewer_nom,
        "fewer_active_params": fewer_act,
        "strict_param_gain": strict_param,
        "strict_memory_gain": strict_mem,
        "any_strict_resource_gain": bool(strict_param or strict_mem),
    }


def evaluate_63_receipt_suite(
    dest_dir: Path,
    *,
    manifest_path: Path,
    raw_data_path: Path | None,
    dry_run_validation: bool = False,
    bootstrap_resamples: int = 10_000,
    rng_seed: int = 42,
    expected_seal_shas: dict[str, str] | None = None,
    ceremony_pins_path: Path | None = None,
    verify_code_digests: bool = True,
) -> dict[str, Any]:
    if not dry_run_validation and (bootstrap_resamples != 10_000 or rng_seed != 42):
        raise PermissionError("Official analysis requires bootstrap_resamples=10000 and rng_seed=42")

    m_audit = verify_manifest(
        manifest_path,
        raw_data_path=raw_data_path if (raw_data_path and raw_data_path.is_file()) else None,
        enforce_frozen_v1=True,
        require_raw_data=bool(raw_data_path and raw_data_path.is_file()),
    )
    manifest_sha = str(m_audit["manifest_sha256"])
    split_id_sets = m_audit["split_id_sets"]

    gate_summary: dict[str, Any] | None = None
    if dry_run_validation:
        split_name = "validation"
        trials_root_name = "confirm_3seed"
        zs_root_name = "zero_shot_baselines"
    else:
        split_name = "test"
        trials_root_name = "final_sealed_test"
        zs_root_name = "zero_shot_test"
        pins = expected_seal_shas
        if pins is None:
            pins = load_ceremony_pins(ceremony_pins_path).get("seal_sha256_by_model", {})
        gate_summary = verify_complete_63_final_test_gate(
            dest_dir,
            manifest_sha=manifest_sha,
            split_id_sets=split_id_sets,
            expected_seal_shas=pins,
            require_all_63=True,
            require_attempt_logs=True,
            verify_code_digests=verify_code_digests,
        )
        frozen_plan = dest_dir / "final_sealed_test" / "PRE_UNSEAL_ANALYSIS_PLAN.md"
        if not frozen_plan.is_file():
            raise PermissionError(f"Missing HF-frozen analysis plan: {frozen_plan}")
        frozen_plan_sha = sha256_file(frozen_plan)
        live_digests = compute_frozen_code_digests(REPO_ROOT) if verify_code_digests else None
        for slug in CANONICAL_PROTOCOL_MODELS:
            seal = json.loads((dest_dir / "final_sealed_test" / slug / "test_unsealed.json").read_text(encoding="utf-8"))
            if seal.get("analysis_plan_sha256") != frozen_plan_sha:
                raise PermissionError(
                    f"Frozen analysis-plan SHA-256 mismatch for {slug}: "
                    f"{seal.get('analysis_plan_sha256')!r} != {frozen_plan_sha!r}"
                )
            if live_digests is not None:
                for hk in ("analysis_plan_sha256", "audit_script_sha256", "evaluator_script_sha256", "bootstrap_engine_sha256"):
                    if seal.get(hk) != live_digests[hk]:
                        raise PermissionError(f"Official evaluator code/plan digest mismatch on {hk} for {slug}")

    expected_ids = split_id_sets[split_name]
    models_analysis: dict[str, Any] = {}

    for slug, spec in CANONICAL_PROTOCOL_MODELS.items():
        arm_matrices: dict[str, dict[str, list[int]]] = {}
        arm_descriptives: dict[str, Any] = {}
        raw_arm_resources: dict[str, tuple[int, float, float]] = {}

        # 1. Ingest & verify all canonical trained arms across exact seeds [42, 1337, 2026]
        for arm in spec["arms"]:
            sfile = dest_dir / "campaigns" / slug / arm / "campaign_summary.json"
            sdata = json.loads(sfile.read_text(encoding="utf-8"))
            wcfg = TrialConfig.model_validate(sdata["selected_winner_config"])

            stores_by_seed: dict[int, PredictionStore] = {}
            per_seed_em: dict[str, float] = {}
            mems_mb: list[float] = []
            nominal_params = 0
            adalora_init_params: int | None = None
            adalora_active_list: list[int] = []

            for s_val in REQUIRED_CONFIRM_SEEDS:
                tid = f"{arm}_winner_s{s_val}_{split_name}"
                tjson = dest_dir / trials_root_name / slug / arm / "trials" / tid / "trial.json"
                verify_strict_trial_receipt(
                    tjson,
                    expected_seed=s_val,
                    expected_split=split_name,
                    expected_split_ids=expected_ids,
                    expected_manifest_sha=manifest_sha,
                    expected_git_sha=spec["git_commit"],
                    expected_model_id=spec["model_id"],
                    expected_model_revision=spec["model_revision"],
                    expected_gpu_name=spec["locked_gpu_name"],
                    expected_memory_metric=spec["locked_memory_metric_name"],
                    expected_winner_cfg=wcfg,
                    raise_on_error=True,
                )
                rec = json.loads(tjson.read_text(encoding="utf-8"))
                m = rec["metrics"]
                per_seed_em[str(s_val)] = float(m["ast_exact_match_accuracy"])
                mems_mb.append(float(m["search_memory_mb"]))
                nominal_params = int(m["trainable_params"])
                if m.get("adalora_init_params") is not None:
                    adalora_init_params = int(m["adalora_init_params"])
                if m.get("adalora_final_active_params") is not None:
                    adalora_active_list.append(int(m["adalora_final_active_params"]))
                stores_by_seed[s_val] = PredictionStore(tjson.parent / f"predictions_{split_name}.jsonl")

            if list(stores_by_seed.keys()) != REQUIRED_CONFIRM_SEEDS:
                raise RuntimeError(f"Exact seed order mismatch for {slug}/{arm}: {list(stores_by_seed.keys())}")
            mat = build_paired_outcome_matrix(stores_by_seed)
            if set(mat.keys()) != expected_ids:
                raise RuntimeError(f"Outcome matrix ID mismatch for {slug}/{arm}")
            arm_matrices[arm] = mat

            em_vals = [per_seed_em[str(s)] for s in REQUIRED_CONFIRM_SEEDS]
            raw_mean_mem = statistics.mean(mems_mb)
            raw_act_params = float(nominal_params)
            desc_entry: dict[str, Any] = {
                "per_seed_exact_match": per_seed_em,
                "mean_exact_match": round(statistics.mean(em_vals), 6),
                "sd_exact_match": round(statistics.stdev(em_vals), 6),
                "mean_search_memory_mb": round(raw_mean_mem, 3),
                "sd_search_memory_mb": round(statistics.stdev(mems_mb), 3),
                "nominal_trainable_params": nominal_params,
                "nominal_delta_vs_b0_pct": round(100.0 * (nominal_params - spec["baseline_params"]) / spec["baseline_params"], 2),
            }
            if adalora_init_params is not None and adalora_active_list:
                mean_act = statistics.mean(adalora_active_list)
                raw_act_params = float(mean_act)
                desc_entry["adalora_init_params"] = adalora_init_params
                desc_entry["adalora_retained_active_params_per_seed"] = adalora_active_list
                desc_entry["adalora_mean_retained_active_params"] = round(mean_act, 1)
                desc_entry["adalora_retained_delta_vs_b0_pct"] = round(
                    100.0 * (mean_act - spec["baseline_params"]) / spec["baseline_params"], 2
                )
            arm_descriptives[arm] = desc_entry
            raw_arm_resources[arm] = (nominal_params, raw_act_params, raw_mean_mem)

        # 2. Ingest & verify hardware-matched deterministic zero-shot baseline (1 run -> aligned across 3 seed slots)
        zs_json = dest_dir / zs_root_name / slug / f"zero_shot_{split_name}.json"
        zs_rec = verify_strict_zero_shot_receipt(
            zs_json,
            expected_split=split_name,
            expected_split_ids=expected_ids,
            expected_manifest_sha=manifest_sha,
            expected_git_sha=spec["git_commit"],
            expected_model_id=spec["model_id"],
            expected_model_revision=spec["model_revision"],
            expected_gpu_name=spec["locked_gpu_name"],
            expected_memory_metric=spec["locked_memory_metric_name"],
            raise_on_error=True,
        )
        assert zs_rec is not None
        zs_store = PredictionStore(zs_json.parent / f"predictions_{split_name}.jsonl")
        zs_single = build_paired_outcome_matrix({42: zs_store})
        if set(zs_single.keys()) != expected_ids:
            raise RuntimeError(f"Zero-shot ID mismatch for {slug}")
        arm_matrices["zero_shot"] = {qid: [vals[0], vals[0], vals[0]] for qid, vals in zs_single.items()}
        zs_mem_raw = float(zs_rec.get("search_memory_mb", zs_rec.get("peak_cuda_allocated_mb", 0.0)))
        arm_descriptives["zero_shot"] = {
            "deterministic_single_run": True,
            "exact_match": float(zs_rec["exact_match_accuracy"]),
            "parse_success_rate": float(zs_rec["parse_success_rate"]),
            "search_memory_mb": zs_mem_raw,
            "nominal_trainable_params": 0,
        }
        raw_arm_resources["zero_shot"] = (0, 0.0, zs_mem_raw)

        # 3. Paired comparisons: ALOE vs Primary Family {B0, B1, B2} (Holm-corrected) + Secondary {borobia_*, zero_shot}
        aloe_mat = arm_matrices["aloe"]
        aloe_nom_p, _, aloe_raw_mem = raw_arm_resources["aloe"]
        primary_raw_p: dict[str, float] = {}
        raw_bca_low_by_comp: dict[str, float] = {}
        raw_comp_records: dict[str, dict[str, Any]] = {}

        all_comparators = [a for a in list(spec["arms"]) + ["zero_shot"] if a != "aloe"]
        for comp in all_comparators:
            c_mat = arm_matrices[comp]
            bca = paired_bootstrap(
                aloe_mat,
                c_mat,
                B=bootstrap_resamples,
                alpha=0.05,
                ci_method="bca",
                rng_seed=rng_seed,
            )
            if bca.ci_method != "bca":
                raise RuntimeError(
                    f"Pre-registered BCa CI failed for {slug}/ALOE vs {comp} (got {bca.ci_method!r}); "
                    "refusing to substitute a percentile fallback."
                )
            pct = paired_bootstrap(
                aloe_mat,
                c_mat,
                B=bootstrap_resamples,
                alpha=0.05,
                ci_method="percentile",
                rng_seed=rng_seed,
            )
            mcn = mcnemar_exact(aloe_mat, c_mat)
            c_nom_p, c_act_p, c_raw_mem = raw_arm_resources[comp]
            rgains = _compute_resource_gains(aloe_nom_p, aloe_raw_mem, comp, c_nom_p, c_act_p, c_raw_mem)
            raw_bca_low_by_comp[comp] = float(bca.ci_low)
            is_primary = comp in PRIMARY_COMPARATORS
            if is_primary:
                primary_raw_p[comp] = bca.p_value

            raw_comp_records[comp] = {
                "family": "primary_holm_3" if is_primary else "secondary_exploratory",
                "mean_em_diff_aloe_minus_comp": round(bca.mean_diff, 6),
                "bca_95_ci": [round(bca.ci_low, 6), round(bca.ci_high, 6)],
                "bca_ci_method_actual": bca.ci_method,
                "percentile_95_ci": [round(pct.ci_low, 6), round(pct.ci_high, 6)],
                "bootstrap_p_raw": round(bca.p_value, 6),
                "cohens_d": round(bca.cohens_d, 4) if bca.cohens_d is not None else None,
                "non_inferior_at_2pp_bca": bool(bca.ci_low >= NON_INFERIORITY_MARGIN),
                "resource_gains_aloe_vs_comp": rgains,
                "superior_unadjusted": bool(bca.superior),
                "mcnemar_majority_vote": {
                    "both_correct": mcn.n_both_correct,
                    "aloe_only": mcn.n_a_only,
                    "comparator_only": mcn.n_b_only,
                    "both_wrong": mcn.n_both_wrong,
                    "discordant_pairs": mcn.discordant_pairs,
                    "exact_p_value": round(mcn.exact_p_value, 6),
                },
                "per_seed_discordance": _per_seed_discordant_counts(aloe_mat, c_mat, REQUIRED_CONFIRM_SEEDS),
            }

        holm_res = holm_bonferroni(primary_raw_p, alpha=0.05)
        for comp, h in holm_res.items():
            rec_c = raw_comp_records[comp]
            holm_sup = bool(h.reject_null and raw_bca_low_by_comp[comp] > 0.0)
            ni_eff = bool(
                (not holm_sup)
                and rec_c["non_inferior_at_2pp_bca"]
                and rec_c["resource_gains_aloe_vs_comp"]["any_strict_resource_gain"]
            )
            if holm_sup:
                claim = "SUPERIOR_EM"
            elif ni_eff:
                claim = "NON_INFERIOR_EFFICIENCY_WIN"
            else:
                claim = "INCONCLUSIVE_OR_INFERIOR"

            rec_c["holm_adjusted_p"] = round(h.adjusted_p, 6)
            rec_c["holm_reject_null_alpha_0_05"] = bool(h.reject_null)
            rec_c["holm_superior"] = holm_sup
            rec_c["composite_claim"] = claim

        models_analysis[slug] = {
            "model_id": spec["model_id"],
            "locked_gpu_name": spec["locked_gpu_name"],
            "locked_memory_metric_name": spec["locked_memory_metric_name"],
            "split": split_name,
            "n_items": len(expected_ids),
            "seeds": list(REQUIRED_CONFIRM_SEEDS),
            "bootstrap_resamples": bootstrap_resamples,
            "rng_seed": rng_seed,
            "non_inferiority_margin_em": NON_INFERIORITY_MARGIN,
            "arms": arm_descriptives,
            "paired_comparisons_aloe_vs": raw_comp_records,
        }

    out_payload: dict[str, Any] = {
        "protocol_tag": "protocol-v1",
        "evaluation_mode": "dry_run_validation_63" if dry_run_validation else "official_sealed_test_63",
        "inferential_scope": (
            "orchestration_dry_run_selection_inclusive_non_inferential"
            if dry_run_validation
            else "pre_registered_confirmatory_unseen_tool_test"
        ),
        "manifest_sha256": manifest_sha,
        "split": split_name,
        "n_items": len(expected_ids),
        "bootstrap_resamples": bootstrap_resamples,
        "rng_seed": rng_seed,
        "models": models_analysis,
    }
    if gate_summary is not None:
        out_payload["attempt_audit"] = gate_summary["attempt_audit"]
    return out_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 63-receipt Protocol-v1 suite with paired bootstrap & Holm")
    parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifests" / "xlam_ipd_v1.json"))
    parser.add_argument("--raw-data", default=str(REPO_ROOT / "data" / "raw" / "xlam" / "xlam_function_calling_60k.json"))
    parser.add_argument("--ceremony-pins", default=str(REPO_ROOT / "docs" / "CEREMONY_PINS.json"))
    parser.add_argument("--dry-run-validation", action="store_true", help="Run on 63 validation receipts without touching test")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    res = evaluate_63_receipt_suite(
        Path(args.dest_dir).resolve(),
        manifest_path=Path(args.manifest).resolve(),
        raw_data_path=Path(args.raw_data).resolve() if args.raw_data else None,
        dry_run_validation=args.dry_run_validation,
        bootstrap_resamples=args.bootstrap,
        rng_seed=args.seed,
        ceremony_pins_path=Path(args.ceremony_pins).resolve(),
    )
    encoded = json.dumps(res, indent=2)
    if args.out:
        atomic_write_text(Path(args.out).resolve(), encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

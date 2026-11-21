#!/usr/bin/env python3
"""Zero-Dependency Standard-Library Campaign & Split Auditor (`verify_campaign.py`).

Designed so offline audit environments (including sandboxes without `torch`, `peft`,
or `botorch` installed) can independently verify structural and cryptographic integrity:
1. Frozen 3-way xLAM split manifest cardinalities (`1,000 / 200 / 400`), frozen split
   SHA-256 digests, category-stratification counts, zero pairwise tool overlap, and
   independent raw-file SHA-256 & tool-set recomputation.
2. Campaign ledger & summary integrity: `verified_done_trials == target_trials` (default
   requiring `20` for search arms and `1` for `borobia_*`, `--allow-smoke-budget` for CI
   smoke runs), unique signatures across all entries, `dataset_manifest_sha256` &
   `is_budget_matched_5pct` checks, path containment under `trials/<trial_id>/`,
   full per-record JSONL metadata & split-ID verification (`200` validation / `400` test),
   and deterministic winner verification against `campaign_summary.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

FROZEN_XLAM_IPD_V1 = {
    "train": {
        "count": 1000,
        "sha256": "d3e47bbba1604c3befa526503363ea3177724b52d691547ea1d5b8999c29c945",
        "categories": {"multi_plausible_tools": 412, "obvious_single_tool": 588},
    },
    "validation": {
        "count": 200,
        "sha256": "664f29bfc62774f40a7780af48ff38bc7bdbd75dff04bc85c34bdb504883bc53",
        "categories": {"multi_plausible_tools": 82, "obvious_single_tool": 118},
    },
    "test": {
        "count": 400,
        "sha256": "26cdf457f083336ceea215049d4a76a553e0b58e909ec48a7e7323d7e19769c0",
        "categories": {"multi_plausible_tools": 165, "obvious_single_tool": 235},
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _extract_item_tools(item: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    raw_tools = item.get("tools", [])
    if isinstance(raw_tools, str):
        raw_tools = json.loads(raw_tools)
    for t in raw_tools:
        if isinstance(t, dict):
            fn = t.get("function", t)
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                names.add(fn["name"])
    raw_ans = item.get("answers", [])
    if isinstance(raw_ans, str):
        raw_ans = json.loads(raw_ans)
    for a in raw_ans:
        if isinstance(a, dict) and isinstance(a.get("name"), str):
            names.add(a["name"])
    return names


def verify_manifest(
    manifest_path: Path,
    *,
    raw_data_path: Path | None = None,
    enforce_frozen_v1: bool = True,
    require_raw_data: bool = False,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tools = data["tools"]
    train_t = set(tools.get("train_tool_names", []))
    val_t = set(tools.get("val_tool_names", []))
    test_t = set(tools.get("test_tool_names", []))

    if train_t & val_t or train_t & test_t or val_t & test_t:
        raise AssertionError("Pairwise tool overlap detected in manifest!")

    splits = data["splits"]
    if enforce_frozen_v1 and set(splits.keys()) != set(FROZEN_XLAM_IPD_V1.keys()):
        raise AssertionError(f"Frozen split set mismatch: {sorted(splits.keys())}")

    split_counts: dict[str, int] = {}
    split_id_sets: dict[str, set[str]] = {}
    all_seen_ids: set[str] = set()
    for sname, sbody in splits.items():
        ids = [str(x) for x in sbody["ids"]]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate IDs within split '{sname}'!")
        if len(ids) != int(sbody["count"]):
            raise AssertionError(f"Split '{sname}' count mismatch: {len(ids)} != {sbody['count']}")
        if sha256_ids(ids) != sbody["sha256"]:
            raise AssertionError(f"Split '{sname}' SHA-256 mismatch!")
        if enforce_frozen_v1:
            exp = FROZEN_XLAM_IPD_V1[sname]
            if len(ids) != exp["count"]:
                raise AssertionError(f"Frozen cardinality mismatch for '{sname}': {len(ids)} != {exp['count']}")
            if sbody["sha256"] != exp["sha256"]:
                raise AssertionError(f"Non-frozen split SHA-256 for '{sname}'!")
            if sbody.get("categories") != exp["categories"]:
                raise AssertionError(
                    f"Category stratification mismatch for '{sname}': {sbody.get('categories')} != {exp['categories']}"
                )
        if all_seen_ids & set(ids):
            raise AssertionError(f"Duplicate example IDs across splits involving '{sname}'!")
        all_seen_ids.update(ids)
        split_counts[sname] = len(ids)
        split_id_sets[sname] = set(ids)

    raw_verified = False
    candidate_raw = raw_data_path or Path("data/raw/xlam/xlam_function_calling_60k.json")
    if require_raw_data and not candidate_raw.is_file():
        raise FileNotFoundError(f"Raw xLAM file required for full audit not found: {candidate_raw}")
    if candidate_raw.is_file():
        expected_raw_sha = data.get("dataset", {}).get("raw_file_sha256")
        if not expected_raw_sha or expected_raw_sha == "UNKNOWN":
            raise AssertionError("Manifest missing dataset.raw_file_sha256!")
        actual_raw_sha = sha256_file(candidate_raw)
        if actual_raw_sha != expected_raw_sha:
            raise AssertionError(f"Raw xLAM SHA-256 mismatch: {actual_raw_sha} != {expected_raw_sha}")
        raw_items = json.loads(candidate_raw.read_text(encoding="utf-8"))
        by_id = {str(it.get("id", idx)): it for idx, it in enumerate(raw_items)}
        recomp: dict[str, set[str]] = {}
        for sname, sbody in splits.items():
            st: set[str] = set()
            for sid in sbody["ids"]:
                sid_s = str(sid)
                if sid_s not in by_id:
                    raise AssertionError(f"Split '{sname}' ID {sid_s} missing from raw xLAM file!")
                st.update(_extract_item_tools(by_id[sid_s]))
            recomp[sname] = st
        if recomp["train"] & recomp["validation"] or recomp["train"] & recomp["test"] or recomp["validation"] & recomp["test"]:
            raise AssertionError("Recomputed raw xLAM tool sets have non-zero pairwise overlap!")
        if recomp["train"] != train_t or recomp["validation"] != val_t or recomp["test"] != test_t:
            raise AssertionError("Manifest declared tool lists diverge from recomputed raw xLAM tools!")
        raw_verified = True

    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_counts": split_counts,
        "split_id_sets": split_id_sets,
        "tool_counts": {
            "train": len(train_t),
            "validation": len(val_t),
            "test": len(test_t),
        },
        "pairwise_tool_overlap": 0,
        "raw_xlam_independently_verified": raw_verified,
        "status": "PASS",
    }


def verify_campaign_dir(
    campaign_dir: Path,
    *,
    expected_manifest_sha: str | None = None,
    expected_split_ids: dict[str, set[str]] | None = None,
    allow_smoke_budget: bool = False,
) -> dict[str, Any]:
    ledger_file = campaign_dir / "campaign_ledger.json"
    if not ledger_file.is_file():
        raise FileNotFoundError(f"No campaign_ledger.json found in {campaign_dir}")

    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    if expected_manifest_sha and ledger.get("manifest_sha256") != expected_manifest_sha:
        raise AssertionError(
            f"Campaign ledger manifest_sha256 ({ledger.get('manifest_sha256')}) != "
            f"verified manifest SHA-256 ({expected_manifest_sha})"
        )

    arm = str(ledger.get("arm", ""))
    entries = ledger.get("entries", [])
    all_sigs = [e["config_signature"] for e in entries]
    if len(all_sigs) != len(set(all_sigs)):
        raise AssertionError(f"Duplicate configuration signatures found across ledger entries in {ledger_file}!")

    done_entries = [e for e in entries if e.get("status") == "DONE"]
    target_trials = int(ledger["target_trials"])
    expected_protocol_target = 1 if arm.startswith("borobia_") else 20
    if not allow_smoke_budget and target_trials != expected_protocol_target:
        raise AssertionError(
            f"Non-protocol target_trials={target_trials} for arm='{arm}' (expected {expected_protocol_target}; "
            "pass --allow-smoke-budget for non-official CI smoke runs)."
        )
    if len(done_entries) != target_trials:
        raise AssertionError(
            f"Incomplete campaign in {ledger_file}: {len(done_entries)} DONE != target_trials ({target_trials})"
        )

    locked_gpu = ledger.get("locked_gpu_name")
    locked_mem = ledger.get("locked_memory_metric_name")

    trials_root = (campaign_dir / "trials").resolve()
    verified_trials = 0
    for e in done_entries:
        tid = str(e["trial_id"])
        trial_dir = (trials_root / tid).resolve()
        if trials_root not in trial_dir.parents:
            raise AssertionError(f"Path traversal in trial_id '{tid}': {trial_dir} escapes {trials_root}")
        tjson = trial_dir / "trial.json"
        if not tjson.is_file():
            raise FileNotFoundError(f"Missing trial.json for DONE entry {tid}")
        trec = json.loads(tjson.read_text(encoding="utf-8"))
        if trec.get("status") != "PASS":
            raise AssertionError(f"Trial {tid} status is {trec.get('status')}, expected PASS")
        if expected_manifest_sha and not allow_smoke_budget:
            if trec.get("dataset_manifest_sha256") != expected_manifest_sha:
                raise AssertionError(
                    f"Trial {tid} dataset_manifest_sha256 ({trec.get('dataset_manifest_sha256')}) != {expected_manifest_sha}"
                )

        metrics = trec.get("metrics") or {}
        l_metrics = e.get("metrics") or {}
        tc_dict = trec.get("trial_config") or {}
        if tc_dict.get("peft_method", "lora") == "lora" and metrics.get("is_budget_matched_5pct") is not True:
            raise AssertionError(f"LoRA trial {tid} failed ±5% parity check: is_budget_matched_5pct={metrics.get('is_budget_matched_5pct')}")

        acc = float(metrics.get("ast_exact_match_accuracy", metrics.get("tool_accuracy", -1.0)))
        mem_mb = float(metrics.get("search_memory_mb") or 0.0)
        params = float(metrics.get("trainable_params") or 0.0)
        if not (math.isfinite(acc) and 0.0 <= acc <= 1.0):
            raise AssertionError(f"Invalid accuracy in {tid}: {acc}")
        if not (math.isfinite(mem_mb) and mem_mb > 0.0):
            raise AssertionError(f"Non-positive or non-finite search_memory_mb in {tid}: {mem_mb}")
        if not (math.isfinite(params) and params > 0.0):
            raise AssertionError(f"Invalid trainable_params in {tid}: {params}")

        if abs(float(l_metrics.get("tool_accuracy", -9.0)) - acc) > 1e-6:
            raise AssertionError(f"Ledger vs trial.json accuracy mismatch in {tid}")
        if abs(float(l_metrics.get("search_memory_mb", -9.0)) - mem_mb) > 1e-3:
            raise AssertionError(f"Ledger vs trial.json search_memory_mb mismatch in {tid}")
        if "trainable_parameters" in l_metrics and abs(float(l_metrics["trainable_parameters"]) - params) > 1e-3:
            raise AssertionError(f"Ledger vs trial.json trainable_params mismatch in {tid}")

        if locked_mem and metrics.get("memory_metric_name") != locked_mem:
            raise AssertionError(f"Memory metric drift in {tid}: {metrics.get('memory_metric_name')} != {locked_mem}")
        obs_gpu = trec.get("gpu_name") or trec.get("hardware")
        if locked_gpu and obs_gpu != locked_gpu:
            raise AssertionError(f"Hardware SKU drift in {tid}: {obs_gpu} != {locked_gpu}")

        pred_path = (trial_dir / "predictions_validation.jsonl").resolve()
        pred_rel = metrics.get("predictions_jsonl_path")
        if pred_rel:
            raw_pred_p = Path(pred_rel)
            if ".." in raw_pred_p.parts:
                raise AssertionError(f"Predictions path escapes trial directory in {tid}: {pred_rel}")
            cand_p = (trial_dir / raw_pred_p.name).resolve()
            if trial_dir not in cand_p.parents:
                raise AssertionError(f"Predictions path escapes trial directory in {tid}: {cand_p}")
            pred_path = cand_p
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing predictions JSONL for {tid}: {pred_path}")

        expected_sha = metrics.get("predictions_sha256")
        if not expected_sha or expected_sha == "UNKNOWN":
            raise AssertionError(f"Missing or UNKNOWN predictions_sha256 for {tid}")
        actual_sha = sha256_file(pred_path)
        if actual_sha != expected_sha:
            raise AssertionError(f"SHA-256 mismatch for {pred_path}: {actual_sha} != {expected_sha}")

        parsed_recs = [
            json.loads(ln) for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        eval_samples = int(metrics.get("eval_samples") or 0)
        if eval_samples <= 0 or len(parsed_recs) != eval_samples:
            raise AssertionError(
                f"Prediction line count ({len(parsed_recs)}) != eval_samples ({eval_samples}) in {tid}"
            )

        rec_ids = {str(r.get("sample_id")) for r in parsed_recs}
        if len(rec_ids) != len(parsed_recs):
            raise AssertionError(f"Duplicate sample_ids inside {pred_path}")
        for r in parsed_recs:
            if r.get("split") != "validation" or r.get("trial_id") != tid or r.get("arm") != arm:
                raise AssertionError(f"Prediction record metadata mismatch inside {pred_path}")
            if r.get("evaluator_version") != "2.0.0-strict":
                raise AssertionError(f"Unexpected evaluator_version in {pred_path}: {r.get('evaluator_version')}")
        recomp_acc = sum(1 for r in parsed_recs if r.get("exact_match") is True) / len(parsed_recs)
        if abs(recomp_acc - acc) > 1e-5:
            raise AssertionError(
                f"Stored trial accuracy ({acc}) != recomputed JSONL exact_match mean ({recomp_acc}) in {tid}"
            )
        if not allow_smoke_budget and expected_split_ids and "validation" in expected_split_ids:
            if rec_ids != expected_split_ids["validation"]:
                raise AssertionError(
                    f"Prediction IDs in {pred_path} ({len(rec_ids)}) do not match frozen 200 validation IDs!"
                )

        verified_trials += 1

    summary_file = campaign_dir / "campaign_summary.json"
    if not allow_smoke_budget and not summary_file.is_file():
        raise FileNotFoundError(f"Missing required campaign_summary.json in {campaign_dir}")
    if summary_file.is_file():
        sdata = json.loads(summary_file.read_text(encoding="utf-8"))
        best_ent = sorted(
            done_entries,
            key=lambda e: (
                -float(e["metrics"].get("exact_match_accuracy", e["metrics"]["tool_accuracy"])),
                float(e["metrics"]["search_memory_mb"]),
                float(e["metrics"]["trainable_parameters"]),
            ),
        )[0]
        if sdata.get("selected_winner_trial_id") != best_ent["trial_id"]:
            raise AssertionError(
                f"campaign_summary.json winner ({sdata.get('selected_winner_trial_id')}) != "
                f"recomputed deterministic winner ({best_ent['trial_id']})"
            )
        # Independently verify Pareto non-dominated set in stdlib
        expected_pareto: list[str] = []
        for a_ent in done_entries:
            am = a_ent["metrics"]
            a_acc = float(am.get("exact_match_accuracy", am["tool_accuracy"]))
            a_par = float(am["trainable_parameters"])
            a_mem = float(am["search_memory_mb"])
            dominated = False
            for b_ent in done_entries:
                if b_ent["trial_id"] == a_ent["trial_id"]:
                    continue
                bm = b_ent["metrics"]
                b_acc = float(bm.get("exact_match_accuracy", bm["tool_accuracy"]))
                b_par = float(bm["trainable_parameters"])
                b_mem = float(bm["search_memory_mb"])
                if (
                    b_acc >= a_acc
                    and b_par <= a_par
                    and b_mem <= a_mem
                    and (b_acc > a_acc or b_par < a_par or b_mem < a_mem)
                ):
                    dominated = True
                    break
            if not dominated:
                expected_pareto.append(a_ent["trial_id"])
        if sorted(sdata.get("pareto_trial_ids", [])) != sorted(expected_pareto):
            raise AssertionError(
                f"campaign_summary.json pareto_trial_ids ({sdata.get('pareto_trial_ids')}) != "
                f"recomputed Pareto set ({expected_pareto})"
            )

    return {
        "campaign_id": ledger["campaign_id"],
        "model_id": ledger["model_id"],
        "arm": arm,
        "verified_done_trials": verified_trials,
        "target_trials": target_trials,
        "unique_signatures": len(set(all_sigs)),
        "locked_gpu_name": locked_gpu,
        "locked_memory_metric_name": locked_mem,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stdlib-only ALOE manifest & campaign verifier")
    parser.add_argument("--manifest", default="data/manifests/xlam_ipd_v1.json")
    parser.add_argument("--raw-data", default="")
    parser.add_argument("--campaign-dir", default="")
    parser.add_argument("--allow-custom-manifest", action="store_true")
    parser.add_argument("--allow-smoke-budget", action="store_true")
    parser.add_argument("--require-raw-data", action="store_true")
    args = parser.parse_args()

    m_ver = verify_manifest(
        Path(args.manifest),
        raw_data_path=Path(args.raw_data) if args.raw_data else None,
        enforce_frozen_v1=not args.allow_custom_manifest,
        require_raw_data=args.require_raw_data,
    )
    split_id_sets = m_ver.pop("split_id_sets", None)
    out: dict[str, Any] = {
        "manifest_verification": m_ver,
    }
    if args.campaign_dir:
        cpath = Path(args.campaign_dir)
        m_sha = m_ver["manifest_sha256"]
        if (cpath / "campaign_ledger.json").is_file():
            out["campaigns"] = [
                verify_campaign_dir(
                    cpath,
                    expected_manifest_sha=m_sha,
                    expected_split_ids=split_id_sets,
                    allow_smoke_budget=args.allow_smoke_budget,
                )
            ]
        else:
            ledgers = sorted(cpath.rglob("campaign_ledger.json"))
            if not ledgers:
                raise FileNotFoundError(f"No campaign_ledger.json files found under {cpath}")
            out["campaigns"] = [
                verify_campaign_dir(
                    lg.parent,
                    expected_manifest_sha=m_sha,
                    expected_split_ids=split_id_sets,
                    allow_smoke_budget=args.allow_smoke_budget,
                )
                for lg in ledgers
            ]

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

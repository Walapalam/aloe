#!/usr/bin/env python3
"""One-Time Immutable Seal Ceremony Builder & Pre-Flight Dry-Run Verifier (`prepare_unseal_ceremony.py`).

By default (`--execute-hf-commit` NOT set), runs strictly in DRY-RUN mode:
1. Audits all 471 pre-unseal validation receipts (248 canonical search + 160 symmetric multi-seed
   search + 60 confirmation + 3 hardware-matched zero-shot validation) and verifies zero local test,
   seal, or attempt artifacts via `build_pre_unseal_ceremony_payloads`.
2. Captures `PRE_UNSEAL_ANALYSIS_PLAN.md` bytes once into memory (TOCTOU-safe) and computes all 8
   frozen orchestration, shim, plan, auditor, and bootstrap-engine SHA-256 digests.
3. Validates every constructed `test_unsealed.json` payload through `verify_immutable_model_seal`
   with its explicit `expected_seal_sha256` in an isolated temporary directory WITHOUT writing to
   `dest_dir` or Hugging Face.

Only when `--execute-hf-commit --confirm-irreversible-unseal` is explicitly passed in Phase II
does it verify complete absence of `final_sealed_test/*`, `zero_shot_test/*`, and
`final_test_attempts/*` on HF and publish all 3 `test_unsealed.json` files + frozen analysis plan
bytes in a SINGLE atomic Hugging Face commit (`parent_commit` CAS-guarded).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloe.tracking.trial_store import atomic_write_text  # noqa: E402
from scripts.final_test_audit import (  # noqa: E402
    build_pre_unseal_ceremony_payloads,
    compute_frozen_code_digests,
    verify_immutable_model_seal,
)
from scripts.run_local_mps_final import _write_worker_shim  # noqa: E402
from scripts.verify_campaign import verify_manifest  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run or execute the one-time Protocol-v1 Seal Ceremony")
    parser.add_argument("--dest-dir", required=True, help="Local synced store (runs/colab_l4_ipd_v1)")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifests" / "xlam_ipd_v1.json"))
    parser.add_argument("--raw-data", default=str(REPO_ROOT / "data" / "raw" / "xlam" / "xlam_function_calling_60k.json"))
    parser.add_argument("--analysis-plan", default=str(REPO_ROOT / "docs" / "PRE_UNSEAL_ANALYSIS_PLAN.md"))
    parser.add_argument("--hf-repo", default="Walapalam/aloe-ipd-v1-runs")
    parser.add_argument("--execute-hf-commit", action="store_true", help="Publish atomic Seal Ceremony commit to HF")
    parser.add_argument("--confirm-irreversible-unseal", action="store_true")
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir).resolve()
    manifest_p = Path(args.manifest).resolve()
    raw_p = Path(args.raw_data).resolve()
    plan_p = Path(args.analysis_plan).resolve()
    if not plan_p.is_file():
        raise FileNotFoundError(f"Frozen statistical analysis plan missing: {plan_p}")

    # Capture exact plan bytes once to eliminate any hash-to-upload TOCTOU race
    plan_bytes = plan_p.read_bytes()

    m_audit = verify_manifest(manifest_p, raw_data_path=raw_p, enforce_frozen_v1=True, require_raw_data=True)
    manifest_sha = str(m_audit["manifest_sha256"])
    split_id_sets = m_audit["split_id_sets"]

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_root = Path(tmp_d)
        shim_f = _write_worker_shim(tmp_root / "shim")
        tmp_plan_f = tmp_root / "PRE_UNSEAL_ANALYSIS_PLAN.md"
        tmp_plan_f.write_bytes(plan_bytes)

        frozen_digests = compute_frozen_code_digests(
            REPO_ROOT,
            analysis_plan_bytes=plan_bytes,
            shim_path=shim_f,
        )
        unsealed_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        payloads = build_pre_unseal_ceremony_payloads(
            dest_dir,
            manifest_sha=manifest_sha,
            split_id_sets=split_id_sets,
            frozen_digests=frozen_digests,
            unsealed_utc=unsealed_utc,
        )

        seal_shas: dict[str, str] = {}
        encoded_by_slug: dict[str, bytes] = {}
        for slug, pdict in sorted(payloads.items()):
            raw_bytes = (json.dumps(pdict, indent=2) + "\n").encode("utf-8")
            expected_sha = hashlib.sha256(raw_bytes).hexdigest()
            tmp_seal = tmp_root / f"{slug}_test_unsealed.json"
            tmp_seal.write_bytes(raw_bytes)
            verified = verify_immutable_model_seal(
                tmp_seal,
                slug=slug,
                manifest_sha=manifest_sha,
                campaigns_model_dir=dest_dir / "campaigns" / slug,
                expected_split_ids=split_id_sets,
                expected_seal_sha256=expected_sha,
                verify_code_digests=True,
                shim_path=shim_f,
                frozen_plan_path=tmp_plan_f,
            )
            seal_shas[slug] = str(verified["seal_sha256"])
            encoded_by_slug[slug] = raw_bytes

    summary = {
        "mode": "EXECUTED_HF_COMMIT" if args.execute_hf_commit else "DRY_RUN_ONLY_NO_UNSEAL",
        "manifest_sha256": manifest_sha,
        **frozen_digests,
        "seal_sha256_by_model": seal_shas,
        "arm_counts_by_model": {s: len(p["winner_signatures"]) for s, p in payloads.items()},
    }

    if not args.execute_hf_commit:
        print(json.dumps(summary, indent=2))
        return 0

    if not args.confirm_irreversible_unseal:
        raise PermissionError("Refusing --execute-hf-commit without --confirm-irreversible-unseal.")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=_resolve_hf_token())
    info = api.dataset_info(args.hf_repo)
    existing_paths = {s.rfilename for s in (info.siblings or [])}
    blocked_remote = sorted(
        p
        for p in existing_paths
        if p.startswith(("final_sealed_test/", "zero_shot_test/", "final_test_attempts/"))
    )
    if blocked_remote:
        raise PermissionError(
            f"Seal Ceremony aborted: pre-existing final-test/seal/attempt paths on {args.hf_repo}: {blocked_remote[:5]}"
        )

    ops = [
        CommitOperationAdd(
            path_in_repo="final_sealed_test/PRE_UNSEAL_ANALYSIS_PLAN.md",
            path_or_fileobj=io.BytesIO(plan_bytes),
        )
    ]
    for slug, raw_bytes in sorted(encoded_by_slug.items()):
        ops.append(
            CommitOperationAdd(
                path_in_repo=f"final_sealed_test/{slug}/test_unsealed.json",
                path_or_fileobj=io.BytesIO(raw_bytes),
            )
        )

    commit_info = api.create_commit(
        repo_id=args.hf_repo,
        repo_type="dataset",
        operations=ops,
        commit_message="[SEAL CEREMONY] Freeze 20 Protocol-v1 winners + PRE_UNSEAL_ANALYSIS_PLAN.md (write-once)",
        parent_commit=info.sha,
    )
    atomic_write_text(dest_dir / "final_sealed_test" / "PRE_UNSEAL_ANALYSIS_PLAN.md", plan_bytes.decode("utf-8"))
    for slug, raw_bytes in sorted(encoded_by_slug.items()):
        local_seal = dest_dir / "final_sealed_test" / slug / "test_unsealed.json"
        atomic_write_text(local_seal, raw_bytes.decode("utf-8"))

    summary["hf_seal_commit_oid"] = getattr(commit_info, "oid", str(commit_info))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Official Protocol-v1 Local Apple Silicon MPS Final-Test & Zero-Shot-Test Runner (`run_local_mps_final.py`).

Enforces read-only pre-committed `test_unsealed.json` verification (with mandatory 64-hex
`--expected-seal-sha256` and all 8 frozen code/shim/plan SHA-256 digests) BEFORE loading weights
or touching `splits["test"]` (400 items), acquires an exclusive local `flock` run lease, records
append-only `LAUNCHED` / `COMPLETED` / `FAILED` attempt logs with unique `attempt_id`s, unifies
`store_dir` and `persist_dir` with idempotent HF upload recovery on restart, executes uniform
`worker_entry --eval-split test --final --seal-file ...` across all 6 Gemma-4-E2B arms (`[42, 1337, 2026]`)
plus `zero_shot_test`, verifies every receipt via `final_test_audit`, and records adapter SHA-256
determinism parity against `confirm_3seed` adapters across both cached and fresh runs.

Supports `--dry-run-validation-only` (`--dry-run-preflight-only`) to verify the seal, pinned git
commit, worker shim SHA, and MPS hardware contract preflight WITHOUT ever accessing `splits["test"]`.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
import uuid

SCRIPT_DIR = Path(__file__).resolve().parent
FYP_ALOE_ROOT = SCRIPT_DIR.parent
if str(FYP_ALOE_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ALOE_ROOT))

from scripts.final_test_audit import WORKER_SHIM_SOURCE  # noqa: E402

MIN_BATTERY_PCT = 80
MODEL_ID = "google/gemma-4-E2B-it"
SLUG = "google--gemma-4-E2B-it"
EXPECTED_GIT_SHA = "8187255cae076f583dce455acaf5019b20bad1e4"
ARMS = ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ffn")
SEEDS = [42, 1337, 2026]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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


def wait_for_safe_power(min_pct: int = MIN_BATTERY_PCT) -> None:
    while True:
        try:
            out = subprocess.check_output(["pmset", "-g", "batt"], text=True)
            on_ac = "AC Power" in out
            m = re.search(r"(\d+)%", out)
            pct = int(m.group(1)) if m else 100
            if on_ac or pct >= min_pct:
                return
            print(f"[BATTERY GUARD] Battery at {pct}% (<{min_pct}%). Sleeping 120s...", flush=True)
            time.sleep(120)
        except Exception:
            return


def _write_worker_shim(shim_dir: Path) -> Path:
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_file = shim_dir / "sitecustomize.py"
    shim_file.write_text(WORKER_SHIM_SOURCE, encoding="utf-8")
    return shim_file


def _has_completed_attempt_local(persist_dir: Path, run_id: str) -> bool:
    adir = persist_dir / "final_test_attempts" / SLUG
    if not adir.is_dir():
        return False
    for jf in adir.glob("*.json"):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            if d.get("run_id") == run_id and d.get("stage") == "COMPLETED":
                return True
        except Exception:
            pass
    return False


def _find_open_launched_attempt_id(persist_dir: Path, run_id: str) -> str | None:
    adir = persist_dir / "final_test_attempts" / SLUG
    if not adir.is_dir():
        return None
    launched_order: list[tuple[str, str, str]] = []
    closed_ids: set[str] = set()
    for jf in sorted(adir.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            if d.get("run_id") != run_id:
                continue
            aid = str(d.get("attempt_id", "")).strip()
            stg = str(d.get("stage", "")).strip()
            if not aid:
                continue
            if stg == "LAUNCHED":
                launched_order.append((str(d.get("utc", "")), jf.name, aid))
            elif stg in ("COMPLETED", "FAILED"):
                closed_ids.add(aid)
        except Exception:
            pass
    launched_order.sort()
    for _utc, _fn, aid in reversed(launched_order):
        if aid not in closed_ids:
            return aid
    return None


def _replay_local_attempts_to_hf(api: Any, *, hf_repo: str, persist_dir: Path) -> None:
    """Idempotently replay any local `final_test_attempts/<slug>/*.json` files under original filenames."""
    from huggingface_hub import CommitOperationAdd

    adir = persist_dir / "final_test_attempts" / SLUG
    if not adir.is_dir():
        return
    ops = []
    for jf in sorted(adir.glob("*.json")):
        rel = f"final_test_attempts/{SLUG}/{jf.name}"
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=io.BytesIO(jf.read_bytes())))
    if ops:
        info = api.dataset_info(hf_repo)
        api.create_commit(
            repo_id=hf_repo,
            repo_type="dataset",
            operations=ops,
            commit_message=f"[mps attempt replay] {len(ops)} local attempt logs for {SLUG}",
            parent_commit=info.sha,
        )


def _log_hf_attempt(
    api: Any,
    *,
    hf_repo: str,
    persist_dir: Path,
    run_id: str,
    stage: str,
    attempt_id: str,
    seal_sha256: str,
    seq: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Publish append-only attempt record to HF FIRST, then persist local mirror."""
    from huggingface_hub import CommitOperationAdd
    from aloe.tracking.trial_store import atomic_write_text

    eff_seq = seq if seq is not None else (1 if stage == "LAUNCHED" else 2)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    payload = {
        "protocol_tag": "protocol-v1",
        "attempt_id": attempt_id,
        "seq": eff_seq,
        "seal_sha256": seal_sha256,
        "slug": SLUG,
        "run_id": run_id,
        "stage": stage,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    rel = (
        f"final_test_attempts/{SLUG}/"
        f"{run_id}__{eff_seq:02d}_{stage}__{ts}__{socket.gethostname()}_{os.getpid()}_{attempt_id}.json"
    )
    info = api.dataset_info(hf_repo)
    api.create_commit(
        repo_id=hf_repo,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(
                path_in_repo=rel,
                path_or_fileobj=io.BytesIO(encoded.encode("utf-8")),
            )
        ],
        commit_message=f"[mps attempt] {stage} {SLUG}/{run_id}",
        parent_commit=info.sha,
    )
    atomic_write_text(persist_dir / rel, encoded)


def _find_confirm_adapter(persist_dir: Path, store_dir: Path, arm: str, seed: int) -> Path | None:
    rel = (
        Path("confirm_3seed")
        / SLUG
        / arm
        / "trials"
        / f"{arm}_winner_s{seed}_validation"
        / "adapter"
        / "adapter_model.safetensors"
    )
    for root in (persist_dir, store_dir):
        cand = root / rel
        if cand.is_file():
            return cand
    return None


def _run_mps_zero_shot_subprocess(
    *,
    work_repo: Path,
    manifest_rel: str,
    raw_rel: str,
    out_dir: Path,
    split_name: str,
    manifest_sha: str,
    expected_ids: set[str],
    seal_path: Path | None,
) -> dict[str, Any]:
    if split_name == "test" and (seal_path is None or not seal_path.is_file()):
        raise PermissionError("Zero-shot 'test' on MPS requires a verified read-only seal file.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"zero_shot_{split_name}.json"
    pred_path = out_dir / f"predictions_{split_name}.jsonl"
    if pred_path.is_file() and not rec_path.is_file():
        pred_path.unlink()

    inline_py = """
import hashlib, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aloe.datasets.model_registry import ModelRegistry
from aloe.datasets.xlam import load_xlam_file
from aloe.evaluation.prediction_store import PredictionStore
from aloe.evaluation.resources import ResourceProfiler, enforce_hardware_contract
from aloe.evaluation.tool_calling import EVALUATOR_VERSION
from aloe.execution.backends.peft_mps import evaluate_model_on_split
from aloe.tracking.trial_store import atomic_write_text, get_git_commit

cfg = json.loads(sys.argv[1])
split_name = str(cfg["split_name"])
if split_name == "test":
    assert Path(cfg["seal_path"]).is_file(), "Missing seal file in MPS zero-shot test subprocess"
work_repo = Path(cfg["work_repo"]).resolve()
git_sha = get_git_commit(work_repo, forbid_dirty=True)
assert git_sha == cfg["expected_git_sha"]

hw = enforce_hardware_contract(cfg["model_id"], "mps", official_campaign=True)
assert hw["gpu_name"] == "Apple M1 Max"

registry = ModelRegistry()
pinned_rev = registry.PINNED_REVISIONS.get(cfg["model_id"], "main")
try:
    snapshot_path = registry.get_snapshot_path(cfg["model_id"])
except Exception:
    snapshot_path = cfg["model_id"]
is_local_dir = Path(str(snapshot_path)).exists()
rev_kw = {"revision": pinned_rev} if (not is_local_dir and pinned_rev not in ("main", "UNKNOWN", "")) else {}

manifest = json.loads((work_repo / cfg["manifest_rel"]).read_text(encoding="utf-8"))
ids_sorted = sorted(str(i) for i in manifest["splits"][split_name]["ids"])
raw_examples = load_xlam_file(work_repo / cfg["raw_rel"])
id_to_ex = {str(ex.id): ex for ex in raw_examples}
eval_examples = [id_to_ex[sid] for sid in ids_sorted]

profiler = ResourceProfiler(sample_interval_ms=100, device="mps")
profiler.start()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    snapshot_path, torch_dtype=torch.bfloat16, trust_remote_code=True, **rev_kw
).to("mps")
tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True, **rev_kw)
if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
    tokenizer.pad_token = tokenizer.eos_token

out_dir = Path(cfg["out_dir"]).resolve()
slug = cfg["model_id"].replace("/", "--")
pred_path = out_dir / f"predictions_{split_name}.jsonl"
pred_store = PredictionStore(pred_path)
ev_out = evaluate_model_on_split(
    model,
    tokenizer,
    eval_examples,
    prediction_store=pred_store,
    split_name=split_name,
    model_id=cfg["model_id"],
    model_revision=pinned_rev,
    arm="zero_shot",
    trial_id=f"zero_shot_{slug}_{split_name}",
    seed=42,
    device="mps",
    max_new_tokens=192,
)
mem_stats = profiler.stop()
raw_sum = ev_out["summary"]
sdict = raw_sum.model_dump() if hasattr(raw_sum, "model_dump") else vars(raw_sum)
h = hashlib.sha256()
with open(pred_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
pred_sha = h.hexdigest()
receipt = {
    "protocol_tag": "protocol-v1",
    "mode": f"zero_shot_base_{split_name}",
    "model_id": cfg["model_id"],
    "model_revision": pinned_rev,
    "arm": "zero_shot",
    "split": split_name,
    "sample_count": int(sdict["total_samples"]),
    "exact_match_accuracy": float(sdict["exact_match_accuracy"]),
    "lenient_exact_match_accuracy": float(sdict.get("lenient_exact_match_accuracy", sdict["exact_match_accuracy"])),
    "tool_name_accuracy": float(sdict.get("tool_selection_accuracy", 0.0)),
    "argument_name_accuracy": float(sdict.get("argument_name_accuracy", 0.0)),
    "argument_value_accuracy": float(sdict.get("argument_value_accuracy", 0.0)),
    "parse_success_rate": float(sdict.get("parse_success_rate", 0.0)),
    "schema_valid_rate": float(sdict.get("schema_valid_rate", 0.0)),
    "trainable_parameters": 0,
    "memory_metric_name": str(mem_stats["memory_metric"]),
    "search_memory_mb": float(mem_stats["search_memory_mb"]),
    "peak_rss_gb": float(mem_stats["peak_rss_gb"]),
    "mps_driver_allocated_gb": float(mem_stats["mps_driver_allocated_gb"]),
    "peak_cuda_allocated_mb": 0.0,
    "wall_seconds": round(time.time() - t0, 3),
    "eval_seconds": float(ev_out["eval_seconds"]),
    "predictions_sha256": pred_sha,
    "evaluator_version": EVALUATOR_VERSION,
    "manifest_sha256": cfg["manifest_sha"],
    "git_commit": git_sha,
    "gpu_name": str(hw["gpu_name"]),
    "hardware_label": str(hw["hardware_label"]),
}
atomic_write_text(out_dir / f"zero_shot_{split_name}.json", json.dumps(receipt, indent=2) + "\\n")
"""
    payload = json.dumps(
        {
            "work_repo": str(work_repo),
            "model_id": MODEL_ID,
            "manifest_rel": manifest_rel,
            "raw_rel": raw_rel,
            "out_dir": str(out_dir),
            "expected_git_sha": EXPECTED_GIT_SHA,
            "manifest_sha": manifest_sha,
            "split_name": split_name,
            "seal_path": str(seal_path) if seal_path else "",
        }
    )
    subprocess.run([sys.executable, "-c", inline_py, payload], cwd=str(work_repo), check=True)
    return json.loads(rec_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sealed final-test & zero-shot-test for Gemma-4-E2B on MPS")
    parser.add_argument("--work-repo", required=True, help="Clean aloe checkout pinned to 8187255cae07...")
    parser.add_argument("--store-dir", required=True, help="Local synced store containing campaigns/ and final_sealed_test/")
    parser.add_argument("--persist-dir", required=True, help="Local persistent MPS output mirror")
    parser.add_argument("--hf-repo", default="Walapalam/aloe-ipd-v1-runs")
    parser.add_argument("--expected-seal-sha256", default="", help="Required 64-hex SHA-256 of test_unsealed.json")
    parser.add_argument("--ceremony-hf-revision", default="", help="Optional HF Seal Ceremony commit OID")
    parser.add_argument("--ceremony-pins", default=str(FYP_ALOE_ROOT / "docs" / "CEREMONY_PINS.json"))
    parser.add_argument("--phases", default="zero_shot_test,final_test")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument(
        "--dry-run-validation-only",
        "--dry-run-preflight-only",
        dest="dry_run_preflight_only",
        action="store_true",
        help="Preflight-verify read-only seal, code digests, git commit, and MPS hardware contract without running test",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from aloe.evaluation.resources import detect_hardware_summary, enforce_hardware_contract
    from aloe.execution.campaign_runner import compute_exact_feasible_pool
    from aloe.pipeline import inspect_to_grammar
    from aloe.tracking.trial_store import atomic_write_text, get_git_commit
    from scripts.final_test_audit import (
        CANONICAL_PROTOCOL_MODELS,
        HEX64_RE,
        load_ceremony_pins,
        verify_immutable_model_seal,
        verify_strict_trial_receipt,
        verify_strict_zero_shot_receipt,
    )
    from scripts.verify_campaign import verify_manifest

    work_repo = Path(args.work_repo).resolve()
    store_dir = Path(args.store_dir).resolve()
    persist_dir = Path(args.persist_dir).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    pins_data = load_ceremony_pins(Path(args.ceremony_pins).resolve())
    resolved_seal_sha = (args.expected_seal_sha256 or (pins_data.get("seal_sha256_by_model") or {}).get(SLUG, "")).strip()
    ceremony_rev = (args.ceremony_hf_revision or pins_data.get("hf_seal_commit_oid") or "").strip()
    if not resolved_seal_sha or not HEX64_RE.fullmatch(resolved_seal_sha):
        raise PermissionError(
            "MPS sealed runner requires a 64-hex --expected-seal-sha256 (or populated docs/CEREMONY_PINS.json)."
        )

    lock_file_path = persist_dir / ".mps_final_runner.lock"
    lock_fh = open(lock_file_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another MPS final runner already holds exclusive lock {lock_file_path}") from exc
    lock_fh.write(f"pid={os.getpid()}\n")
    lock_fh.flush()

    try:
        shim_file = _write_worker_shim(Path("/tmp/aloe_mps_final_shim"))
        os.environ["PYTHONPATH"] = f"{shim_file.parent}:{work_repo}" + (
            f":{os.environ['PYTHONPATH']}" if os.environ.get("PYTHONPATH") else ""
        )

        git_sha = get_git_commit(work_repo, forbid_dirty=True)
        if git_sha != EXPECTED_GIT_SHA:
            raise RuntimeError(f"Work repo SHA mismatch: {git_sha} != {EXPECTED_GIT_SHA}")

        manifest_rel = "data/manifests/xlam_ipd_v1.json"
        raw_rel = "data/raw/xlam/xlam_function_calling_60k.json"
        m_audit = verify_manifest(
            work_repo / manifest_rel,
            raw_data_path=work_repo / raw_rel,
            enforce_frozen_v1=True,
            require_raw_data=True,
        )
        manifest_sha = str(m_audit["manifest_sha256"])
        split_id_sets = m_audit["split_id_sets"]

        hw = enforce_hardware_contract(MODEL_ID, "mps", official_campaign=True)
        exact_gpu_name = str(hw["gpu_name"])
        spec = CANONICAL_PROTOCOL_MODELS[SLUG]
        if exact_gpu_name != spec["locked_gpu_name"]:
            raise RuntimeError(f"MPS hardware mismatch: {exact_gpu_name} != {spec['locked_gpu_name']}")

        hf_token = "" if args.dry_run_preflight_only else _resolve_hf_token()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        api = HfApi(token=hf_token) if hf_token else None

        if not args.dry_run_preflight_only:
            snapshot_download(
                repo_id=args.hf_repo,
                repo_type="dataset",
                local_dir=str(store_dir),
                allow_patterns=[
                    f"campaigns/{SLUG}/**",
                    f"confirm_3seed/{SLUG}/**",
                    "final_sealed_test/PRE_UNSEAL_ANALYSIS_PLAN.md",
                    f"final_sealed_test/{SLUG}/**",
                    f"zero_shot_test/{SLUG}/**",
                    f"final_test_attempts/{SLUG}/**",
                ],
                ignore_patterns=["**/checkpoint-*/*"],
                token=hf_token,
            )
            if ceremony_rev:
                for rel_ceremony_f in (
                    "final_sealed_test/PRE_UNSEAL_ANALYSIS_PLAN.md",
                    f"final_sealed_test/{SLUG}/test_unsealed.json",
                ):
                    hf_hub_download(
                        repo_id=args.hf_repo,
                        repo_type="dataset",
                        filename=rel_ceremony_f,
                        revision=ceremony_rev,
                        local_dir=str(store_dir),
                        token=hf_token,
                    )
            # Unify store_dir and persist_dir so restored runs and attempt logs are visible to resumption
            if store_dir != persist_dir:
                for sub_rel in (
                    Path("final_sealed_test") / SLUG,
                    Path("zero_shot_test") / SLUG,
                    Path("final_test_attempts") / SLUG,
                ):
                    src_sub = store_dir / sub_rel
                    dst_sub = persist_dir / sub_rel
                    if src_sub.is_dir():
                        dst_sub.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)

        seal_path = store_dir / "final_sealed_test" / SLUG / "test_unsealed.json"
        frozen_plan = store_dir / "final_sealed_test" / "PRE_UNSEAL_ANALYSIS_PLAN.md"
        seal_info = verify_immutable_model_seal(
            seal_path,
            slug=SLUG,
            manifest_sha=manifest_sha,
            campaigns_model_dir=store_dir / "campaigns" / SLUG,
            expected_split_ids=split_id_sets,
            expected_seal_sha256=resolved_seal_sha,
            verify_code_digests=True,
            shim_path=shim_file,
            frozen_plan_path=frozen_plan if frozen_plan.is_file() or not args.dry_run_preflight_only else None,
        )
        winners_by_arm = seal_info["winners_by_arm"]
        print(f"[SEAL VERIFIED READ-ONLY] {SLUG} seal_sha256={seal_info['seal_sha256'][:16]}...", flush=True)

        if args.dry_run_preflight_only:
            print("[PREFLIGHT DRY-RUN OK] Seal, frozen code digests, git commit, and MPS hardware verified.", flush=True)
            return 0

        assert api is not None
        seal_sha_hex = str(seal_info["seal_sha256"])
        _replay_local_attempts_to_hf(api, hf_repo=args.hf_repo, persist_dir=persist_dir)
        phase_list = [p.strip() for p in args.phases.split(",") if p.strip()]
        req_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        desc, grammar = inspect_to_grammar(MODEL_ID, backend="mps")
        baseline_params = int(compute_exact_feasible_pool(desc, grammar, parity_tol=0.05)["baseline_params"])

        # Phase 0: Zero-shot on sealed 400-item test split
        if "zero_shot_test" in phase_list:
            wait_for_safe_power()
            zs_dir = persist_dir / "zero_shot_test" / SLUG
            zs_rec_path = zs_dir / "zero_shot_test.json"
            zs_run_id = f"zero_shot_{SLUG}_test"
            if (
                verify_strict_zero_shot_receipt(
                    zs_rec_path,
                    expected_split="test",
                    expected_split_ids=split_id_sets["test"],
                    expected_manifest_sha=manifest_sha,
                    expected_git_sha=EXPECTED_GIT_SHA,
                    expected_model_id=MODEL_ID,
                    expected_model_revision=spec["model_revision"],
                    expected_gpu_name=exact_gpu_name,
                    expected_memory_metric=spec["locked_memory_metric_name"],
                )
                is not None
            ):
                api.upload_folder(
                    repo_id=args.hf_repo,
                    repo_type="dataset",
                    folder_path=str(zs_dir),
                    path_in_repo=f"zero_shot_test/{SLUG}",
                    commit_message=f"[mps zero_shot_test] {SLUG} 400-item test PASS (idempotent sync)",
                )
                if not _has_completed_attempt_local(persist_dir, zs_run_id):
                    open_aid = _find_open_launched_attempt_id(persist_dir, zs_run_id)
                    if open_aid is None:
                        open_aid = uuid.uuid4().hex[:12]
                        _log_hf_attempt(
                            api,
                            hf_repo=args.hf_repo,
                            persist_dir=persist_dir,
                            run_id=zs_run_id,
                            stage="LAUNCHED",
                            attempt_id=open_aid,
                            seal_sha256=seal_sha_hex,
                            seq=1,
                        )
                    _log_hf_attempt(
                        api,
                        hf_repo=args.hf_repo,
                        persist_dir=persist_dir,
                        run_id=zs_run_id,
                        stage="COMPLETED",
                        attempt_id=open_aid,
                        seal_sha256=seal_sha_hex,
                        seq=2,
                    )
                print(f"[MPS ZERO-SHOT CACHED & SYNCED] {zs_run_id}", flush=True)
            else:
                if _has_completed_attempt_local(persist_dir, zs_run_id):
                    raise PermissionError(f"Completed run cannot be re-executed: {zs_run_id}")
                orphan_aid = _find_open_launched_attempt_id(persist_dir, zs_run_id)
                if orphan_aid is not None:
                    _log_hf_attempt(
                        api,
                        hf_repo=args.hf_repo,
                        persist_dir=persist_dir,
                        run_id=zs_run_id,
                        stage="FAILED",
                        attempt_id=orphan_aid,
                        seal_sha256=seal_sha_hex,
                        seq=2,
                        extra={"error": "preempted_or_unclosed_prior_attempt"},
                    )
                zs_aid = uuid.uuid4().hex[:12]
                _log_hf_attempt(
                    api,
                    hf_repo=args.hf_repo,
                    persist_dir=persist_dir,
                    run_id=zs_run_id,
                    stage="LAUNCHED",
                    attempt_id=zs_aid,
                    seal_sha256=seal_sha_hex,
                    seq=1,
                )
                try:
                    _run_mps_zero_shot_subprocess(
                        work_repo=work_repo,
                        manifest_rel=manifest_rel,
                        raw_rel=raw_rel,
                        out_dir=zs_dir,
                        split_name="test",
                        manifest_sha=manifest_sha,
                        expected_ids=split_id_sets["test"],
                        seal_path=seal_path,
                    )
                    verify_strict_zero_shot_receipt(
                        zs_rec_path,
                        expected_split="test",
                        expected_split_ids=split_id_sets["test"],
                        expected_manifest_sha=manifest_sha,
                        expected_git_sha=EXPECTED_GIT_SHA,
                        expected_model_id=MODEL_ID,
                        expected_model_revision=spec["model_revision"],
                        expected_gpu_name=exact_gpu_name,
                        expected_memory_metric=spec["locked_memory_metric_name"],
                        raise_on_error=True,
                    )
                except Exception as zs_exc:
                    _log_hf_attempt(
                        api,
                        hf_repo=args.hf_repo,
                        persist_dir=persist_dir,
                        run_id=zs_run_id,
                        stage="FAILED",
                        attempt_id=zs_aid,
                        seal_sha256=seal_sha_hex,
                        seq=2,
                        extra={"error": str(zs_exc)[:300]},
                    )
                    raise
                api.upload_folder(
                    repo_id=args.hf_repo,
                    repo_type="dataset",
                    folder_path=str(zs_dir),
                    path_in_repo=f"zero_shot_test/{SLUG}",
                    commit_message=f"[mps zero_shot_test] {SLUG} 400-item test PASS",
                )
                _log_hf_attempt(
                    api,
                    hf_repo=args.hf_repo,
                    persist_dir=persist_dir,
                    run_id=zs_run_id,
                    stage="COMPLETED",
                    attempt_id=zs_aid,
                    seal_sha256=seal_sha_hex,
                    seq=2,
                )

        # Phase 1: Uniform 3-seed final-test runs (`worker_entry --eval-split test --final --seal-file ...`)
        determinism_records: list[dict[str, Any]] = []
        if "final_test" in phase_list:
            for arm in req_arms:
                wcfg = winners_by_arm[arm]
                local_trials_dir = work_repo / "runs" / "final_sealed_test" / SLUG / arm / "trials"
                mirror_trials_dir = persist_dir / "final_sealed_test" / SLUG / arm / "trials"
                local_trials_dir.mkdir(parents=True, exist_ok=True)

                for s_val in SEEDS:
                    wait_for_safe_power()
                    tid = f"{arm}_winner_s{s_val}_test"
                    done_mirror = mirror_trials_dir / tid / "trial.json"
                    was_cached = verify_strict_trial_receipt(
                        done_mirror,
                        expected_seed=s_val,
                        expected_split="test",
                        expected_split_ids=split_id_sets["test"],
                        expected_manifest_sha=manifest_sha,
                        expected_git_sha=EXPECTED_GIT_SHA,
                        expected_model_id=MODEL_ID,
                        expected_model_revision=spec["model_revision"],
                        expected_gpu_name=exact_gpu_name,
                        expected_memory_metric=spec["locked_memory_metric_name"],
                        expected_winner_cfg=wcfg,
                    )
                    run_aid: str | None = None
                    if not was_cached:
                        if _has_completed_attempt_local(persist_dir, tid):
                            raise PermissionError(f"Completed run cannot be re-executed: {tid}")
                        orphan_aid = _find_open_launched_attempt_id(persist_dir, tid)
                        if orphan_aid is not None:
                            _log_hf_attempt(
                                api,
                                hf_repo=args.hf_repo,
                                persist_dir=persist_dir,
                                run_id=tid,
                                stage="FAILED",
                                attempt_id=orphan_aid,
                                seal_sha256=seal_sha_hex,
                                seq=2,
                                extra={"error": "preempted_or_unclosed_prior_attempt"},
                            )
                        seeded_cfg = wcfg.model_copy(
                            update={
                                "trial_id": tid,
                                "arm": arm,
                                "seed": s_val,
                                "dataset_manifest_path": manifest_rel,
                                "dataset_raw_path": raw_rel,
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
                            "test",
                            "--parity-tol",
                            "0.05",
                            "--baseline-params",
                            str(baseline_params),
                            "--official-campaign",
                            "--forbid-dirty-git",
                            "--mirror-dir",
                            str(mirror_trials_dir),
                            "--final",
                            "--seal-file",
                            str(seal_path),
                        ]
                        if seeded_cfg.peft_method == "lora":
                            cmd.append("--enforce-parity")

                        assert str(detect_hardware_summary().get("gpu_name") or "") == exact_gpu_name
                        run_aid = uuid.uuid4().hex[:12]
                        _log_hf_attempt(
                            api,
                            hf_repo=args.hf_repo,
                            persist_dir=persist_dir,
                            run_id=tid,
                            stage="LAUNCHED",
                            attempt_id=run_aid,
                            seal_sha256=seal_sha_hex,
                            seq=1,
                        )
                        print(f"=== [MPS FINAL TEST] {MODEL_ID} :: {tid} ===", flush=True)
                        try:
                            subprocess.run(cmd, cwd=str(work_repo), check=True)
                            verify_strict_trial_receipt(
                                done_mirror,
                                expected_seed=s_val,
                                expected_split="test",
                                expected_split_ids=split_id_sets["test"],
                                expected_manifest_sha=manifest_sha,
                                expected_git_sha=EXPECTED_GIT_SHA,
                                expected_model_id=MODEL_ID,
                                expected_model_revision=spec["model_revision"],
                                expected_gpu_name=exact_gpu_name,
                                expected_memory_metric=spec["locked_memory_metric_name"],
                                expected_winner_cfg=wcfg,
                                raise_on_error=True,
                            )
                        except Exception as run_exc:
                            _log_hf_attempt(
                                api,
                                hf_repo=args.hf_repo,
                                persist_dir=persist_dir,
                                run_id=tid,
                                stage="FAILED",
                                attempt_id=run_aid,
                                seal_sha256=seal_sha_hex,
                                seq=2,
                                extra={"error": str(run_exc)[:300]},
                            )
                            raise
                    else:
                        print(f"[MPS FINAL CACHED — ENSURING HF PUBLICATION] {tid}", flush=True)

                    # Bonus MPS adapter SHA-256 reproducibility check across BOTH cached and fresh seeds
                    final_st = mirror_trials_dir / tid / "adapter" / "adapter_model.safetensors"
                    conf_st = _find_confirm_adapter(persist_dir, store_dir, arm, s_val)
                    if final_st.is_file() and conf_st is not None:
                        f_sha = _sha256_file(final_st)
                        c_sha = _sha256_file(conf_st)
                        determinism_records.append(
                            {
                                "arm": arm,
                                "seed": s_val,
                                "confirm_adapter_sha256": c_sha,
                                "final_adapter_sha256": f_sha,
                                "bit_identical": f_sha == c_sha,
                            }
                        )

                    api.upload_folder(
                        repo_id=args.hf_repo,
                        repo_type="dataset",
                        folder_path=str(persist_dir / "final_sealed_test" / SLUG / arm),
                        path_in_repo=f"final_sealed_test/{SLUG}/{arm}",
                        ignore_patterns=["**/checkpoint-*/*"],
                        commit_message=f"[mps final_test] {SLUG}/{arm} s={s_val} PASS",
                    )
                    if not _has_completed_attempt_local(persist_dir, tid):
                        comp_aid = run_aid or _find_open_launched_attempt_id(persist_dir, tid)
                        if comp_aid is None:
                            comp_aid = uuid.uuid4().hex[:12]
                            _log_hf_attempt(
                                api,
                                hf_repo=args.hf_repo,
                                persist_dir=persist_dir,
                                run_id=tid,
                                stage="LAUNCHED",
                                attempt_id=comp_aid,
                                seal_sha256=seal_sha_hex,
                                seq=1,
                            )
                        _log_hf_attempt(
                            api,
                            hf_repo=args.hf_repo,
                            persist_dir=persist_dir,
                            run_id=tid,
                            stage="COMPLETED",
                            attempt_id=comp_aid,
                            seal_sha256=seal_sha_hex,
                            seq=2,
                        )

            if determinism_records:
                det_path = persist_dir / "final_sealed_test" / SLUG / "mps_adapter_determinism_audit.json"
                atomic_write_text(det_path, json.dumps(determinism_records, indent=2) + "\n")
                api.upload_file(
                    repo_id=args.hf_repo,
                    repo_type="dataset",
                    path_or_fileobj=str(det_path),
                    path_in_repo=f"final_sealed_test/{SLUG}/mps_adapter_determinism_audit.json",
                    commit_message=f"[mps determinism] {SLUG} adapter SHA-256 comparison",
                )

        print("🏁 LOCAL MPS GEMMA-4-E2B SEALED FINAL TEST COMPLETE!", flush=True)
        return 0
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())

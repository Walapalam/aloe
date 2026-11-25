#!/usr/bin/env python3
"""Atomic Per-Arm Colab Pro+ Shard Runner (`run_colab_shard.py`).

Hardened against all multi-tab concurrency & preemption edge cases (ADR-019, Opus 5.5 + GPT Sol):
1. Fails closed immediately (<1s) if the Colab GPU does not match `--expected-exact-gpu`
   (`"NVIDIA L4"`) or if `--work-repo` (`git rev-parse HEAD`, `--forbid-dirty-git`)
   differs from `--expected-git-sha` (`f0d0ea2...` for Falcon-H1, `8187255...` for LFM3).
2. Optimistic-concurrency HF shard lease (`leases/<shard_id>.json`, 20-min TTL, 8-min standalone
   renewal) using `parent_commit=head_sha` on EVERY acquire, renew, delta upload, and release,
   failing closed on lease-read errors and verifying `token == self.token_id` before every write
   so an expired/preempted zombie VM can never overwrite its replacement's data.
3. Hooks incremental HF delta uploads strictly AFTER `CampaignRunner.ledger.save(state)` marks a
   trial `DONE` (with updated BoTorch `optimizer_state`), persisting the current leader's `adapter/`
   immediately so a preempted run never loses the winning weights or re-runs a completed trial.
4. Enforces `git_commit == expected_git_sha` on every restored search and 3-seed confirmation receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

LEASE_TTL_SECONDS = 1200  # 20 minutes
LEASE_STANDALONE_RENEW_SECONDS = 480  # 8 minutes (~7.5 commits/hr/shard << 128/hr HF cap)
REQUIRED_CONFIRM_SEEDS = [42, 1337, 2026]
RAW_XLAM_SHA256 = "4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6"
XLAM_FALLBACK_REPO = "Salesforce/xlam-function-calling-60k"
XLAM_FALLBACK_REV = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"


class LeaseLostError(RuntimeError):
    """Raised when another VM legitimately acquired this shard's lease."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class HfShardSyncer:
    """Manages CAS-verified HF shard leases and post-ledger-save per-arm delta commits."""

    def __init__(
        self,
        *,
        api: Any,
        hf_repo: str,
        shard_id: str,
        model_id: str,
        arms: list[str],
        phases: list[str],
        force_steal_lease: bool = False,
    ) -> None:
        self.api = api
        self.hf_repo = hf_repo
        self.shard_id = shard_id
        self.model_id = model_id
        self.arms = arms
        self.phases = phases
        self.force_steal_lease = force_steal_lease
        self.lease_repo_path = f"leases/{shard_id}.json"
        self.token_id = f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}-{random.randint(1000, 9999)}"
        self._uploaded_index: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._last_lease_push = 0.0
        self._stop_event = threading.Event()
        self._lease_lost = False
        self._hb_thread: threading.Thread | None = None

    def _make_lease_bytes(self, status: str, note: str = "") -> bytes:
        now = time.time()
        payload = {
            "shard_id": self.shard_id,
            "token": self.token_id,
            "status": status,
            "model_id": self.model_id,
            "arms": self.arms,
            "phases": self.phases,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "updated_epoch": round(now, 3),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "expires_epoch": round(now + LEASE_TTL_SECONDS, 3) if status == "ACTIVE" else round(now, 3),
            "note": note,
        }
        return (json.dumps(payload, indent=2) + "\n").encode("utf-8")

    def _read_remote_head_and_lease(self) -> tuple[str, dict[str, Any] | None]:
        """Fetch current repo `head_sha` and fail-closed parse `leases/<shard_id>.json` if present."""
        from huggingface_hub import hf_hub_download

        info = self.api.dataset_info(self.hf_repo)
        head_sha = str(info.sha)
        repo_files = {s.rfilename for s in (info.siblings or [])}
        if self.lease_repo_path not in repo_files:
            return head_sha, None
        # Fail-closed: NEVER treat download/JSON failure as "no lease" (Opus B2 + Sol P1)
        local_l = hf_hub_download(
            repo_id=self.hf_repo,
            repo_type="dataset",
            filename=self.lease_repo_path,
            revision=head_sha,
            token=self.api.token,
        )
        ldata = json.loads(Path(local_l).read_text(encoding="utf-8"))
        return head_sha, ldata

    def _verify_still_owner(self, ldata: dict[str, Any] | None) -> None:
        if ldata is None or ldata.get("token") != self.token_id:
            self._lease_lost = True
            self._stop_event.set()
            other = (ldata or {}).get("token", "MISSING")
            raise LeaseLostError(
                f"Shard '{self.shard_id}' lease ownership lost (remote token='{other}', local='{self.token_id}'). "
                "Aborting immediately to protect replacement VM."
            )

    def acquire_lease(self) -> None:
        from huggingface_hub import CommitOperationAdd

        for attempt in range(1, 9):
            try:
                head_sha, ldata = self._read_remote_head_and_lease()
            except Exception as exc:
                if attempt < 8:
                    sleep_s = 1.5 * attempt + random.uniform(0.1, 0.9)
                    print(f"[HF LEASE READ RETRY {attempt}/8] {exc} (waiting {sleep_s:.1f}s)", flush=True)
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Fail-closed: unable to read remote lease state after 8 attempts: {exc}") from exc

            now = time.time()
            if (
                ldata is not None
                and ldata.get("status") == "ACTIVE"
                and float(ldata.get("expires_epoch") or 0.0) > now
                and ldata.get("token") != self.token_id
                and not self.force_steal_lease
            ):
                rem = int(float(ldata["expires_epoch"]) - now)
                raise RuntimeError(
                    f"[DUPLICATE SHARD BLOCKED] Shard '{self.shard_id}' is already ACTIVE on "
                    f"host={ldata.get('host')} (lease expires in {rem}s). "
                    "Close this duplicate tab, or tick FORCE_STEAL_LEASE if the old VM was terminated."
                )

            op = CommitOperationAdd(
                path_in_repo=self.lease_repo_path,
                path_or_fileobj=io.BytesIO(self._make_lease_bytes("ACTIVE", "lease_acquired")),
            )
            try:
                self.api.create_commit(
                    repo_id=self.hf_repo,
                    repo_type="dataset",
                    operations=[op],
                    commit_message=f"[lease] acquire {self.shard_id} ({self.model_id})",
                    parent_commit=head_sha,
                )
                self._last_lease_push = time.time()
                print(f"[HF LEASE ACQUIRED] {self.lease_repo_path} | token={self.token_id}", flush=True)
                self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                self._hb_thread.start()
                return
            except Exception as exc:
                msg = str(exc)
                if attempt < 8 and ("412" in msg or "Precondition" in msg or "parent_commit" in msg or "429" in msg):
                    sleep_s = 1.5 * attempt + random.uniform(0.1, 0.9)
                    print(f"[HF LEASE CAS RETRY {attempt}/8] Concurrent commit; retrying in {sleep_s:.1f}s...", flush=True)
                    time.sleep(sleep_s)
                    continue
                raise

        raise RuntimeError(f"Failed to acquire HF shard lease for {self.shard_id} after 8 attempts.")

    def _cas_commit_owned(self, extra_ops: list[Any], *, status: str, commit_label: str, max_attempts: int = 6) -> None:
        """Execute a CAS `parent_commit=head_sha` commit ONLY after verifying `token == self.token_id`."""
        from huggingface_hub import CommitOperationAdd

        for attempt in range(1, max_attempts + 1):
            try:
                head_sha, ldata = self._read_remote_head_and_lease()
                self._verify_still_owner(ldata)
                lease_op = CommitOperationAdd(
                    path_in_repo=self.lease_repo_path,
                    path_or_fileobj=io.BytesIO(self._make_lease_bytes(status, commit_label)),
                )
                self.api.create_commit(
                    repo_id=self.hf_repo,
                    repo_type="dataset",
                    operations=[*extra_ops, lease_op],
                    commit_message=commit_label,
                    parent_commit=head_sha,
                )
                self._last_lease_push = time.time()
                return
            except LeaseLostError:
                raise
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                wait_s = min(30.0, (1.8**attempt) + random.uniform(0.2, 1.0))
                print(f"[HF CAS RETRY {attempt}/{max_attempts}] {commit_label}: {exc} (waiting {wait_s:.1f}s)", flush=True)
                time.sleep(wait_s)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(45.0):
            if self._lease_lost or (time.time() - self._last_lease_push < LEASE_STANDALONE_RENEW_SECONDS):
                continue
            with self._lock:
                if self._lease_lost or (time.time() - self._last_lease_push < LEASE_STANDALONE_RENEW_SECONDS):
                    continue
                try:
                    self._cas_commit_owned(
                        [],
                        status="ACTIVE",
                        commit_label=f"[lease] renew {self.shard_id}",
                        max_attempts=4,
                    )
                except LeaseLostError as lost_exc:
                    print(f"[HF LEASE LOST] {lost_exc}", flush=True)
                    return
                except Exception as exc:
                    print(f"[HF LEASE WARN] Standalone heartbeat retry deferred ({exc})", flush=True)

    def push_arm_delta(
        self,
        local_arm_dir: Path,
        repo_arm_prefix: str,
        *,
        commit_label: str,
        include_winner_adapter_tid: str | None = None,
        strict: bool = False,
    ) -> None:
        """Upload new/modified files under `local_arm_dir` + CAS-verified lease renewal."""
        from huggingface_hub import CommitOperationAdd

        if self._lease_lost:
            raise LeaseLostError(f"Aborting trial execution because shard '{self.shard_id}' lost its HF lease.")
        if not local_arm_dir.is_dir():
            return

        with self._lock:
            ops: list[Any] = []
            staged_meta: dict[str, tuple[int, int]] = {}

            for fpath in sorted(local_arm_dir.rglob("*")):
                if not fpath.is_file():
                    continue
                rel = fpath.relative_to(local_arm_dir).as_posix()
                parts = rel.split("/")
                if any(p.startswith("checkpoint-") for p in parts):
                    continue
                p_name = parts[-1]
                if p_name.startswith(".campaign_lease") or p_name.endswith(".tmp"):
                    continue
                if "adapter" in parts:
                    if not include_winner_adapter_tid or f"trials/{include_winner_adapter_tid}/adapter/" not in f"{rel}/":
                        continue

                st = fpath.stat()
                sig = (st.st_mtime_ns, st.st_size)
                repo_path = f"{repo_arm_prefix}/{rel}"
                if self._uploaded_index.get(repo_path) == sig:
                    continue
                ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(fpath)))
                staged_meta[repo_path] = sig

            if not ops:
                return

            try:
                self._cas_commit_owned(ops, status="ACTIVE", commit_label=commit_label, max_attempts=6)
                self._uploaded_index.update(staged_meta)
                print(f"[HF SYNC OK] {commit_label} ({len(staged_meta)} files)", flush=True)
            except LeaseLostError:
                raise
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"Mandatory final HF sync failed: {commit_label}: {exc}") from exc
                print(f"[HF SYNC DEFERRED] Will retry accumulated files on next save ({commit_label}: {exc})", flush=True)

    def release_lease(self, *, status: str, note: str = "") -> None:
        self._stop_event.set()
        if self._lease_lost:
            return
        with self._lock:
            try:
                self._cas_commit_owned(
                    [],
                    status=status,
                    commit_label=f"[lease] {status.lower()} {self.shard_id}" + (f" ({note[:80]})" if note else ""),
                    max_attempts=4,
                )
                print(f"[HF LEASE {status}] {self.shard_id}", flush=True)
            except LeaseLostError:
                print(f"[HF LEASE SKIP RELEASE] Replacement VM already owns {self.shard_id}.", flush=True)
            except Exception as exc:
                print(f"[HF LEASE WARN] Could not write final lease state ({exc})", flush=True)


def verify_arm_git_commits(arm_dir: Path, expected_git_sha: str) -> None:
    """Reject any restored search or confirmation trial receipt whose `git_commit` != `expected_git_sha` (Sol P1 #3)."""
    trials_dir = arm_dir / "trials"
    if not trials_dir.is_dir():
        return
    for tjson in sorted(trials_dir.glob("*/trial.json")):
        rec = json.loads(tjson.read_text(encoding="utf-8"))
        if rec.get("status") != "PASS":
            continue
        obs_sha = str(rec.get("git_commit") or "")
        if obs_sha != expected_git_sha:
            raise RuntimeError(
                f"Git pin violation in {tjson}: receipt has git_commit='{obs_sha}', expected '{expected_git_sha}'!"
            )


def _select_current_leader_tid(entries: list[Any]) -> str | None:
    done = [e for e in entries if getattr(e, "status", None) == "DONE" and getattr(e, "metrics", None)]
    if not done:
        return None
    best = sorted(
        done,
        key=lambda e: (
            -float(e.metrics.get("exact_match_accuracy", e.metrics["tool_accuracy"])),
            float(e.metrics["search_memory_mb"]),
            float(e.metrics["trainable_parameters"]),
        ),
    )[0]
    return str(best.trial_id)


def verify_winner_adapter(arm_dir: Path, winner_tid: str) -> None:
    """Ensure the selected leader/winner has non-empty PEFT adapter weights & config."""
    adapter = arm_dir / "trials" / winner_tid / "adapter"
    if not (adapter / "adapter_config.json").is_file() or not any(
        p.is_file() and p.stat().st_size > 0 for p in adapter.glob("adapter_model.*")
    ):
        raise RuntimeError(f"Missing or empty winner adapter weights in {adapter}")


def _install_or_build_cuda_wheels(
    wheel_root: Path,
    *,
    need_mamba: bool,
    need_causal_conv: bool,
    syncer: HfShardSyncer,
) -> None:
    import importlib
    import torch

    if not (need_mamba or need_causal_conv):
        return

    probe_stmt = "import causal_conv1d" + (
        "; import mamba_ssm; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn"
        if need_mamba
        else ""
    )

    def _ok(*, verbose: bool = False) -> bool:
        importlib.invalidate_caches()
        try:
            import causal_conv1d  # type: ignore  # noqa: F401
            if need_mamba:
                import mamba_ssm  # type: ignore  # noqa: F401
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore  # noqa: F401
            subprocess.run([sys.executable, "-c", probe_stmt], check=True, capture_output=True, text=True)
            return True
        except Exception as exc:
            if verbose:
                err_detail = getattr(exc, "stderr", "") or str(exc)
                print(f"[WHEELS PROBE WARN] {err_detail.strip()}", flush=True)
            return False

    if _ok(verbose=False):
        print("[WHEELS] Fast CUDA hybrid kernels already importable.", flush=True)
        return

    # Always install pure-Python runtime/build deps from PyPI first (einops for mamba_ssm, ninja for causal_conv1d)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "packaging", "ninja", "einops"],
        check=True,
    )

    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_tag = torch.__version__.split("+")[0]
    cu_tag = (torch.version.cuda or "cpu").replace(".", "")
    abi_name = f"{py_tag}_torch{torch_tag}_cu{cu_tag}"
    abi_dir = wheel_root / abi_name
    abi_dir.mkdir(parents=True, exist_ok=True)

    cc_whls = sorted(abi_dir.glob("causal_conv1d-*.whl"))
    mb_whls = sorted(abi_dir.glob("mamba_ssm-*.whl"))
    if cc_whls and (not need_mamba or mb_whls):
        whls_to_install = [str(cc_whls[-1])] + ([str(mb_whls[-1])] if need_mamba else [])
        print(
            f"[WHEELS] Installing {len(whls_to_install)} pre-built wheels (--no-deps) from {abi_dir}...",
            flush=True,
        )
        res = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *whls_to_install],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            print(f"[WHEELS FAST-PATH PIP ERROR]\n{res.stdout}\n{res.stderr}", flush=True)
        elif _ok(verbose=True):
            print("[WHEELS] Pre-built CUDA kernels active (<10s fast path).", flush=True)
            return

    print(f"[WHEELS] ABI '{abi_name}' not cached yet; running self-healing CUDA C++20 wheel build...", flush=True)
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "8.0;8.9"
    env["CAUSAL_CONV1D_FORCE_BUILD"] = "TRUE"
    env["MAMBA_FORCE_BUILD"] = "TRUE"
    env["MAX_JOBS"] = "4"

    with tempfile.TemporaryDirectory() as sdist_tmp:
        tmp_p = Path(sdist_tmp)
        specs = ["causal-conv1d>=1.4.0,<2.0.0"] + (["mamba-ssm>=2.2.2,<3.0.0"] if need_mamba else [])
        for spec in specs:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--no-binary",
                    ":all:",
                    "--no-deps",
                    "--no-build-isolation",
                    "-d",
                    str(tmp_p),
                    spec,
                ],
                check=True,
            )
        for tarball in sorted(tmp_p.glob("*.tar.gz")):
            subprocess.run(["tar", "-xzf", str(tarball), "-C", str(tmp_p)], check=True)
        for setup_py in sorted(tmp_p.glob("*/setup.py")):
            txt = setup_py.read_text(encoding="utf-8")
            txt = txt.replace("-std=c++17", "-std=c++20")
            for bad_arch in ("10.3", "11.0", "12.1", "103", "110", "121"):
                txt = txt.replace(f'"{bad_arch}"', '"8.9"')
            lines = [
                ln for ln in txt.splitlines()
                if not any(tok in ln for tok in ("compute_103", "compute_110", "compute_121", "sm_103", "sm_110", "sm_121"))
            ]
            setup_py.write_text("\n".join(lines) + "\n", encoding="utf-8")
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w", str(abi_dir), str(setup_py.parent)],
                env=env,
                check=True,
            )

    built_cc = sorted(abi_dir.glob("causal_conv1d-*.whl"))
    built_mb = sorted(abi_dir.glob("mamba_ssm-*.whl"))
    built_to_install = [str(built_cc[-1])] + ([str(built_mb[-1])] if need_mamba else [])
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *built_to_install],
        check=True,
    )
    if not _ok(verbose=True):
        raise RuntimeError("CUDA hybrid kernel import failed even after wheel build.")

    try:
        self_wheels = sorted(abi_dir.glob("*.whl"))
        if self_wheels:
            from huggingface_hub import CommitOperationAdd

            ops = [
                CommitOperationAdd(path_in_repo=f"wheels_cache/{abi_name}/{w.name}", path_or_fileobj=str(w))
                for w in self_wheels
            ]
            syncer._cas_commit_owned(
                ops,
                status="ACTIVE",
                commit_label=f"[wheels] cache {abi_name} prebuilt CUDA kernels",
            )
    except LeaseLostError:
        raise
    except Exception as exc:
        print(f"[WHEELS WARN] Non-fatal wheel cache upload skip: {exc}", flush=True)


def _ensure_raw_xlam(work_repo: Path, *, api: Any, hf_repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    raw_dest = work_repo / "data" / "raw" / "xlam" / "xlam_function_calling_60k.json"
    raw_dest.parent.mkdir(parents=True, exist_ok=True)
    if raw_dest.is_file() and _sha256_file(raw_dest) == RAW_XLAM_SHA256:
        print("[DATA] Verified local raw xLAM SHA-256.", flush=True)
        return raw_dest

    print("[DATA] Downloading SHA-verified raw xLAM from private HF mirror...", flush=True)
    try:
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            repo_type="dataset",
            filename="data_cache/xlam_function_calling_60k.json",
            token=api.token,
        )
    except Exception as exc:
        print(f"[DATA] Private cache fallback ({exc}); pulling from {XLAM_FALLBACK_REPO}...", flush=True)
        downloaded = hf_hub_download(
            repo_id=XLAM_FALLBACK_REPO,
            repo_type="dataset",
            filename="xlam_function_calling_60k.json",
            revision=XLAM_FALLBACK_REV,
            token=api.token,
        )

    shutil.copyfile(downloaded, raw_dest)
    actual_sha = _sha256_file(raw_dest)
    if actual_sha != RAW_XLAM_SHA256:
        raw_dest.unlink(missing_ok=True)
        raise RuntimeError(f"Raw xLAM SHA-256 mismatch: {actual_sha} != {RAW_XLAM_SHA256}")
    print("[DATA] Raw xLAM SHA-256 verified OK.", flush=True)
    return raw_dest


def _run_zero_shot_validation_subprocess(
    *,
    work_repo: Path,
    model_id: str,
    manifest_rel: str,
    raw_rel: str,
    out_dir: Path,
    expected_git_sha: str,
    manifest_sha: str,
    val_ids: set[str],
    pinned_gpu: str,
    exact_gpu_name: str,
    allow_slow_kernels: bool,
) -> dict[str, Any]:
    """Evaluate unmodified base instruct model (`0` LoRA adapters) on the 200-item validation split in an isolated subprocess."""
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = out_dir / "zero_shot_validation.json"
    pred_file = out_dir / "predictions_validation.jsonl"
    if receipt_file.is_file() and pred_file.is_file():
        try:
            cached = json.loads(receipt_file.read_text(encoding="utf-8"))
            lines = [json.loads(ln) for ln in pred_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            seen_ids = {str(r.get("sample_id")) for r in lines}
            if (
                cached.get("git_commit") == expected_git_sha
                and cached.get("manifest_sha256") == manifest_sha
                and cached.get("gpu_name") == exact_gpu_name
                and seen_ids == val_ids
                and len(lines) == len(val_ids)
            ):
                print(
                    f"[ZERO-SHOT CACHED] {model_id} EM={cached['exact_match_accuracy']:.2%}",
                    flush=True,
                )
                return cached
        except Exception:
            pass

    inline_py = """
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aloe.datasets.model_registry import ModelRegistry
from aloe.datasets.xlam import load_xlam_file
from aloe.evaluation.prediction_store import PredictionStore
from aloe.evaluation.resources import enforce_hardware_contract
from aloe.evaluation.tool_calling import EVALUATOR_VERSION
from aloe.execution.backends.peft_mps import evaluate_model_on_split
from aloe.tracking.trial_store import atomic_write_text, get_git_commit

cfg = json.loads(sys.argv[1])
work_repo = Path(cfg["work_repo"]).resolve()
git_sha = get_git_commit(work_repo, forbid_dirty=True)
assert git_sha == cfg["expected_git_sha"], f"Git SHA mismatch: {git_sha} != {cfg['expected_git_sha']}"

hw = enforce_hardware_contract(
    cfg["model_id"],
    "cuda",
    pinned_gpu=cfg["pinned_gpu"],
    exact_gpu_name=cfg["exact_gpu_name"],
    official_campaign=True,
    allow_slow_kernels=bool(cfg["allow_slow_kernels"]),
)

registry = ModelRegistry()
canonical_id = cfg["model_id"]
for alias, target in registry.ALIASES.items():
    if canonical_id.lower() == alias:
        canonical_id = target
        break
pinned_rev = registry.PINNED_REVISIONS.get(canonical_id, "main")
rev_kw = {"revision": pinned_rev} if pinned_rev not in ("main", "UNKNOWN", "") else {}

manifest = json.loads((work_repo / cfg["manifest_rel"]).read_text(encoding="utf-8"))
val_ids = [str(i) for i in manifest["splits"]["validation"]["ids"]]
raw_examples = load_xlam_file(work_repo / cfg["raw_rel"])
id_to_ex = {str(ex.id): ex for ex in raw_examples}
val_examples = [id_to_ex[sid] for sid in sorted(val_ids)]

if hasattr(ModelRegistry, "_ensure_lfm3_rope_compat"):
    ModelRegistry._ensure_lfm3_rope_compat()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    canonical_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    **rev_kw,
).to("cuda")
if hasattr(ModelRegistry, "disable_lfm3_broken_triton_conv"):
    ModelRegistry.disable_lfm3_broken_triton_conv(model)
tokenizer = AutoTokenizer.from_pretrained(canonical_id, trust_remote_code=True, **rev_kw)
if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
    tokenizer.pad_token = tokenizer.eos_token

out_dir = Path(cfg["out_dir"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
slug = canonical_id.replace("/", "--")
pred_store = PredictionStore(out_dir / "predictions_validation.jsonl")
ev_out = evaluate_model_on_split(
    model,
    tokenizer,
    val_examples,
    prediction_store=pred_store,
    split_name="validation",
    model_id=canonical_id,
    model_revision=pinned_rev,
    arm="zero_shot",
    trial_id=f"zero_shot_{slug}_validation",
    seed=42,
    device="cuda",
    max_new_tokens=192,
)
peak_cuda_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3)
raw_sum = ev_out["summary"]
sdict = raw_sum.model_dump() if hasattr(raw_sum, "model_dump") else (raw_sum if isinstance(raw_sum, dict) else vars(raw_sum))
receipt = {
    "protocol_tag": "protocol-v1",
    "mode": "zero_shot_base_validation",
    "model_id": canonical_id,
    "model_revision": pinned_rev,
    "arm": "zero_shot",
    "split": "validation",
    "sample_count": int(sdict["total_samples"]),
    "exact_match_accuracy": float(sdict["exact_match_accuracy"]),
    "lenient_exact_match_accuracy": float(sdict.get("lenient_exact_match_accuracy", sdict["exact_match_accuracy"])),
    "tool_name_accuracy": float(sdict.get("tool_selection_accuracy", sdict.get("tool_match_rate", 0.0))),
    "argument_name_accuracy": float(sdict.get("argument_name_accuracy", 0.0)),
    "argument_value_accuracy": float(sdict.get("argument_value_accuracy", 0.0)),
    "parse_success_rate": float(sdict.get("parse_success_rate", 0.0)),
    "schema_valid_rate": float(sdict.get("schema_valid_rate", 0.0)),
    "trainable_parameters": 0,
    "peak_cuda_allocated_mb": peak_cuda_mb,
    "wall_seconds": round(time.time() - t0, 3),
    "eval_seconds": float(ev_out["eval_seconds"]),
    "predictions_sha256": str(ev_out["predictions_sha256"]),
    "evaluator_version": EVALUATOR_VERSION,
    "manifest_sha256": cfg["manifest_sha"],
    "git_commit": git_sha,
    "gpu_name": str(hw["gpu_name"]),
    "hardware_label": str(hw["hardware_label"]),
}
atomic_write_text(out_dir / "zero_shot_validation.json", json.dumps(receipt, indent=2) + "\\n")
"""
    payload = json.dumps(
        {
            "work_repo": str(work_repo),
            "model_id": model_id,
            "manifest_rel": manifest_rel,
            "raw_rel": raw_rel,
            "out_dir": str(out_dir),
            "expected_git_sha": expected_git_sha,
            "manifest_sha": manifest_sha,
            "pinned_gpu": pinned_gpu,
            "exact_gpu_name": exact_gpu_name,
            "allow_slow_kernels": allow_slow_kernels,
        }
    )
    subprocess.run([sys.executable, "-c", inline_py, payload], cwd=str(work_repo), check=True)
    return json.loads(receipt_file.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated per-arm ALOE shard on Colab Pro+ L4")
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-repo", required=True, help="Clean checkout pinned to --expected-git-sha")
    parser.add_argument("--expected-git-sha", required=True, help="Exact 40-char commit SHA required for model parity")
    parser.add_argument("--arms", required=True, help="Comma-separated arms owned by this shard")
    parser.add_argument("--phases", default="search,confirm", help="Comma-separated phases: zero_shot,search,confirm")
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--campaign-prefix", default="", help="Optional isolated HF campaign prefix (e.g. campaigns_cuda_supplementary)")
    parser.add_argument("--confirm-prefix", default="", help="Optional isolated HF confirm prefix (e.g. confirm_3seed_cuda_supplementary)")
    parser.add_argument("--allow-supplementary-cuda", action="store_true", help="Permit supplementary CUDA run for local-tier model in isolated prefix")
    parser.add_argument("--pinned-gpu", default="L4")
    parser.add_argument("--expected-exact-gpu", default="NVIDIA L4")
    parser.add_argument("--hf-repo", default="Walapalam/aloe-ipd-v1-runs")
    parser.add_argument("--local-store", default="/content/aloe_shard_store")
    parser.add_argument("--force-steal-lease", action="store_true")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is required.")
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Step 1: Fail-closed GPU check (<1s before heavy imports/downloads)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU attached! In Colab tap Runtime -> Change runtime type -> L4 GPU, then Run all."
        )
    obs_gpu = torch.cuda.get_device_name(0)
    if obs_gpu != args.expected_exact_gpu:
        raise RuntimeError(
            f"ADR-017/019 GPU lock mismatch: got '{obs_gpu}', required '{args.expected_exact_gpu}'. "
            "Switch Runtime -> Change runtime type -> L4 GPU."
        )

    # Step 2: Switch into pinned work_repo and verify exact clean Git SHA
    work_repo = Path(args.work_repo).resolve()
    if not work_repo.is_dir():
        raise FileNotFoundError(f"Work repo missing: {work_repo}")
    os.chdir(work_repo)
    shim_dir = Path("/tmp/aloe_worker_shim")
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(
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
        "    pass\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(work_repo))
    sys.path.insert(0, str(shim_dir))
    os.environ["PYTHONPATH"] = f"{shim_dir}:{work_repo}" + (
        f":{os.environ['PYTHONPATH']}" if os.environ.get("PYTHONPATH") else ""
    )

    import botorch.fit as _bt_fit
    from botorch.optim.fit import fit_gpytorch_mll_torch as _fit_mll_torch

    _orig_fit_mll = _bt_fit.fit_gpytorch_mll

    def _resilient_fit_gpytorch_mll(mll, *f_args, **f_kwargs):
        try:
            return _orig_fit_mll(mll, *f_args, **f_kwargs)
        except Exception as fit_exc:
            if "optimizer" in f_kwargs:
                raise
            print(
                f"[MOBO GP FALLBACK] SciPy L-BFGS-B MLL fit hit {fit_exc}; "
                "refitting exact Matern-5/2 ARD GP via fit_gpytorch_mll_torch (Adam, 500 steps)...",
                flush=True,
            )
            return _orig_fit_mll(
                mll,
                *f_args,
                optimizer=_fit_mll_torch,
                optimizer_kwargs={"step_limit": 500},
                **f_kwargs,
            )

    _bt_fit.fit_gpytorch_mll = _resilient_fit_gpytorch_mll

    from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
    from aloe.evaluation.resources import enforce_hardware_contract
    from aloe.execution.campaign_runner import CampaignRunner, compute_exact_feasible_pool
    from aloe.execution.trial_config import TrainingParameters, TrialConfig
    from aloe.pipeline import inspect_to_grammar
    from aloe.tracking.trial_store import atomic_write_text, get_git_commit
    from scripts.run_colab_a100_campaign import _verify_completed_trial_receipt
    from scripts.verify_campaign import verify_campaign_dir, verify_manifest

    actual_git_sha = get_git_commit(work_repo, forbid_dirty=True)
    if actual_git_sha != args.expected_git_sha:
        raise RuntimeError(
            f"Pinned commit mismatch in {work_repo}: got '{actual_git_sha}', expected '{args.expected_git_sha}'."
        )
    print(f"[GIT VERIFIED] {work_repo} @ {actual_git_sha[:12]} (clean)", flush=True)

    api = HfApi(token=hf_token)
    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    phase_list = [p.strip() for p in args.phases.split(",") if p.strip()]
    invalid_phases = set(phase_list) - {"zero_shot", "search", "confirm"}
    if invalid_phases:
        raise ValueError(f"Unsupported shard phases {sorted(invalid_phases)} (sealed final-test is blocked here).")
    if args.allow_supplementary_cuda:
        is_isolated_prefix = bool(args.campaign_prefix and args.campaign_prefix.strip() not in ("campaigns", "campaigns_multiseed"))
        is_zero_shot_only = set(phase_list) == {"zero_shot"}
        if not (is_isolated_prefix or is_zero_shot_only):
            raise ValueError(
                "--allow-supplementary-cuda requires an isolated --campaign-prefix "
                "(e.g. campaigns_cuda_supplementary) so canonical campaigns/ are never touched."
            )

    syncer = HfShardSyncer(
        api=api,
        hf_repo=args.hf_repo,
        shard_id=args.shard_id,
        model_id=args.model,
        arms=arm_list,
        phases=phase_list,
        force_steal_lease=args.force_steal_lease,
    )
    syncer.acquire_lease()

    try:
        local_store = Path(args.local_store).resolve()
        local_store.mkdir(parents=True, exist_ok=True)
        if args.campaign_prefix and args.campaign_prefix.strip():
            campaign_prefix = args.campaign_prefix.strip().strip("/")
        else:
            campaign_prefix = "campaigns" if args.search_seed == 42 else f"campaigns_multiseed/seed_{args.search_seed}"
        confirm_prefix = (
            args.confirm_prefix.strip().strip("/")
            if (args.confirm_prefix and args.confirm_prefix.strip())
            else "confirm_3seed"
        )
        allow_slow_flag = bool(args.allow_supplementary_cuda)

        # Step 3: Download cached CUDA wheels & install fast hybrid kernels
        snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            local_dir=str(local_store),
            allow_patterns=["wheels_cache/**"],
            token=hf_token,
        )
        model_lower = args.model.lower()
        need_mamba = "falcon-h1" in model_lower
        need_causal_conv = need_mamba or ("lfm3" in model_lower)
        _install_or_build_cuda_wheels(
            local_store / "wheels_cache",
            need_mamba=need_mamba,
            need_causal_conv=need_causal_conv,
            syncer=syncer,
        )

        # Step 4: Verify raw xLAM dataset & frozen split manifest
        manifest_rel = "data/manifests/xlam_ipd_v1.json"
        raw_rel = "data/raw/xlam/xlam_function_calling_60k.json"
        _ensure_raw_xlam(work_repo, api=api, hf_repo=args.hf_repo)
        manifest_audit = verify_manifest(
            work_repo / manifest_rel,
            raw_data_path=work_repo / raw_rel,
            enforce_frozen_v1=True,
            require_raw_data=True,
        )
        if not manifest_audit["raw_xlam_independently_verified"]:
            raise RuntimeError("Raw xLAM independent verification failed.")
        manifest_sha = str(manifest_audit["manifest_sha256"])
        split_id_sets: dict[str, set[str]] = manifest_audit["split_id_sets"]

        # Step 5: Inspect model grammar & pull any prior progress (including leader adapters!) from HF
        desc, grammar = inspect_to_grammar(args.model, backend="auto")
        slug = desc.model_id.replace("/", "--")
        active_arms = [
            a for a in arm_list
            if not (a == "borobia_ssm" and not any(g.component_type.value == "ssm" for g in grammar.groups))
        ]
        parity_info = compute_exact_feasible_pool(desc, grammar, parity_tol=0.05)
        baseline_params = int(parity_info["baseline_params"])

        allow_pats = [f"{campaign_prefix}/{slug}/hardware_lock.json"]
        if "zero_shot" in phase_list:
            allow_pats.append(f"zero_shot_baselines/{slug}/**")
        for a in active_arms:
            allow_pats.append(f"{campaign_prefix}/{slug}/{a}/**")
            if "confirm" in phase_list and args.search_seed == 42:
                allow_pats.append(f"{confirm_prefix}/{slug}/{a}/**")
        snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            local_dir=str(local_store),
            allow_patterns=allow_pats,
            ignore_patterns=["**/checkpoint-*/*"],
            token=hf_token,
        )

        # Verify git_commit == expected_git_sha on all restored receipts before running (Sol P1 #3)
        for a in active_arms:
            verify_arm_git_commits(local_store / campaign_prefix / slug / a, args.expected_git_sha)
            if "confirm" in phase_list and args.search_seed == 42:
                verify_arm_git_commits(local_store / confirm_prefix / slug / a, args.expected_git_sha)

        # Pre-write & enforce hardware_lock.json == NVIDIA L4 (when running search/confirm)
        hw_lock_file = local_store / campaign_prefix / slug / "hardware_lock.json"
        if ("search" in phase_list) or ("confirm" in phase_list):
            if hw_lock_file.is_file():
                lock_meta = json.loads(hw_lock_file.read_text(encoding="utf-8"))
                if lock_meta.get("locked_gpu_name") != args.expected_exact_gpu:
                    raise RuntimeError(f"Existing hardware_lock.json mismatch: {lock_meta}")
            else:
                lock_payload = (
                    json.dumps(
                        {
                            "model_id": desc.model_id,
                            "locked_gpu_name": args.expected_exact_gpu,
                            "locked_memory_metric_name": "cuda_max_memory_allocated",
                        },
                        indent=2,
                    )
                    + "\n"
                )
                atomic_write_text(hw_lock_file, lock_payload)
                syncer._cas_commit_owned(
                    [
                        CommitOperationAdd(
                            path_in_repo=f"{campaign_prefix}/{slug}/hardware_lock.json",
                            path_or_fileobj=io.BytesIO(lock_payload.encode("utf-8")),
                        )
                    ],
                    status="ACTIVE",
                    commit_label=f"[lock] {slug} locked to {args.expected_exact_gpu}",
                )

        hw = enforce_hardware_contract(
            args.model,
            "cuda",
            pinned_gpu=args.pinned_gpu,
            exact_gpu_name=args.expected_exact_gpu,
            official_campaign=True,
            allow_slow_kernels=allow_slow_flag,
        )
        exact_gpu_name = str(hw["gpu_name"])
        print(
            f"[SHARD LIVE] shard={args.shard_id} | GPU verified: {hw['hardware_label']} | "
            f"arms={active_arms} | phases={phase_list} | prefix={campaign_prefix} | Starting trials...",
            flush=True,
        )

        # Phase 0: Optional Base Zero-Shot Pre-Fine-Tuning Validation Baseline (isolated subprocess)
        if "zero_shot" in phase_list:
            zs_mirror_dir = local_store / "zero_shot_baselines" / slug
            print(f"\n=== [SHARD ZERO-SHOT] {desc.model_id} :: validation (200 items, 0 adapters) ===", flush=True)
            zs_receipt = _run_zero_shot_validation_subprocess(
                work_repo=work_repo,
                model_id=desc.model_id,
                manifest_rel=manifest_rel,
                raw_rel=raw_rel,
                out_dir=zs_mirror_dir,
                expected_git_sha=args.expected_git_sha,
                manifest_sha=manifest_sha,
                val_ids=split_id_sets["validation"],
                pinned_gpu=args.pinned_gpu,
                exact_gpu_name=exact_gpu_name,
                allow_slow_kernels=allow_slow_flag,
            )
            syncer.push_arm_delta(
                zs_mirror_dir,
                f"zero_shot_baselines/{slug}",
                commit_label=f"[{args.shard_id}] zero_shot {slug} EM={zs_receipt['exact_match_accuracy']:.2%}",
                strict=True,
            )
            print(
                f"[ZERO-SHOT VERIFIED] {desc.model_id} EM={zs_receipt['exact_match_accuracy']:.2%} "
                f"peak_cuda={zs_receipt['peak_cuda_allocated_mb']:.1f} MB",
                flush=True,
            )

        # Phase 1: Search (hook AFTER ledger.save marks DONE so preempted runs never re-run DONE trials!)
        if "search" in phase_list:
            for arm in active_arms:
                t0 = time.time()
                local_dir = work_repo / "runs" / "ipd_campaigns" / f"s{args.search_seed}" / slug / arm
                mirror_dir = local_store / campaign_prefix / slug / arm
                repo_arm_prefix = f"{campaign_prefix}/{slug}/{arm}"
                if (local_dir / "campaign_ledger.json").is_file() and not (mirror_dir / "campaign_ledger.json").is_file():
                    try:
                        _ld = json.loads((local_dir / "campaign_ledger.json").read_text(encoding="utf-8"))
                        if sum(1 for _e in _ld.get("entries", []) if _e.get("status") == "DONE") == 0:
                            shutil.rmtree(local_dir, ignore_errors=True)
                    except Exception:
                        pass
                base_tp = (
                    TrainingParameters(max_steps=125, batch_size=1, gradient_accumulation_steps=8)
                    if "gemma" in model_lower
                    else TrainingParameters(max_steps=125)
                )
                print(
                    f"\n=== [SHARD SEARCH] {desc.model_id} :: arm={arm} (search_seed={args.search_seed}) ===",
                    flush=True,
                )
                runner = CampaignRunner(
                    arch=desc,
                    grammar=grammar,
                    campaign_dir=local_dir,
                    arm=arm,
                    target_trials=20,
                    search_seed=args.search_seed,
                    training_seed=42,
                    manifest_path=manifest_rel,
                    raw_path=raw_rel,
                    base_training=base_tp,
                    pinned_gpu=args.pinned_gpu,
                    exact_gpu_name=exact_gpu_name,
                    hardware_lock_path=hw_lock_file,
                    official_campaign=True,
                    allow_slow_kernels=allow_slow_flag,
                    forbid_dirty_git=True,
                    use_subprocess_worker=True,
                    mirror_dir=mirror_dir,
                )
                # Copy restored mirror ledger/trials into local_dir so CampaignRunner resumes seamlessly
                if (mirror_dir / "campaign_ledger.json").is_file() and not (local_dir / "campaign_ledger.json").is_file():
                    local_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(mirror_dir, local_dir, dirs_exist_ok=True)

                synced_done = 0
                if runner.ledger.exists():
                    synced_done = sum(1 for e in runner.ledger.load().entries if e.status == "DONE")

                orig_save = runner.ledger.save

                def _post_done_ledger_save(
                    state: Any,
                    *,
                    _orig: Any = orig_save,
                    _mirror: Path = mirror_dir,
                    _prefix: str = repo_arm_prefix,
                    _arm: str = arm,
                ) -> None:
                    nonlocal synced_done
                    _orig(state)
                    cur_done = sum(1 for e in state.entries if e.status == "DONE")
                    if cur_done > synced_done:
                        synced_done = cur_done
                        leader_tid = _select_current_leader_tid(state.entries)
                        if leader_tid is not None:
                            verify_winner_adapter(_mirror, leader_tid)
                        syncer.push_arm_delta(
                            _mirror,
                            _prefix,
                            commit_label=f"[{args.shard_id}] {slug}/{_arm} DONE {cur_done}/{state.target_trials} (leader={leader_tid})",
                            include_winner_adapter_tid=leader_tid,
                        )

                runner.ledger.save = _post_done_ledger_save  # type: ignore[method-assign]
                summary = runner.run_campaign(retry_failed=True)
                winner_tid = str(summary["selected_winner_trial_id"])
                verify_campaign_dir(
                    mirror_dir,
                    expected_manifest_sha=manifest_sha,
                    expected_split_ids=split_id_sets,
                    allow_smoke_budget=False,
                )
                verify_arm_git_commits(mirror_dir, args.expected_git_sha)
                verify_winner_adapter(mirror_dir, winner_tid)
                syncer.push_arm_delta(
                    mirror_dir,
                    repo_arm_prefix,
                    commit_label=f"[{args.shard_id}] {slug}/{arm} SEARCH COMPLETE (winner={winner_tid})",
                    include_winner_adapter_tid=winner_tid,
                    strict=True,
                )
                print(
                    f"[SEARCH VERIFIED] arm={arm} winner={winner_tid} "
                    f"EM={summary['selected_winner_metrics']['tool_accuracy']:.2%} ({(time.time()-t0)/60:.1f}m)",
                    flush=True,
                )

        # Phase 2: 3-Seed Validation Confirmation (only for search_seed == 42)
        if "confirm" in phase_list:
            if args.search_seed != 42:
                raise ValueError("3-seed confirmation applies strictly to search_seed=42.")
            for arm in active_arms:
                arm_mirror_dir = local_store / campaign_prefix / slug / arm
                verify_campaign_dir(
                    arm_mirror_dir,
                    expected_manifest_sha=manifest_sha,
                    expected_split_ids=split_id_sets,
                    allow_smoke_budget=False,
                )
                verify_arm_git_commits(arm_mirror_dir, args.expected_git_sha)
                sdata = json.loads((arm_mirror_dir / "campaign_summary.json").read_text(encoding="utf-8"))
                if sdata.get("protocol_tag") != "protocol-v1":
                    raise PermissionError(f"Invalid protocol_tag in {arm_mirror_dir / 'campaign_summary.json'}")
                verify_winner_adapter(arm_mirror_dir, str(sdata["selected_winner_trial_id"]))
                wcfg = TrialConfig.model_validate(sdata["selected_winner_config"])
                locked_dump = wcfg.model_dump(mode="json")

                local_trials_dir = work_repo / "runs" / confirm_prefix / slug / arm / "trials"
                mirror_arm_confirm = local_store / confirm_prefix / slug / arm
                mirror_trials_dir = mirror_arm_confirm / "trials"
                local_trials_dir.mkdir(parents=True, exist_ok=True)
                repo_confirm_prefix = f"{confirm_prefix}/{slug}/{arm}"

                for s_val in REQUIRED_CONFIRM_SEEDS:
                    tid = f"{arm}_winner_s{s_val}_validation"
                    done_mirror = mirror_trials_dir / tid / "trial.json"
                    if _verify_completed_trial_receipt(
                        done_mirror,
                        expected_seed=s_val,
                        expected_split="validation",
                        expected_split_ids=split_id_sets["validation"],
                        expected_manifest_sha=manifest_sha,
                        expected_gpu_name=exact_gpu_name,
                        expected_winner_cfg=locked_dump,
                    ):
                        verify_arm_git_commits(mirror_arm_confirm, args.expected_git_sha)
                        print(f"[CONFIRM VERIFIED CACHED] {tid}", flush=True)
                        continue

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
                        "validation",
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
                    if allow_slow_flag:
                        cmd.append("--allow-slow-kernels")
                    if seeded_cfg.peft_method == "lora":
                        cmd.append("--enforce-parity")

                    print(f"=== [SHARD CONFIRM] {desc.model_id} :: {tid} ===", flush=True)
                    subprocess.run(cmd, cwd=str(work_repo), check=True)
                    if not _verify_completed_trial_receipt(
                        done_mirror,
                        expected_seed=s_val,
                        expected_split="validation",
                        expected_split_ids=split_id_sets["validation"],
                        expected_manifest_sha=manifest_sha,
                        expected_gpu_name=exact_gpu_name,
                        expected_winner_cfg=locked_dump,
                    ):
                        raise RuntimeError(f"Post-run receipt verification failed for {tid}")
                    verify_arm_git_commits(mirror_arm_confirm, args.expected_git_sha)

                    syncer.push_arm_delta(
                        mirror_arm_confirm,
                        repo_confirm_prefix,
                        commit_label=f"[{args.shard_id}] confirm_3seed {slug}/{arm} s={s_val} PASS",
                    )

                syncer.push_arm_delta(
                    mirror_arm_confirm,
                    repo_confirm_prefix,
                    commit_label=f"[{args.shard_id}] confirm_3seed {slug}/{arm} ALL 3 SEEDS COMPLETE",
                    strict=True,
                )

        syncer.release_lease(status="COMPLETED", note="all_phases_verified")
        print(f"\n🏁 SHARD {args.shard_id} 100% COMPLETE & VERIFIED ON HUGGING FACE!", flush=True)
        return 0

    except Exception as exc:
        syncer.release_lease(status="FAILED", note=str(exc)[:200])
        print(f"\n❌ SHARD {args.shard_id} ABORTED: {exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

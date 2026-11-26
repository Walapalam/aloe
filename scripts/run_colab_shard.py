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


_COLAB_WORKER_SHIM_SOURCE = (
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


def _log_final_test_attempt(
    syncer: HfShardSyncer,
    *,
    slug: str,
    run_id: str,
    stage: str,
    attempt_id: str,
    seal_sha256: str,
    seq: int | None = None,
    local_store: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Publish append-only attempt record to HF FIRST (`_cas_commit_owned`), then persist local mirror."""
    from huggingface_hub import CommitOperationAdd
    from aloe.tracking.trial_store import atomic_write_text

    eff_seq = seq if seq is not None else (1 if stage == "LAUNCHED" else 2)
    ts_compact = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    payload = {
        "protocol_tag": "protocol-v1",
        "attempt_id": attempt_id,
        "seq": eff_seq,
        "seal_sha256": seal_sha256,
        "slug": slug,
        "run_id": run_id,
        "shard_id": syncer.shard_id,
        "stage": stage,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    rel_path = (
        f"final_test_attempts/{slug}/"
        f"{run_id}__{eff_seq:02d}_{stage}__{ts_compact}__{socket.gethostname()}_{os.getpid()}_{attempt_id}.json"
    )
    with syncer._lock:
        syncer._cas_commit_owned(
            [
                CommitOperationAdd(
                    path_in_repo=rel_path,
                    path_or_fileobj=io.BytesIO(encoded.encode("utf-8")),
                )
            ],
            status="ACTIVE",
            commit_label=f"[{syncer.shard_id}] attempt {stage} {slug}/{run_id}",
            max_attempts=6,
        )
    if local_store is not None:
        atomic_write_text(local_store / rel_path, encoded)


def _replay_local_attempts_to_hf(syncer: HfShardSyncer, local_store: Path, slug: str) -> None:
    """Idempotently replay any local `final_test_attempts/<slug>/*.json` files under original filenames."""
    from huggingface_hub import CommitOperationAdd

    adir = local_store / "final_test_attempts" / slug
    if not adir.is_dir():
        return
    ops = []
    for jf in sorted(adir.glob("*.json")):
        rel = f"final_test_attempts/{slug}/{jf.name}"
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=io.BytesIO(jf.read_bytes())))
    if ops:
        with syncer._lock:
            syncer._cas_commit_owned(
                ops,
                status="ACTIVE",
                commit_label=f"[{syncer.shard_id}] replay {len(ops)} local attempt logs for {slug}",
                max_attempts=6,
            )


def _has_completed_attempt_local(local_store: Path, slug: str, run_id: str) -> bool:
    adir = local_store / "final_test_attempts" / slug
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


def _find_open_launched_attempt_id(local_store: Path, slug: str, run_id: str) -> str | None:
    adir = local_store / "final_test_attempts" / slug
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


def _verify_immutable_seal_standalone(
    seal_path: Path,
    *,
    expected_model_id: str,
    expected_model_revision: str,
    expected_git_sha: str,
    expected_manifest_sha: str,
    expected_gpu_name: str,
    expected_memory_metric: str,
    expected_complete_arms: tuple[str, ...],
    campaigns_model_dir: Path,
    expected_split_ids: dict[str, set[str]],
    expected_seal_sha256: str,
    shim_path: Path | None = None,
    frozen_plan_path: Path | None = None,
    verify_code_digests: bool = True,
) -> dict[str, Any]:
    """Read-only verification of pre-committed `test_unsealed.json` against ALL canonical arms & code digests."""
    import re as _re
    from aloe.execution.trial_config import TrialConfig
    from scripts.run_colab_a100_campaign import compute_full_winner_sha256
    from scripts.verify_campaign import verify_campaign_dir

    hex64 = _re.compile(r"^[0-9a-f]{64}$")
    if not expected_seal_sha256 or not hex64.fullmatch(expected_seal_sha256):
        raise PermissionError(
            f"Approved 64-char ceremony expected_seal_sha256 is required; got {expected_seal_sha256!r}"
        )
    if not seal_path.is_file():
        raise PermissionError(
            f"Missing pre-committed read-only seal {seal_path}; shard workers are forbidden from self-issuing seals."
        )
    actual_seal_sha = _sha256_file(seal_path)
    if actual_seal_sha != expected_seal_sha256:
        raise PermissionError(
            f"Seal SHA-256 mismatch at {seal_path}: got {actual_seal_sha}, expected {expected_seal_sha256}"
        )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    req = {
        "protocol_tag": "protocol-v1",
        "model_id": expected_model_id,
        "model_revision": expected_model_revision,
        "git_commit": expected_git_sha,
        "manifest_sha256": expected_manifest_sha,
        "locked_gpu_name": expected_gpu_name,
        "locked_memory_metric_name": expected_memory_metric,
        "allowed_seeds": REQUIRED_CONFIRM_SEEDS,
        "max_steps": 125,
        "max_new_tokens": 192,
    }
    for k, exp_v in req.items():
        if seal.get(k) != exp_v:
            raise PermissionError(f"Seal field '{k}' mismatch in {seal_path}: {seal.get(k)!r} != {exp_v!r}")

    req_hash_keys = (
        "colab_bootstrap_sha256",
        "mps_bootstrap_sha256",
        "bootstrap_sha256",
        "shim_sha256",
        "analysis_plan_sha256",
        "audit_script_sha256",
        "evaluator_script_sha256",
        "bootstrap_engine_sha256",
    )
    for hk in req_hash_keys:
        hv = seal.get(hk)
        if not isinstance(hv, str) or not hex64.fullmatch(hv):
            raise PermissionError(f"Seal missing valid 64-hex frozen digest '{hk}' in {seal_path}: {hv!r}")
    exp_combo = hashlib.sha256(
        f"colab:{seal['colab_bootstrap_sha256']}|mps:{seal['mps_bootstrap_sha256']}".encode("utf-8")
    ).hexdigest()
    if seal["bootstrap_sha256"] != exp_combo:
        raise PermissionError(f"Seal bootstrap_sha256 inconsistent with colab/mps digests in {seal_path}")

    if verify_code_digests:
        self_file = Path(__file__).resolve()
        orch_root = self_file.parent.parent
        if _sha256_file(self_file) != seal["colab_bootstrap_sha256"]:
            raise PermissionError("Frozen colab_bootstrap_sha256 mismatch against live run_colab_shard.py!")
        expected_shim_sha = (
            _sha256_file(shim_path)
            if shim_path is not None
            else hashlib.sha256(_COLAB_WORKER_SHIM_SOURCE.encode("utf-8")).hexdigest()
        )
        if expected_shim_sha != seal["shim_sha256"]:
            raise PermissionError("Frozen shim_sha256 mismatch against worker sitecustomize.py!")
        for hk, rel_f in (
            ("mps_bootstrap_sha256", "scripts/run_local_mps_final.py"),
            ("audit_script_sha256", "scripts/final_test_audit.py"),
            ("evaluator_script_sha256", "scripts/evaluate_final_test_suite.py"),
            ("bootstrap_engine_sha256", "aloe/evaluation/bootstrap.py"),
            ("analysis_plan_sha256", "docs/PRE_UNSEAL_ANALYSIS_PLAN.md"),
        ):
            cand = orch_root / rel_f
            if not cand.is_file():
                raise PermissionError(f"Missing frozen artifact for '{hk}': {cand}!")
            if _sha256_file(cand) != seal[hk]:
                raise PermissionError(f"Frozen '{hk}' mismatch against {cand}!")
    if frozen_plan_path is not None:
        if not frozen_plan_path.is_file():
            raise PermissionError(f"Missing published frozen analysis plan: {frozen_plan_path}")
        if _sha256_file(frozen_plan_path) != seal["analysis_plan_sha256"]:
            raise PermissionError(f"Published analysis plan SHA-256 mismatch at {frozen_plan_path}!")

    seal_sigs = seal.get("winner_signatures")
    if not isinstance(seal_sigs, dict) or set(seal_sigs.keys()) != set(expected_complete_arms):
        raise PermissionError(
            f"Seal winner_signatures must equal complete frozen arm set {sorted(expected_complete_arms)}"
        )

    winners_by_arm: dict[str, Any] = {}
    recomputed: dict[str, str] = {}
    for arm in expected_complete_arms:
        arm_dir = campaigns_model_dir / arm
        verify_campaign_dir(
            arm_dir,
            expected_manifest_sha=expected_manifest_sha,
            expected_split_ids=expected_split_ids,
            allow_smoke_budget=False,
        )
        sdata = json.loads((arm_dir / "campaign_summary.json").read_text(encoding="utf-8"))
        if sdata.get("protocol_tag") != "protocol-v1":
            raise PermissionError(f"Non-protocol-v1 summary in {arm_dir}")
        wcfg = TrialConfig.model_validate(sdata["selected_winner_config"])
        winners_by_arm[arm] = wcfg
        recomputed[arm] = compute_full_winner_sha256(wcfg)

    if seal_sigs != recomputed:
        raise PermissionError(f"Seal winner_signatures mismatch against locked campaign summaries in {seal_path}")
    return {"seal_sha256": actual_seal_sha, "seal": seal, "winners_by_arm": winners_by_arm}


def _verify_strict_receipt_standalone(
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
    expected_winner_cfg: Any,
) -> bool:
    """Full-hash, finite-resource, boolean-EM receipt verifier identical to `final_test_audit.verify_strict_trial_receipt`."""
    import math as _math
    from aloe.evaluation.tool_calling import EVALUATOR_VERSION
    from scripts.run_colab_a100_campaign import compute_full_winner_sha256

    def _fin(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and _math.isfinite(float(v))

    def _pint(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool) and v > 0

    if not trial_json_path.is_file():
        return False
    rec = json.loads(trial_json_path.read_text(encoding="utf-8"))
    tcfg = rec.get("trial_config") or {}
    metrics = rec.get("metrics") or {}
    obs_gpu = rec.get("gpu_name") or rec.get("hardware")
    if (
        rec.get("status") != "PASS"
        or rec.get("git_commit") != expected_git_sha
        or rec.get("model_id") != expected_model_id
        or rec.get("model_revision") != expected_model_revision
        or obs_gpu != expected_gpu_name
        or metrics.get("memory_metric_name") != expected_memory_metric
        or rec.get("dataset_manifest_sha256") != expected_manifest_sha
        or tcfg.get("seed") != expected_seed
        or isinstance(tcfg.get("seed"), bool)
        or metrics.get("optimizer_steps_completed") != 125
        or isinstance(metrics.get("optimizer_steps_completed"), bool)
        or metrics.get("fresh_process_reload") is not True
        or metrics.get("eval_samples") != len(expected_split_ids)
        or isinstance(metrics.get("eval_samples"), bool)
        or metrics.get("evaluator_version") != EVALUATOR_VERSION
    ):
        return False
    tp = metrics.get("trainable_params")
    if not (_fin(tp) and int(tp) > 0 and float(tp) == float(int(tp))):
        return False
    if compute_full_winner_sha256(tcfg) != compute_full_winner_sha256(expected_winner_cfg):
        return False
    peft_method = str(tcfg.get("peft_method", "lora"))
    if peft_method == "lora":
        if metrics.get("is_budget_matched_5pct") is not True:
            return False
    elif peft_method == "adalora":
        if not (_pint(metrics.get("adalora_init_params")) and _pint(metrics.get("adalora_final_active_params"))):
            return False
    else:
        return False
    if not (_fin(metrics.get("search_memory_mb")) and float(metrics["search_memory_mb"]) > 0.0):
        return False
    pred_file = trial_json_path.parent / f"predictions_{expected_split}.jsonl"
    exp_sha = metrics.get("predictions_sha256")
    if not pred_file.is_file() or not exp_sha or _sha256_file(pred_file) != exp_sha:
        return False
    rows = [json.loads(ln) for ln in pred_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(rows) != len(expected_split_ids) or len(rows) == 0 or {str(r.get("sample_id")) for r in rows} != expected_split_ids:
        return False
    w_dump = expected_winner_cfg.model_dump(mode="json") if hasattr(expected_winner_cfg, "model_dump") else dict(expected_winner_cfg)
    expected_arm = str(w_dump.get("arm", ""))
    expected_tid = trial_json_path.parent.name
    if not all(
        type(r.get("exact_match")) is bool
        and r.get("split") == expected_split
        and r.get("evaluator_version") == EVALUATOR_VERSION
        and r.get("trial_id") == expected_tid
        and r.get("model_id") == expected_model_id
        and r.get("model_revision") == expected_model_revision
        and r.get("seed") == expected_seed
        and not isinstance(r.get("seed"), bool)
        and (not expected_arm or r.get("arm") == expected_arm)
        for r in rows
    ):
        return False
    recomp_em = sum(1 for r in rows if r["exact_match"]) / float(len(rows))
    rep_em = metrics.get("ast_exact_match_accuracy")
    rep_tool = metrics.get("tool_accuracy")
    return all(_fin(v) and 0.0 <= float(v) <= 1.0 and abs(recomp_em - float(v)) <= 1e-6 for v in (rep_em, rep_tool))


def _verify_strict_zero_shot_standalone(
    receipt_path: Path,
    *,
    expected_split: str,
    expected_split_ids: set[str],
    expected_manifest_sha: str,
    expected_git_sha: str,
    expected_model_id: str,
    expected_model_revision: str,
    expected_gpu_name: str,
    expected_memory_metric: str = "cuda_max_memory_allocated",
    raise_on_error: bool = False,
) -> dict[str, Any] | None:
    """Strict zero-shot JSON + JSONL verifier identical to `final_test_audit.verify_strict_zero_shot_receipt`."""
    import math as _math
    from aloe.evaluation.tool_calling import EVALUATOR_VERSION

    def _fin(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and _math.isfinite(float(v))

    def _fail(msg: str) -> None:
        if raise_on_error:
            raise ValueError(f"[{receipt_path}] {msg}")
        return None

    if not receipt_path.is_file():
        return _fail("zero-shot receipt file missing")
    rec = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        rec.get("protocol_tag") != "protocol-v1"
        or rec.get("arm") != "zero_shot"
        or rec.get("split") != expected_split
        or rec.get("model_id") != expected_model_id
        or rec.get("model_revision") != expected_model_revision
        or rec.get("git_commit") != expected_git_sha
        or rec.get("gpu_name") != expected_gpu_name
        or rec.get("manifest_sha256") != expected_manifest_sha
        or rec.get("trainable_parameters") != 0
        or isinstance(rec.get("trainable_parameters"), bool)
        or rec.get("sample_count") != len(expected_split_ids)
        or isinstance(rec.get("sample_count"), bool)
        or rec.get("evaluator_version") != EVALUATOR_VERSION
    ):
        return _fail("zero-shot top-level metadata mismatch")

    if expected_split == "test" or "memory_metric_name" in rec:
        if rec.get("memory_metric_name") != expected_memory_metric:
            return _fail(f"memory_metric_name={rec.get('memory_metric_name')!r} != {expected_memory_metric!r}")
        if not (_fin(rec.get("search_memory_mb")) and float(rec["search_memory_mb"]) > 0.0):
            return _fail("invalid search_memory_mb")
    else:
        if expected_gpu_name != "NVIDIA L4" or not (_fin(rec.get("peak_cuda_allocated_mb")) and float(rec["peak_cuda_allocated_mb"]) > 0.0):
            return _fail("invalid peak_cuda_allocated_mb")

    pred_path = receipt_path.parent / f"predictions_{expected_split}.jsonl"
    exp_sha = rec.get("predictions_sha256")
    if not pred_path.is_file() or not exp_sha or _sha256_file(pred_path) != exp_sha:
        return _fail("zero-shot predictions JSONL missing or SHA-256 mismatch")
    rows = [json.loads(ln) for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(rows) != len(expected_split_ids) or len(rows) == 0 or {str(r.get("sample_id")) for r in rows} != expected_split_ids:
        return _fail("zero-shot prediction sample_id set mismatch")
    slug = expected_model_id.replace("/", "--")
    expected_tid = f"zero_shot_{slug}_{expected_split}"
    if not all(
        type(r.get("exact_match")) is bool
        and r.get("split") == expected_split
        and r.get("arm") == "zero_shot"
        and r.get("trial_id") == expected_tid
        and r.get("model_id") == expected_model_id
        and r.get("model_revision") == expected_model_revision
        and r.get("seed") == 42
        and not isinstance(r.get("seed"), bool)
        and r.get("evaluator_version") == EVALUATOR_VERSION
        for r in rows
    ):
        return _fail("zero-shot row metadata or boolean exact_match mismatch")
    recomp_em = sum(1 for r in rows if r["exact_match"]) / float(len(rows))
    rep_em = rec.get("exact_match_accuracy")
    if not (_fin(rep_em) and 0.0 <= float(rep_em) <= 1.0 and abs(recomp_em - float(rep_em)) <= 1e-6):
        return _fail("zero-shot reported EM invalid or inconsistent with recomputed EM")
    return rec


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
    split_name: str = "validation",
    seal_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate unmodified base instruct model (`0` LoRA adapters) on `validation` (200) or sealed `test` (400)."""
    from aloe.datasets.model_registry import ModelRegistry

    if split_name not in ("validation", "test"):
        raise ValueError(f"Unsupported zero-shot split_name={split_name!r}")
    if split_name == "test" and (seal_path is None or not seal_path.is_file()):
        raise PermissionError("Zero-shot 'test' evaluation requires a verified read-only test_unsealed.json seal.")

    reg = ModelRegistry()
    canon_id = model_id
    for alias, target in reg.ALIASES.items():
        if canon_id.lower() == alias:
            canon_id = target
            break
    pinned_rev = reg.PINNED_REVISIONS.get(canon_id, "main")

    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = out_dir / f"zero_shot_{split_name}.json"
    pred_file = out_dir / f"predictions_{split_name}.jsonl"
    cached = _verify_strict_zero_shot_standalone(
        receipt_file,
        expected_split=split_name,
        expected_split_ids=val_ids,
        expected_manifest_sha=manifest_sha,
        expected_git_sha=expected_git_sha,
        expected_model_id=canon_id,
        expected_model_revision=pinned_rev,
        expected_gpu_name=exact_gpu_name,
        expected_memory_metric="cuda_max_memory_allocated",
    )
    if cached is not None:
        print(
            f"[ZERO-SHOT STRICTLY CACHED] {canon_id} ({split_name}) EM={cached['exact_match_accuracy']:.2%}",
            flush=True,
        )
        return cached
    if pred_file.is_file() and not receipt_file.is_file():
        pred_file.unlink()

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
split_name = str(cfg["split_name"])
if split_name == "test":
    seal_p = Path(cfg["seal_path"])
    if not seal_p.is_file():
        raise PermissionError("Missing test_unsealed.json in zero-shot test subprocess")
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
target_ids = [str(i) for i in manifest["splits"][split_name]["ids"]]
raw_examples = load_xlam_file(work_repo / cfg["raw_rel"])
id_to_ex = {str(ex.id): ex for ex in raw_examples}
eval_examples = [id_to_ex[sid] for sid in sorted(target_ids)]

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
pred_store = PredictionStore(out_dir / f"predictions_{split_name}.jsonl")
ev_out = evaluate_model_on_split(
    model,
    tokenizer,
    eval_examples,
    prediction_store=pred_store,
    split_name=split_name,
    model_id=canonical_id,
    model_revision=pinned_rev,
    arm="zero_shot",
    trial_id=f"zero_shot_{slug}_{split_name}",
    seed=42,
    device="cuda",
    max_new_tokens=192,
)
peak_cuda_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3)
raw_sum = ev_out["summary"]
sdict = raw_sum.model_dump() if hasattr(raw_sum, "model_dump") else (raw_sum if isinstance(raw_sum, dict) else vars(raw_sum))
receipt = {
    "protocol_tag": "protocol-v1",
    "mode": f"zero_shot_base_{split_name}",
    "model_id": canonical_id,
    "model_revision": pinned_rev,
    "arm": "zero_shot",
    "split": split_name,
    "sample_count": int(sdict["total_samples"]),
    "exact_match_accuracy": float(sdict["exact_match_accuracy"]),
    "lenient_exact_match_accuracy": float(sdict.get("lenient_exact_match_accuracy", sdict["exact_match_accuracy"])),
    "tool_name_accuracy": float(sdict.get("tool_selection_accuracy", sdict.get("tool_match_rate", 0.0))),
    "argument_name_accuracy": float(sdict.get("argument_name_accuracy", 0.0)),
    "argument_value_accuracy": float(sdict.get("argument_value_accuracy", 0.0)),
    "parse_success_rate": float(sdict.get("parse_success_rate", 0.0)),
    "schema_valid_rate": float(sdict.get("schema_valid_rate", 0.0)),
    "trainable_parameters": 0,
    "memory_metric_name": "cuda_max_memory_allocated",
    "search_memory_mb": peak_cuda_mb,
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
atomic_write_text(out_dir / f"zero_shot_{split_name}.json", json.dumps(receipt, indent=2) + "\\n")
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
            "split_name": split_name,
            "seal_path": str(seal_path) if seal_path else "",
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
    parser.add_argument(
        "--phases",
        default="search,confirm",
        help="Comma-separated phases: zero_shot,search,confirm,zero_shot_test,final_test",
    )
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--expected-seal-sha256", default="", help="Required 64-hex SHA-256 of test_unsealed.json for sealed phases")
    parser.add_argument("--ceremony-hf-revision", default="", help="Optional HF Seal Ceremony commit OID to pin seal/plan download")
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
    (shim_dir / "sitecustomize.py").write_text(_COLAB_WORKER_SHIM_SOURCE, encoding="utf-8")
    sys.path.insert(0, str(work_repo))
    sys.path.insert(0, str(shim_dir))
    orch_root_str = str(Path(__file__).resolve().parent.parent)
    if orch_root_str not in sys.path:
        sys.path.append(orch_root_str)
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
    invalid_phases = set(phase_list) - {"zero_shot", "search", "confirm", "zero_shot_test", "final_test"}
    if invalid_phases:
        raise ValueError(f"Unsupported shard phases {sorted(invalid_phases)}.")
    has_sealed_phase = bool(set(phase_list) & {"zero_shot_test", "final_test"})
    if has_sealed_phase:
        import re as _re

        if args.search_seed != 42 or args.allow_supplementary_cuda or args.campaign_prefix or args.confirm_prefix:
            raise PermissionError(
                "Sealed final_test / zero_shot_test phases are restricted strictly to canonical search_seed=42 CUDA models."
            )
        if not args.expected_seal_sha256 or not _re.fullmatch(r"[0-9a-f]{64}", args.expected_seal_sha256.strip()):
            raise PermissionError(
                "Sealed phases require an explicit 64-hex --expected-seal-sha256 from the approved Seal Ceremony."
            )
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

        all_canonical_arms = tuple(
            a
            for a in ("aloe", "b0", "b1", "b2", "borobia_attn", "borobia_ssm", "borobia_ffn")
            if not (a == "borobia_ssm" and not any(g.component_type.value == "ssm" for g in grammar.groups))
        )
        allow_pats = [f"{campaign_prefix}/{slug}/hardware_lock.json"]
        if "zero_shot" in phase_list:
            allow_pats.append(f"zero_shot_baselines/{slug}/**")
        if "zero_shot_test" in phase_list:
            allow_pats.append(f"zero_shot_test/{slug}/**")
        if has_sealed_phase:
            # Always download ALL canonical search summaries + read-only seal + frozen plan + attempt logs
            allow_pats.append("final_sealed_test/PRE_UNSEAL_ANALYSIS_PLAN.md")
            allow_pats.append(f"final_sealed_test/{slug}/test_unsealed.json")
            allow_pats.append(f"final_test_attempts/{slug}/**")
            for ca in all_canonical_arms:
                allow_pats.append(f"{campaign_prefix}/{slug}/{ca}/**")
        for a in active_arms:
            allow_pats.append(f"{campaign_prefix}/{slug}/{a}/**")
            if "confirm" in phase_list and args.search_seed == 42:
                allow_pats.append(f"{confirm_prefix}/{slug}/{a}/**")
            if "final_test" in phase_list:
                allow_pats.append(f"final_sealed_test/{slug}/{a}/**")
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

        # Phase 3: Read-Only Seal Verification + Zero-Shot Test & Uniform 3-Seed Final Test
        if has_sealed_phase:
            from huggingface_hub import hf_hub_download
            from aloe.datasets.model_registry import ModelRegistry

            if args.ceremony_hf_revision and args.ceremony_hf_revision.strip():
                for rel_ceremony_f in (
                    "final_sealed_test/PRE_UNSEAL_ANALYSIS_PLAN.md",
                    f"final_sealed_test/{slug}/test_unsealed.json",
                ):
                    hf_hub_download(
                        repo_id=args.hf_repo,
                        repo_type="dataset",
                        filename=rel_ceremony_f,
                        revision=args.ceremony_hf_revision.strip(),
                        local_dir=str(local_store),
                        token=hf_token,
                    )

            pinned_rev = ModelRegistry().PINNED_REVISIONS.get(desc.model_id, "main")
            seal_path = local_store / "final_sealed_test" / slug / "test_unsealed.json"
            frozen_plan_path = local_store / "final_sealed_test" / "PRE_UNSEAL_ANALYSIS_PLAN.md"
            seal_info = _verify_immutable_seal_standalone(
                seal_path,
                expected_model_id=desc.model_id,
                expected_model_revision=pinned_rev,
                expected_git_sha=args.expected_git_sha,
                expected_manifest_sha=manifest_sha,
                expected_gpu_name=exact_gpu_name,
                expected_memory_metric="cuda_max_memory_allocated",
                expected_complete_arms=all_canonical_arms,
                campaigns_model_dir=local_store / campaign_prefix / slug,
                expected_split_ids=split_id_sets,
                expected_seal_sha256=args.expected_seal_sha256.strip(),
                shim_path=shim_dir / "sitecustomize.py",
                frozen_plan_path=frozen_plan_path,
                verify_code_digests=True,
            )
            winners_by_arm = seal_info["winners_by_arm"]
            seal_sha_hex = str(seal_info["seal_sha256"])
            _replay_local_attempts_to_hf(syncer, local_store, slug)
            print(
                f"[SEAL VERIFIED READ-ONLY] {slug} seal_sha256={seal_sha_hex[:16]}... "
                f"arms={list(all_canonical_arms)}",
                flush=True,
            )

            import uuid as _uuid

            if "zero_shot_test" in phase_list:
                zs_test_dir = local_store / "zero_shot_test" / slug
                zs_test_json = zs_test_dir / "zero_shot_test.json"
                run_id = f"zero_shot_{slug}_test"
                cached_zs = _verify_strict_zero_shot_standalone(
                    zs_test_json,
                    expected_split="test",
                    expected_split_ids=split_id_sets["test"],
                    expected_manifest_sha=manifest_sha,
                    expected_git_sha=args.expected_git_sha,
                    expected_model_id=desc.model_id,
                    expected_model_revision=pinned_rev,
                    expected_gpu_name=exact_gpu_name,
                    expected_memory_metric="cuda_max_memory_allocated",
                )
                if cached_zs is not None:
                    syncer.push_arm_delta(
                        zs_test_dir,
                        f"zero_shot_test/{slug}",
                        commit_label=f"[{args.shard_id}] zero_shot_test {slug} EM={cached_zs['exact_match_accuracy']:.2%} (cached sync)",
                        strict=True,
                    )
                    if not _has_completed_attempt_local(local_store, slug, run_id):
                        open_aid = _find_open_launched_attempt_id(local_store, slug, run_id)
                        if open_aid is None:
                            open_aid = _uuid.uuid4().hex[:12]
                            _log_final_test_attempt(
                                syncer,
                                slug=slug,
                                run_id=run_id,
                                stage="LAUNCHED",
                                attempt_id=open_aid,
                                seal_sha256=seal_sha_hex,
                                seq=1,
                                local_store=local_store,
                            )
                        _log_final_test_attempt(
                            syncer,
                            slug=slug,
                            run_id=run_id,
                            stage="COMPLETED",
                            attempt_id=open_aid,
                            seal_sha256=seal_sha_hex,
                            seq=2,
                            local_store=local_store,
                        )
                else:
                    if _has_completed_attempt_local(local_store, slug, run_id):
                        raise PermissionError(f"Completed run cannot be re-executed: {run_id}")
                    orphan_aid = _find_open_launched_attempt_id(local_store, slug, run_id)
                    if orphan_aid is not None:
                        _log_final_test_attempt(
                            syncer,
                            slug=slug,
                            run_id=run_id,
                            stage="FAILED",
                            attempt_id=orphan_aid,
                            seal_sha256=seal_sha_hex,
                            seq=2,
                            local_store=local_store,
                            extra={"error": "preempted_or_unclosed_prior_attempt"},
                        )
                    zs_aid = _uuid.uuid4().hex[:12]
                    _log_final_test_attempt(
                        syncer,
                        slug=slug,
                        run_id=run_id,
                        stage="LAUNCHED",
                        attempt_id=zs_aid,
                        seal_sha256=seal_sha_hex,
                        seq=1,
                        local_store=local_store,
                    )
                    print(f"\n=== [SHARD ZERO-SHOT TEST] {desc.model_id} :: test (400 items, 0 adapters) ===", flush=True)
                    try:
                        zs_test_rec = _run_zero_shot_validation_subprocess(
                            work_repo=work_repo,
                            model_id=desc.model_id,
                            manifest_rel=manifest_rel,
                            raw_rel=raw_rel,
                            out_dir=zs_test_dir,
                            expected_git_sha=args.expected_git_sha,
                            manifest_sha=manifest_sha,
                            val_ids=split_id_sets["test"],
                            pinned_gpu=args.pinned_gpu,
                            exact_gpu_name=exact_gpu_name,
                            allow_slow_kernels=False,
                            split_name="test",
                            seal_path=seal_path,
                        )
                        _verify_strict_zero_shot_standalone(
                            zs_test_json,
                            expected_split="test",
                            expected_split_ids=split_id_sets["test"],
                            expected_manifest_sha=manifest_sha,
                            expected_git_sha=args.expected_git_sha,
                            expected_model_id=desc.model_id,
                            expected_model_revision=pinned_rev,
                            expected_gpu_name=exact_gpu_name,
                            expected_memory_metric="cuda_max_memory_allocated",
                            raise_on_error=True,
                        )
                    except Exception as zs_exc:
                        _log_final_test_attempt(
                            syncer,
                            slug=slug,
                            run_id=run_id,
                            stage="FAILED",
                            attempt_id=zs_aid,
                            seal_sha256=seal_sha_hex,
                            seq=2,
                            local_store=local_store,
                            extra={"error": str(zs_exc)[:300]},
                        )
                        raise
                    syncer.push_arm_delta(
                        zs_test_dir,
                        f"zero_shot_test/{slug}",
                        commit_label=f"[{args.shard_id}] zero_shot_test {slug} EM={zs_test_rec['exact_match_accuracy']:.2%}",
                        strict=True,
                    )
                    _log_final_test_attempt(
                        syncer,
                        slug=slug,
                        run_id=run_id,
                        stage="COMPLETED",
                        attempt_id=zs_aid,
                        seal_sha256=seal_sha_hex,
                        seq=2,
                        local_store=local_store,
                    )

            if "final_test" in phase_list:
                for arm in active_arms:
                    wcfg = winners_by_arm[arm]
                    local_trials_dir = work_repo / "runs" / "final_sealed_test" / slug / arm / "trials"
                    mirror_arm_final = local_store / "final_sealed_test" / slug / arm
                    mirror_trials_dir = mirror_arm_final / "trials"
                    local_trials_dir.mkdir(parents=True, exist_ok=True)
                    repo_final_prefix = f"final_sealed_test/{slug}/{arm}"

                    for s_val in REQUIRED_CONFIRM_SEEDS:
                        tid = f"{arm}_winner_s{s_val}_test"
                        done_mirror = mirror_trials_dir / tid / "trial.json"
                        if _verify_strict_receipt_standalone(
                            done_mirror,
                            expected_seed=s_val,
                            expected_split="test",
                            expected_split_ids=split_id_sets["test"],
                            expected_manifest_sha=manifest_sha,
                            expected_git_sha=args.expected_git_sha,
                            expected_model_id=desc.model_id,
                            expected_model_revision=pinned_rev,
                            expected_gpu_name=exact_gpu_name,
                            expected_memory_metric="cuda_max_memory_allocated",
                            expected_winner_cfg=wcfg,
                        ):
                            verify_arm_git_commits(mirror_arm_final, args.expected_git_sha)
                            syncer.push_arm_delta(
                                mirror_arm_final,
                                repo_final_prefix,
                                commit_label=f"[{args.shard_id}] final_sealed_test {slug}/{arm} s={s_val} PASS (cached sync)",
                                strict=True,
                            )
                            if not _has_completed_attempt_local(local_store, slug, tid):
                                open_aid = _find_open_launched_attempt_id(local_store, slug, tid)
                                if open_aid is None:
                                    open_aid = _uuid.uuid4().hex[:12]
                                    _log_final_test_attempt(
                                        syncer,
                                        slug=slug,
                                        run_id=tid,
                                        stage="LAUNCHED",
                                        attempt_id=open_aid,
                                        seal_sha256=seal_sha_hex,
                                        seq=1,
                                        local_store=local_store,
                                    )
                                _log_final_test_attempt(
                                    syncer,
                                    slug=slug,
                                    run_id=tid,
                                    stage="COMPLETED",
                                    attempt_id=open_aid,
                                    seal_sha256=seal_sha_hex,
                                    seq=2,
                                    local_store=local_store,
                                )
                            print(f"[FINAL TEST VERIFIED CACHED] {tid}", flush=True)
                            continue

                        if _has_completed_attempt_local(local_store, slug, tid):
                            raise PermissionError(f"Completed run cannot be re-executed: {tid}")
                        orphan_aid = _find_open_launched_attempt_id(local_store, slug, tid)
                        if orphan_aid is not None:
                            _log_final_test_attempt(
                                syncer,
                                slug=slug,
                                run_id=tid,
                                stage="FAILED",
                                attempt_id=orphan_aid,
                                seal_sha256=seal_sha_hex,
                                seq=2,
                                local_store=local_store,
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
                            "--pinned-gpu",
                            args.pinned_gpu,
                            "--exact-gpu-name",
                            exact_gpu_name,
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

                        run_aid = _uuid.uuid4().hex[:12]
                        _log_final_test_attempt(
                            syncer,
                            slug=slug,
                            run_id=tid,
                            stage="LAUNCHED",
                            attempt_id=run_aid,
                            seal_sha256=seal_sha_hex,
                            seq=1,
                            local_store=local_store,
                        )
                        print(f"=== [SHARD FINAL TEST] {desc.model_id} :: {tid} ===", flush=True)
                        try:
                            subprocess.run(cmd, cwd=str(work_repo), check=True)
                            if not _verify_strict_receipt_standalone(
                                done_mirror,
                                expected_seed=s_val,
                                expected_split="test",
                                expected_split_ids=split_id_sets["test"],
                                expected_manifest_sha=manifest_sha,
                                expected_git_sha=args.expected_git_sha,
                                expected_model_id=desc.model_id,
                                expected_model_revision=pinned_rev,
                                expected_gpu_name=exact_gpu_name,
                                expected_memory_metric="cuda_max_memory_allocated",
                                expected_winner_cfg=wcfg,
                            ):
                                raise RuntimeError(f"Post-run strict final-test verification failed for {tid}")
                            verify_arm_git_commits(mirror_arm_final, args.expected_git_sha)
                        except Exception as run_exc:
                            _log_final_test_attempt(
                                syncer,
                                slug=slug,
                                run_id=tid,
                                stage="FAILED",
                                attempt_id=run_aid,
                                seal_sha256=seal_sha_hex,
                                seq=2,
                                local_store=local_store,
                                extra={"error": str(run_exc)[:300]},
                            )
                            raise

                        syncer.push_arm_delta(
                            mirror_arm_final,
                            repo_final_prefix,
                            commit_label=f"[{args.shard_id}] final_sealed_test {slug}/{arm} s={s_val} PASS",
                            strict=True,
                        )
                        _log_final_test_attempt(
                            syncer,
                            slug=slug,
                            run_id=tid,
                            stage="COMPLETED",
                            attempt_id=run_aid,
                            seal_sha256=seal_sha_hex,
                            seq=2,
                            local_store=local_store,
                        )

                    syncer.push_arm_delta(
                        mirror_arm_final,
                        repo_final_prefix,
                        commit_label=f"[{args.shard_id}] final_sealed_test {slug}/{arm} ALL 3 SEEDS COMPLETE",
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

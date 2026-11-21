"""Trial Store, Atomic Persistence, and Campaign Resume Ledger (Stage 9 / Protocol v1).

Records, indexes, and replays experimental trials under Result Schema Envelope 1.1:
- Atomic `NamedTemporaryFile + flush + os.fsync + os.replace` writes (zero half-written JSON).
- Real runtime hardware detection (never hardcoding Apple MPS on Linux/CUDA).
- Transactional CampaignLedger (`PENDING -> RUNNING -> DONE / FAILED`) supporting
  mid-campaign crash recovery and optional Google Drive persistent mirror sync.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aloe.evaluation.resources import detect_hardware_summary
from aloe.execution.trial_config import TrialConfig


def _fsync_dir(dir_path: Path) -> None:
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_text(target: Path, content: str) -> None:
    """Atomically write UTF-8 text via temp file + fsync + os.replace + directory fsync."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        _fsync_dir(target.parent)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy `src` to `dst` using same-directory temp file + fsync + os.replace."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with open(src, "rb") as src_h, tempfile.NamedTemporaryFile(
            mode="wb", dir=dst.parent, delete=False
        ) as dst_h:
            tmp_path = Path(dst_h.name)
            shutil.copyfileobj(src_h, dst_h, length=1024 * 1024)
            dst_h.flush()
            os.fsync(dst_h.fileno())
        os.replace(tmp_path, dst)
        _fsync_dir(dst.parent)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def get_git_commit(repo_dir: Path | str | None = None, *, forbid_dirty: bool = False) -> str:
    """Retrieve current git commit SHA, appending '-dirty' if tracked files are modified."""
    target_dir = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[2]
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        sha = res.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty.returncode != 0:
            sha = "UNKNOWN"
        elif dirty.stdout.strip():
            sha = f"{sha}-dirty"
    except Exception:
        sha = "UNKNOWN"

    if forbid_dirty and (sha == "UNKNOWN" or sha.endswith("-dirty")):
        raise RuntimeError(
            f"Official campaign requires a clean committed Git tree, got '{sha}' in {target_dir}"
        )
    return sha


def get_file_sha256(path: Path | str) -> str:
    """Compute sha256 hex digest of a file."""
    p = Path(path)
    if not p.is_file():
        return "UNKNOWN"
    h = hashlib.sha256()
    with open(p, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_environment_metadata() -> dict[str, Any]:
    """Capture runtime library versions, OS, and actual detected hardware metadata."""
    import peft
    import torch
    import transformers

    hw = detect_hardware_summary()
    libs: dict[str, str] = {
        "torch": torch.__version__,
        "peft": peft.__version__,
        "transformers": transformers.__version__,
    }
    try:
        import botorch  # type: ignore
        libs["botorch"] = botorch.__version__
    except Exception:
        pass
    return {
        "hardware": hw["hardware_label"],
        "device": hw["device"],
        "gpu_name": hw["gpu_name"],
        "cuda_version": hw["cuda_version"],
        "nvidia_driver": hw["nvidia_driver"],
        "fast_mamba_kernels": hw["fast_mamba_kernels"],
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "library_versions": libs,
    }


class TrialMetrics(BaseModel):
    """Measured empirical costs and performance metrics for a trial run."""

    model_config = ConfigDict(extra="ignore")

    peak_memory_gb: float
    peak_rss_gb: float
    mps_driver_allocated_gb: float = 0.0
    cuda_max_memory_allocated_gb: float = 0.0
    cuda_max_memory_reserved_gb: float = 0.0
    search_memory_mb: float | None = None
    memory_metric_name: str = "process_rss_sampled_100ms"
    train_seconds: float
    eval_seconds: float = 0.0
    latency_per_step_sec: float
    optimizer_steps_completed: int = 0
    examples_seen: int = 0
    trainable_params: int
    adalora_init_params: int | None = None
    adalora_final_active_params: int | None = None
    total_params: int
    trainable_ratio: float
    adapter_size_mb: float
    initial_train_loss: float = 0.0
    final_train_loss: float = 0.0
    val_loss: float | None = None
    eval_samples: int | None = None
    evaluator_version: str | None = None
    predictions_jsonl_path: str | None = None
    predictions_sha256: str | None = None
    tool_accuracy: float | None = None
    ast_parse_success_rate: float | None = None
    ast_schema_valid_rate: float | None = None
    ast_tool_selection_accuracy: float | None = None
    ast_argument_name_accuracy: float | None = None
    ast_argument_value_accuracy: float | None = None
    ast_exact_match_accuracy: float | None = None
    ast_lenient_exact_match_accuracy: float | None = None
    param_delta_vs_baseline_pct: float | None = None
    is_budget_matched_5pct: bool | None = None
    resumed_from_checkpoint: bool = False
    fresh_process_reload: bool = False


class TrialResultEnvelope(BaseModel):
    """Standardized Result Schema Envelope for an experimental trial."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1.0"
    protocol_tag: str = "protocol-v1"
    trial_id: str
    arm: str = "aloe"
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    git_commit: str
    dataset_manifest_sha256: str
    model_revision: str = "UNKNOWN"
    hardware: str
    gpu_name: str | None = None
    library_versions: dict[str, str]
    model_id: str
    backend: str
    trial_config: dict[str, Any]
    metrics: TrialMetrics | None = None
    status: Literal["PASS", "FAIL", "INVALID_CONFIG"] = "PASS"
    failures: list[str] = Field(default_factory=list)
    not_determined: list[str] = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> TrialResultEnvelope:
        return cls.model_validate_json(json_str)


class TrialStore:
    """Local repository store for managing, reading, and replaying trial artifacts."""

    def __init__(
        self,
        runs_dir: Path | str = "runs/trials",
        mirror_dir: Path | str | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None
        if self.mirror_dir is not None:
            self.mirror_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: TrialResultEnvelope) -> Path:
        """Atomically save trial envelope to runs/trials/<trial_id>/trial.json (+ optional Drive mirror)."""
        trial_dir = self.runs_dir / record.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        record_file = trial_dir / "trial.json"
        payload = record.to_json()
        atomic_write_text(record_file, payload)

        if self.mirror_dir is not None:
            mirror_trial_dir = self.mirror_dir / record.trial_id
            mirror_trial_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(mirror_trial_dir / "trial.json", payload)
        return record_file

    def sync_file_to_mirror(self, trial_id: str, local_file: Path) -> None:
        """Atomically mirror a single artifact file (e.g., 25-step checkpoint or JSONL) to Drive."""
        if self.mirror_dir is None or not local_file.is_file():
            return
        src_dir = self.runs_dir / trial_id
        try:
            rel = local_file.relative_to(src_dir)
        except ValueError:
            rel = Path(local_file.name)
        target = self.mirror_dir / trial_id / rel
        atomic_copy_file(local_file, target)

    def remove_file_from_mirror(self, trial_id: str, filename: str) -> None:
        """Remove an obsolete temporary checkpoint file from Drive mirror once adapter is finalized."""
        if self.mirror_dir is None:
            return
        (self.mirror_dir / trial_id / filename).unlink(missing_ok=True)

    def restore_trial_from_mirror(self, trial_id: str) -> list[str]:
        """Restore mid-trial checkpoints and partial JSONL predictions from Drive mirror into ephemeral local runs_dir."""
        if self.mirror_dir is None:
            return []
        mirror_trial_dir = self.mirror_dir / trial_id
        if not mirror_trial_dir.is_dir():
            return []
        local_trial_dir = self.runs_dir / trial_id
        local_trial_dir.mkdir(parents=True, exist_ok=True)
        restored: list[str] = []
        for item in mirror_trial_dir.rglob("*"):
            if item.is_file() and not item.name.endswith(".tmp"):
                rel = item.relative_to(mirror_trial_dir)
                dst = local_trial_dir / rel
                if not dst.is_file() or item.stat().st_mtime > dst.stat().st_mtime:
                    atomic_copy_file(item, dst)
                    restored.append(str(rel))
        return restored

    def sync_trial_artifacts_to_mirror(self, trial_id: str) -> None:
        """Atomically copy predictions JSONL and adapter files for `trial_id` into Google Drive mirror."""
        if self.mirror_dir is None:
            return
        src_dir = self.runs_dir / trial_id
        if not src_dir.is_dir():
            return
        dst_dir = self.mirror_dir / trial_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for item in src_dir.rglob("*"):
            if item.is_file() and not item.name.endswith(".tmp"):
                rel = item.relative_to(src_dir)
                target = dst_dir / rel
                atomic_copy_file(item, target)

    def get(self, trial_id: str) -> TrialResultEnvelope:
        """Load a trial result envelope by trial ID."""
        record_file = self.runs_dir / trial_id / "trial.json"
        if not record_file.is_file():
            raise FileNotFoundError(f"Trial record not found at: {record_file}")
        return TrialResultEnvelope.from_json(record_file.read_text(encoding="utf-8"))

    def list_trials(self, *, strict: bool = False) -> list[TrialResultEnvelope]:
        """List recorded trial results. Raises on corrupt records when `strict=True`."""
        records: list[TrialResultEnvelope] = []
        for path in sorted(self.runs_dir.glob("*/trial.json")):
            try:
                records.append(TrialResultEnvelope.from_json(path.read_text(encoding="utf-8")))
            except Exception as err:
                if strict:
                    raise ValueError(f"Corrupt trial record at {path}: {err}") from err
                continue
        return records

    def replay(
        self,
        trial_id: str,
        backend_runner: Any = None,
        loss_tolerance: float = 0.05,
    ) -> TrialResultEnvelope:
        """Replay a recorded trial and assert metrics match within specified tolerance."""
        stored = self.get(trial_id)
        config = TrialConfig.model_validate(stored.trial_config)

        if backend_runner is None:
            from aloe.execution.backends.peft_mps import PeftMPSBackend
            backend_runner = PeftMPSBackend()

        replay_id = f"replay_{trial_id}"
        replay_result = backend_runner.run_trial(config, trial_id=replay_id)

        if stored.metrics and replay_result.metrics:
            assert stored.metrics.trainable_params == replay_result.metrics.trainable_params, (
                f"Trainable param mismatch on replay: {stored.metrics.trainable_params} vs "
                f"{replay_result.metrics.trainable_params}"
            )
            loss_diff = abs(stored.metrics.final_train_loss - replay_result.metrics.final_train_loss)
            assert loss_diff <= loss_tolerance, (
                f"Final train loss difference ({loss_diff:.4f}) exceeded tolerance ({loss_tolerance}) "
                f"between original ({stored.metrics.final_train_loss:.4f}) and replay "
                f"({replay_result.metrics.final_train_loss:.4f})"
            )

        return replay_result


class LedgerTrialEntry(BaseModel):
    """Transactional state entry for one trial inside a CampaignLedger."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    trial_index: int
    config_signature: str
    trial_config: dict[str, Any]
    status: Literal["PENDING", "RUNNING", "DONE", "FAILED"] = "PENDING"
    attempts: int = 0
    metrics: dict[str, float] | None = None
    memory_metric_name: str | None = None
    gpu_name: str | None = None
    predictions_jsonl: str | None = None
    last_error: str | None = None
    updated_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class CampaignLedgerState(BaseModel):
    """Persistent state of a resumable optimization or baseline campaign."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    protocol_tag: str = "protocol-v1"
    model_id: str
    model_revision: str
    arm: str
    search_seed: int
    training_seed: int
    target_trials: int
    baseline_params: int
    parity_tolerance: float
    pinned_gpu: str | None = None
    locked_gpu_name: str | None = None
    locked_memory_metric_name: str | None = None
    training_signature: str = ""
    manifest_path: str
    manifest_sha256: str
    optimizer_state: dict[str, Any] = Field(default_factory=dict)
    entries: list[LedgerTrialEntry] = Field(default_factory=list)


class CampaignLedger:
    """Atomic JSON campaign state ledger supporting deterministic crash recovery."""

    def __init__(self, ledger_path: Path | str, mirror_path: Path | str | None = None) -> None:
        self.path = Path(ledger_path)
        self.mirror_path = Path(mirror_path) if mirror_path else None

    def exists(self) -> bool:
        return self.path.is_file() or (self.mirror_path is not None and self.mirror_path.is_file())

    def load(self) -> CampaignLedgerState:
        if not self.path.is_file() and self.mirror_path is not None and self.mirror_path.is_file():
            atomic_copy_file(self.mirror_path, self.path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Campaign ledger not found: {self.path}")
        return CampaignLedgerState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, state: CampaignLedgerState) -> None:
        payload = state.model_dump_json(indent=2)
        atomic_write_text(self.path, payload)
        if self.mirror_path is not None:
            atomic_write_text(self.mirror_path, payload)

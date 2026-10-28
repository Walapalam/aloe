"""Trial Store and Result Envelopes (Stage 9 / Gate C).

Records, indexes, and replays experimental trials under Result Schema Envelope 1.0.
A trial you cannot replay is not a result.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aloe.execution.trial_config import TrialConfig


def get_git_commit(repo_dir: Path | str = ".") -> str:
    """Retrieve current git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def get_file_sha256(path: Path | str) -> str:
    """Compute sha256 hex digest of a file."""
    p = Path(path)
    if not p.is_file():
        return "UNKNOWN"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def get_environment_metadata() -> dict[str, Any]:
    """Capture library versions, OS, and Apple Silicon hardware metadata."""
    import peft
    import torch
    import transformers

    return {
        "hardware": "Apple M1 Max (64 GB unified memory), macOS, MPS",
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "library_versions": {
            "torch": torch.__version__,
            "peft": peft.__version__,
            "transformers": transformers.__version__,
        },
    }


class TrialMetrics(BaseModel):
    """Measured empirical costs and performance metrics for a trial run."""

    model_config = ConfigDict(extra="ignore")

    peak_memory_gb: float               # Primary Apple Silicon Pareto objective: peak RSS (ADR-004)
    peak_rss_gb: float                  # Peak process RSS sampled at 100ms
    mps_driver_allocated_gb: float      # Diagnostic: PyTorch MPS driver buffer allocation
    train_seconds: float
    latency_per_step_sec: float
    trainable_params: int
    total_params: int
    trainable_ratio: float
    adapter_size_mb: float
    initial_train_loss: float = 0.0
    final_train_loss: float = 0.0
    val_loss: float | None = None
    tool_accuracy: float | None = None
    ast_parse_success_rate: float | None = None
    ast_tool_selection_accuracy: float | None = None
    ast_argument_name_accuracy: float | None = None
    ast_argument_value_accuracy: float | None = None
    ast_exact_match_accuracy: float | None = None
    param_delta_vs_baseline_pct: float | None = None
    is_budget_matched_5pct: bool | None = None
    fresh_process_reload: bool = False


class TrialResultEnvelope(BaseModel):
    """Standardized Result Schema Envelope (schema_version 1.0) for an experimental trial."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1.0"
    trial_id: str
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    git_commit: str
    dataset_manifest_sha256: str
    hardware: str
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

    def __init__(self, runs_dir: Path | str = "runs/trials") -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: TrialResultEnvelope) -> Path:
        """Save trial envelope to runs/trials/<trial_id>/trial.json."""
        trial_dir = self.runs_dir / record.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        record_file = trial_dir / "trial.json"
        record_file.write_text(record.to_json(), encoding="utf-8")
        return record_file

    def get(self, trial_id: str) -> TrialResultEnvelope:
        """Load a trial result envelope by trial ID."""
        record_file = self.runs_dir / trial_id / "trial.json"
        if not record_file.is_file():
            raise FileNotFoundError(f"Trial record not found at: {record_file}")
        return TrialResultEnvelope.from_json(record_file.read_text(encoding="utf-8"))

    def list_trials(self) -> list[TrialResultEnvelope]:
        """List all recorded trial results in the store."""
        records = []
        for path in sorted(self.runs_dir.glob("*/trial.json")):
            try:
                records.append(TrialResultEnvelope.from_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return records

    def replay(
        self,
        trial_id: str,
        backend_runner: Any = None,
        loss_tolerance: float = 0.05,
    ) -> TrialResultEnvelope:
        """Replay a recorded trial and assert metrics match within specified tolerance.

        A trial that cannot replay to the same result is invalid.
        """
        stored = self.get(trial_id)
        config = TrialConfig.model_validate(stored.trial_config)

        if backend_runner is None:
            from aloe.execution.backends.peft_mps import PeftMPSBackend
            backend_runner = PeftMPSBackend()

        # Execute replayed trial in an isolated directory
        replay_id = f"replay_{trial_id}"
        replay_result = backend_runner.run_trial(config, trial_id=replay_id)

        # Assert deterministic reproducibility
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

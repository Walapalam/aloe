"""Machine-readable EvalCard generator for ALOE.

Adheres to October Gate E specification and reproducible AI reporting principles
informed by Raff (NeurIPS 2019) and EvalCards standards.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _get_git_commit() -> str:
    """Safely obtain current repository git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_installed_libraries() -> dict[str, str]:
    """Retrieve versions of key machine learning libraries."""
    versions = {}
    for pkg in ["torch", "transformers", "peft", "botorch", "gpytorch", "datasets", "numpy"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "installed"))
        except ImportError:
            pass
    return versions


class EvalCardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    base_commit: str
    parameters_total: int
    trainable_parameters: int
    parameter_fraction: float


class EvalCardSearchPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_settings: dict[str, Any]
    grammar_sha256: str | None = None


class EvalCardMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_accuracy_mean: float
    tool_accuracy_std: float = Field(description="Sample standard deviation across seeds with ddof=1")
    ci_95_bootstrap: list[float] = Field(description="95% Confidence Interval [ci_low, ci_high]")
    p_value_vs_uniform: float
    p_value_bonferroni: float | None = None
    cohens_d: float | None = None
    peak_memory_rss_gb: float
    memory_metric: str = Field(
        default="process_rss_unified",
        description="Physical metric used: 'process_rss_unified' for Apple Silicon MPS or 'cuda_max_memory_allocated' for CUDA",
    )


class EvalCardOptimizer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impl: str = Field(description="Optimizer implementation, e.g. 'mobo_botorch_matern52', 'random_search'")
    kernel: str | None = None
    acquisition: str | None = None
    ref_point: list[float] | None = None
    n_init: int | None = None
    bo_seed: int | None = None
    botorch_version: str | None = None


class EvalCardEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hardware_tier: str = Field(description="Hardware platform: 'apple_silicon_mps', 'gcp_l4_cuda', 'colab_a100'")
    os: str
    python_version: str
    backend: str
    library_versions: dict[str, str] = Field(default_factory=dict)
    seeds_evaluated: list[int] = Field(default_factory=list)

    @field_validator("seeds_evaluated")
    @classmethod
    def validate_seeds(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("seeds_evaluated must contain at least one seed.")
        return v


class EvalCard(BaseModel):
    """Machine-readable reproducibility card for ALOE evaluation checkpoints."""

    model_config = ConfigDict(extra="forbid")

    eval_card_version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: EvalCardModel
    search_point: EvalCardSearchPoint
    metrics: EvalCardMetrics
    optimizer: EvalCardOptimizer
    environment: EvalCardEnvironment
    baseline_trial_id: str | None = None
    dataset_manifest_sha256: str | None = None
    eval_card_sha256: str = ""

    def compute_sha256(self) -> str:
        """Compute deterministic SHA-256 fingerprint over content (excluding eval_card_sha256)."""
        data = self.model_dump()
        data["eval_card_sha256"] = ""
        canonical_json = json.dumps(data, sort_keys=True, indent=None)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def finalize(self) -> EvalCard:
        """Compute and seal the card with its SHA-256 fingerprint."""
        self.eval_card_sha256 = self.compute_sha256()
        return self

    def save(self, path: Path | str) -> Path:
        """Save EvalCard to JSON file."""
        if not self.eval_card_sha256:
            self.finalize()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> EvalCard:
        """Load and validate an EvalCard from JSON file."""
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        card = cls.model_validate_json(content)
        # Verify integrity if hash was present
        if card.eval_card_sha256:
            expected = card.compute_sha256()
            if card.eval_card_sha256 != expected:
                raise ValueError(f"EvalCard SHA-256 integrity check failed: {card.eval_card_sha256} != {expected}")
        return card

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        parameters_total: int,
        trainable_parameters: int,
        group_settings: dict[str, Any],
        metrics: EvalCardMetrics,
        optimizer: EvalCardOptimizer,
        hardware_tier: str,
        backend: str,
        seeds: list[int],
        baseline_trial_id: str | None = None,
        dataset_manifest_sha256: str | None = None,
        grammar_sha256: str | None = None,
        base_commit: str | None = None,
    ) -> EvalCard:
        """Construct, validate, and seal an EvalCard."""
        param_frac = (trainable_parameters / parameters_total) if parameters_total > 0 else 0.0

        card = cls(
            model=EvalCardModel(
                model_id=model_id,
                base_commit=base_commit or _get_git_commit(),
                parameters_total=parameters_total,
                trainable_parameters=trainable_parameters,
                parameter_fraction=param_frac,
            ),
            search_point=EvalCardSearchPoint(
                group_settings=group_settings,
                grammar_sha256=grammar_sha256,
            ),
            metrics=metrics,
            optimizer=optimizer,
            environment=EvalCardEnvironment(
                hardware_tier=hardware_tier,
                os=f"{platform.system()} {platform.release()}",
                python_version=sys.version.split()[0],
                backend=backend,
                library_versions=_get_installed_libraries(),
                seeds_evaluated=seeds,
            ),
            baseline_trial_id=baseline_trial_id,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
        return card.finalize()

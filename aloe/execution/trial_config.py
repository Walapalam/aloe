"""Trial Configuration Data Model (Stage 9 / Gate C).

Represents a concrete, reproducible point in the architectural LoRA search space.
Serializes losslessly to and from JSON. Converts directly to PEFT LoraConfig with
backend-aware group rank_pattern, alpha_pattern, and validated target_modules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aloe.architecture.schema import ArchitectureDescription
from aloe.search_space.schema import SearchGrammar


class InvalidConfigError(ValueError):
    """Raised when a trial configuration is structurally invalid or targets excluded modules."""


class GroupTrialSetting(BaseModel):
    """Configuration for a specific architectural module group."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    rank: int = 8
    alpha: int = 16

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v <= 0:
            raise InvalidConfigError(f"LoRA rank must be positive, got {v}")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: int) -> int:
        if v <= 0:
            raise InvalidConfigError(f"LoRA alpha must be positive, got {v}")
        return v


class TrainingParameters(BaseModel):
    """Execution and optimizer parameters for the trial run."""

    model_config = ConfigDict(extra="ignore")

    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 50
    max_seq_len: int = 256
    warmup_steps: int = 5
    weight_decay: float = 0.01
    precision: Literal["bf16", "fp32"] = "bf16"
    adapter_dtype: Literal["fp32", "bf16"] = "fp32"


class TrialConfig(BaseModel):
    """A concrete, fully specified trial configuration in the ALOE search space."""

    model_config = ConfigDict(extra="ignore")

    trial_id: str = Field(default_factory=lambda: f"trial_{uuid.uuid4().hex[:8]}")
    model_id: str
    model_revision: str = "main"
    backend: str = "peft_mps"
    dataset_manifest_path: str = "data/manifests/xlam_pilot_v1.json"
    dataset_raw_path: str = "data/raw/xlam/xlam_function_calling_60k.json"
    seed: int = 42
    group_settings: dict[str, GroupTrialSetting] = Field(default_factory=dict)
    training: TrainingParameters = Field(default_factory=TrainingParameters)
    explicit_target_modules: list[str] | None = None
    description: str = ""

    @model_validator(mode="after")
    def validate_at_least_one_active(self) -> TrialConfig:
        if self.explicit_target_modules is not None:
            if len(self.explicit_target_modules) == 0:
                raise InvalidConfigError("explicit_target_modules cannot be empty if specified")
            return self

        if not self.group_settings:
            # Empty group settings is only permissible if explicit targets are given
            return self

        any_enabled = any(g.enabled for g in self.group_settings.values())
        if not any_enabled:
            raise InvalidConfigError("At least one architectural module group must be enabled.")
        return self

    def validate_against_grammar(self, grammar: SearchGrammar) -> None:
        """Validate configuration against a model search grammar and its exclusion list."""
        if grammar.model_id != self.model_id:
            # If alias or slug used, check model suffix
            g_name = grammar.model_id.split("/")[-1].lower()
            m_name = self.model_id.split("/")[-1].lower()
            if g_name != m_name:
                raise InvalidConfigError(
                    f"Grammar model_id '{grammar.model_id}' does not match config '{self.model_id}'"
                )

        # 1. Check explicit target modules against exclusions
        if self.explicit_target_modules:
            for target in self.explicit_target_modules:
                if target in grammar.excluded:
                    reason = grammar.excluded[target]
                    raise InvalidConfigError(
                        f"Target module '{target}' is excluded by architecture grammar: {reason}"
                    )
                if "conv1d" in target.lower():
                    raise InvalidConfigError(
                        f"Target module '{target}' is a Conv1D kernel excluded by architectural convention."
                    )

        # 2. Check enabled groups against grammar groups
        grammar_group_types = {g.component_type.value: g for g in grammar.groups}
        for g_name, setting in self.group_settings.items():
            if setting.enabled:
                if g_name not in grammar_group_types:
                    raise InvalidConfigError(
                        f"Configured group '{g_name}' does not exist in model grammar groups: "
                        f"{list(grammar_group_types.keys())}"
                    )
                # Check for any excluded targets in this group
                g_targets = grammar_group_types[g_name].targets
                for t in g_targets:
                    if t in grammar.excluded:
                        raise InvalidConfigError(
                            f"Group '{g_name}' includes excluded target module '{t}'"
                        )

    def _explicit_rank_and_alpha(self) -> tuple[int, int]:
        """Resolve rank and alpha pair for explicit_target_modules consistently across config and parameter counting."""
        return min(
            ((g.rank, g.alpha) for g in self.group_settings.values() if g.enabled),
            default=(8, 16),
        )

    def to_peft_lora_config(self, grammar: SearchGrammar | None = None) -> tuple[dict[str, Any], Any]:
        """Convert TrialConfig to PEFT LoraConfig keyword arguments and LoraConfig instance."""
        from peft import LoraConfig, TaskType

        if grammar is not None:
            self.validate_against_grammar(grammar)

        target_modules: list[str] = []
        rank_pattern: dict[str, int] = {}
        alpha_pattern: dict[str, int] = {}

        if self.explicit_target_modules:
            explicit_r, explicit_alpha = self._explicit_rank_and_alpha()
            for mod in self.explicit_target_modules:
                if grammar and mod in grammar.excluded:
                    raise InvalidConfigError(
                        f"Explicit target '{mod}' is in grammar exclusions: {grammar.excluded[mod]}"
                    )
                target_modules.append(mod)
                rank_pattern[mod] = explicit_r
                alpha_pattern[mod] = explicit_alpha
        elif grammar is not None:
            grammar_group_types = {g.component_type.value: g for g in grammar.groups}
            for g_name, g_dim in grammar_group_types.items():
                setting = self.group_settings.get(g_name)
                if setting is not None and setting.enabled:
                    for target in g_dim.targets:
                        if target in grammar.excluded:
                            raise InvalidConfigError(
                                f"Module '{target}' in group '{g_name}' is excluded: {grammar.excluded[target]}"
                            )
                        target_modules.append(target)
                        rank_pattern[target] = setting.rank
                        alpha_pattern[target] = setting.alpha

        if not target_modules:
            raise InvalidConfigError("No valid target modules resolved for adaptation.")

        default_r = min(rank_pattern.values()) if rank_pattern else 8
        default_alpha = min(alpha_pattern.values()) if alpha_pattern else 16

        peft_kwargs = {
            "r": default_r,
            "lora_alpha": default_alpha,
            "target_modules": target_modules,
            "rank_pattern": rank_pattern,
            "alpha_pattern": alpha_pattern,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }

        lora_config = LoraConfig(**peft_kwargs)
        return peft_kwargs, lora_config

    def compute_trainable_params(
        self,
        arch: ArchitectureDescription,
        grammar: SearchGrammar | None = None,
    ) -> int:
        """Compute exact trainable LoRA parameter count prior to model instantiation (FR04).

        For each targeted linear module with shape (d_out, d_in) and assigned rank r,
        LoRA injects matrices A (r x d_in) and B (d_out x r), contributing r * (d_in + d_out)
        trainable parameters.
        """
        if grammar is not None:
            self.validate_against_grammar(grammar)

        shapes: dict[str, list[int]] = {
            m.name: m.shape for m in arch.modules if m.shape and len(m.shape) == 2
        }

        total_params = 0
        if self.explicit_target_modules:
            explicit_r, _ = self._explicit_rank_and_alpha()
            for mod in self.explicit_target_modules:
                if mod not in shapes:
                    raise InvalidConfigError(f"Module '{mod}' has no 2D linear shape in architecture.")
                d_out, d_in = shapes[mod]
                total_params += explicit_r * (d_in + d_out)
        elif grammar is not None:
            grammar_group_types = {g.component_type.value: g for g in grammar.groups}
            for g_name, g_dim in grammar_group_types.items():
                setting = self.group_settings.get(g_name)
                if setting is not None and setting.enabled:
                    for target in g_dim.targets:
                        if target not in shapes:
                            raise InvalidConfigError(
                                f"Target '{target}' in group '{g_name}' has no 2D linear shape."
                            )
                        d_out, d_in = shapes[target]
                        total_params += setting.rank * (d_in + d_out)
        return total_params

    def validate_parameter_budget_parity(
        self,
        arch: ArchitectureDescription,
        baseline_params: int,
        tolerance: float = 0.05,
        grammar: SearchGrammar | None = None,
        enforce: bool = True,
    ) -> dict[str, Any]:
        """Verify trainable parameter count parity within +/- tolerance (default 5%, FR04)."""
        if baseline_params <= 0:
            raise ValueError(f"baseline_params must be positive, got {baseline_params}")
        candidate_params = self.compute_trainable_params(arch=arch, grammar=grammar)
        rel_delta = (candidate_params - baseline_params) / baseline_params
        within_budget = abs(rel_delta) <= tolerance
        if enforce and not within_budget:
            raise InvalidConfigError(
                f"Parameter budget parity violation: candidate={candidate_params:,} vs "
                f"baseline={baseline_params:,} (delta={rel_delta:+.2%}, tolerance=±{tolerance:.1%})"
            )
        return {
            "candidate_params": candidate_params,
            "baseline_params": baseline_params,
            "relative_delta": round(rel_delta, 6),
            "relative_delta_pct": round(rel_delta * 100.0, 2),
            "tolerance": tolerance,
            "within_budget": within_budget,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize TrialConfig losslessly to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> TrialConfig:
        """Deserialize TrialConfig from JSON string."""
        return cls.model_validate_json(json_str)

    def save(self, path: Path | str) -> Path:
        """Save TrialConfig to a JSON file on disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_file(cls, path: Path | str) -> TrialConfig:
        """Load TrialConfig from a JSON file on disk."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file does not exist: {p}")
        return cls.from_json(p.read_text(encoding="utf-8"))

"""Trial Configuration Data Model (Stage 9 / Protocol v1).

Represents a concrete, reproducible point in the architectural LoRA / AdaLoRA search space.
Serializes losslessly to and from JSON. Converts directly to PEFT LoraConfig or AdaLoraConfig
with exact shape-based parameter counting, ±5% parity enforcement, and Borobia single-group
exact-integer-rank control generation.
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
    """Raised when a trial configuration is structurally invalid or violates budget parity."""


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
    max_steps: int = 125
    max_seq_len: int = 1024
    max_new_tokens: int = 192
    warmup_steps: int = 10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: Literal["cosine", "constant"] = "cosine"
    checkpoint_every_steps: int = 25
    precision: Literal["bf16", "fp32"] = "bf16"
    adapter_dtype: Literal["fp32", "bf16"] = "fp32"


class TrialConfig(BaseModel):
    """A concrete, fully specified trial configuration in the ALOE search space."""

    model_config = ConfigDict(extra="ignore")

    trial_id: str = Field(default_factory=lambda: f"trial_{uuid.uuid4().hex[:8]}")
    arm: str = "aloe"
    peft_method: Literal["lora", "adalora"] = "lora"
    adalora_init_r: int = 12
    adalora_target_r: int = 8
    adalora_tinit: int = 20
    adalora_tfinal: int = 20
    adalora_delta_t: int = 5
    model_id: str
    model_revision: str = "main"
    backend: str = "peft_mps"
    dataset_manifest_path: str = "data/manifests/xlam_ipd_v1.json"
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
            return self

        any_enabled = any(g.enabled for g in self.group_settings.values())
        if not any_enabled:
            raise InvalidConfigError("At least one architectural module group must be enabled.")
        return self

    def canonical_signature(self) -> str:
        """Canonical structural + LR signature used for strict search deduplication."""
        parts = [f"method={self.peft_method}", f"lr={self.training.learning_rate:.8g}"]
        if self.peft_method == "adalora":
            parts.append(f"init_r={self.adalora_init_r}:target_r={self.adalora_target_r}")
        for g_name in sorted(self.group_settings.keys()):
            gs = self.group_settings[g_name]
            if gs.enabled:
                parts.append(f"{g_name}:{gs.rank}:{gs.alpha}")
            else:
                parts.append(f"{g_name}:OFF")
        return "|".join(parts)

    def validate_against_grammar(self, grammar: SearchGrammar) -> None:
        """Validate configuration against a model search grammar and its exclusion list."""
        if grammar.model_id != self.model_id:
            g_name = grammar.model_id.split("/")[-1].lower()
            m_name = self.model_id.split("/")[-1].lower()
            if g_name != m_name:
                raise InvalidConfigError(
                    f"Grammar model_id '{grammar.model_id}' does not match config '{self.model_id}'"
                )

        if self.explicit_target_modules:
            for target in self.explicit_target_modules:
                if target in grammar.excluded:
                    reason = grammar.excluded[target]
                    raise InvalidConfigError(
                        f"Target module '{target}' is excluded by architecture grammar: {reason}"
                    )
                if "conv1d" in target.lower() or target.lower().endswith(".conv.conv"):
                    raise InvalidConfigError(
                        f"Target module '{target}' is a Conv1D kernel excluded by architectural convention."
                    )

        grammar_group_types = {g.component_type.value: g for g in grammar.groups}
        for g_name, setting in self.group_settings.items():
            if setting.enabled:
                if g_name not in grammar_group_types:
                    raise InvalidConfigError(
                        f"Configured group '{g_name}' does not exist in model grammar groups: "
                        f"{list(grammar_group_types.keys())}"
                    )
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
        """Convert TrialConfig to PEFT LoraConfig or AdaLoraConfig."""
        from peft import AdaLoraConfig, LoraConfig, TaskType

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

        if self.peft_method == "adalora":
            total_step = max(self.training.max_steps, 10)
            tinit = min(self.adalora_tinit, max(1, total_step // 5))
            tfinal = min(self.adalora_tfinal, max(1, total_step // 5))
            default_alpha = next(iter(alpha_pattern.values()), self.adalora_target_r * 2)
            adalora_kwargs: dict[str, Any] = {
                "init_r": self.adalora_init_r,
                "target_r": self.adalora_target_r,
                "lora_alpha": default_alpha,
                "target_modules": target_modules,
                "tinit": tinit,
                "tfinal": tfinal,
                "deltaT": max(1, self.adalora_delta_t),
                "total_step": total_step,
                "lora_dropout": 0.0,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM,
            }
            return adalora_kwargs, AdaLoraConfig(**adalora_kwargs)

        default_r = min(rank_pattern.values()) if rank_pattern else 8
        default_alpha = min(alpha_pattern.values()) if alpha_pattern else 16

        peft_kwargs: dict[str, Any] = {
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
        """Compute exact trainable LoRA / post-pruning AdaLoRA parameter count from real 2D module shapes."""
        if grammar is not None:
            self.validate_against_grammar(grammar)

        shapes: dict[str, list[int]] = {
            m.name: m.shape for m in arch.modules if m.shape and len(m.shape) == 2
        }

        if self.peft_method == "adalora":
            # Effective final budget uses adalora_target_r * (d_in + d_out + 1) across active targets
            active_targets: list[str] = []
            if self.explicit_target_modules:
                active_targets = list(self.explicit_target_modules)
            elif grammar is not None:
                for g_dim in grammar.groups:
                    s = self.group_settings.get(g_dim.component_type.value)
                    if s is not None and s.enabled:
                        active_targets.extend(g_dim.targets)
            total_ada = 0
            for mod in active_targets:
                if mod not in shapes:
                    raise InvalidConfigError(f"Module '{mod}' has no 2D linear shape in architecture.")
                d_out, d_in = shapes[mod]
                total_ada += self.adalora_target_r * (d_in + d_out + 1)
            return total_ada

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
        return self.model_dump_json(indent=indent)

    def to_file(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def save(self, path: Path | str) -> Path:
        return self.to_file(path)

    @classmethod
    def from_json(cls, json_str: str) -> TrialConfig:
        return cls.model_validate_json(json_str)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_file(cls, path: Path | str) -> TrialConfig:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file does not exist: {p}")
        return cls.from_json(p.read_text(encoding="utf-8"))


def solve_borobia_ablation_configs(
    arch: ArchitectureDescription,
    grammar: SearchGrammar,
    baseline_params: int,
    *,
    tolerance: float = 0.05,
    seed: int = 42,
    manifest_path: str = "data/manifests/xlam_ipd_v1.json",
    training: TrainingParameters | None = None,
) -> dict[str, dict[str, Any]]:
    """Solve exact integer-rank Borobia-style single-operator placement controls matching `baseline_params`.

    Placements generated per model:
    - `borobia_attn`: enables `local_attention` and/or `global_attention` only.
    - `borobia_ssm`: enables `ssm` (Mamba-2 or short-conv projections) only (omitted if N/A, e.g. Gemma 4).
    - `borobia_ffn`: enables `ffn` only.
    """
    shapes = {m.name: m.shape for m in arch.modules if m.shape and len(m.shape) == 2}
    group_UnitCost: dict[str, int] = {}
    for g in grammar.groups:
        unit = sum(shapes[t][0] + shapes[t][1] for t in g.targets)
        group_UnitCost[g.component_type.value] = unit

    all_group_names = [g.component_type.value for g in grammar.groups]
    placements: dict[str, list[str]] = {
        "borobia_attn": [n for n in all_group_names if "attention" in n],
        "borobia_ssm": [n for n in all_group_names if n == "ssm"],
        "borobia_ffn": [n for n in all_group_names if n == "ffn"],
    }

    solved: dict[str, dict[str, Any]] = {}
    for arm_name, active_groups in placements.items():
        if not active_groups:
            continue
        unit_sum = sum(group_UnitCost[g] for g in active_groups)
        base_r = max(1, int(round(baseline_params / unit_sum)))
        cand_ranks = sorted({max(1, base_r - 1), base_r, base_r + 1})
        exact_r = min(cand_ranks, key=lambda r_cand: abs(r_cand * unit_sum - baseline_params))
        alpha = exact_r * 2
        gs = {
            g_name: (
                GroupTrialSetting(enabled=True, rank=exact_r, alpha=alpha)
                if g_name in active_groups
                else GroupTrialSetting(enabled=False, rank=4, alpha=8)
            )
            for g_name in all_group_names
        }
        cfg = TrialConfig(
            trial_id=f"{arm_name}_s{seed}",
            arm=arm_name,
            model_id=arch.model_id,
            seed=seed,
            dataset_manifest_path=manifest_path,
            group_settings=gs,
            training=training or TrainingParameters(),
            description=f"Borobia-style single-operator control ({arm_name}, solved integer r={exact_r})",
        )
        parity = cfg.validate_parameter_budget_parity(
            arch=arch,
            baseline_params=baseline_params,
            tolerance=tolerance,
            grammar=grammar,
            enforce=True,
        )
        solved[arm_name] = {
            "config": cfg,
            "solved_rank": exact_r,
            "solved_alpha": alpha,
            "parity": parity,
        }
    return solved

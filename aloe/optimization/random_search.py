"""Random search configuration sampler with strict parameter-budget parity for ALOE (Baseline B1).

Shares the exact shape-derived ±5% feasible candidate pool with ALOE (including disabled
groups `enabled=False` and optional shared learning-rate choices), samples without replacement,
keeps Common Random Numbers (`seed=42`) for training, and refuses out-of-budget candidates.
"""
from __future__ import annotations

import itertools
import random

from aloe.architecture.schema import ArchitectureDescription
from aloe.execution.trial_config import (
    GroupTrialSetting,
    InvalidConfigError,
    TrainingParameters,
    TrialConfig,
)
from aloe.search_space.schema import SearchGrammar


class RandomSearchSampler:
    """Samples distinct, budget-matched TrialConfigs from a SearchGrammar."""

    def __init__(
        self,
        grammar: SearchGrammar,
        seed: int = 42,
        training_seed: int = 42,
        arch: ArchitectureDescription | None = None,
        lr_choices: list[float] | None = None,
        dataset_manifest_path: str = "data/manifests/xlam_ipd_v1.json",
        dataset_raw_path: str = "data/raw/xlam/xlam_function_calling_60k.json",
    ) -> None:
        self.grammar = grammar
        self.arch = arch
        self.seed = seed
        self.training_seed = training_seed
        self.rng = random.Random(seed)
        self.lr_choices = sorted(lr_choices) if lr_choices else None
        self.dataset_manifest_path = dataset_manifest_path
        self.dataset_raw_path = dataset_raw_path

        self._group_unit_params: dict[str, int] = {}
        if self.arch is not None:
            shapes = {
                m.name: m.shape for m in self.arch.modules if m.shape and len(m.shape) == 2
            }
            for g in self.grammar.groups:
                self._group_unit_params[g.component_type.value] = sum(
                    shapes[t][0] + shapes[t][1] for t in g.targets if t in shapes
                )

    def estimate_trainable_parameters(self, settings: dict[str, GroupTrialSetting]) -> int:
        """Compute exact parameter count when `arch` is provided; fallback to unit-test estimate only if `arch is None`."""
        total = 0
        if self._group_unit_params:
            for g in self.grammar.groups:
                setting = settings.get(g.component_type.value)
                if setting and setting.enabled:
                    total += setting.rank * self._group_unit_params[g.component_type.value]
            return total

        d_est = 2048
        for g in self.grammar.groups:
            setting = settings.get(g.component_type.value)
            if setting and setting.enabled:
                total += 2 * setting.rank * d_est * len(g.targets)
        return total

    def _enumerate_feasible_pool(
        self,
        target_param_count: int | None,
        param_tolerance: float,
    ) -> list[tuple[dict[str, GroupTrialSetting], float | None]]:
        group_options: list[list[GroupTrialSetting]] = []
        for g in self.grammar.groups:
            ranks = sorted(set(g.rank))
            alphas = sorted(set(g.alpha))
            opts = [GroupTrialSetting(enabled=False, rank=ranks[0], alpha=alphas[0])]
            for r in ranks:
                for a in alphas:
                    opts.append(GroupTrialSetting(enabled=True, rank=r, alpha=a))
            group_options.append(opts)

        lrs: list[float | None] = list(self.lr_choices) if self.lr_choices else [None]
        pool: list[tuple[dict[str, GroupTrialSetting], float | None]] = []

        for combo in itertools.product(*group_options):
            gs = {g.component_type.value: s for g, s in zip(self.grammar.groups, combo)}
            if not any(s.enabled for s in gs.values()):
                continue
            if target_param_count is not None:
                p = self.estimate_trainable_parameters(gs)
                delta = abs(p - target_param_count) / target_param_count
                if delta > param_tolerance:
                    continue
            for lr in lrs:
                pool.append((gs, lr))
        return pool

    def sample_candidate(
        self,
        trial_id: str,
        target_param_count: int | None = None,
        param_tolerance: float = 0.05,
        max_retries: int = 200,
        training: TrainingParameters | None = None,
    ) -> TrialConfig:
        """Sample a single strictly budget-feasible candidate configuration."""
        del max_retries
        pool = self._enumerate_feasible_pool(target_param_count, param_tolerance)
        if not pool:
            raise InvalidConfigError(
                f"No feasible configuration found within ±{param_tolerance:.1%} of target={target_param_count}"
            )
        chosen_gs, chosen_lr = self.rng.choice(pool)
        tp = (training.model_copy() if training is not None else TrainingParameters())
        if chosen_lr is not None:
            tp.learning_rate = chosen_lr
        return TrialConfig(
            trial_id=trial_id,
            arm="b1",
            model_id=self.grammar.model_id,
            seed=self.training_seed,
            dataset_manifest_path=self.dataset_manifest_path,
            dataset_raw_path=self.dataset_raw_path,
            group_settings=chosen_gs,
            training=tp,
        )

    def sample_batch(
        self,
        count: int,
        prefix: str = "trial",
        target_param_count: int | None = None,
        param_tolerance: float = 0.05,
        training: TrainingParameters | None = None,
    ) -> list[TrialConfig]:
        """Sample `count` distinct, strictly budget-matched candidates without replacement."""
        pool = self._enumerate_feasible_pool(target_param_count, param_tolerance)
        if not pool:
            raise InvalidConfigError(
                f"No feasible configurations exist within ±{param_tolerance:.1%} of target={target_param_count}"
            )
        if count > len(pool):
            raise InvalidConfigError(
                f"Requested {count} unique candidates, but feasible pool has only {len(pool)}"
            )
        sampled = self.rng.sample(pool, count)
        candidates: list[TrialConfig] = []
        for i, (gs, lr) in enumerate(sampled):
            tp = (training.model_copy() if training is not None else TrainingParameters())
            if lr is not None:
                tp.learning_rate = lr
            candidates.append(
                TrialConfig(
                    trial_id=f"{prefix}_{i + 1:03d}",
                    arm="b1",
                    model_id=self.grammar.model_id,
                    seed=self.training_seed,
                    dataset_manifest_path=self.dataset_manifest_path,
                    dataset_raw_path=self.dataset_raw_path,
                    group_settings=gs,
                    training=tp,
                )
            )
        return candidates

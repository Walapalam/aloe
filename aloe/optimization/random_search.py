"""Random search configuration sampler with parameter budget matching for ALOE."""
from __future__ import annotations

import random
from typing import Any

from aloe.execution.trial_config import GroupTrialSetting, TrainingParameters, TrialConfig
from aloe.search_space.schema import SearchGrammar


class RandomSearchSampler:
    """Samples valid TrialConfig configurations from a SearchGrammar.

    Supports strict parameter budget matching:
    |TrainableParams_sampled - TrainableParams_target| / TrainableParams_target <= tolerance
    """

    def __init__(
        self,
        grammar: SearchGrammar,
        seed: int = 42,
        dataset_manifest_path: str = "data/manifests/xlam_pilot_v1.json",
        dataset_raw_path: str = "data/raw/xlam/xlam_function_calling_60k.json",
    ) -> None:
        self.grammar = grammar
        self.rng = random.Random(seed)
        self.dataset_manifest_path = dataset_manifest_path
        self.dataset_raw_path = dataset_raw_path

    def estimate_trainable_parameters(self, settings: dict[str, GroupTrialSetting]) -> int:
        """Estimate trainable parameters across all active target modules in groups.

        For each LoRA linear target: num_params = 2 * r * d_model (approximation).
        Assuming average hidden dimension ~ 2048 for small SLMs (<3B).
        """
        d_est = 2048
        total = 0
        for g in self.grammar.groups:
            setting = settings.get(g.component_type.value)
            if setting and setting.enabled:
                r = setting.rank
                num_targets = len(g.targets)
                total += 2 * r * d_est * num_targets
        return total

    def sample_candidate(
        self,
        trial_id: str,
        target_param_count: int | None = None,
        param_tolerance: float = 0.05,
        max_retries: int = 200,
        training: TrainingParameters | None = None,
    ) -> TrialConfig:
        """Sample a candidate configuration from the grammar.

        If target_param_count is provided, attempts to find a candidate within tolerance.
        """
        best_candidate: dict[str, GroupTrialSetting] | None = None
        min_delta = float("inf")

        for _ in range(max_retries):
            sampled_settings: dict[str, GroupTrialSetting] = {}
            for g in self.grammar.groups:
                chosen_r = self.rng.choice(g.rank)
                chosen_a = self.rng.choice(g.alpha)
                sampled_settings[g.component_type.value] = GroupTrialSetting(
                    enabled=True,
                    rank=chosen_r,
                    alpha=chosen_a,
                )

            if target_param_count is None:
                best_candidate = sampled_settings
                break

            est_params = self.estimate_trainable_parameters(sampled_settings)
            delta = abs(est_params - target_param_count) / target_param_count

            if delta <= param_tolerance:
                best_candidate = sampled_settings
                break

            if delta < min_delta:
                min_delta = delta
                best_candidate = sampled_settings

        if best_candidate is None:
            raise RuntimeError("Failed to sample candidate from grammar.")

        return TrialConfig(
            trial_id=trial_id,
            model_id=self.grammar.model_id,
            seed=self.rng.randint(1, 10000),
            dataset_manifest_path=self.dataset_manifest_path,
            dataset_raw_path=self.dataset_raw_path,
            group_settings=best_candidate,
            training=training or TrainingParameters(),
        )

    def sample_batch(
        self,
        count: int,
        prefix: str = "trial",
        target_param_count: int | None = None,
        param_tolerance: float = 0.05,
    ) -> list[TrialConfig]:
        """Sample multiple distinct candidates."""
        candidates: list[TrialConfig] = []
        for i in range(count):
            tid = f"{prefix}_{i+1:03d}"
            cand = self.sample_candidate(
                trial_id=tid,
                target_param_count=target_param_count,
                param_tolerance=param_tolerance,
            )
            candidates.append(cand)
        return candidates

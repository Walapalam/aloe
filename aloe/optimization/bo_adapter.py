"""BoTorch Search Space Adapter for ALOE.

Maps an architecture-derived SearchGrammar into continuous bounds [0, 1]^D,
supporting discrete/continuous transformation, canonical collapse for disabled groups,
injectivity, and parameter-budget calculation.

Conforms to ADR-016 (D=6 formulation for 3-group models) and October Gate D backlog.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import numpy as np
import torch

from aloe.execution.trial_config import GroupTrialSetting, TrainingParameters, TrialConfig
from aloe.search_space.schema import GroupDimensions, SearchGrammar


class BOSearchAdapter:
    """Adapts an ALOE SearchGrammar to normalized continuous search space [0, 1]^D for BoTorch.

    Parameter mapping per group g:
    - u_rank in [0, 1]: mapped to rank_choice in {0, 4, 8, 16, 32}, where 0 means disabled.
      (Discretized into len(ranks)+1 equal intervals).
    - u_alpha in [0, 1]: mapped to alpha_choice in {8, 16, 32, 64, 128}.
      (Discretized into len(alphas) equal intervals). When rank=0, alpha collapses canonically.

    For Gemma 4-E2B (3 groups: local_attn, global_attn, ffn), D = 3 * 2 = 6 dimensions.
    For Falcon-H1 1.5B (3 groups: global_attn, ssm, ffn), D = 3 * 2 = 6 dimensions.
    """

    def __init__(
        self,
        grammar: SearchGrammar,
        dataset_manifest_path: str = "data/manifests/xlam_pilot_v1.json",
        dataset_raw_path: str = "data/raw/xlam/xlam_function_calling_60k.json",
        d_model_est: int = 2048,
    ) -> None:
        self.grammar = grammar
        self.dataset_manifest_path = dataset_manifest_path
        self.dataset_raw_path = dataset_raw_path
        self.d_model_est = d_model_est

        self.groups: list[GroupDimensions] = list(grammar.groups)
        self.group_names: list[str] = [g.component_type.value for g in self.groups]
        self.num_groups: int = len(self.groups)
        self.dim: int = self.num_groups * 2

        # Cache available choices per group
        # rank_choices: [0] + g.rank (where 0 means disabled)
        self.rank_choices: dict[str, list[int]] = {}
        self.alpha_choices: dict[str, list[int]] = {}
        for g in self.groups:
            name = g.component_type.value
            ranks = [0] + sorted(list(set(g.rank)))
            alphas = sorted(list(set(g.alpha)))
            self.rank_choices[name] = ranks
            self.alpha_choices[name] = alphas

    @property
    def bounds(self) -> torch.Tensor:
        """Return 2 x D tensor of lower and upper bounds in [0, 1]."""
        return torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64)

    def get_bounds_list(self) -> tuple[list[float], list[float]]:
        """Return (lower_bounds, upper_bounds) as lists."""
        return [0.0] * self.dim, [1.0] * self.dim

    def encode(self, settings: Mapping[str, GroupTrialSetting]) -> torch.Tensor:
        """Encode a dictionary of GroupTrialSetting into a continuous vector in [0, 1]^D.

        Places values at the midpoint of their discrete intervals to maximize tolerance to small noise.
        """
        vector = []
        for g in self.groups:
            name = g.component_type.value
            ranks = self.rank_choices[name]
            alphas = self.alpha_choices[name]

            setting = settings.get(name)
            if setting is None or not setting.enabled:
                r_val = 0
                a_val = alphas[0]  # canonical collapse
            else:
                r_val = setting.rank if setting.rank in ranks else ranks[1]
                a_val = setting.alpha if setting.alpha in alphas else alphas[0]

            # Find discrete indices
            r_idx = ranks.index(r_val)
            a_idx = alphas.index(a_val)

            # Midpoint mapping: (idx + 0.5) / num_choices
            u_r = (r_idx + 0.5) / len(ranks)
            u_a = (a_idx + 0.5) / len(alphas)

            vector.extend([u_r, u_a])

        return torch.tensor(vector, dtype=torch.float64)

    def decode(
        self,
        vector: Sequence[float] | torch.Tensor,
        trial_id: str = "trial_000",
        training: TrainingParameters | None = None,
    ) -> TrialConfig:
        """Decode a continuous vector in [0, 1]^D into a validated TrialConfig."""
        if isinstance(vector, torch.Tensor):
            vals = vector.detach().cpu().flatten().tolist()
        else:
            vals = list(vector)

        if len(vals) != self.dim:
            raise ValueError(f"Expected vector of dimension {self.dim}, got {len(vals)}")

        group_settings: dict[str, GroupTrialSetting] = {}

        for i, g in enumerate(self.groups):
            name = g.component_type.value
            ranks = self.rank_choices[name]
            alphas = self.alpha_choices[name]

            u_r = float(vals[2 * i])
            u_a = float(vals[2 * i + 1])

            # Clamp to [0, 1]
            u_r = min(max(u_r, 0.0), 1.0)
            u_a = min(max(u_a, 0.0), 1.0)

            # Discretize: index = min(floor(u * N), N - 1)
            r_idx = min(int(math.floor(u_r * len(ranks))), len(ranks) - 1)
            a_idx = min(int(math.floor(u_a * len(alphas))), len(alphas) - 1)

            selected_rank = ranks[r_idx]
            selected_alpha = alphas[a_idx]

            if selected_rank == 0:
                # Canonical collapse: disabled group
                group_settings[name] = GroupTrialSetting(
                    enabled=False,
                    rank=ranks[1],  # canonical minimum valid rank
                    alpha=alphas[0],  # canonical minimum valid alpha
                )
            else:
                group_settings[name] = GroupTrialSetting(
                    enabled=True,
                    rank=selected_rank,
                    alpha=selected_alpha,
                )

        return TrialConfig(
            trial_id=trial_id,
            model_id=self.grammar.model_id,
            dataset_manifest_path=self.dataset_manifest_path,
            dataset_raw_path=self.dataset_raw_path,
            training=training or TrainingParameters(),
            group_settings=group_settings,
        )

    def estimate_trainable_parameters(self, settings: Mapping[str, GroupTrialSetting]) -> int:
        """Compute exact/approximate parameter count for active LoRA adapters."""
        total = 0
        for g in self.groups:
            name = g.component_type.value
            setting = settings.get(name)
            if setting and setting.enabled and setting.rank > 0:
                r = setting.rank
                num_targets = len(g.targets)
                total += 2 * r * self.d_model_est * num_targets
        return total

    def enumerate_all(self) -> list[dict[str, GroupTrialSetting]]:
        """Enumerate all distinct non-trivial configurations in the search space."""
        import itertools

        group_choices = []
        for g in self.groups:
            name = g.component_type.value
            ranks = self.rank_choices[name]
            alphas = self.alpha_choices[name]

            # List of possible GroupTrialSetting for this group
            choices = []
            # Disabled state (canonical)
            choices.append(GroupTrialSetting(enabled=False, rank=ranks[1], alpha=alphas[0]))
            # Active states
            for r in ranks:
                if r == 0:
                    continue
                for a in alphas:
                    choices.append(GroupTrialSetting(enabled=True, rank=r, alpha=a))
            group_choices.append(choices)

        all_configs = []
        for combo in itertools.product(*group_choices):
            cfg = {g.component_type.value: s for g, s in zip(self.groups, combo)}
            # Filter out all-disabled trivial point
            if any(s.enabled for s in cfg.values()):
                all_configs.append(cfg)

        return all_configs

    def generate_uniform_sample(
        self,
        n_samples: int,
        seed: int = 42,
    ) -> list[dict[str, GroupTrialSetting]]:
        """Draw a seeded uniform sample of distinct configurations from the discrete space.

        Guarantees zero duplicates for initial designs, avoiding Sobol discretization collisions.
        """
        all_cfgs = self.enumerate_all()
        if n_samples > len(all_cfgs):
            raise ValueError(f"Requested {n_samples} samples but space only contains {len(all_cfgs)}")

        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(len(all_cfgs), size=n_samples, replace=False)
        return [all_cfgs[i] for i in selected_indices]

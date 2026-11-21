"""BoTorch Search Space Adapter for ALOE (ADR-016 / Protocol v1).

Maps an architecture-derived SearchGrammar into normalized bounds [0, 1]^D with:
- Exact shape-based parameter counting whenever `arch` is provided (eliminating 2048 width approximation).
- Scrambled Sobol initial design mapped onto the exact ±5% feasible candidate pool without duplicates.
- Pre-encoded discrete feasible candidate pool tensor for exact `optimize_acqf_discrete` acquisition.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence
import numpy as np
import torch

from aloe.architecture.schema import ArchitectureDescription
from aloe.execution.trial_config import GroupTrialSetting, TrainingParameters, TrialConfig
from aloe.search_space.schema import GroupDimensions, SearchGrammar


def group_settings_signature(
    settings: Mapping[str, GroupTrialSetting],
    learning_rate: float | None = None,
) -> str:
    """Canonical hashable string signature for a group-settings dict (+ optional LR)."""
    parts: list[str] = []
    if learning_rate is not None:
        parts.append(f"lr={learning_rate:.8g}")
    for k in sorted(settings.keys()):
        s = settings[k]
        if s.enabled:
            parts.append(f"{k}:{s.rank}:{s.alpha}")
        else:
            parts.append(f"{k}:OFF")
    return "|".join(parts)


class BOSearchAdapter:
    """Adapts an ALOE SearchGrammar to normalized continuous search space [0, 1]^D for BoTorch."""

    def __init__(
        self,
        grammar: SearchGrammar,
        arch: ArchitectureDescription | None = None,
        feasible_configs: list[dict[str, GroupTrialSetting]] | None = None,
        lr_choices: list[float] | None = None,
        dataset_manifest_path: str = "data/manifests/xlam_ipd_v1.json",
        dataset_raw_path: str = "data/raw/xlam/xlam_function_calling_60k.json",
        d_model_est: int = 2048,
    ) -> None:
        self.grammar = grammar
        self.arch = arch
        self.dataset_manifest_path = dataset_manifest_path
        self.dataset_raw_path = dataset_raw_path
        self.d_model_est = d_model_est
        self.lr_choices = sorted(lr_choices) if lr_choices else None

        self.groups: list[GroupDimensions] = list(grammar.groups)
        self.group_names: list[str] = [g.component_type.value for g in self.groups]
        self.num_groups: int = len(self.groups)
        self.dim: int = self.num_groups * 2 + (1 if self.lr_choices else 0)

        self.rank_choices: dict[str, list[int]] = {}
        self.alpha_choices: dict[str, list[int]] = {}
        for g in self.groups:
            name = g.component_type.value
            ranks = [0] + sorted(list(set(g.rank)))
            alphas = sorted(list(set(g.alpha)))
            self.rank_choices[name] = ranks
            self.alpha_choices[name] = alphas

        # Precompute exact per-group unit parameter cost sum(d_in + d_out) if arch is available
        self._group_unit_params: dict[str, int] = {}
        if self.arch is not None:
            shapes = {
                m.name: m.shape for m in self.arch.modules if m.shape and len(m.shape) == 2
            }
            for g in self.groups:
                self._group_unit_params[g.component_type.value] = sum(
                    shapes[t][0] + shapes[t][1] for t in g.targets if t in shapes
                )

        self.feasible_configs: list[dict[str, GroupTrialSetting]] | None = (
            list(feasible_configs) if feasible_configs is not None else None
        )

    @property
    def bounds(self) -> torch.Tensor:
        """Return 2 x D tensor of lower and upper bounds in [0, 1]."""
        return torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64)

    def get_bounds_list(self) -> tuple[list[float], list[float]]:
        return [0.0] * self.dim, [1.0] * self.dim

    def encode(
        self,
        settings: Mapping[str, GroupTrialSetting],
        learning_rate: float | None = None,
    ) -> torch.Tensor:
        """Encode GroupTrialSetting (+ optional LR) into midpoint coordinates in [0, 1]^D."""
        vector: list[float] = []
        for g in self.groups:
            name = g.component_type.value
            ranks = self.rank_choices[name]
            alphas = self.alpha_choices[name]

            setting = settings.get(name)
            if setting is None or not setting.enabled:
                r_val = 0
                a_val = alphas[0]
            else:
                r_val = setting.rank if setting.rank in ranks else ranks[1]
                a_val = setting.alpha if setting.alpha in alphas else alphas[0]

            r_idx = ranks.index(r_val)
            a_idx = alphas.index(a_val)
            vector.extend([(r_idx + 0.5) / len(ranks), (a_idx + 0.5) / len(alphas)])

        if self.lr_choices:
            lr_val = learning_rate if learning_rate in self.lr_choices else self.lr_choices[len(self.lr_choices) // 2]
            lr_idx = self.lr_choices.index(lr_val)
            vector.append((lr_idx + 0.5) / len(self.lr_choices))

        return torch.tensor(vector, dtype=torch.float64)

    def decode(
        self,
        vector: Sequence[float] | torch.Tensor,
        trial_id: str = "trial_000",
        training: TrainingParameters | None = None,
        seed: int = 42,
    ) -> TrialConfig:
        """Decode a continuous vector in [0, 1]^D into a validated TrialConfig."""
        vals = vector.detach().cpu().flatten().tolist() if isinstance(vector, torch.Tensor) else list(vector)
        if len(vals) != self.dim:
            raise ValueError(f"Expected vector of dimension {self.dim}, got {len(vals)}")

        group_settings: dict[str, GroupTrialSetting] = {}
        for i, g in enumerate(self.groups):
            name = g.component_type.value
            ranks = self.rank_choices[name]
            alphas = self.alpha_choices[name]

            u_r = min(max(float(vals[2 * i]), 0.0), 1.0)
            u_a = min(max(float(vals[2 * i + 1]), 0.0), 1.0)

            r_idx = min(int(math.floor(u_r * len(ranks))), len(ranks) - 1)
            a_idx = min(int(math.floor(u_a * len(alphas))), len(alphas) - 1)

            selected_rank = ranks[r_idx]
            selected_alpha = alphas[a_idx]

            if selected_rank == 0:
                group_settings[name] = GroupTrialSetting(
                    enabled=False,
                    rank=ranks[1],
                    alpha=alphas[0],
                )
            else:
                group_settings[name] = GroupTrialSetting(
                    enabled=True,
                    rank=selected_rank,
                    alpha=selected_alpha,
                )

        train_params = (training.model_copy() if training is not None else TrainingParameters())
        if self.lr_choices:
            u_lr = min(max(float(vals[-1]), 0.0), 1.0)
            lr_idx = min(int(math.floor(u_lr * len(self.lr_choices))), len(self.lr_choices) - 1)
            train_params.learning_rate = self.lr_choices[lr_idx]

        return TrialConfig(
            trial_id=trial_id,
            model_id=self.grammar.model_id,
            seed=seed,
            dataset_manifest_path=self.dataset_manifest_path,
            dataset_raw_path=self.dataset_raw_path,
            training=train_params,
            group_settings=group_settings,
        )

    def estimate_trainable_parameters(self, settings: Mapping[str, GroupTrialSetting]) -> int:
        """Compute exact parameter count when `arch` is provided; fallback to unit-test `d_model_est` only if `arch is None`."""
        total = 0
        if self._group_unit_params:
            for g in self.groups:
                name = g.component_type.value
                setting = settings.get(name)
                if setting and setting.enabled and setting.rank > 0:
                    total += setting.rank * self._group_unit_params[name]
            return total

        for g in self.groups:
            name = g.component_type.value
            setting = settings.get(name)
            if setting and setting.enabled and setting.rank > 0:
                total += 2 * setting.rank * self.d_model_est * len(g.targets)
        return total

    def enumerate_all(self) -> list[dict[str, GroupTrialSetting]]:
        """Enumerate all distinct non-trivial configurations in the search space."""
        group_choices = []
        for g in self.groups:
            name = g.component_type.value
            ranks = self.rank_choices[name]
            alphas = self.alpha_choices[name]
            choices = [GroupTrialSetting(enabled=False, rank=ranks[1], alpha=alphas[0])]
            for r in ranks:
                if r == 0:
                    continue
                for a in alphas:
                    choices.append(GroupTrialSetting(enabled=True, rank=r, alpha=a))
            group_choices.append(choices)

        all_configs = []
        for combo in itertools.product(*group_choices):
            cfg = {g.component_type.value: s for g, s in zip(self.groups, combo)}
            if any(s.enabled for s in cfg.values()):
                all_configs.append(cfg)
        return all_configs

    def build_discrete_candidate_pool(
        self,
        *,
        exclude_signatures: set[str] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[dict[str, GroupTrialSetting], float | None, str]]]:
        """Build `(N x D tensor, metadata_list)` over the exact feasible candidate pool (or full space if unconstrained)."""
        base_pool = self.feasible_configs if self.feasible_configs is not None else self.enumerate_all()
        excl = exclude_signatures or set()
        lrs: list[float | None] = list(self.lr_choices) if self.lr_choices else [None]

        vecs: list[torch.Tensor] = []
        meta: list[tuple[dict[str, GroupTrialSetting], float | None, str]] = []
        for gs in base_pool:
            for lr in lrs:
                sig = group_settings_signature(gs, lr)
                if sig in excl:
                    continue
                vecs.append(self.encode(gs, learning_rate=lr))
                meta.append((gs, lr, sig))

        if not vecs:
            return torch.empty((0, self.dim), dtype=torch.float64), []
        return torch.stack(vecs), meta

    def generate_sobol_sample(
        self,
        n_samples: int = 5,
        seed: int = 42,
        *,
        exclude_signatures: set[str] | None = None,
    ) -> list[tuple[dict[str, GroupTrialSetting], float | None]]:
        """Draw `n_samples` unique scrambled-Sobol points snapped to the nearest feasible discrete states."""
        pool_tensor, pool_meta = self.build_discrete_candidate_pool(exclude_signatures=exclude_signatures)
        if len(pool_meta) < n_samples:
            raise ValueError(
                f"Requested {n_samples} unique Sobol samples, but feasible pool has only {len(pool_meta)}"
            )

        sobol = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        selected: list[tuple[dict[str, GroupTrialSetting], float | None]] = []
        used_indices: set[int] = set()

        # Draw Sobol points until `n_samples` unique discrete states are chosen
        max_draws = max(n_samples * 16, 64)
        for _ in range(max_draws):
            if len(selected) >= n_samples:
                break
            pt = sobol.draw(1).to(dtype=torch.float64)  # (1, D)
            dists = torch.cdist(pt, pool_tensor).squeeze(0)  # (N,)
            order = torch.argsort(dists).tolist()
            for idx in order:
                if idx not in used_indices:
                    used_indices.add(idx)
                    gs, lr, _ = pool_meta[idx]
                    selected.append((gs, lr))
                    break

        return selected

    def generate_uniform_sample(
        self,
        n_samples: int,
        seed: int = 42,
    ) -> list[dict[str, GroupTrialSetting]]:
        """Draw a seeded uniform sample of distinct configurations (retained for unit-test compatibility)."""
        base_pool = self.feasible_configs if self.feasible_configs is not None else self.enumerate_all()
        if n_samples > len(base_pool):
            raise ValueError(f"Requested {n_samples} samples but space only contains {len(base_pool)}")
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(len(base_pool), size=n_samples, replace=False)
        return [base_pool[i] for i in selected_indices]

"""Multi-Objective Bayesian Optimization (MOBO) for ALOE.

Implements October Gate D specification and ADR-016:
- 3-Objective formulation: Tool Accuracy (max), Trainable Params (min), Peak Process RSS (min).
- Exact Matérn-5/2 kernel with ScaleKernel and Standardize outcome transform.
- qLogNEHVI acquisition function with discrete candidate snapping via BOSearchAdapter.
- Seeded uniform initial design avoiding Sobol discretization duplicates.
- Explicit fallback safety: raises ImportError if BoTorch is missing unless allow_fallback=True.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence
import numpy as np
import torch

from aloe.execution.trial_config import GroupTrialSetting, TrainingParameters, TrialConfig
from aloe.optimization.bo_adapter import BOSearchAdapter
from aloe.optimization.objectives import DEFAULT_ALOE_OBJECTIVES, ObjectiveConfig

logger = logging.getLogger(__name__)


class MOBOptimizer:
    """Multi-Objective Bayesian Optimizer for architecture-aware PEFT search.

    Manages sequential exploration of the discrete/continuous hybrid search space
    using Gaussian Process surrogates and Log-space Hypervolume Improvement.
    """

    def __init__(
        self,
        adapter: BOSearchAdapter,
        objectives: list[ObjectiveConfig] | None = None,
        ref_point: list[float] | None = None,
        memory_metric: str = "process_rss_unified",
        seed: int = 42,
        n_init: int = 6,
        allow_fallback: bool = False,
    ) -> None:
        self.adapter = adapter
        self.objectives = objectives or DEFAULT_ALOE_OBJECTIVES
        self.memory_metric = memory_metric
        self.seed = seed
        self.n_init = n_init
        self.allow_fallback = allow_fallback

        # Check BoTorch availability
        self._has_botorch = self._check_botorch_available()
        if not self._has_botorch and not self.allow_fallback:
            raise ImportError(
                "BoTorch and GPyTorch are required for MOBOptimizer. "
                "Install them with 'pip install botorch gpytorch' or pass allow_fallback=True "
                "to allow seeded random search fallback."
            )

        self.impl = "mobo_botorch_matern52" if self._has_botorch else "random_fallback"

        # Reference point in BoTorch maximization convention:
        # tool_accuracy: 0.0 (0% accuracy)
        # trainable_parameters: -100_000_000.0 (max 100M params)
        # peak_process_rss_mb: -24_000.0 (max 24 GB)
        self.ref_point = ref_point or [0.0, -100_000_000.0, -24_000.0]

        # Initial design generation
        self.initial_configs: list[dict[str, GroupTrialSetting]] = self.adapter.generate_uniform_sample(
            n_samples=self.n_init, seed=self.seed
        )
        self.trial_counter: int = 0

        # Storage for observed trials
        self.observed_X: list[torch.Tensor] = []
        self.observed_Y: list[list[float]] = []  # In BoTorch maximization space [acc, -params, -rss]
        self.evaluated_trial_ids: list[str] = []

    @staticmethod
    def _check_botorch_available() -> bool:
        """Check if BoTorch and GPyTorch packages can be imported."""
        try:
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
            return True
        except ImportError:
            return False

    def suggest(self, q: int = 1) -> list[TrialConfig]:
        """Suggest the next batch of q candidate TrialConfigs."""
        candidates: list[TrialConfig] = []

        for _ in range(q):
            trial_id = f"trial_mobo_{self.trial_counter:04d}"

            # Step 1: Initial exploration phase
            if self.trial_counter < self.n_init:
                group_settings = self.initial_configs[self.trial_counter]
                config = TrialConfig(
                    trial_id=trial_id,
                    model_id=self.adapter.grammar.model_id,
                    dataset_manifest_path=self.adapter.dataset_manifest_path,
                    dataset_raw_path=self.adapter.dataset_raw_path,
                    group_settings=group_settings,
                )
            else:
                # Step 2: Active acquisition phase
                if self._has_botorch:
                    config = self._suggest_botorch(trial_id)
                else:
                    config = self._suggest_fallback(trial_id)

            candidates.append(config)
            self.trial_counter += 1

        return candidates

    def _suggest_fallback(self, trial_id: str) -> TrialConfig:
        """Fallback suggestion using seeded parameter-budget random search."""
        rng = np.random.default_rng(self.seed + self.trial_counter)
        all_cfgs = self.adapter.enumerate_all()
        chosen = all_cfgs[int(rng.integers(0, len(all_cfgs)))]
        return TrialConfig(
            trial_id=trial_id,
            model_id=self.adapter.grammar.model_id,
            dataset_manifest_path=self.adapter.dataset_manifest_path,
            dataset_raw_path=self.adapter.dataset_raw_path,
            group_settings=chosen,
        )

    def _suggest_botorch(self, trial_id: str) -> TrialConfig:
        """Propose candidate using BoTorch with Matérn-5/2 kernel and qLogNEHVI."""
        from botorch.acquisition.multi_objective.logei import (  # type: ignore
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.fit import fit_gpytorch_mll  # type: ignore
        from botorch.models import ModelListGP, SingleTaskGP  # type: ignore
        from botorch.models.transforms.outcome import Standardize  # type: ignore
        from botorch.optim import optimize_acqf  # type: ignore
        from gpytorch.kernels import MaternKernel, ScaleKernel  # type: ignore
        from gpytorch.mlls import ExactMarginalLogLikelihood  # type: ignore

        train_X = torch.stack(self.observed_X).to(dtype=torch.float64)
        train_Y = torch.tensor(self.observed_Y, dtype=torch.float64)

        # Fit independent GPs for accuracy and memory with Matérn-5/2 ARD kernel
        models = []
        for i in range(train_Y.shape[-1]):
            y_i = train_Y[:, i : i + 1]
            covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.adapter.dim))
            gp = SingleTaskGP(
                train_X,
                y_i,
                covar_module=covar,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                logger.warning("GPyTorch MLL fitting warning: %s", e)
            models.append(gp)

        model_list = ModelListGP(*models)

        ref_tensor = torch.tensor(self.ref_point, dtype=torch.float64)
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model_list,
            ref_point=ref_tensor,
            X_baseline=train_X,
            prune_baseline=True,
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.adapter.bounds,
            q=1,
            num_restarts=5,
            raw_samples=32,
        )

        x_cand = candidates[0]
        return self.adapter.decode(x_cand, trial_id=trial_id)

    def observe(
        self,
        config: TrialConfig,
        metrics: Mapping[str, float],
    ) -> None:
        """Record an observed evaluation result for a TrialConfig."""
        # Validate required metrics
        if "tool_accuracy" not in metrics:
            raise KeyError("Missing required metric 'tool_accuracy' in observation.")
        if "peak_process_rss_mb" not in metrics:
            raise KeyError("Missing required metric 'peak_process_rss_mb' in observation.")

        acc = float(metrics["tool_accuracy"])
        mem = float(metrics["peak_process_rss_mb"])

        # Parameter count is calculated deterministically from adapter
        params = float(
            metrics.get(
                "trainable_parameters",
                self.adapter.estimate_trainable_parameters(config.group_settings),
            )
        )

        # Transform to BoTorch maximization convention: [acc, -params, -mem]
        y_obs = [acc, -params, -mem]

        x_vec = self.adapter.encode(config.group_settings)

        self.observed_X.append(x_vec)
        self.observed_Y.append(y_obs)
        self.evaluated_trial_ids.append(config.trial_id)

    def state_dict(self) -> dict[str, Any]:
        """Serialize optimizer state for checkpointing and replayability."""
        return {
            "impl": self.impl,
            "seed": self.seed,
            "n_init": self.n_init,
            "trial_counter": self.trial_counter,
            "memory_metric": self.memory_metric,
            "ref_point": list(self.ref_point),
            "observed_X": [x.tolist() for x in self.observed_X],
            "observed_Y": self.observed_Y,
            "evaluated_trial_ids": list(self.evaluated_trial_ids),
        }

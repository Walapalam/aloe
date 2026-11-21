"""Multi-Objective Bayesian Optimization (MOBO) for ALOE (ADR-016 / Protocol v1).

Implements the exact Proposal & IPD search protocol:
- 5 unique scrambled-Sobol initial points + 15 sequential qLogNEHVI points (`n_init=5`).
- Exact discrete optimization (`optimize_acqf_discrete`) directly over the ±5% feasible
  candidate pool (`550` Gemma / `430` Falcon / `11,375` LFM3 in memory-safe chunks),
  eliminating post-hoc modulo snapping and duplicate configurations.
- Strict fail-closed `allow_fallback=False` for official campaigns.
- Full `state_dict()` and `load_state_dict()` serialization for deterministic crash recovery.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Literal, Mapping
import numpy as np
import torch

from aloe.execution.trial_config import GroupTrialSetting, TrainingParameters, TrialConfig
from aloe.optimization.bo_adapter import BOSearchAdapter, group_settings_signature
from aloe.optimization.objectives import DEFAULT_ALOE_OBJECTIVES, ObjectiveConfig

logger = logging.getLogger(__name__)


class MOBOptimizer:
    """Multi-Objective Bayesian Optimizer for architecture-aware PEFT search."""

    def __init__(
        self,
        adapter: BOSearchAdapter,
        objectives: list[ObjectiveConfig] | None = None,
        ref_point: list[float] | None = None,
        memory_metric: str = "process_rss_unified",
        seed: int = 42,
        training_seed: int = 42,
        n_init: int = 5,
        init_strategy: Literal["sobol", "uniform"] = "sobol",
        allow_fallback: bool = False,
        base_training: TrainingParameters | None = None,
        discrete_chunk_size: int = 2048,
    ) -> None:
        self.adapter = adapter
        self.objectives = objectives or DEFAULT_ALOE_OBJECTIVES
        self.memory_metric = memory_metric
        self.seed = seed
        self.training_seed = training_seed
        self.n_init = n_init
        self.init_strategy = init_strategy
        self.allow_fallback = allow_fallback
        self.base_training = base_training or TrainingParameters()
        self.discrete_chunk_size = discrete_chunk_size

        self._has_botorch = self._check_botorch_available()
        if not self._has_botorch and not self.allow_fallback:
            raise ImportError(
                "BoTorch and GPyTorch are required for MOBOptimizer. "
                "Install them with 'pip install botorch gpytorch' or pass allow_fallback=True "
                "for offline unit tests."
            )

        self.impl = (
            "mobo_sobol_qlognehvi_discrete"
            if self._has_botorch and self.adapter.feasible_configs is not None
            else ("mobo_botorch_matern52" if self._has_botorch else "random_fallback")
        )

        # Reference point in BoTorch maximization convention: [acc, -params, -mem_mb].
        # Unless explicitly overridden, calibrated automatically from the Sobol initial design nadir (ADR-018)
        # so accuracy, ±5% parameter budget, and memory fluctuations contribute balanced hypervolume volume.
        self._custom_ref_point = ref_point is not None
        self.ref_point = list(ref_point) if ref_point is not None else [0.0, -100_000_000.0, -40_000.0]
        self.ref_point_calibrated: bool = self._custom_ref_point

        self.observed_signatures: set[str] = set()
        self.pending_signatures: set[str] = set()

        if self.init_strategy == "sobol":
            sobol_pairs = self.adapter.generate_sobol_sample(n_samples=self.n_init, seed=self.seed)
            self.initial_configs: list[dict[str, GroupTrialSetting]] = [p[0] for p in sobol_pairs]
            self._initial_lrs: list[float | None] = [p[1] for p in sobol_pairs]
        else:
            self.initial_configs = self.adapter.generate_uniform_sample(
                n_samples=self.n_init, seed=self.seed
            )
            self._initial_lrs = [None] * self.n_init

        self.trial_counter: int = 0
        self.observed_X: list[torch.Tensor] = []
        self.observed_Y: list[list[float]] = []
        self.sobol_Y: list[list[float]] = []
        self.evaluated_trial_ids: list[str] = []

    @staticmethod
    def _check_botorch_available() -> bool:
        try:
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
            return True
        except ImportError:
            return False

    def _build_trial_config(
        self,
        trial_id: str,
        group_settings: dict[str, GroupTrialSetting],
        lr: float | None = None,
    ) -> TrialConfig:
        tp = self.base_training.model_copy()
        if lr is not None:
            tp.learning_rate = lr
        cfg = TrialConfig(
            trial_id=trial_id,
            arm="aloe",
            model_id=self.adapter.grammar.model_id,
            seed=self.training_seed,
            dataset_manifest_path=self.adapter.dataset_manifest_path,
            dataset_raw_path=self.adapter.dataset_raw_path,
            group_settings=group_settings,
            training=tp,
        )
        sig = group_settings_signature(
            group_settings,
            tp.learning_rate if self.adapter.lr_choices else None,
        )
        self.pending_signatures.add(sig)
        return cfg

    def suggest(self, q: int = 1) -> list[TrialConfig]:
        """Suggest the next batch of `q` unique, budget-feasible candidate TrialConfigs."""
        candidates: list[TrialConfig] = []

        for _ in range(q):
            trial_id = f"trial_mobo_{self.trial_counter:04d}"

            if self.trial_counter < self.n_init:
                gs = self.initial_configs[self.trial_counter]
                lr = self._initial_lrs[self.trial_counter]
                sig = group_settings_signature(gs, lr if self.adapter.lr_choices else None)
                if sig in (self.observed_signatures | self.pending_signatures):
                    # Draw replacement unique Sobol candidate
                    repl = self.adapter.generate_sobol_sample(
                        n_samples=1,
                        seed=self.seed + 1000 + self.trial_counter,
                        exclude_signatures=self.observed_signatures | self.pending_signatures,
                    )[0]
                    gs, lr = repl
                config = self._build_trial_config(trial_id, gs, lr)
            else:
                if self._has_botorch:
                    config = self._suggest_botorch(trial_id)
                else:
                    config = self._suggest_fallback(trial_id)

            candidates.append(config)
            self.trial_counter += 1

        return candidates

    def _suggest_fallback(self, trial_id: str) -> TrialConfig:
        rng = np.random.default_rng(self.seed + self.trial_counter)
        _, pool_meta = self.adapter.build_discrete_candidate_pool(
            exclude_signatures=self.observed_signatures | self.pending_signatures
        )
        if not pool_meta:
            raise RuntimeError("Exhausted all unique candidates in discrete search pool.")
        gs, lr, _ = pool_meta[int(rng.integers(0, len(pool_meta)))]
        return self._build_trial_config(trial_id, gs, lr)

    def _calibrate_ref_point_if_ready(self) -> None:
        """Lock empirical nadir-minus-margin reference point strictly from `n_init` Sobol observations (ADR-018)."""
        req = max(1, self.n_init)
        if self.ref_point_calibrated or len(self.sobol_Y) < req:
            return
        init_Y = torch.tensor(self.sobol_Y[:req], dtype=torch.float64)
        y_min = init_Y.min(dim=0).values
        y_max = init_Y.max(dim=0).values
        fallback_span = torch.maximum(torch.abs(y_min) * 0.02, torch.tensor([0.05, 1000.0, 50.0], dtype=torch.float64))
        span = torch.maximum(y_max - y_min, fallback_span)
        calibrated = (y_min - 0.15 * span).tolist()
        self.ref_point = [float(v) for v in calibrated]
        self.ref_point_calibrated = True

    def _suggest_botorch(self, trial_id: str) -> TrialConfig:
        """Propose next unique candidate via Matérn-5/2 ARD GP + discrete qLogNEHVI."""
        from botorch.acquisition.multi_objective.logei import (  # type: ignore
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.fit import fit_gpytorch_mll  # type: ignore
        from botorch.models import ModelListGP, SingleTaskGP  # type: ignore
        from botorch.models.transforms.outcome import Standardize  # type: ignore
        from botorch.optim import optimize_acqf_discrete  # type: ignore
        from botorch.sampling.normal import SobolQMCNormalSampler  # type: ignore
        from gpytorch.kernels import MaternKernel, ScaleKernel  # type: ignore
        from gpytorch.mlls import ExactMarginalLogLikelihood  # type: ignore

        self._calibrate_ref_point_if_ready()

        torch.manual_seed(self.seed + self.trial_counter)
        train_X = torch.stack(self.observed_X).to(dtype=torch.float64)
        train_Y = torch.tensor(self.observed_Y, dtype=torch.float64)

        models = []
        for i in range(train_Y.shape[-1]):
            y_i = train_Y[:, i : i + 1]
            # Do NOT mutate observed outcomes with synthetic jitter; omit Standardize if variance == 0
            has_variance = y_i.shape[0] > 1 and float(torch.std(y_i)) > 1e-12
            covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.adapter.dim))
            gp = SingleTaskGP(
                train_X,
                y_i,
                covar_module=covar,
                outcome_transform=Standardize(m=1) if has_variance else None,
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                if has_variance:
                    fit_gpytorch_mll(mll)
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(f"Official MOBO GP MLL fitting failed on objective {i}: {e}") from e
                logger.warning("GPyTorch MLL fitting warning: %s", e)
            models.append(gp)

        model_list = ModelListGP(*models)
        ref_tensor = torch.tensor(self.ref_point, dtype=torch.float64)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=self.seed + self.trial_counter)
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model_list,
            ref_point=ref_tensor,
            X_baseline=train_X,
            sampler=sampler,
            prune_baseline=True,
        )

        # Evaluate directly over the exact remaining feasible discrete pool (chunked for LFM3 11,375 states)
        excl = self.observed_signatures | self.pending_signatures
        pool_tensor, pool_meta = self.adapter.build_discrete_candidate_pool(exclude_signatures=excl)
        if len(pool_meta) == 0:
            raise RuntimeError(
                "Exhausted all unique candidates in the exact discrete feasible pool; "
                "refusing unconstrained continuous fallback."
            )

        if len(pool_meta) <= self.discrete_chunk_size:
            with torch.no_grad():
                vals_small = acq_func(pool_tensor.unsqueeze(1))
                if not bool(torch.isfinite(vals_small).all()):
                    raise RuntimeError(
                        "Non-finite qLogNEHVI values in discrete candidate pool; "
                        "refusing arbitrary candidate selection."
                    )
            best_x, best_acq_val = optimize_acqf_discrete(acq_function=acq_func, q=1, choices=pool_tensor)
            if not bool(torch.isfinite(best_acq_val).all()):
                raise RuntimeError("Non-finite qLogNEHVI optimum returned by optimize_acqf_discrete.")
            dists = torch.cdist(best_x, pool_tensor).squeeze(0)
            best_idx = int(torch.argmin(dists).item())
        else:
            best_val = -float("inf")
            best_idx = 0
            with torch.no_grad():
                for start in range(0, len(pool_meta), self.discrete_chunk_size):
                    end = min(start + self.discrete_chunk_size, len(pool_meta))
                    chunk = pool_tensor[start:end].unsqueeze(1)  # (B, 1, D)
                    vals = acq_func(chunk)  # (B,)
                    if not bool(torch.isfinite(vals).all()):
                        raise RuntimeError(
                            f"Non-finite qLogNEHVI values in discrete chunk {start}:{end}; "
                            "refusing arbitrary candidate selection."
                        )
                    chunk_max_idx = int(torch.argmax(vals).item())
                    chunk_max_val = float(vals[chunk_max_idx].item())
                    if chunk_max_val > best_val:
                        best_val = chunk_max_val
                        best_idx = start + chunk_max_idx
            if not math.isfinite(best_val):
                raise RuntimeError("No finite qLogNEHVI acquisition value was obtained.")
        gs, lr, _ = pool_meta[best_idx]
        return self._build_trial_config(trial_id, gs, lr)

    def observe(
        self,
        config: TrialConfig,
        metrics: Mapping[str, float],
    ) -> None:
        """Record a real observed evaluation result for a TrialConfig."""
        if "tool_accuracy" not in metrics:
            raise KeyError("Missing required metric 'tool_accuracy' in observation.")
        mem_val = metrics.get("search_memory_mb", metrics.get("peak_process_rss_mb"))
        if mem_val is None:
            raise KeyError("Missing required metric 'peak_process_rss_mb' (or 'search_memory_mb') in observation.")

        acc = float(metrics["tool_accuracy"])
        mem = float(mem_val)
        params = float(
            metrics.get(
                "trainable_parameters",
                self.adapter.estimate_trainable_parameters(config.group_settings),
            )
        )
        if not (math.isfinite(acc) and math.isfinite(mem) and math.isfinite(params)):
            raise ValueError(f"Non-finite objective observation: acc={acc}, params={params}, mem={mem}")

        y_obs = [acc, -params, -mem]
        lr_opt = config.training.learning_rate if self.adapter.lr_choices else None
        x_vec = self.adapter.encode(config.group_settings, learning_rate=lr_opt)
        sig = group_settings_signature(config.group_settings, lr_opt)

        self.pending_signatures.discard(sig)
        self.observed_signatures.add(sig)
        self.observed_X.append(x_vec)
        self.observed_Y.append(y_obs)
        # Track Sobol-phase observations explicitly by trial_id index (< n_init) or pre-calibration sequence
        is_sobol_trial = True
        if config.trial_id.startswith("trial_mobo_"):
            try:
                t_idx = int(config.trial_id.rsplit("_", 1)[1])
                is_sobol_trial = t_idx < self.n_init
            except ValueError:
                is_sobol_trial = len(self.sobol_Y) < self.n_init
        elif len(self.sobol_Y) >= self.n_init:
            is_sobol_trial = False
        if is_sobol_trial:
            self.sobol_Y.append(y_obs)
        self.evaluated_trial_ids.append(config.trial_id)
        self._calibrate_ref_point_if_ready()

    def state_dict(self) -> dict[str, Any]:
        """Serialize complete optimizer state for deterministic checkpoint recovery."""
        return {
            "impl": self.impl,
            "seed": self.seed,
            "training_seed": self.training_seed,
            "n_init": self.n_init,
            "init_strategy": self.init_strategy,
            "trial_counter": self.trial_counter,
            "memory_metric": self.memory_metric,
            "ref_point": list(self.ref_point),
            "ref_point_calibrated": self.ref_point_calibrated,
            "observed_X": [x.tolist() for x in self.observed_X],
            "observed_Y": self.observed_Y,
            "sobol_Y": self.sobol_Y,
            "evaluated_trial_ids": list(self.evaluated_trial_ids),
            "observed_signatures": sorted(self.observed_signatures),
            "pending_signatures": sorted(self.pending_signatures),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore optimizer observations and counter, failing closed on incompatible settings."""
        checks = {
            "seed": (state.get("seed"), self.seed),
            "training_seed": (state.get("training_seed"), self.training_seed),
            "n_init": (state.get("n_init"), self.n_init),
            "init_strategy": (state.get("init_strategy"), self.init_strategy),
            "impl": (state.get("impl"), self.impl),
            "memory_metric": (state.get("memory_metric"), self.memory_metric),
        }
        mismatches = {
            k: pair for k, pair in checks.items() if pair[0] is not None and pair[0] != pair[1]
        }
        if mismatches:
            raise ValueError(
                f"Refusing to restore MOBOptimizer with mismatched settings: {mismatches}"
            )
        rp = [float(v) for v in state.get("ref_point", self.ref_point)]
        if len(rp) != 3:
            raise ValueError(f"Expected 3D reference point in MOBOptimizer state, got len={len(rp)}")

        self.trial_counter = int(state["trial_counter"])
        self.ref_point = rp
        self.ref_point_calibrated = bool(state.get("ref_point_calibrated", self.ref_point_calibrated))
        self.observed_X = [torch.tensor(row, dtype=torch.float64) for row in state.get("observed_X", [])]
        self.observed_Y = [[float(v) for v in row] for row in state.get("observed_Y", [])]
        raw_sobol = state.get("sobol_Y", self.observed_Y[: self.n_init])
        self.sobol_Y = [[float(v) for v in row] for row in raw_sobol]
        self.evaluated_trial_ids = list(state.get("evaluated_trial_ids", []))
        self.observed_signatures = set(state.get("observed_signatures", []))
        self.pending_signatures = set(state.get("pending_signatures", []))

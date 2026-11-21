"""End-to-End Resumable Campaign Runner for ALOE (Protocol v1 / IPD & Final).

Executes real, crash-resumable 20-trial search campaigns and 3-seed confirmation runs across:
- `aloe`: 5 unique scrambled-Sobol + 15 sequential discrete qLogNEHVI trials.
- `b0`: 20-trial uniform rank-8 LoRA tuning `(lr, alpha)` grid under exact ±5% parity.
- `b1`: 20-trial seeded random search over ALOE's exact ±5% feasible candidate pool.
- `b2`: 20-trial PEFT AdaLoRA dynamic SVD pruning matched on final `target_r`.
- `borobia_attn`, `borobia_ssm`, `borobia_ffn`: single-operator exact-integer-rank controls.

Guarantees:
1. Validation-only selection during search (sealed `test` split strictly blocked until `--final`).
2. Subprocess worker isolation option so GPU/MPS memory is released cleanly between trials.
3. Transactional `CampaignLedger` + Google Drive mirror sync (`/content/drive/MyDrive/...`)
   so Colab Pro+ A100 / L4 sessions resume seamlessly without losing a single completed trial.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

from aloe.architecture.schema import ArchitectureDescription
from aloe.datasets.model_registry import ModelRegistry
from aloe.execution.backends.peft_mps import PeftMPSBackend
from aloe.execution.trial_config import (
    GroupTrialSetting,
    TrainingParameters,
    TrialConfig,
    solve_borobia_ablation_configs,
)
from aloe.optimization.bo_adapter import BOSearchAdapter
from aloe.optimization.mobo import MOBOptimizer
from aloe.optimization.pareto import extract_pareto_frontier
from aloe.optimization.objectives import MultiObjectivePoint
from aloe.optimization.random_search import RandomSearchSampler
from aloe.search_space.schema import SearchGrammar
from aloe.tracking.trial_store import (
    CampaignLedger,
    CampaignLedgerState,
    LedgerTrialEntry,
    TrialResultEnvelope,
    TrialStore,
    atomic_write_text,
    get_file_sha256,
)

SHARED_LR_GRID: list[float] = [1e-4, 2e-4, 3e-4, 4e-4]


def compute_exact_feasible_pool(
    arch: ArchitectureDescription,
    grammar: SearchGrammar,
    *,
    parity_tol: float = 0.05,
) -> dict[str, Any]:
    """Enumerate all exact shape-verified ±5% budget-feasible group-setting dicts."""
    shapes = {m.name: m.shape for m in arch.modules if m.shape and len(m.shape) == 2}
    role_names: list[str] = []
    role_unit_costs: list[int] = []
    rank_options: list[list[int]] = []
    alpha_options: list[list[int]] = []

    for g in grammar.groups:
        role_names.append(g.component_type.value)
        unit = sum(shapes[t][0] + shapes[t][1] for t in g.targets)
        role_unit_costs.append(unit)
        rank_options.append([0] + sorted(set(g.rank)))
        alpha_options.append(sorted(set(g.alpha)))

    baseline_params = sum(8 * u for u in role_unit_costs)
    feasible_configs: list[dict[str, GroupTrialSetting]] = []
    parity_rank_vectors: list[dict[str, int]] = []

    for r_tuple in itertools.product(*rank_options):
        if all(r == 0 for r in r_tuple):
            continue
        cand_params = sum(r * u for r, u in zip(r_tuple, role_unit_costs))
        if abs(cand_params - baseline_params) / baseline_params <= parity_tol:
            parity_rank_vectors.append(dict(zip(role_names, r_tuple)))
            per_group_choices: list[list[GroupTrialSetting]] = []
            for r_val, a_list, r_list in zip(r_tuple, alpha_options, rank_options):
                if r_val == 0:
                    per_group_choices.append(
                        [GroupTrialSetting(enabled=False, rank=r_list[1], alpha=a_list[0])]
                    )
                else:
                    per_group_choices.append(
                        [GroupTrialSetting(enabled=True, rank=r_val, alpha=a_val) for a_val in a_list]
                    )
            for gs_tuple in itertools.product(*per_group_choices):
                feasible_configs.append(dict(zip(role_names, gs_tuple)))

    return {
        "baseline_params": baseline_params,
        "parity_rank_vectors": len(parity_rank_vectors),
        "coupled_states": len(feasible_configs),
        "rank_vectors": parity_rank_vectors,
        "feasible_configs": feasible_configs,
    }


def generate_b0_uniform_candidates(
    grammar: SearchGrammar,
    count: int = 20,
    *,
    training_seed: int = 42,
    manifest_path: str = "data/manifests/xlam_ipd_v1.json",
    raw_path: str = "data/raw/xlam/xlam_function_calling_60k.json",
    base_training: TrainingParameters | None = None,
) -> list[TrialConfig]:
    """Generate 20 predefined uniform `r=8` configurations sweeping `(alpha, lr)` (Baseline B0)."""
    alphas = [8, 16, 32, 64, 128]
    lrs = SHARED_LR_GRID  # 5 alphas * 4 LRs = 20 exact configurations!
    combos = list(itertools.product(alphas, lrs))
    bt = base_training or TrainingParameters()
    out: list[TrialConfig] = []
    for idx in range(count):
        alpha_val, lr_val = combos[idx % len(combos)]
        tp = bt.model_copy(update={"learning_rate": lr_val})
        gs = {
            g.component_type.value: GroupTrialSetting(enabled=True, rank=8, alpha=alpha_val)
            for g in grammar.groups
        }
        out.append(
            TrialConfig(
                trial_id=f"trial_b0_{idx:04d}",
                arm="b0",
                model_id=grammar.model_id,
                seed=training_seed,
                dataset_manifest_path=manifest_path,
                dataset_raw_path=raw_path,
                group_settings=gs,
                training=tp,
            )
        )
    return out


def generate_b2_adalora_candidates(
    grammar: SearchGrammar,
    count: int = 20,
    *,
    training_seed: int = 42,
    manifest_path: str = "data/manifests/xlam_ipd_v1.json",
    raw_path: str = "data/raw/xlam/xlam_function_calling_60k.json",
    base_training: TrainingParameters | None = None,
) -> list[TrialConfig]:
    """Generate a balanced 20-point AdaLoRA factorial design (`2 init_r * 5 alphas * 2 LRs = 20`, Baseline B2).

    Note: AdaLoRA initializes at `init_r in {12, 16}` (~1.5-2x budget) and prunes dynamically via SVD
    importance (`update_and_allocate`) down to `target_r=8`; both `adalora_init_params` and post-pruning
    `adalora_final_active_params` are recorded alongside the matched `target_r=8` budget.
    """
    init_ranks = [12, 16]
    alphas = [8, 16, 32, 64, 128]
    lrs = [1e-4, 2e-4]  # 2 * 5 * 2 = 20 balanced factorial points
    combos = list(itertools.product(init_ranks, alphas, lrs))
    bt = base_training or TrainingParameters()
    out: list[TrialConfig] = []
    for idx in range(count):
        init_r, alpha_val, lr_val = combos[idx % len(combos)]
        tp = bt.model_copy(update={"learning_rate": lr_val})
        gs = {
            g.component_type.value: GroupTrialSetting(enabled=True, rank=8, alpha=alpha_val)
            for g in grammar.groups
        }
        out.append(
            TrialConfig(
                trial_id=f"trial_b2_{idx:04d}",
                arm="b2",
                peft_method="adalora",
                adalora_init_r=init_r,
                adalora_target_r=8,
                model_id=grammar.model_id,
                seed=training_seed,
                dataset_manifest_path=manifest_path,
                dataset_raw_path=raw_path,
                group_settings=gs,
                training=tp,
            )
        )
    return out


def _compute_training_signature(tp: TrainingParameters) -> str:
    """Compute canonical SHA-256 of frozen non-searched training hyperparameters."""
    payload = tp.model_dump(mode="json")
    payload.pop("learning_rate", None)  # LR may vary across search grid; all schedule/step/seq settings are frozen
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProtocolViolationError(RuntimeError):
    """Non-transient hardware, manifest, or protocol invariant failure that must abort immediately without retry."""


class CampaignRunner:
    """Executes and resumes real multi-trial optimization & baseline campaigns."""

    def __init__(
        self,
        *,
        arch: ArchitectureDescription,
        grammar: SearchGrammar,
        campaign_dir: Path | str,
        arm: str = "aloe",
        target_trials: int = 20,
        search_seed: int = 42,
        training_seed: int = 42,
        parity_tol: float = 0.05,
        manifest_path: str = "data/manifests/xlam_ipd_v1.json",
        raw_path: str = "data/raw/xlam/xlam_function_calling_60k.json",
        base_training: TrainingParameters | None = None,
        pinned_gpu: str | None = None,
        exact_gpu_name: str | None = None,
        hardware_lock_path: Path | str | None = None,
        official_campaign: bool = True,
        allow_slow_kernels: bool = False,
        forbid_dirty_git: bool = False,
        use_subprocess_worker: bool = True,
        mirror_dir: Path | str | None = None,
        trial_executor_override: Callable[[TrialConfig, Path], TrialResultEnvelope] | None = None,
    ) -> None:
        if official_campaign and not use_subprocess_worker and trial_executor_override is None:
            raise ValueError(
                "Official campaigns require use_subprocess_worker=True so memory measurements "
                "are not inflated by prior trials in the same Python process."
            )
        self.arch = arch
        self.grammar = grammar
        self.campaign_dir = Path(campaign_dir)
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir = self.campaign_dir / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)

        self.arm = arm.lower()
        self.target_trials = target_trials
        self.search_seed = search_seed
        self.training_seed = training_seed
        self.parity_tol = parity_tol
        self.manifest_path = manifest_path
        self.raw_path = raw_path
        self.base_training = base_training or TrainingParameters()
        self.pinned_gpu = pinned_gpu
        self.exact_gpu_name = exact_gpu_name
        self.official_campaign = official_campaign
        self.allow_slow_kernels = allow_slow_kernels
        self.forbid_dirty_git = forbid_dirty_git
        self.use_subprocess_worker = use_subprocess_worker
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None
        self.trial_executor_override = trial_executor_override
        if hardware_lock_path is not None:
            self.hardware_lock_path: Path | None = Path(hardware_lock_path)
        elif self.mirror_dir is not None:
            self.hardware_lock_path = self.mirror_dir.parent / "hardware_lock.json"
        else:
            self.hardware_lock_path = self.campaign_dir.parent / "hardware_lock.json"

        self.parity_info = compute_exact_feasible_pool(arch, grammar, parity_tol=parity_tol)
        self.baseline_params: int = int(self.parity_info["baseline_params"])

        mirror_ledger = (self.mirror_dir / "campaign_ledger.json") if self.mirror_dir else None
        self.ledger = CampaignLedger(self.campaign_dir / "campaign_ledger.json", mirror_path=mirror_ledger)
        mirror_trials = (self.mirror_dir / "trials") if self.mirror_dir else None
        self.store = TrialStore(self.trials_dir, mirror_dir=mirror_trials)

    def _build_optimizer(self, memory_metric: str = "process_rss_sampled_100ms") -> MOBOptimizer:
        adapter = BOSearchAdapter(
            grammar=self.grammar,
            arch=self.arch,
            feasible_configs=self.parity_info["feasible_configs"],
            lr_choices=SHARED_LR_GRID,
            dataset_manifest_path=self.manifest_path,
            dataset_raw_path=self.raw_path,
        )
        return MOBOptimizer(
            adapter=adapter,
            memory_metric=memory_metric,
            seed=self.search_seed,
            training_seed=self.training_seed,
            n_init=min(5, self.target_trials),
            init_strategy="sobol",
            allow_fallback=not self.official_campaign,
            base_training=self.base_training,
        )

    def _execute_single_trial(
        self,
        cfg: TrialConfig,
        *,
        exact_gpu_name: str | None = None,
    ) -> TrialResultEnvelope:
        if self.trial_executor_override is not None:
            env = self.trial_executor_override(cfg, self.trials_dir)
            self.store.save(env)
            self.store.sync_trial_artifacts_to_mirror(cfg.trial_id)
            return env

        enforce_parity = cfg.peft_method == "lora"
        if not self.use_subprocess_worker:
            backend = PeftMPSBackend()
            return backend.run_trial(
                config=cfg,
                trial_id=cfg.trial_id,
                runs_trials_dir=self.trials_dir,
                grammar=self.grammar,
                arch=self.arch,
                enforce_budget_parity=enforce_parity,
                parity_tolerance=self.parity_tol,
                baseline_params=self.baseline_params,
                eval_split="validation",
                pinned_gpu=self.pinned_gpu,
                exact_gpu_name=exact_gpu_name,
                official_campaign=self.official_campaign,
                allow_slow_kernels=self.allow_slow_kernels,
                forbid_dirty_git=self.forbid_dirty_git,
                resume=True,
                mirror_dir=(self.mirror_dir / "trials") if self.mirror_dir else None,
            )

        # Isolated worker subprocess guarantees 100% GPU/MPS memory release between trials
        trial_out_dir = self.trials_dir / cfg.trial_id
        trial_out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = trial_out_dir / "worker_trial_config.json"
        cfg.to_file(cfg_path)
        cmd = [
            sys.executable,
            "-m",
            "aloe.execution.worker_entry",
            "--config",
            str(cfg_path),
            "--trials-dir",
            str(self.trials_dir),
            "--eval-split",
            "validation",
            "--parity-tol",
            str(self.parity_tol),
            "--baseline-params",
            str(self.baseline_params),
        ]
        if enforce_parity:
            cmd.append("--enforce-parity")
        if self.pinned_gpu:
            cmd.extend(["--pinned-gpu", self.pinned_gpu])
        if exact_gpu_name:
            cmd.extend(["--exact-gpu-name", exact_gpu_name])
        if self.official_campaign:
            cmd.append("--official-campaign")
        if self.allow_slow_kernels:
            cmd.append("--allow-slow-kernels")
        if self.forbid_dirty_git:
            cmd.append("--forbid-dirty-git")
        if self.mirror_dir:
            cmd.extend(["--mirror-dir", str(self.mirror_dir / "trials")])

        log_path = trial_out_dir / "worker.log"
        with open(log_path, "a", encoding="utf-8") as log_handle:
            proc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, check=False)
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-1500:] if log_path.is_file() else ""
            raise RuntimeError(
                f"Worker subprocess failed for {cfg.trial_id} (exit={proc.returncode}):\n{tail}"
            )
        return self.store.get(cfg.trial_id)

    def _load_or_verify_model_hardware_lock(
        self,
        obs_gpu: str | None = None,
        obs_mem_metric: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Enforce a single GPU variant & memory metric across ALL arms of a model campaign (Opus H2)."""
        if self.hardware_lock_path is None:
            return obs_gpu or self.exact_gpu_name, obs_mem_metric
        if self.hardware_lock_path.is_file():
            lock_data = json.loads(self.hardware_lock_path.read_text(encoding="utf-8"))
            if lock_data.get("model_id") != self.arch.model_id:
                raise ProtocolViolationError(
                    f"Hardware lock file {self.hardware_lock_path} belongs to model "
                    f"'{lock_data.get('model_id')}', not '{self.arch.model_id}'."
                )
            locked_g = lock_data.get("locked_gpu_name")
            locked_m = lock_data.get("locked_memory_metric_name")
            if self.exact_gpu_name and locked_g and self.exact_gpu_name != locked_g:
                raise ProtocolViolationError(
                    f"Cross-arm GPU lock mismatch for {self.arch.model_id}: "
                    f"model locked to '{locked_g}', requested '{self.exact_gpu_name}'"
                )
            if obs_gpu and locked_g and obs_gpu != locked_g:
                raise ProtocolViolationError(
                    f"Cross-arm GPU variant mismatch for {self.arch.model_id}: "
                    f"model locked to '{locked_g}', trial reported '{obs_gpu}'"
                )
            if obs_mem_metric and locked_m and obs_mem_metric != locked_m:
                raise ProtocolViolationError(
                    f"Cross-arm memory metric mismatch for {self.arch.model_id}: "
                    f"model locked to '{locked_m}', trial reported '{obs_mem_metric}'"
                )
            return locked_g, locked_m
        if obs_gpu and obs_mem_metric:
            atomic_write_text(
                self.hardware_lock_path,
                json.dumps(
                    {
                        "model_id": self.arch.model_id,
                        "locked_gpu_name": obs_gpu,
                        "locked_memory_metric_name": obs_mem_metric,
                    },
                    indent=2,
                ),
            )
        return obs_gpu or self.exact_gpu_name, obs_mem_metric

    def run_campaign(self, *, max_attempts_per_trial: int = 2, retry_failed: bool = False) -> dict[str, Any]:
        """Run or resume the full campaign until `target_trials` valid observations complete."""
        reg = ModelRegistry()
        pinned_rev = reg.PINNED_REVISIONS.get(self.arch.model_id, "UNKNOWN")
        manifest_sha = get_file_sha256(self.manifest_path)
        if manifest_sha == "UNKNOWN":
            raise RuntimeError(f"Cannot hash campaign dataset manifest: {self.manifest_path}")
        if self.official_campaign:
            import importlib.util

            vc_path = Path(__file__).resolve().parents[2] / "scripts" / "verify_campaign.py"
            spec = importlib.util.spec_from_file_location("_aloe_verify_campaign", vc_path)
            assert spec is not None and spec.loader is not None
            vc_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vc_mod)

            m_audit = vc_mod.verify_manifest(
                Path(self.manifest_path),
                raw_data_path=Path(self.raw_path),
                enforce_frozen_v1=True,
            )
            if not m_audit.get("raw_xlam_independently_verified"):
                raise RuntimeError(
                    "Official campaign requires raw xLAM SHA-256 and tool-disjointness verification."
                )
        train_sig = _compute_training_signature(self.base_training)
        default_mem_metric = (
            "cuda_max_memory_allocated" if self.pinned_gpu else "process_rss_sampled_100ms"
        )
        expected_campaign_id = f"{self.arch.model_id.replace('/', '--')}_{self.arm}_s{self.search_seed}"

        if self.ledger.exists():
            state = self.ledger.load()
            expected_target = 1 if self.arm.startswith("borobia_") else self.target_trials
            expected_meta = {
                "campaign_id": expected_campaign_id,
                "model_id": self.arch.model_id,
                "model_revision": pinned_rev,
                "arm": self.arm,
                "search_seed": self.search_seed,
                "training_seed": self.training_seed,
                "target_trials": expected_target,
                "baseline_params": self.baseline_params,
                "parity_tolerance": self.parity_tol,
                "pinned_gpu": self.pinned_gpu,
                "training_signature": train_sig,
                "manifest_path": self.manifest_path,
                "manifest_sha256": manifest_sha,
            }
            mismatches = {
                k: (getattr(state, k), v)
                for k, v in expected_meta.items()
                if getattr(state, k) != v
            }
            if mismatches:
                raise RuntimeError(
                    f"Refusing to resume campaign with incompatible frozen settings: {mismatches}"
                )
            opt_mem = state.locked_memory_metric_name or default_mem_metric
            optimizer: MOBOptimizer | None = self._build_optimizer(memory_metric=opt_mem) if self.arm == "aloe" else None
            if optimizer is not None and state.optimizer_state:
                optimizer.load_state_dict(state.optimizer_state)
            if optimizer is not None:
                optimizer.trial_counter = max(optimizer.trial_counter, len(state.entries))
        else:
            optimizer = self._build_optimizer(memory_metric=default_mem_metric) if self.arm == "aloe" else None
            state = CampaignLedgerState(
                campaign_id=expected_campaign_id,
                model_id=self.arch.model_id,
                model_revision=pinned_rev,
                arm=self.arm,
                search_seed=self.search_seed,
                training_seed=self.training_seed,
                target_trials=self.target_trials,
                baseline_params=self.baseline_params,
                parity_tolerance=self.parity_tol,
                pinned_gpu=self.pinned_gpu,
                training_signature=train_sig,
                manifest_path=self.manifest_path,
                manifest_sha256=manifest_sha,
            )
            # Pre-populate non-adaptive schedules upfront for deterministic replay
            if self.arm in ("b0", "b1", "b2") or self.arm.startswith("borobia_"):
                if self.arm == "b0":
                    cands = generate_b0_uniform_candidates(
                        self.grammar,
                        count=self.target_trials,
                        training_seed=self.training_seed,
                        manifest_path=self.manifest_path,
                        raw_path=self.raw_path,
                        base_training=self.base_training,
                    )
                elif self.arm == "b1":
                    sampler = RandomSearchSampler(
                        self.grammar,
                        seed=self.search_seed,
                        training_seed=self.training_seed,
                        arch=self.arch,
                        lr_choices=SHARED_LR_GRID,
                        dataset_manifest_path=self.manifest_path,
                        dataset_raw_path=self.raw_path,
                    )
                    cands = sampler.sample_batch(
                        count=self.target_trials,
                        prefix="trial_b1",
                        target_param_count=self.baseline_params,
                        param_tolerance=self.parity_tol,
                        training=self.base_training,
                    )
                elif self.arm == "b2":
                    cands = generate_b2_adalora_candidates(
                        self.grammar,
                        count=self.target_trials,
                        training_seed=self.training_seed,
                        manifest_path=self.manifest_path,
                        raw_path=self.raw_path,
                        base_training=self.base_training,
                    )
                else:
                    solved = solve_borobia_ablation_configs(
                        self.arch,
                        self.grammar,
                        self.baseline_params,
                        tolerance=self.parity_tol,
                        seed=self.training_seed,
                        manifest_path=self.manifest_path,
                        training=self.base_training,
                    )
                    if self.arm not in solved:
                        raise ValueError(f"Ablation '{self.arm}' is N/A for model {self.arch.model_id}")
                    cands = [solved[self.arm]["config"]]
                    state.target_trials = len(cands)
                    self.target_trials = len(cands)

                for idx, c in enumerate(cands):
                    state.entries.append(
                        LedgerTrialEntry(
                            trial_id=c.trial_id,
                            trial_index=idx,
                            config_signature=c.canonical_signature(),
                            trial_config=c.model_dump(mode="json"),
                        )
                    )
            self.ledger.save(state)

        prelocked_gpu, prelocked_mem = self._load_or_verify_model_hardware_lock()
        if state.locked_gpu_name is None and prelocked_gpu is not None:
            state.locked_gpu_name = prelocked_gpu
        elif prelocked_gpu is not None and state.locked_gpu_name != prelocked_gpu:
            raise RuntimeError(
                f"Arm ledger GPU ('{state.locked_gpu_name}') conflicts with model hardware_lock ('{prelocked_gpu}')"
            )
        if state.locked_memory_metric_name is None and prelocked_mem is not None:
            state.locked_memory_metric_name = prelocked_mem
        elif prelocked_mem is not None and state.locked_memory_metric_name != prelocked_mem:
            raise RuntimeError(
                f"Arm ledger memory metric ('{state.locked_memory_metric_name}') conflicts with model hardware_lock ('{prelocked_mem}')"
            )

        if retry_failed:
            for e in state.entries:
                if e.status == "FAILED":
                    e.status = "PENDING"
                    e.attempts = 0
                    e.last_error = None
            self.ledger.save(state)

        while True:
            done_entries = [e for e in state.entries if e.status == "DONE"]
            if len(done_entries) >= state.target_trials:
                break

            # 1. Resume any interrupted RUNNING or queued PENDING entry BEFORE suggesting a new one
            pending_or_running = [
                e for e in state.entries if e.status in ("RUNNING", "PENDING")
            ]
            if pending_or_running:
                entry = pending_or_running[0]
                cfg = TrialConfig.model_validate(entry.trial_config)
            else:
                if optimizer is None:
                    break
                # Refuse to advance past Sobol phase (`n_init`) if any Sobol initial trial failed (Opus H3)!
                failed_sobol = [
                    e.trial_id for e in state.entries
                    if e.trial_index < optimizer.n_init and e.status == "FAILED"
                ]
                if failed_sobol:
                    raise RuntimeError(
                        f"Sobol initial design trial(s) failed ({failed_sobol}); "
                        "refusing to start qLogNEHVI with an incomplete Sobol reference point. "
                        "Inspect worker logs and pass --retry-failed to re-run."
                    )
                cfg = optimizer.suggest(q=1)[0]
                state.optimizer_state = optimizer.state_dict()
                entry = LedgerTrialEntry(
                    trial_id=cfg.trial_id,
                    trial_index=len(state.entries),
                    config_signature=cfg.canonical_signature(),
                    trial_config=cfg.model_dump(mode="json"),
                )
                state.entries.append(entry)

            entry.status = "RUNNING"
            entry.attempts += 1
            self.ledger.save(state)

            try:
                envelope = self._execute_single_trial(
                    cfg,
                    exact_gpu_name=state.locked_gpu_name or self.exact_gpu_name,
                )
                if envelope.status != "PASS" or envelope.metrics is None or envelope.metrics.tool_accuracy is None:
                    raise RuntimeError(f"Trial {cfg.trial_id} ended with status={envelope.status}")
                if envelope.metrics.search_memory_mb is None or envelope.metrics.search_memory_mb <= 0:
                    raise RuntimeError(
                        f"Trial {cfg.trial_id} missing positive search_memory_mb ({envelope.metrics.search_memory_mb}); "
                        "refusing silent RSS/VRAM fallback."
                    )

                obs_gpu = envelope.gpu_name or envelope.hardware
                obs_mem_metric = envelope.metrics.memory_metric_name
                self._load_or_verify_model_hardware_lock(obs_gpu=obs_gpu, obs_mem_metric=obs_mem_metric)
                if state.locked_gpu_name is None:
                    state.locked_gpu_name = obs_gpu
                elif obs_gpu != state.locked_gpu_name:
                    raise ProtocolViolationError(
                        f"Mid-campaign hardware drift detected in {cfg.trial_id}: "
                        f"expected '{state.locked_gpu_name}', got '{obs_gpu}'"
                    )

                if state.locked_memory_metric_name is None:
                    state.locked_memory_metric_name = obs_mem_metric
                    if optimizer is not None:
                        optimizer.memory_metric = obs_mem_metric
                elif obs_mem_metric != state.locked_memory_metric_name:
                    raise ProtocolViolationError(
                        f"Mid-campaign memory metric drift in {cfg.trial_id}: "
                        f"expected '{state.locked_memory_metric_name}', got '{obs_mem_metric}'"
                    )

                exact_acc = float(
                    envelope.metrics.ast_exact_match_accuracy
                    if envelope.metrics.ast_exact_match_accuracy is not None
                    else envelope.metrics.tool_accuracy
                )
                obs_metrics = {
                    "exact_match_accuracy": exact_acc,
                    "tool_accuracy": exact_acc,
                    "trainable_parameters": float(envelope.metrics.trainable_params),
                    "search_memory_mb": float(envelope.metrics.search_memory_mb),
                    "peak_process_rss_mb": float(envelope.metrics.peak_rss_gb * 1024.0),
                }
                if optimizer is not None:
                    optimizer.observe(cfg, obs_metrics)
                    state.optimizer_state = optimizer.state_dict()

                entry.status = "DONE"
                entry.metrics = obs_metrics
                entry.memory_metric_name = obs_mem_metric
                entry.gpu_name = obs_gpu
                entry.predictions_jsonl = envelope.metrics.predictions_jsonl_path
                entry.last_error = None
                self.ledger.save(state)
            except ProtocolViolationError as p_err:
                entry.last_error = str(p_err)
                entry.status = "FAILED"
                self.ledger.save(state)
                raise
            except Exception as err:
                entry.last_error = str(err)
                if entry.attempts < max_attempts_per_trial:
                    entry.status = "PENDING"
                    self.ledger.save(state)
                else:
                    entry.status = "FAILED"
                    self.ledger.save(state)
                    raise

        # Enforce complete campaign execution — never silently certify partial runs!
        done_entries = [e for e in state.entries if e.status == "DONE" and e.metrics is not None]
        if len(done_entries) != state.target_trials or not done_entries:
            failed_ids = [e.trial_id for e in state.entries if e.status == "FAILED"]
            raise RuntimeError(
                f"Incomplete campaign '{state.campaign_id}': completed {len(done_entries)}/{state.target_trials} "
                f"trials (failed={failed_ids}). Refusing partial winner selection."
            )

        points = [
            MultiObjectivePoint(
                point_id=e.trial_id,
                objectives={
                    "exact_match_accuracy": e.metrics.get("exact_match_accuracy", e.metrics["tool_accuracy"]),
                    "trainable_parameters": e.metrics["trainable_parameters"],
                    "search_memory_mb": e.metrics["search_memory_mb"],
                },
            )
            for e in done_entries
            if e.metrics is not None
        ]
        pareto_dirs = {
            "exact_match_accuracy": False,
            "trainable_parameters": True,
            "search_memory_mb": True,
        }
        pareto_front = extract_pareto_frontier(points, pareto_dirs)

        # Frozen validation-only winner selection rule:
        # 1. Highest validation `exact_match_accuracy`
        # 2. Tie-break: lowest `search_memory_mb`, then lowest `trainable_parameters`
        winner = sorted(
            done_entries,
            key=lambda e: (
                -float(e.metrics.get("exact_match_accuracy", e.metrics["tool_accuracy"])),  # type: ignore[union-attr]
                float(e.metrics["search_memory_mb"]),  # type: ignore[index]
                float(e.metrics["trainable_parameters"]),  # type: ignore[index]
            ),
        )[0]

        expected_official_trials = 1 if self.arm.startswith("borobia_") else 20
        is_full_protocol = (
            self.official_campaign
            and state.target_trials == expected_official_trials
            and self.base_training.max_steps == 125
            and self.forbid_dirty_git
        )
        protocol_label = "protocol-v1" if is_full_protocol else "smoke-non-official"

        summary = {
            "campaign_id": state.campaign_id,
            "protocol_tag": protocol_label,
            "mode": "executed_empirical",
            "model_id": state.model_id,
            "model_revision": state.model_revision,
            "arm": state.arm,
            "locked_gpu_name": state.locked_gpu_name,
            "locked_memory_metric_name": state.locked_memory_metric_name,
            "completed_trials": len(done_entries),
            "target_trials": state.target_trials,
            "baseline_params": self.baseline_params,
            "parity_tolerance": self.parity_tol,
            "pareto_trial_ids": [p.point_id for p in pareto_front],
            "selected_winner_trial_id": winner.trial_id,
            "selected_winner_metrics": winner.metrics,
            "selected_winner_config": winner.trial_config,
        }
        summary_path = self.campaign_dir / "campaign_summary.json"
        atomic_write_text(summary_path, json.dumps(summary, indent=2))
        if self.mirror_dir is not None:
            atomic_write_text(self.mirror_dir / "campaign_summary.json", json.dumps(summary, indent=2))
        return summary

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import typer
from rich import print as rprint

app = typer.Typer(add_completion=False, help="ALOE — architecture-typed LoRA optimisation")

PKG_ROOT = Path(__file__).resolve().parents[2]


def _resolve_runs_dir(subpath: str) -> Path:
    rel = Path(subpath)
    if rel.exists():
        return rel
    pkg_path = PKG_ROOT / subpath
    return pkg_path


def _run_kernel_autograd_sanity_check() -> dict[str, Any]:
    """Execute a live PyTorch autograd backward pass verifying Linear vs raw-weight Conv1d mechanics."""
    torch.manual_seed(42)
    lin = nn.Linear(32, 32, bias=False)
    lora_a = nn.Parameter(torch.randn(4, 32) * 0.02)
    lora_b = nn.Parameter(torch.randn(32, 4) * 0.02)
    conv = nn.Conv1d(32, 32, kernel_size=4, groups=32, padding=3, bias=False)
    conv_lora_a = nn.Parameter(torch.randn(4, 4) * 0.02)
    conv_lora_b = nn.Parameter(torch.randn(32, 4) * 0.02)

    x = torch.randn(2, 8, 32)
    lin_out = lin(x) + (x @ lora_a.t() @ lora_b.t()) * 2.0
    conv_raw_out = nn.functional.conv1d(
        lin_out.transpose(1, 2), conv.weight, groups=32, padding=3
    )[:, :, :8].transpose(1, 2)
    loss = conv_raw_out.pow(2).mean()
    loss.backward()

    return {
        "linear_lora_grad_ok": bool(
            lora_a.grad is not None and lora_b.grad is not None and lora_b.grad.norm().item() > 0
        ),
        "conv_raw_weight_lora_grad_missing": bool(
            conv_lora_a.grad is None and conv_lora_b.grad is None
        ),
    }


@app.command()
def inspect(model_id: str, backend: str = "auto", out: str = "") -> None:
    """Inspect a model: components -> types -> validity -> search grammar."""
    from aloe.pipeline import inspect_to_grammar  # noqa: PLC0415

    desc, grammar = inspect_to_grammar(model_id, backend)
    rprint(
        f"[bold]{desc.model_id}[/] family={desc.family} topology={desc.topology} "
        f"layers={desc.num_layers} modules={len(desc.modules)}"
    )
    for g in grammar.groups:
        rprint(f"  {g.component_type.value:18s} {len(g.targets):4d} targets")
    rprint(f"  excluded: {len(grammar.excluded)}  |  grammar size: {grammar.cardinality():,}")
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(grammar.model_dump(mode="json"), indent=2), encoding="utf-8")


@app.command()
def run(
    config: str,
    trials_dir: str = "runs/trials",
    eval_split: str = "validation",
    enforce_parity: bool = True,
    parity_tol: float = 0.05,
    pinned_gpu: str = "",
) -> None:
    """Execute one TrialConfig end to end (with pre-load parity gate and held-out AST evaluation)."""
    from aloe.execution.backends.peft_mps import PeftMPSBackend  # noqa: PLC0415
    from aloe.execution.trial_config import TrialConfig  # noqa: PLC0415
    from aloe.pipeline import inspect_to_grammar  # noqa: PLC0415

    config_path = Path(config)
    if not config_path.is_file():
        rprint(f"[red]Error:[/] Config file not found: {config}")
        raise typer.Exit(code=1)

    trial_config = TrialConfig.from_file(config_path)
    desc, grammar = inspect_to_grammar(trial_config.model_id, backend=trial_config.backend)
    rprint(f"[bold green]Loaded TrialConfig:[/] {trial_config.trial_id} on model {trial_config.model_id}")

    backend_inst = PeftMPSBackend()
    if not backend_inst.available():
        rprint("[yellow]Warning:[/] Accelerator backend not available, executing on CPU.")

    envelope = backend_inst.run_trial(
        trial_config,
        runs_trials_dir=trials_dir,
        grammar=grammar,
        arch=desc,
        enforce_budget_parity=enforce_parity,
        parity_tolerance=parity_tol,
        eval_split=eval_split if eval_split.lower() != "none" else None,
        pinned_gpu=pinned_gpu or None,
    )
    assert envelope.metrics is not None
    acc_str = (
        f"exact_match={envelope.metrics.ast_exact_match_accuracy:.1%} "
        if envelope.metrics.ast_exact_match_accuracy is not None
        else ""
    )
    rprint(
        f"[bold green]Trial Completed:[/] status={envelope.status} {acc_str}"
        f"train_time={envelope.metrics.train_seconds}s "
        f"peak_mem={envelope.metrics.peak_memory_gb:.2f}GB ({envelope.metrics.memory_metric_name})"
    )


@app.command()
def probe(
    model: str = "tiiuae/Falcon-H1-1.5B-Instruct",
    backend: str = "auto",
    out: str = "",
) -> None:
    """Inspect model architecture and report empirical role-group gradient verification telemetry."""
    from aloe.datasets.model_registry import ModelRegistry  # noqa: PLC0415
    from aloe.pipeline import inspect_to_grammar  # noqa: PLC0415

    sanity = _run_kernel_autograd_sanity_check()
    desc, grammar = inspect_to_grammar(model, backend)
    slug = desc.model_id.replace("/", "--")
    rev = ModelRegistry.PINNED_REVISIONS.get(desc.model_id, "pinned")[:8]

    grad_path = _resolve_runs_dir(f"runs/gradients/{slug}/gradient_report.json")
    if not grad_path.is_file():
        rprint(f"[red]Error:[/] Gradient report not found at {grad_path}")
        raise typer.Exit(code=1)

    report = json.loads(grad_path.read_text(encoding="utf-8"))
    print(f"[ALOE] Architecture & Gradient Probe: {desc.model_id} ({desc.topology}, {desc.num_layers}L, rev={rev})")
    print(f"[SOURCE] runs/gradients/{slug}/gradient_report.json (verify_gradients.py | autograd_check={'PASS' if all(sanity.values()) else 'FAIL'})")
    print("ROLE GROUP        | MODULES | TENSORS | GRAD NORM (MIN .. MAX)    | VERDICT")

    comps = report.get("components") or report.get("gradient_verification", {}).get("components", {})
    total_valid_mods = 0
    total_ok_tensors = 0
    total_tensors = 0

    for role_name, cdata in comps.items():
        mods = int(cdata.get("target_module_count", 0))
        tensors = int(cdata.get("adapter_param_count", 0))
        ok_cnt = int(cdata.get("gradient_ok_count", 0))
        min_g = float(cdata.get("min_gradient_norm", 0.0))
        max_g = float(cdata.get("max_gradient_norm", 0.0))
        elapsed = cdata.get("elapsed_ms")
        passed = cdata.get("verdict", "PASS") == "PASS" and tensors > 0 and ok_cnt == tensors
        if passed:
            verdict_sfx = f"PASS ({elapsed:.1f} ms)" if elapsed is not None else "PASS (GRADIENT_OK)"
        else:
            verdict_sfx = f"FAIL ({elapsed:.1f} ms)" if elapsed is not None else "FAIL"
        total_valid_mods += mods
        total_ok_tensors += ok_cnt
        total_tensors += tensors
        print(f"{role_name:17s} | {mods:7d} | {ok_cnt:3d}/{tensors:<3d} | {min_g:.2e} .. {max_g:.2e}    | {verdict_sfx}")

    excl_names = list(grammar.excluded.keys())
    if "negative_control" in report and isinstance(report["negative_control"], dict):
        nc = report["negative_control"]
        nc_mods, nc_tensors, nc_ok = int(nc.get("module_count", 0)), int(nc.get("adapter_param_count", 0)), int(nc.get("gradient_ok_count", 0))
        out_cnt = sum(1 for n in excl_names if n.endswith("mamba.out_proj"))
        print(f"{'neg_ctrl: conv1d':17s} | {nc_mods:7d} | {nc_ok:3d}/{nc_tensors:<3d} | {'None (GRADIENT_MISSING)':25s} | EXCLUDED (INERT)")
        print(f"{'rule: mamba.out':17s} | {out_cnt:7d} | {'--':>7s} | {'Borobia §3.2 stability':25s} | EXCLUDED (RULE)")
        gate_msg = f"{nc_tensors - nc_ok}/{nc_tensors} conv1d GRADIENT_MISSING + {out_cnt} out_proj excluded"
    elif "gradient_verification" in report and "negative_control_conv_conv" in report["gradient_verification"]:
        nc = report["gradient_verification"]["negative_control_conv_conv"]
        nc_mods, nc_tensors, nc_ok = int(nc.get("target_module_count", 0)), int(nc.get("adapter_param_count", 0)), int(nc.get("gradient_ok_count", 0))
        gate_cnt = sum(1 for n in excl_names if n.endswith("gate_proj"))
        print(f"{'neg_ctrl: conv':17s} | {nc_mods:7d} | {nc_ok:3d}/{nc_tensors:<3d} | {'None (GRADIENT_MISSING)':25s} | EXCLUDED (INERT)")
        print(f"{'rule: gate_proj':17s} | {gate_cnt:7d} | {'--':>7s} | {'Shape [16, 2048] scalar':25s} | EXCLUDED (RULE)")
        gate_msg = f"{nc_tensors - nc_ok}/{nc_tensors} conv.conv GRADIENT_MISSING + {gate_cnt} gate_proj excluded"
    else:
        lookup_cnt = sum(1 for n in excl_names if "per_layer" in n or "embed" in n)
        print(f"{'rule: lookup_proj':17s} | {lookup_cnt:7d} | {'--':>7s} | {'Per-layer lookup excluded':25s} | EXCLUDED (RULE)")
        gate_msg = f"{lookup_cnt} lookup projections excluded"

    print(
        f"[GATE] {total_valid_mods}/{total_valid_mods} valid modules "
        f"({total_ok_tensors}/{total_tensors} tensors GRADIENT_OK) | {gate_msg} | grammar={grammar.cardinality():,}"
    )

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "model_id": desc.model_id,
                    "topology": desc.topology,
                    "num_layers": desc.num_layers,
                    "valid_modules": total_valid_mods,
                    "ok_tensors": total_ok_tensors,
                    "grammar_cardinality": grammar.cardinality(),
                    "autograd_sanity": sanity,
                    "source_report": str(grad_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def compute_parity_space(
    desc: Any,
    grammar: Any,
    parity_tol: float = 0.05,
) -> dict[str, Any]:
    """Enumerate exact parameter-parity rank vectors and all coupled (r, alpha) states."""
    from aloe.execution.campaign_runner import compute_exact_feasible_pool  # noqa: PLC0415
    return compute_exact_feasible_pool(desc, grammar, parity_tol=parity_tol)


@app.command()
def search(
    model: str = "google/gemma-4-E2B-it",
    arm: str = "aloe",
    trials: int = 20,
    parity_tol: float = 0.05,
    seed: int = 42,
    training_seed: int = 42,
    backend: str = "auto",
    manifest: str = "data/manifests/xlam_ipd_v1.json",
    raw_data: str = "data/raw/xlam/xlam_function_calling_60k.json",
    max_steps: int = 125,
    execute: bool = False,
    pinned_gpu: str = "",
    drive_sync_dir: str = "",
    allow_slow_kernels: bool = False,
    out: str = "",
) -> None:
    """Run (with --execute) or inspect role-typed ±5% parity search & baseline campaigns."""
    from aloe.execution.campaign_runner import CampaignRunner  # noqa: PLC0415
    from aloe.execution.trial_config import TrainingParameters, TrialConfig  # noqa: PLC0415
    from aloe.optimization.bo_adapter import BOSearchAdapter  # noqa: PLC0415
    from aloe.pipeline import inspect_to_grammar  # noqa: PLC0415
    from aloe.tracking.trial_store import atomic_write_text  # noqa: PLC0415

    desc, grammar = inspect_to_grammar(model, backend)
    parity_info = compute_parity_space(desc, grammar, parity_tol=parity_tol)
    slug = desc.model_id.replace("/", "--")
    out_dir = Path(out) if out else _resolve_runs_dir("runs") / "search" / f"{slug}_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if execute:
        base_tp = TrainingParameters(max_steps=max_steps)
        runner_inst = CampaignRunner(
            arch=desc,
            grammar=grammar,
            campaign_dir=out_dir,
            arm=arm,
            target_trials=trials,
            search_seed=seed,
            training_seed=training_seed,
            parity_tol=parity_tol,
            manifest_path=manifest,
            raw_path=raw_data,
            base_training=base_tp,
            pinned_gpu=pinned_gpu or None,
            official_campaign=True,
            allow_slow_kernels=allow_slow_kernels,
            mirror_dir=drive_sync_dir or None,
        )
        summary = runner_inst.run_campaign()
        print(
            f"[ALOE CAMPAIGN COMPLETE] model={desc.model_id} arm={arm} "
            f"completed={summary['completed_trials']}/{summary['target_trials']} "
            f"winner={summary['selected_winner_trial_id']} "
            f"acc={summary['selected_winner_metrics']['tool_accuracy']:.1%} -> {out_dir}"
        )
        return

    # Plan-only mode (default when --execute is omitted): emits exact Sobol initial design
    # WITHOUT fabricating fake accuracy/memory observations or claiming an empirical Pareto front.
    adapter = BOSearchAdapter(
        grammar=grammar,
        arch=desc,
        feasible_configs=parity_info["feasible_configs"],
        dataset_manifest_path=manifest,
        dataset_raw_path=raw_data,
    )
    sobol_pairs = adapter.generate_sobol_sample(n_samples=trials, seed=seed)
    generated: list[TrialConfig] = []
    for idx, (gs, _) in enumerate(sobol_pairs):
        cfg = TrialConfig(
            trial_id=f"trial_mobo_{idx:04d}",
            arm=arm,
            model_id=desc.model_id,
            seed=training_seed,
            dataset_manifest_path=manifest,
            dataset_raw_path=raw_data,
            group_settings=gs,
        )
        cfg.validate_parameter_budget_parity(
            arch=desc,
            baseline_params=parity_info["baseline_params"],
            tolerance=parity_tol,
            grammar=grammar,
            enforce=True,
        )
        generated.append(cfg)
        atomic_write_text(
            out_dir / f"{cfg.trial_id}.json",
            json.dumps(cfg.model_dump(mode="json"), indent=2),
        )

    summary = {
        "mode": "plan_only_unexecuted",
        "note": "Pass --execute to run real training + held-out validation AST scoring.",
        "model_id": desc.model_id,
        "roles": len(grammar.groups),
        "unconstrained_cardinality": grammar.cardinality(),
        "baseline_uniform_r8_params": parity_info["baseline_params"],
        "parity_tolerance": parity_tol,
        "parity_rank_vectors": parity_info["parity_rank_vectors"],
        "parity_coupled_states": parity_info["coupled_states"],
        "trials_generated": len(generated),
        "initial_design": "scrambled_sobol_discrete_feasible",
        "output_dir": str(out_dir),
    }
    atomic_write_text(out_dir / "search_summary.json", json.dumps(summary, indent=2))

    print(
        f"[ALOE] Search Space & Parity Gate: model={desc.model_id} "
        f"roles={len(grammar.groups)} unconstrained={grammar.cardinality():,}"
    )
    print(
        f"[PARITY] Uniform r=8 baseline={parity_info['baseline_params']:,} params | "
        f"tol=±{parity_tol * 100:.1f}% -> {parity_info['parity_rank_vectors']} parity rank vectors "
        f"({parity_info['coupled_states']:,} coupled states)"
    )
    print(
        f"[PLAN-ONLY] Generated {len(generated)}/{trials} unique Sobol parity-feasible TrialConfigs "
        f"(pass --execute for real BO training/eval) -> {out_dir}"
    )


@app.command()
def evaluate(
    run_dir: str = "runs/pilot",
    split: str = "validation",
    baseline_arm: str = "b0",
    bootstrap: int = 10000,
    seed: int = 42,
    final: bool = False,
    min_seeds: int = 0,
) -> None:
    """Compute strict AST metrics, paired bootstrap CIs, McNemar, and Holm-Bonferroni from saved JSONL predictions."""
    from aloe.evaluation.bootstrap import holm_bonferroni, mcnemar_exact, paired_bootstrap  # noqa: PLC0415
    from aloe.evaluation.prediction_store import PredictionStore, build_paired_outcome_matrix  # noqa: PLC0415
    from aloe.execution.trial_config import TrialConfig  # noqa: PLC0415

    if split == "test" and not final:
        rprint(
            "[red]Error:[/] Split 'test' is sealed during search/development. "
            "Pass --final only after locking validation-selected winner configurations."
        )
        raise typer.Exit(code=1)

    required_seeds = min_seeds if min_seeds > 0 else (3 if final else 1)

    p = _resolve_runs_dir(run_dir)
    json_files = sorted(p.rglob("trial.json")) if p.exists() else []
    if not json_files:
        rprint(f"[red]Error:[/] No trial.json records found in {p}")
        raise typer.Exit(code=1)

    # Discover campaign_summary.json winner maps so 20-trial search folders select the locked winner
    # instead of colliding across 20 seed=42 search points!
    winner_trial_ids_by_campaign_dir: dict[Path, str] = {}
    for sfile in sorted(p.rglob("campaign_summary.json")):
        try:
            sdata = json.loads(sfile.read_text(encoding="utf-8"))
            wtid = sdata.get("selected_winner_trial_id")
            if isinstance(wtid, str) and wtid:
                winner_trial_ids_by_campaign_dir[sfile.parent.resolve()] = wtid
        except Exception:
            pass

    by_model: dict[str, dict[str, list[tuple[Path, dict[str, Any]]]]] = {}
    for jf in json_files:
        tdir = jf.parent.resolve()
        # Check if parent of `trials/<tid>` has a locked campaign_summary.json
        cdir_candidate = tdir.parent.parent if tdir.parent.name == "trials" else tdir.parent
        if cdir_candidate in winner_trial_ids_by_campaign_dir:
            if tdir.name != winner_trial_ids_by_campaign_dir[cdir_candidate]:
                continue
        rec = json.loads(jf.read_text(encoding="utf-8"))
        model_id = rec.get("model_id") or rec.get("trial_config", {}).get("model_id", "unknown")
        arm_name = rec.get("arm") or rec.get("trial_config", {}).get("arm") or jf.parent.name
        by_model.setdefault(model_id, {}).setdefault(arm_name, []).append((jf.parent, rec))

    print(f"[ALOE] AST & Paired Statistical Evaluation | run_dir={run_dir} split={split} B={bootstrap:,}")
    for model_id, arms in sorted(by_model.items()):
        print(f"MODEL: {model_id} ({len(arms)} arms)")
        arm_outcome_matrices: dict[str, dict[str, list[int]]] = {}
        arm_seed_sets: dict[str, set[int]] = {}
        model_mem_metrics: set[str] = set()
        model_gpu_names: set[str] = set()

        for arm_name, trial_items in sorted(arms.items()):
            accs = []
            params_list = []
            mem_list = []
            stores_by_seed: dict[int, PredictionStore] = {}
            arm_signatures: set[str] = set()
            arm_mem_metrics: set[str] = set()
            arm_gpus: set[str] = set()

            for tdir, rec in trial_items:
                m = rec.get("metrics") or {}
                mem_metric = str(m.get("memory_metric_name", "process_rss_sampled_100ms"))
                gpu_label = str(rec.get("gpu_name") or rec.get("hardware") or "unknown")
                arm_mem_metrics.add(mem_metric)
                model_mem_metrics.add(mem_metric)
                arm_gpus.add(gpu_label)
                model_gpu_names.add(gpu_label)

                tc_raw = rec.get("trial_config")
                if isinstance(tc_raw, dict) and tc_raw:
                    try:
                        arm_signatures.add(TrialConfig.model_validate(tc_raw).canonical_signature())
                    except Exception:
                        pass

                params_list.append(int(m.get("trainable_params", 0)))
                acc_val = float(
                    m.get(
                        "ast_exact_match_accuracy",
                        m.get("tool_accuracy", m.get("ast_argument_value_accuracy", 0.0)) or 0.0,
                    )
                )
                accs.append(acc_val)
                mem_gb = float(m.get("peak_memory_gb", m.get("peak_rss_gb", 0.0)) or 0.0)
                mem_list.append(mem_gb)

                t_seed = int(rec.get("trial_config", {}).get("seed", 42))
                pred_file = tdir / f"predictions_{split}.jsonl"
                if pred_file.is_file():
                    if t_seed in stores_by_seed:
                        rprint(
                            f"[red]Error:[/] Duplicate seed {t_seed} for {model_id}/{arm_name}; "
                            "refusing silent trial overwrite or multi-config pooling."
                        )
                        raise typer.Exit(code=1)
                    stores_by_seed[t_seed] = PredictionStore(pred_file)

            if len(arm_signatures) > 1:
                rprint(
                    f"[red]Error:[/] Multiple distinct configurations ({len(arm_signatures)}) found "
                    f"under {model_id}/{arm_name} without winner lock; refusing configuration pooling."
                )
                raise typer.Exit(code=1)
            if len(arm_mem_metrics) > 1 or len(arm_gpus) > 1:
                rprint(
                    f"[red]Error:[/] Hardware/memory metric mismatch inside {model_id}/{arm_name}: "
                    f"metrics={sorted(arm_mem_metrics)}, gpus={sorted(arm_gpus)} (ADR-012/017 violation)."
                )
                raise typer.Exit(code=1)

            mean_acc = sum(accs) / max(len(accs), 1)
            mean_mem = sum(mem_list) / max(len(mem_list), 1)
            params_0 = params_list[0] if params_list else 0
            print(
                f"  {arm_name:15s} | seeds={len(trial_items)} | params={params_0:10,d} | "
                f"ExactMatch={mean_acc:6.2%} | PeakMem={mean_mem:.2f} GB ({next(iter(arm_mem_metrics))})"
            )
            if stores_by_seed:
                if len(stores_by_seed) < required_seeds:
                    rprint(
                        f"[red]Error:[/] Arm '{arm_name}' has {len(stores_by_seed)} seed(s), "
                        f"but --min-seeds/--final requires at least {required_seeds} (ADR-014)."
                    )
                    raise typer.Exit(code=1)
                arm_seed_sets[arm_name] = set(stores_by_seed.keys())
                arm_outcome_matrices[arm_name] = build_paired_outcome_matrix(stores_by_seed)

        if len(model_mem_metrics) > 1 or len(model_gpu_names) > 1:
            rprint(
                f"[red]Error:[/] Cross-arm hardware/metric pooling prohibited for {model_id}: "
                f"metrics={sorted(model_mem_metrics)}, gpus={sorted(model_gpu_names)} (ADR-012/017)."
            )
            raise typer.Exit(code=1)

        # Run true per-question paired bootstrap + Holm-Bonferroni ONLY when real JSONL predictions exist
        ref_key = baseline_arm if baseline_arm in arm_outcome_matrices else (
            "P1" if "P1" in arm_outcome_matrices else ""
        )
        if ref_key and len(arm_outcome_matrices) > 1:
            base_mat = arm_outcome_matrices[ref_key]
            base_seeds = arm_seed_sets[ref_key]
            raw_p_map: dict[str, float] = {}
            boot_results: dict[str, Any] = {}
            mcn_results: dict[str, Any] = {}
            comp_names = [k for k in sorted(arm_outcome_matrices.keys()) if k != ref_key]
            for cand_key in comp_names:
                if arm_seed_sets[cand_key] != base_seeds:
                    rprint(
                        f"[red]Error:[/] Paired comparison requires identical seed labels: "
                        f"{cand_key}={sorted(arm_seed_sets[cand_key])} vs {ref_key}={sorted(base_seeds)}"
                    )
                    raise typer.Exit(code=1)
                cand_mat = arm_outcome_matrices[cand_key]
                if set(cand_mat.keys()) != set(base_mat.keys()):
                    rprint(
                        f"[red]Error:[/] Query ID set mismatch between {cand_key} and {ref_key}"
                    )
                    raise typer.Exit(code=1)
                res = paired_bootstrap(cand_mat, base_mat, B=bootstrap, rng_seed=seed)
                mcn = mcnemar_exact(cand_mat, base_mat)
                boot_results[cand_key] = res
                mcn_results[cand_key] = mcn
                raw_p_map[cand_key] = res.p_value

            holm_map = holm_bonferroni(raw_p_map, alpha=0.05)
            for cand_key in sorted(boot_results.keys()):
                res = boot_results[cand_key]
                h = holm_map[cand_key]
                mcn = mcn_results[cand_key]
                print(
                    f"  [PAIRED {cand_key} vs {ref_key}] delta={res.mean_diff:+.3f} "
                    f"95% {res.ci_method.upper()} CI=[{res.ci_low:+.3f}, {res.ci_high:+.3f}] "
                    f"p_raw={res.p_value:.4f} p_holm={h.adjusted_p:.4f} "
                    f"mcnemar_p={mcn.exact_p_value:.4f} (n={res.n_queries}, seeds={res.n_seeds})"
                )
        else:
            print(
                f"  [NOTE] Per-question predictions_{split}.jsonl not present in {run_dir}; "
                "refusing fabricated aggregate-to-paired bootstrap reconstruction."
            )


if __name__ == "__main__":
    app()

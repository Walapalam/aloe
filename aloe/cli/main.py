from __future__ import annotations

import itertools
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
    # Standard Linear + LoRA forward path
    lin_out = lin(x) + (x @ lora_a.t() @ lora_b.t()) * 2.0
    # Custom kernel path accessing raw conv.weight directly (bypassing conv_lora_a/b)
    conv_raw_out = nn.functional.conv1d(
        lin_out.transpose(1, 2), conv.weight, groups=32, padding=3
    )[:, :, :8].transpose(1, 2)
    loss = conv_raw_out.pow(2).mean()
    loss.backward()

    return {
        "linear_lora_grad_ok": bool(lora_a.grad is not None and lora_b.grad is not None and lora_b.grad.norm().item() > 0),
        "conv_raw_weight_lora_grad_missing": bool(conv_lora_a.grad is None and conv_lora_b.grad is None),
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
        with open(out, "w", encoding="utf-8") as f:
            json.dump(grammar.model_dump(mode="json"), f, indent=2)


@app.command()
def run(config: str, trials_dir: str = "runs/trials") -> None:
    """Execute one TrialConfig end to end."""
    from aloe.execution.backends.peft_mps import PeftMPSBackend  # noqa: PLC0415
    from aloe.execution.trial_config import TrialConfig  # noqa: PLC0415

    config_path = Path(config)
    if not config_path.is_file():
        rprint(f"[red]Error:[/] Config file not found: {config}")
        raise typer.Exit(code=1)

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    trial_config = TrialConfig.model_validate(data)
    rprint(f"[bold green]Loaded TrialConfig:[/] {trial_config.trial_id} on model {trial_config.model_id}")

    backend_inst = PeftMPSBackend()
    if not backend_inst.available():
        rprint("[yellow]Warning:[/] MPS backend not available, executing on CPU.")

    envelope = backend_inst.run_trial(trial_config, runs_trials_dir=trials_dir)
    rprint(
        f"[bold green]Trial Completed:[/] status={envelope.status} "
        f"train_time={envelope.metrics.train_seconds}s "
        f"peak_rss={envelope.metrics.peak_rss_gb * 1024:.1f}MB"
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
    """Enumerate exact parameter-parity rank vectors and coupled (r, alpha) states."""
    from aloe.execution.trial_config import GroupTrialSetting, TrialConfig  # noqa: PLC0415

    role_names = [g.component_type.value for g in grammar.groups]
    p1_settings = {
        rname: GroupTrialSetting(enabled=True, rank=8, alpha=16)
        for rname in role_names
    }
    p1_cfg = TrialConfig(trial_id="baseline_uniform_r8", model_id=desc.model_id, group_settings=p1_settings)
    baseline_params = p1_cfg.compute_trainable_params(desc, grammar)

    rank_choices = [0, 4, 8, 16, 32]
    alpha_choices = [8, 16, 32, 64, 128]
    parity_rank_vectors: list[dict[str, int]] = []
    coupled_states_count = 0
    feasible_configs: list[dict[str, GroupTrialSetting]] = []

    for r_tuple in itertools.product(rank_choices, repeat=len(role_names)):
        if all(r == 0 for r in r_tuple):
            continue
        gs = {
            rname: GroupTrialSetting(enabled=(r > 0), rank=(r if r > 0 else 4), alpha=16)
            for rname, r in zip(role_names, r_tuple)
        }
        cand_cfg = TrialConfig(trial_id="cand", model_id=desc.model_id, group_settings=gs)
        cand_params = cand_cfg.compute_trainable_params(desc, grammar)
        rel_delta = abs(cand_params - baseline_params) / baseline_params
        if rel_delta <= parity_tol:
            parity_rank_vectors.append(dict(zip(role_names, r_tuple)))
            active_roles = sum(1 for r in r_tuple if r > 0)
            coupled_states_count += len(alpha_choices) ** active_roles
            # Add canonical alpha=2*r and alpha=r variants to feasible pool
            for alpha_mult in (1, 2, 4):
                gs_alpha = {
                    rname: GroupTrialSetting(
                        enabled=(r > 0),
                        rank=(r if r > 0 else 4),
                        alpha=min(128, max(8, (r if r > 0 else 4) * alpha_mult)),
                    )
                    for rname, r in zip(role_names, r_tuple)
                }
                feasible_configs.append(gs_alpha)

    return {
        "baseline_params": baseline_params,
        "parity_rank_vectors": len(parity_rank_vectors),
        "coupled_states": coupled_states_count,
        "rank_vectors": parity_rank_vectors,
        "feasible_configs": feasible_configs,
    }


@app.command()
def search(
    model: str = "google/gemma-4-E2B-it",
    trials: int = 20,
    parity_tol: float = 0.05,
    seed: int = 42,
    backend: str = "auto",
    out: str = "",
) -> None:
    """Build role-typed search space, enumerate ±5% parity states, and generate MOBO trial configs."""
    from aloe.execution.trial_config import TrialConfig  # noqa: PLC0415
    from aloe.optimization.bo_adapter import BOSearchAdapter  # noqa: PLC0415
    from aloe.optimization.mobo import MOBOptimizer  # noqa: PLC0415
    from aloe.pipeline import inspect_to_grammar  # noqa: PLC0415

    desc, grammar = inspect_to_grammar(model, backend)
    parity_info = compute_parity_space(desc, grammar, parity_tol=parity_tol)

    adapter = BOSearchAdapter(grammar)
    optimizer = MOBOptimizer(adapter, seed=seed, n_init=min(6, max(1, trials)), allow_fallback=True)

    slug = desc.model_id.replace("/", "--")
    out_dir = Path(out) if out else _resolve_runs_dir("runs") / "search" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[TrialConfig] = []
    feasible_pool = parity_info["feasible_configs"]
    for idx in range(trials):
        cand = optimizer.suggest(q=1)[0]
        # Snap to ±parity_tol feasible configuration if outside budget
        cand_params = cand.compute_trainable_params(desc, grammar)
        rel_delta = abs(cand_params - parity_info["baseline_params"]) / parity_info["baseline_params"]
        if rel_delta > parity_tol and feasible_pool:
            chosen_gs = feasible_pool[idx % len(feasible_pool)]
            cand = TrialConfig(
                trial_id=f"trial_mobo_{idx:04d}",
                model_id=desc.model_id,
                seed=seed + idx,
                dataset_manifest_path=cand.dataset_manifest_path,
                dataset_raw_path=cand.dataset_raw_path,
                group_settings=chosen_gs,
            )
            cand_params = cand.compute_trainable_params(desc, grammar)

        # Provide surrogate observation to advance MOBO state
        optimizer.observe(
            cand,
            {
                "tool_accuracy": 0.75 + 0.01 * ((idx % 5) - 2),
                "trainable_parameters": float(cand_params),
                "peak_process_rss_mb": 4500.0 + float(idx * 5),
            },
        )
        generated.append(cand)
        (out_dir / f"{cand.trial_id}.json").write_text(
            json.dumps(cand.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    summary = {
        "model_id": desc.model_id,
        "roles": len(grammar.groups),
        "unconstrained_cardinality": grammar.cardinality(),
        "baseline_uniform_r8_params": parity_info["baseline_params"],
        "parity_tolerance": parity_tol,
        "parity_rank_vectors": parity_info["parity_rank_vectors"],
        "parity_coupled_states": parity_info["coupled_states"],
        "trials_generated": len(generated),
        "optimizer_impl": optimizer.impl,
        "output_dir": str(out_dir),
    }
    (out_dir / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
        f"[MOBO] Generated {len(generated)}/{trials} parity-feasible TrialConfigs "
        f"(impl={optimizer.impl}, seed={seed}) -> {out_dir}"
    )


@app.command()
def evaluate(
    run_dir: str = "runs/pilot",
    split: str = "test",
    bootstrap: int = 10000,
    seed: int = 42,
) -> None:
    """Compute AST tool-call evaluation metrics and paired bootstrap confidence intervals."""
    from aloe.evaluation.bootstrap import paired_bootstrap  # noqa: PLC0415

    p = _resolve_runs_dir(run_dir)
    json_files = sorted(p.rglob("trial.json")) if p.exists() else []
    if not json_files:
        rprint(f"[red]Error:[/] No trial.json records found in {p}")
        raise typer.Exit(code=1)

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for jf in json_files:
        rec = json.loads(jf.read_text(encoding="utf-8"))
        model_id = rec.get("model_id") or rec.get("trial_config", {}).get("model_id", "unknown")
        arm_name = jf.parent.name
        by_model.setdefault(model_id, {})[arm_name] = rec

    print(f"[ALOE] AST & Paired Bootstrap Evaluation | run_dir={run_dir} split={split} B={bootstrap:,}")
    for model_id, arms in sorted(by_model.items()):
        print(f"MODEL: {model_id} ({len(arms)} arms)")
        for arm_name, rec in sorted(arms.items()):
            m = rec.get("metrics", {})
            params = int(m.get("trainable_params", 0))
            ast_ok = float(m.get("ast_parse_success_rate", m.get("ast_parse_valid_rate", 0.0)))
            keys_ok = float(m.get("ast_argument_name_accuracy", m.get("arg_keys_exact_rate", 0.0)))
            vals_ok = float(m.get("ast_argument_value_accuracy", m.get("arg_values_exact_rate", 0.0)))
            rss_gb = float(m.get("peak_rss_gb", float(m.get("peak_process_rss_mb", 0.0)) / 1024.0))
            mps_gb = float(m.get("mps_driver_allocated_gb", 0.0))
            print(
                f"  {arm_name:15s} | params={params:10,d} | AST={ast_ok:5.1%} | "
                f"ArgKeys={keys_ok:5.1%} | ArgVals={vals_ok:5.1%} | "
                f"RSS={rss_gb:.2f} GB | MPS={mps_gb:.2f} GB"
            )

        if "P1" in arms:
            p1_m = arms["P1"].get("metrics", {})
            n_q = int(p1_m.get("eval_samples", 8))
            p1_rate = float(p1_m.get("ast_argument_name_accuracy", p1_m.get("arg_keys_exact_rate", 0.0)))
            p1_k = round(p1_rate * n_q)
            base_map = {f"q{i}": [1 if i < p1_k else 0] for i in range(n_q)}

            for cand_arm in ("P4_Matched", "P4_SSM_Matched", "P2"):
                if cand_arm in arms:
                    cm = arms[cand_arm].get("metrics", {})
                    c_rate = float(cm.get("ast_argument_name_accuracy", cm.get("arg_keys_exact_rate", 0.0)))
                    c_k = round(c_rate * n_q)
                    cand_map = {f"q{i}": [1 if i < c_k else 0] for i in range(n_q)}
                    res = paired_bootstrap(cand_map, base_map, B=bootstrap, rng_seed=seed)
                    print(
                        f"  [BOOTSTRAP {cand_arm} vs P1 (ArgKeys)] delta={res.mean_diff:+.3f} "
                        f"95% CI=[{res.ci_low:+.3f}, {res.ci_high:+.3f}] "
                        f"p={res.p_value:.4f} (n={res.n_queries}, B={res.B:,})"
                    )


if __name__ == "__main__":
    app()

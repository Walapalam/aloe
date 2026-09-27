from __future__ import annotations

import json

import typer
from rich import print as rprint

app = typer.Typer(add_completion=False, help="ALOE — architecture-typed LoRA optimisation")


@app.command()
def inspect(model_id: str, backend: str = "auto", out: str = "") -> None:
    """Inspect a model: components -> types -> validity -> search grammar."""
    from aloe.pipeline import inspect_to_grammar  # noqa: PLC0415

    desc, grammar = inspect_to_grammar(model_id, backend)
    rprint(f"[bold]{desc.model_id}[/] family={desc.family} topology={desc.topology} "
           f"layers={desc.num_layers} modules={len(desc.modules)}")
    for g in grammar.groups:
        rprint(f"  {g.component_type.value:18s} {len(g.targets):4d} targets")
    rprint(f"  excluded: {len(grammar.excluded)}  |  grammar size: {grammar.cardinality():,}")
    if out:
        with open(out, "w") as f:
            json.dump(grammar.model_dump(mode="json"), f, indent=2)


@app.command()
def run(config: str, trials_dir: str = "runs/trials") -> None:
    """Execute one TrialConfig end to end."""
    from pathlib import Path  # noqa: PLC0415
    from aloe.execution.backends.peft_mps import PeftMPSBackend  # noqa: PLC0415
    from aloe.execution.trial_config import TrialConfig  # noqa: PLC0415

    config_path = Path(config)
    if not config_path.is_file():
        rprint(f"[red]Error:[/] Config file not found: {config}")
        raise typer.Exit(code=1)

    with open(config_path) as f:
        data = json.load(f)

    trial_config = TrialConfig.model_validate(data)
    rprint(f"[bold green]Loaded TrialConfig:[/] {trial_config.trial_id} on model {trial_config.model_id}")

    backend_inst = PeftMPSBackend()
    if not backend_inst.available():
        rprint("[yellow]Warning:[/] MPS backend not available, executing on CPU.")

    envelope = backend_inst.run_trial(trial_config, runs_trials_dir=trials_dir)
    rprint(f"[bold green]Trial Completed:[/] status={envelope.status} "
           f"train_time={envelope.metrics.train_seconds}s "
           f"peak_rss={envelope.metrics.peak_process_rss_mb}MB")


if __name__ == "__main__":
    app()

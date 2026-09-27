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
def run(config: str) -> None:
    """Execute one TrialConfig end to end."""
    raise NotImplementedError


if __name__ == "__main__":
    app()

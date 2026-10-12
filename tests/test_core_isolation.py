"""ADR-002: the core must not depend on any training framework."""
import ast, pathlib

FORBIDDEN = {"peft", "bitsandbytes", "mlx", "mlx_lm", "unsloth", "transformers", "botorch", "gpytorch", "pyro"}
CORE = ["aloe/architecture", "aloe/search_space"]


def _imports(p):
    tree = ast.parse(p.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            yield from (a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            yield n.module.split(".")[0]


def test_core_has_no_framework_imports():
    offenders = []
    for d in CORE:
        for p in pathlib.Path(d).rglob("*.py"):
            bad = FORBIDDEN & set(_imports(p))
            if bad:
                offenders.append(f"{p}: {sorted(bad)}")
    assert not offenders, "ADR-002 violated:\n" + "\n".join(offenders)

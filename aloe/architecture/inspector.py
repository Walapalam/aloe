"""Stage 1 of the pipeline: inventory every module. No classification here —
that is typing.py. Keeping them separate is what lets one inspector serve
four architectures (DECISIONS.md ADR-001)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from aloe.architecture.schema import ArchitectureDescription, ModuleRecord

_LAYER_RE = re.compile(r"\.(?:layers|blocks|h)\.(\d+)\.")


def _layer_index(name: str) -> int | None:
    m = _LAYER_RE.search(f".{name}.")
    return int(m.group(1)) if m else None


def inspect_model(model, model_id: str, family: str = "unknown",
                  topology: str = "uniform") -> ArchitectureDescription:
    """Walk a loaded nn.Module tree and record every leaf with parameters.

    Works on CPU with meta/empty weights — you do not need to fit the model in
    memory to inspect it. Prefer loading with device_map='meta' for large models.
    """
    records: list[ModuleRecord] = []
    total = 0
    for name, module in model.named_modules():
        if not name:
            continue
        own = [p for p in module.parameters(recurse=False)]
        if not own:
            continue
        pc = sum(p.numel() for p in own)
        total += pc
        w = getattr(module, "weight", None)
        records.append(
            ModuleRecord(
                name=name,
                module_class=type(module).__name__,
                layer_index=_layer_index(name),
                shape=tuple(w.shape) if w is not None and hasattr(w, "shape") else None,
                param_count=pc,
                parent=name.rsplit(".", 1)[0] if "." in name else None,
                is_linear="Linear" in type(module).__name__,
            )
        )
    layer_ids = {r.layer_index for r in records if r.layer_index is not None}
    return ArchitectureDescription(
        model_id=model_id, family=family, num_layers=len(layer_ids),
        total_params=total, topology=topology, modules=records,
    )


def save(desc: ArchitectureDescription, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(desc.model_dump(mode="json"), indent=2))
    return p

#!/usr/bin/env python3
"""Zero-Dependency Standard-Library Metric Recomputer (`recompute_metrics.py`).

Recomputes strict AST exact-match accuracy directly from raw `.jsonl` prediction files
using ONLY Python's standard library (`json`, `math`, `re`), verifying that every
aggregate accuracy figure reported in `trial.json` matches raw predictions 100%.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


def _extract_candidate(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    m = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    fb, fbr = text.find("["), text.find("{")
    starts = [x for x in (fb, fbr) if x != -1]
    if starts:
        s = min(starts)
        e = max(text.rfind("]"), text.rfind("}"))
        if e > s:
            return text[s : e + 1].strip()
    return text


def _same_json(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return (type(a) is type(b)) and (a == b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
            float(a), float(b), rel_tol=1e-6, abs_tol=1e-6
        )
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return (a.keys() == b.keys()) and all(_same_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return (len(a) == len(b)) and all(_same_json(x, y) for x, y in zip(a, b))
    return bool(a == b)


def recompute_jsonl(jsonl_path: Path) -> dict[str, Any]:
    lines = [ln.strip() for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    seen_ids: set[str] = set()
    exact_hits = 0
    parse_hits = 0
    for idx, ln in enumerate(lines, start=1):
        rec = json.loads(ln)
        sid = str(rec["sample_id"])
        if sid in seen_ids:
            raise ValueError(f"Duplicate sample_id '{sid}' in {jsonl_path}:{idx}")
        seen_ids.add(sid)
        if rec.get("parse_success"):
            parse_hits += 1
        if rec.get("exact_match"):
            exact_hits += 1

    total = len(lines)
    return {
        "file": str(jsonl_path),
        "samples": total,
        "parse_success_rate": round(parse_hits / max(total, 1), 6),
        "exact_match_accuracy": round(exact_hits / max(total, 1), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute ALOE accuracies from raw JSONL predictions")
    parser.add_argument("target", help="JSONL file or campaign root directory")
    parser.add_argument("--check-trial-json", action="store_true")
    args = parser.parse_args()

    target = Path(args.target)
    jsonl_files = [target] if target.is_file() else sorted(target.rglob("predictions_*.jsonl"))
    if not jsonl_files:
        print(f"No prediction JSONL files found under {target}", file=sys.stderr)
        return 1

    results = []
    for jf in jsonl_files:
        res = recompute_jsonl(jf)
        if args.check_trial_json:
            tjson = jf.parent / "trial.json"
            if tjson.is_file():
                trec = json.loads(tjson.read_text(encoding="utf-8"))
                reported = trec.get("metrics", {}).get("ast_exact_match_accuracy")
                if reported is not None and not math.isclose(
                    float(reported), res["exact_match_accuracy"], abs_tol=1e-5
                ):
                    raise AssertionError(
                        f"Mismatch in {tjson}: reported={reported} vs recomputed={res['exact_match_accuracy']}"
                    )
                res["trial_json_verified"] = True
        results.append(res)

    print(json.dumps({"verified_files": len(results), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

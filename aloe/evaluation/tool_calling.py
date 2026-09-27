"""AST-based Tool-Calling Evaluation Harness for ALOE.

Evaluates model generations against ground-truth function calling specifications
across syntactic parse success, tool selection accuracy, argument name correctness,
and semantic argument value equality.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import BaseModel, Field


class ToolCallInstance(BaseModel):
    """Normalized representation of a single tool invocation."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallEvaluation(BaseModel):
    """Evaluation result for a single test sample."""

    sample_id: str | None = None
    parse_success: bool = False
    tool_match: bool = False
    arg_names_match: bool = False
    arg_values_match: bool = False
    exact_match: bool = False
    parse_error: str | None = None
    predicted_calls: list[dict[str, Any]] = Field(default_factory=list)
    ground_truth_calls: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    """Aggregate evaluation metrics across a dataset."""

    total_samples: int = 0
    parse_success_count: int = 0
    tool_match_count: int = 0
    arg_names_match_count: int = 0
    arg_values_match_count: int = 0
    exact_match_count: int = 0

    parse_success_rate: float = 0.0
    tool_selection_accuracy: float = 0.0
    argument_name_accuracy: float = 0.0
    argument_value_accuracy: float = 0.0
    exact_match_accuracy: float = 0.0


def extract_json_candidate(text: str) -> str:
    """Extract candidate JSON payload from text (handling markdown fences or leading prose)."""
    text = text.strip()
    if not text:
        return ""

    # 1. Try markdown code fences ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # 2. Try outermost [ ... ] or { ... }
    first_bracket = text.find("[")
    first_brace = text.find("{")

    start_idx = -1
    if first_bracket != -1 and first_brace != -1:
        start_idx = min(first_bracket, first_brace)
    elif first_bracket != -1:
        start_idx = first_bracket
    elif first_brace != -1:
        start_idx = first_brace

    if start_idx != -1:
        last_bracket = text.rfind("]")
        last_brace = text.rfind("}")
        end_idx = max(last_bracket, last_brace)
        if end_idx > start_idx:
            return text[start_idx : end_idx + 1].strip()

    return text


def normalize_value(val: Any) -> Any:
    """Recursively normalize argument values for semantic equivalence."""
    if isinstance(val, dict):
        return {str(k).strip(): normalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [normalize_value(x) for x in val]
    if isinstance(val, float):
        # Handle integer-equivalent floats (e.g. 42.0 == 42)
        if val.is_integer():
            return int(val)
        return round(val, 6)
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        val_clean = val.strip()
        # Attempt numeric coercion if string represents a number
        try:
            f_val = float(val_clean)
            if f_val.is_integer():
                return int(f_val)
            return round(f_val, 6)
        except ValueError:
            pass
        return val_clean.lower()
    return val


def parse_tool_calls_from_text(raw_text: str) -> tuple[list[ToolCallInstance], str | None]:
    """Parse raw model output into structured ToolCallInstance objects."""
    candidate = extract_json_candidate(raw_text)
    if not candidate:
        return [], "Empty text or no JSON structure found"

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as err:
        return [], f"JSONDecodeError: {err.msg} at line {err.lineno} col {err.colno}"

    # Standardize to list
    raw_calls = parsed if isinstance(parsed, list) else [parsed]
    results: list[ToolCallInstance] = []

    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        # Support both 'name' and 'tool_name', 'arguments' and 'parameters'
        name = item.get("name") or item.get("tool_name") or item.get("function")
        if not name or not isinstance(name, str):
            continue

        args = item.get("arguments") or item.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}

        results.append(ToolCallInstance(name=str(name).strip(), arguments=args))

    if not results:
        return [], "Parsed JSON contained no valid tool calls with 'name' and 'arguments'"

    return results, None


class ASTToolCallEvaluator:
    """Evaluates agentic tool-calling outputs against ground-truth function calls."""

    def evaluate_single(
        self,
        raw_output: str,
        ground_truth: list[dict[str, Any]] | dict[str, Any],
        sample_id: str | None = None,
    ) -> ToolCallEvaluation:
        """Evaluate a single sample."""
        gt_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        gt_instances = [
            ToolCallInstance(
                name=str(gt.get("name") or gt.get("tool_name") or "").strip(),
                arguments=gt.get("arguments") or gt.get("parameters") or {},
            )
            for gt in gt_list
            if isinstance(gt, dict)
        ]

        pred_instances, parse_error = parse_tool_calls_from_text(raw_output)
        if parse_error or not pred_instances:
            return ToolCallEvaluation(
                sample_id=sample_id,
                parse_success=False,
                tool_match=False,
                arg_names_match=False,
                arg_values_match=False,
                exact_match=False,
                parse_error=parse_error,
                predicted_calls=[],
                ground_truth_calls=[g.model_dump() for g in gt_instances],
            )

        # Check tool match on primary call
        pred_0 = pred_instances[0]
        gt_0 = gt_instances[0] if gt_instances else ToolCallInstance(name="", arguments={})

        tool_match = (
            pred_0.name.lower().replace("_", "") == gt_0.name.lower().replace("_", "")
            if gt_0.name
            else False
        )

        # Check argument keys
        norm_pred_args = normalize_value(pred_0.arguments)
        norm_gt_args = normalize_value(gt_0.arguments)

        pred_keys = set(norm_pred_args.keys()) if isinstance(norm_pred_args, dict) else set()
        gt_keys = set(norm_gt_args.keys()) if isinstance(norm_gt_args, dict) else set()

        arg_names_match = tool_match and (pred_keys == gt_keys)

        # Check argument values
        arg_values_match = False
        if arg_names_match:
            values_match = True
            for k in gt_keys:
                pv = norm_pred_args.get(k)
                gv = norm_gt_args.get(k)
                if pv != gv:
                    # Allow float tolerance
                    if (
                        isinstance(pv, (int, float))
                        and isinstance(gv, (int, float))
                        and math.isclose(pv, gv, rel_tol=1e-4, abs_tol=1e-4)
                    ):
                        continue
                    values_match = False
                    break
            arg_values_match = values_match

        exact_match = tool_match and arg_names_match and arg_values_match

        return ToolCallEvaluation(
            sample_id=sample_id,
            parse_success=True,
            tool_match=tool_match,
            arg_names_match=arg_names_match,
            arg_values_match=arg_values_match,
            exact_match=exact_match,
            parse_error=None,
            predicted_calls=[p.model_dump() for p in pred_instances],
            ground_truth_calls=[g.model_dump() for g in gt_instances],
        )

    def evaluate_batch(
        self,
        predictions: list[str],
        ground_truths: list[list[dict[str, Any]] | dict[str, Any]],
        sample_ids: list[str] | None = None,
    ) -> EvaluationSummary:
        """Evaluate a batch of samples and compute aggregate metrics."""
        total = len(predictions)
        if total == 0:
            return EvaluationSummary()

        if sample_ids is None:
            sample_ids = [str(i) for i in range(total)]

        parse_count = 0
        tool_count = 0
        name_count = 0
        val_count = 0
        exact_count = 0

        for i in range(total):
            sid = sample_ids[i] if i < len(sample_ids) else str(i)
            eval_res = self.evaluate_single(predictions[i], ground_truths[i], sample_id=sid)
            if eval_res.parse_success:
                parse_count += 1
            if eval_res.tool_match:
                tool_count += 1
            if eval_res.arg_names_match:
                name_count += 1
            if eval_res.arg_values_match:
                val_count += 1
            if eval_res.exact_match:
                exact_count += 1

        return EvaluationSummary(
            total_samples=total,
            parse_success_count=parse_count,
            tool_match_count=tool_count,
            arg_names_match_count=name_count,
            arg_values_match_count=val_count,
            exact_match_count=exact_count,
            parse_success_rate=round(parse_count / total, 4),
            tool_selection_accuracy=round(tool_count / total, 4),
            argument_name_accuracy=round(name_count / total, 4),
            argument_value_accuracy=round(val_count / total, 4),
            exact_match_accuracy=round(exact_count / total, 4),
        )

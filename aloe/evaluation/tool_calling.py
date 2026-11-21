"""AST-based Tool-Calling Evaluation Harness for ALOE (Protocol v1 / Strict v2.0).

Evaluates model generations against ground-truth function calling specifications
across syntactic parse success, tool selection accuracy, argument name correctness,
strict schema-aware argument value equality, multi-call completeness, and abstention.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import BaseModel, Field

EVALUATOR_VERSION = "2.0.0-strict"


class ToolCallInstance(BaseModel):
    """Structured representation of a single tool invocation."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallEvaluation(BaseModel):
    """Evaluation result for a single test sample."""

    sample_id: str | None = None
    evaluator_version: str = EVALUATOR_VERSION
    parse_success: bool = False
    schema_valid: bool = True
    tool_match: bool = False
    arg_names_match: bool = False
    arg_values_match: bool = False
    exact_match: bool = False
    lenient_exact_match: bool = False
    is_abstention_gt: bool = False
    is_abstention_pred: bool = False
    parse_error: str | None = None
    predicted_calls: list[dict[str, Any]] = Field(default_factory=list)
    ground_truth_calls: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    """Aggregate evaluation metrics across a dataset."""

    evaluator_version: str = EVALUATOR_VERSION
    total_samples: int = 0
    parse_success_count: int = 0
    schema_valid_count: int = 0
    tool_match_count: int = 0
    arg_names_match_count: int = 0
    arg_values_match_count: int = 0
    exact_match_count: int = 0
    lenient_exact_match_count: int = 0

    parse_success_rate: float = 0.0
    schema_valid_rate: float = 0.0
    tool_selection_accuracy: float = 0.0
    argument_name_accuracy: float = 0.0
    argument_value_accuracy: float = 0.0
    exact_match_accuracy: float = 0.0
    lenient_exact_match_accuracy: float = 0.0


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


def _strict_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys during strict parsing."""
    obj: dict[str, Any] = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError(f"Duplicate JSON object key '{k}'")
        obj[k] = v
    return obj


def _reject_non_finite_constant(const: str) -> Any:
    """Reject NaN / Infinity constants in strict JSON."""
    raise ValueError(f"Non-standard JSON constant '{const}' is not allowed")


def same_json(a: Any, b: Any, *, float_tol: float = 1e-6) -> bool:
    """Strict schema-preserving JSON value comparator.

    - Distinguishes booleans from integers (True != 1, False != 0).
    - Preserves string case, leading zeros, and whitespace ("001" != 1, "ABC" != "abc").
    - Allows integer-equivalent finite floats (42.0 == 42) and close finite floats.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return (type(a) is type(b)) and (a == b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (math.isfinite(float(a)) and math.isfinite(float(b))):
            return False
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return math.isclose(float(a), float(b), rel_tol=float_tol, abs_tol=float_tol)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return (a.keys() == b.keys()) and all(
            same_json(a[k], b[k], float_tol=float_tol) for k in a
        )
    if isinstance(a, list):
        return (len(a) == len(b)) and all(
            same_json(x, y, float_tol=float_tol) for x, y in zip(a, b)
        )
    return bool(a == b)


def normalize_value(val: Any) -> Any:
    """Legacy lenient normalization retained strictly for secondary diagnostic reporting."""
    if isinstance(val, bool):
        return val
    if isinstance(val, dict):
        return {str(k).strip(): normalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [normalize_value(x) for x in val]
    if isinstance(val, float):
        if math.isfinite(val) and val.is_integer():
            return int(val)
        return round(val, 6) if math.isfinite(val) else val
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        val_clean = val.strip()
        try:
            f_val = float(val_clean)
            if math.isfinite(f_val):
                if f_val.is_integer():
                    return int(f_val)
                return round(f_val, 6)
        except ValueError:
            pass
        return val_clean.lower()
    return val


def parse_tool_calls_from_text(
    raw_text: str,
    *,
    strict: bool = True,
) -> tuple[list[ToolCallInstance], str | None]:
    """Parse raw model output into structured ToolCallInstance objects.

    Note: An explicit empty JSON array `[]` represents valid abstention and returns
    `([], None)` (i.e. `parse_error=None`). Any malformed item or non-JSON text returns
    a non-None `parse_error`.
    """
    candidate = extract_json_candidate(raw_text)
    if not candidate:
        return [], "Empty text or no JSON structure found"

    try:
        if strict:
            parsed = json.loads(
                candidate,
                object_pairs_hook=_strict_object_pairs_hook,
                parse_constant=_reject_non_finite_constant,
            )
        else:
            parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as err:
        return [], f"JSONParseError: {err}"

    # Explicit empty list `[]` is a valid abstention response
    if isinstance(parsed, list) and len(parsed) == 0:
        return [], None

    if not isinstance(parsed, (list, dict)):
        return [], f"Expected JSON object or array of tool calls, got {type(parsed).__name__}"

    raw_calls = parsed if isinstance(parsed, list) else [parsed]
    results: list[ToolCallInstance] = []

    for idx, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            return [], f"Tool call at index {idx} is not a JSON object ({type(item).__name__})"

        name = item.get("name") if "name" in item else (item.get("tool_name") or item.get("function"))
        if not isinstance(name, str) or not name.strip():
            return [], f"Tool call at index {idx} missing valid string 'name'"

        has_args_key = any(k in item for k in ("arguments", "parameters"))
        raw_args = item.get("arguments") if "arguments" in item else item.get("parameters", {})
        if raw_args is None and not has_args_key:
            raw_args = {}

        if isinstance(raw_args, str):
            try:
                if strict:
                    raw_args = json.loads(
                        raw_args,
                        object_pairs_hook=_strict_object_pairs_hook,
                        parse_constant=_reject_non_finite_constant,
                    )
                else:
                    raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError) as err:
                return [], f"Malformed arguments JSON string at call {idx}: {err}"

        if not isinstance(raw_args, dict):
            return [], f"Arguments at call {idx} must be a JSON object, got {type(raw_args).__name__}"

        clean_name = name if strict else name.strip()
        results.append(ToolCallInstance(name=clean_name, arguments=raw_args))

    return results, None


def _matches_json_schema_type(value: Any, expected_type: str) -> bool:
    """Validate a Python JSON value against a tool parameter's declared type (fail-closed on unknown types)."""
    t = expected_type.lower().strip()
    if t in ("str", "string"):
        return isinstance(value, str)
    if t in ("int", "integer"):
        return isinstance(value, int) and not isinstance(value, bool)
    if t in ("float", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if t in ("bool", "boolean"):
        return isinstance(value, bool)
    if t in ("list", "array", "tuple"):
        return isinstance(value, list)
    if t in ("dict", "object", "map"):
        return isinstance(value, dict)
    if t in ("null", "none"):
        return value is None
    if t == "any":
        return True
    return False


def _validate_declared_tool_names(
    calls: list[ToolCallInstance],
    available_tools: list[dict[str, Any]] | None,
) -> tuple[bool, str | None]:
    if available_tools is None:
        return True, None
    declared: set[str] = set()
    for tool in available_tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            declared.add(fn["name"])
    for idx, call in enumerate(calls):
        if call.name not in declared:
            return False, f"Call {idx} invoked undeclared tool '{call.name}'"
    return True, None


def validate_calls_against_schema(
    calls: list[ToolCallInstance],
    available_tools: list[dict[str, Any]] | None,
) -> tuple[bool, str | None]:
    """Validate predicted tool calls against sample's declared tool schemas (xLAM / OpenAI format)."""
    if available_tools is None:
        return True, None

    tool_map: dict[str, dict[str, Any]] = {}
    for tool in available_tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            tool_map[fn["name"]] = fn

    for idx, call in enumerate(calls):
        if call.name not in tool_map:
            return False, f"Call {idx} invoked undeclared tool '{call.name}'"
        schema = tool_map[call.name]
        params = schema.get("parameters", {})
        if not isinstance(params, dict):
            continue

        # Support both JSON Schema (`properties` + `required` + `additionalProperties`) and xLAM flat parameter dicts
        if "properties" in params and isinstance(params.get("properties"), dict):
            props = params["properties"]
            if params.get("additionalProperties") is False:
                extra = sorted(set(call.arguments.keys()) - set(props.keys()))
                if extra:
                    return False, f"Call {idx} ('{call.name}') has undeclared arguments: {extra}"
            required = set(params.get("required", []))
            for req_key in required:
                if req_key not in call.arguments:
                    return False, f"Call {idx} ('{call.name}') missing required argument '{req_key}'"
            for arg_k, arg_v in call.arguments.items():
                if arg_k in props and isinstance(props[arg_k], dict) and "type" in props[arg_k]:
                    p_type = props[arg_k]["type"]
                    if isinstance(p_type, str) and not _matches_json_schema_type(arg_v, p_type):
                        return False, f"Call {idx} arg '{arg_k}' failed type '{p_type}'"
        else:
            for param_name, param_spec in params.items():
                if not isinstance(param_spec, dict):
                    continue
                p_type = str(param_spec.get("type", ""))
                is_optional = "optional" in p_type.lower() or "default" in param_spec
                if not is_optional and param_name not in call.arguments:
                    return False, f"Call {idx} ('{call.name}') missing required argument '{param_name}'"
                if param_name in call.arguments and p_type:
                    base_type = p_type.split(",")[0].strip()
                    if not _matches_json_schema_type(call.arguments[param_name], base_type):
                        return False, f"Call {idx} arg '{param_name}' failed type '{base_type}'"

    return True, None


def _normalize_ground_truth(
    ground_truth: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[ToolCallInstance]:
    """Normalize ground-truth calls strictly — raising ValueError on malformed labels instead of silent coercion."""
    if ground_truth is None:
        return []
    if isinstance(ground_truth, list):
        gt_list = ground_truth
    elif isinstance(ground_truth, dict):
        gt_list = [ground_truth] if ground_truth else []
    else:
        raise ValueError(f"Ground truth must be list, dict, or None; got {type(ground_truth).__name__}")

    instances: list[ToolCallInstance] = []
    for idx, gt in enumerate(gt_list):
        if not isinstance(gt, dict):
            raise ValueError(f"Ground truth call {idx} must be a dict, got {type(gt).__name__}")
        name = gt.get("name") or gt.get("tool_name") or gt.get("function")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Ground truth call {idx} missing valid tool 'name'")
        args = gt.get("arguments") if "arguments" in gt else gt.get("parameters", {})
        if args is None:
            args = {}
        elif isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception as err:
                raise ValueError(f"Malformed ground truth arguments JSON at call {idx}: {err}") from err
        if not isinstance(args, dict):
            raise ValueError(f"Ground truth arguments at call {idx} must be a JSON object")
        instances.append(ToolCallInstance(name=name, arguments=args))
    return instances


def _lenient_calls_match(
    pred: list[ToolCallInstance],
    gt: list[ToolCallInstance],
    *,
    ordered: bool = False,
) -> bool:
    if len(pred) != len(gt):
        return False
    if ordered:
        for p, g in zip(pred, gt):
            if p.name.lower().replace("_", "") != g.name.lower().replace("_", ""):
                return False
            if normalize_value(p.arguments) != normalize_value(g.arguments):
                return False
        return True

    used = [False] * len(gt)
    for p in pred:
        p_name = p.name.lower().replace("_", "")
        p_args = normalize_value(p.arguments)
        found = False
        for j, g in enumerate(gt):
            if not used[j] and p_name == g.name.lower().replace("_", "") and p_args == normalize_value(g.arguments):
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True


def _match_unordered_multiset(
    pred: list[ToolCallInstance],
    gt: list[ToolCallInstance],
) -> tuple[bool, bool, bool]:
    """Order-insensitive multiplicity-preserving matching over all tool calls."""
    if len(pred) != len(gt):
        return False, False, False
    if len(gt) == 0:
        return True, True, True

    # 1. Exact multiset match (name + keys + strict values)
    used_exact = [False] * len(gt)
    all_exact = True
    for p in pred:
        found = False
        for j, g in enumerate(gt):
            if not used_exact[j] and p.name == g.name and same_json(p.arguments, g.arguments):
                used_exact[j] = True
                found = True
                break
        if not found:
            all_exact = False
            break
    if all_exact:
        return True, True, True

    # 2. Check tool names + arg keys match
    used_keys = [False] * len(gt)
    all_keys = True
    for p in pred:
        found = False
        for j, g in enumerate(gt):
            if not used_keys[j] and p.name == g.name and (p.arguments.keys() == g.arguments.keys()):
                used_keys[j] = True
                found = True
                break
        if not found:
            all_keys = False
            break

    # 3. Check tool names multiset match
    used_names = [False] * len(gt)
    all_names = True
    for p in pred:
        found = False
        for j, g in enumerate(gt):
            if not used_names[j] and p.name == g.name:
                used_names[j] = True
                found = True
                break
        if not found:
            all_names = False
            break

    return all_names, (all_names and all_keys), False


class ASTToolCallEvaluator:
    """Evaluates agentic tool-calling outputs against ground-truth function calls.

    Defaults to strict protocol-v1 evaluation (`strict=True`):
    - Scores ALL predicted and ground-truth calls (rejecting extra or missing calls).
    - Preserves exact tool-name case/underscores and strict JSON argument types/strings.
    - Supports abstention (`[]` vs `[]` -> `exact_match=True`).
    - Validates against `available_tools` JSON schemas when provided.
    """

    def __init__(
        self,
        *,
        strict: bool = True,
        ordered_calls: bool = False,
    ) -> None:
        self.strict = strict
        self.ordered_calls = ordered_calls

    def evaluate_single(
        self,
        raw_output: str,
        ground_truth: list[dict[str, Any]] | dict[str, Any] | None,
        sample_id: str | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> ToolCallEvaluation:
        """Evaluate a single generated response against ground truth."""
        gt_instances = _normalize_ground_truth(ground_truth)
        is_abstention_gt = len(gt_instances) == 0

        pred_instances, parse_error = parse_tool_calls_from_text(raw_output, strict=self.strict)
        if parse_error is not None:
            return ToolCallEvaluation(
                sample_id=sample_id,
                parse_success=False,
                schema_valid=False,
                tool_match=False,
                arg_names_match=False,
                arg_values_match=False,
                exact_match=False,
                lenient_exact_match=False,
                is_abstention_gt=is_abstention_gt,
                is_abstention_pred=False,
                parse_error=parse_error,
                predicted_calls=[],
                ground_truth_calls=[g.model_dump() for g in gt_instances],
            )

        is_abstention_pred = len(pred_instances) == 0
        schema_ok, schema_err = validate_calls_against_schema(pred_instances, available_tools)
        gt_schema_ok, _ = validate_calls_against_schema(gt_instances, available_tools)
        declared_names_ok, declared_err = _validate_declared_tool_names(pred_instances, available_tools)

        lenient_ok = _lenient_calls_match(pred_instances, gt_instances, ordered=self.ordered_calls)

        if not self.strict:
            same_count = len(pred_instances) == len(gt_instances)
            if self.ordered_calls:
                tool_match = same_count and all(
                    p.name.lower().replace("_", "") == g.name.lower().replace("_", "")
                    for p, g in zip(pred_instances, gt_instances)
                )
                arg_names_match = tool_match and all(
                    set(normalize_value(p.arguments).keys()) == set(normalize_value(g.arguments).keys())
                    for p, g in zip(pred_instances, gt_instances)
                )
            else:
                p_names = sorted(p.name.lower().replace("_", "") for p in pred_instances)
                g_names = sorted(g.name.lower().replace("_", "") for g in gt_instances)
                tool_match = same_count and (p_names == g_names)
                p_keys = sorted(
                    (p.name.lower().replace("_", ""), tuple(sorted(normalize_value(p.arguments).keys())))
                    for p in pred_instances
                )
                g_keys = sorted(
                    (g.name.lower().replace("_", ""), tuple(sorted(normalize_value(g.arguments).keys())))
                    for g in gt_instances
                )
                arg_names_match = tool_match and (p_keys == g_keys)
            arg_values_match = arg_names_match and lenient_ok
            exact_match = arg_values_match
        else:
            if self.ordered_calls:
                same_count = len(pred_instances) == len(gt_instances)
                pairs = list(zip(pred_instances, gt_instances))
                tool_match = same_count and all(p.name == g.name for p, g in pairs)
                arg_names_match = tool_match and all(
                    p.arguments.keys() == g.arguments.keys() for p, g in pairs
                )
                arg_values_match = arg_names_match and all(
                    same_json(p.arguments, g.arguments) for p, g in pairs
                )
            else:
                tool_match, arg_names_match, arg_values_match = _match_unordered_multiset(
                    pred_instances, gt_instances
                )
            # Enforce full schema compliance when ground truth itself conforms to the schema;
            # if an upstream scraped RapidAPI schema contradicts its own ground truth (29/1600 xLAM items),
            # enforce declared tool-name membership + strict AST value equality so oracle accuracy == 100%.
            effective_schema_gate = schema_ok if gt_schema_ok else declared_names_ok
            exact_match = bool(effective_schema_gate and tool_match and arg_names_match and arg_values_match)

        return ToolCallEvaluation(
            sample_id=sample_id,
            parse_success=True,
            schema_valid=schema_ok,
            tool_match=tool_match,
            arg_names_match=arg_names_match,
            arg_values_match=arg_values_match,
            exact_match=exact_match,
            lenient_exact_match=lenient_ok,
            is_abstention_gt=is_abstention_gt,
            is_abstention_pred=is_abstention_pred,
            parse_error=schema_err if not schema_ok else (declared_err if not declared_names_ok else None),
            predicted_calls=[p.model_dump() for p in pred_instances],
            ground_truth_calls=[g.model_dump() for g in gt_instances],
        )

    def evaluate_batch(
        self,
        predictions: list[str],
        ground_truths: list[list[dict[str, Any]] | dict[str, Any] | None],
        sample_ids: list[str] | None = None,
        available_tools_list: list[list[dict[str, Any]] | None] | None = None,
    ) -> EvaluationSummary:
        """Evaluate a batch of samples and compute aggregate metrics."""
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Predictions length ({len(predictions)}) != ground_truths length ({len(ground_truths)})"
            )
        total = len(predictions)
        if sample_ids is not None and len(sample_ids) != total:
            raise ValueError(
                f"sample_ids length ({len(sample_ids)}) != predictions length ({total})"
            )
        if available_tools_list is not None and len(available_tools_list) != total:
            raise ValueError(
                f"available_tools_list length ({len(available_tools_list)}) != predictions length ({total})"
            )
        if total == 0:
            return EvaluationSummary()

        if sample_ids is None:
            sample_ids = [str(i) for i in range(total)]

        parse_count = 0
        schema_count = 0
        tool_count = 0
        name_count = 0
        val_count = 0
        exact_count = 0
        lenient_count = 0

        for idx, (pred, gt, sid) in enumerate(zip(predictions, ground_truths, sample_ids)):
            tools = available_tools_list[idx] if available_tools_list is not None else None
            res = self.evaluate_single(pred, gt, sample_id=sid, available_tools=tools)
            if res.parse_success:
                parse_count += 1
            if res.schema_valid:
                schema_count += 1
            if res.tool_match:
                tool_count += 1
            if res.arg_names_match:
                name_count += 1
            if res.arg_values_match:
                val_count += 1
            if res.exact_match:
                exact_count += 1
            if res.lenient_exact_match:
                lenient_count += 1

        return EvaluationSummary(
            total_samples=total,
            parse_success_count=parse_count,
            schema_valid_count=schema_count,
            tool_match_count=tool_count,
            arg_names_match_count=name_count,
            arg_values_match_count=val_count,
            exact_match_count=exact_count,
            lenient_exact_match_count=lenient_count,
            parse_success_rate=round(parse_count / total, 6),
            schema_valid_rate=round(schema_count / total, 6),
            tool_selection_accuracy=round(tool_count / total, 6),
            argument_name_accuracy=round(name_count / total, 6),
            argument_value_accuracy=round(val_count / total, 6),
            exact_match_accuracy=round(exact_count / total, 6),
            lenient_exact_match_accuracy=round(lenient_count / total, 6),
        )

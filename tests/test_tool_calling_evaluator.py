"""Unit tests for AST-based tool calling evaluator (Strict Protocol-v1 + Lenient Diagnostic)."""
from __future__ import annotations

from aloe.evaluation.tool_calling import (
    ASTToolCallEvaluator,
    extract_json_candidate,
    normalize_value,
    parse_tool_calls_from_text,
    same_json,
)


def test_extract_json_candidate() -> None:
    assert extract_json_candidate('{"name": "search", "arguments": {"q": "test"}}') == (
        '{"name": "search", "arguments": {"q": "test"}}'
    )

    fenced = """Here is the tool call:
```json
{"name": "get_weather", "arguments": {"city": "London"}}
```
Hope that helps!"""
    assert extract_json_candidate(fenced) == '{"name": "get_weather", "arguments": {"city": "London"}}'

    unfenced = 'Sure! {"name": "fetch", "arguments": {"id": 101}} is the function.'
    assert extract_json_candidate(unfenced) == '{"name": "fetch", "arguments": {"id": 101}}'


def test_same_json_strict_and_normalize_value_diagnostic() -> None:
    # Strict comparator preserves case, string vs int, and bool vs int
    assert same_json(42.0, 42) is True
    assert same_json("001", 1) is False
    assert same_json(True, 1) is False
    assert same_json("CMB", "cmb") is False

    # Legacy diagnostic helper remains available for secondary lenient_exact_match
    assert normalize_value(42.0) == 42
    assert normalize_value("100.0") == 100
    assert normalize_value("  Hello World  ") == "hello world"
    assert normalize_value({"b": 2, "a": 1}) == {"a": 1, "b": 2}


def test_parse_tool_calls_from_text() -> None:
    raw = '[{"name": "order_food", "arguments": {"item": "pizza", "quantity": 2}}]'
    calls, err = parse_tool_calls_from_text(raw)
    assert err is None
    assert len(calls) == 1
    assert calls[0].name == "order_food"
    assert calls[0].arguments == {"item": "pizza", "quantity": 2}


def test_evaluator_strict_vs_lenient_case() -> None:
    strict_eval = ASTToolCallEvaluator(strict=True)
    output_exact = '{"name": "book_flight", "arguments": {"origin": "CMB", "dest": "LHR", "price": 450.0}}'
    gt_exact = {"name": "book_flight", "arguments": {"origin": "CMB", "dest": "LHR", "price": 450}}

    res = strict_eval.evaluate_single(output_exact, gt_exact)
    assert res.parse_success is True
    assert res.tool_match is True
    assert res.arg_names_match is True
    assert res.arg_values_match is True
    assert res.exact_match is True

    # Case mismatch fails strict exact_match while setting secondary lenient_exact_match=True
    gt_lower = {"name": "book_flight", "arguments": {"origin": "cmb", "dest": "lhr", "price": 450}}
    res_case = strict_eval.evaluate_single(output_exact, gt_lower)
    assert res_case.exact_match is False
    assert res_case.lenient_exact_match is True


def test_evaluator_mismatched_arguments() -> None:
    evaluator = ASTToolCallEvaluator()
    output = '{"name": "book_flight", "arguments": {"origin": "CMB", "dest": "JFK"}}'
    gt = {"name": "book_flight", "arguments": {"origin": "CMB", "dest": "LHR"}}

    res = evaluator.evaluate_single(output, gt)
    assert res.parse_success is True
    assert res.tool_match is True
    assert res.arg_names_match is True
    assert res.arg_values_match is False
    assert res.exact_match is False


def test_evaluator_batch_aggregation() -> None:
    evaluator = ASTToolCallEvaluator()
    preds = [
        '{"name": "f1", "arguments": {"x": 1}}',
        '{"name": "f2", "arguments": {"y": 2}}',
        'Invalid output without JSON',
        '{"name": "wrong_tool", "arguments": {"z": 3}}',
    ]
    gts = [
        {"name": "f1", "arguments": {"x": 1}},
        {"name": "f2", "arguments": {"y": 99}},
        {"name": "f3", "arguments": {}},
        {"name": "target_tool", "arguments": {"z": 3}},
    ]

    summary = evaluator.evaluate_batch(preds, gts)
    assert summary.total_samples == 4
    assert summary.parse_success_count == 3
    assert summary.parse_success_rate == 0.75
    assert summary.tool_match_count == 2
    assert summary.exact_match_count == 1
    assert summary.exact_match_accuracy == 0.25

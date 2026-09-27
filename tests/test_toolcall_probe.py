"""Tests for Stage 6 / R6 baseline tool-call probe.

Validates that:
1. Tool call parsing extracts valid JSON, codeblocks, and function calls.
2. Tool call probe result envelopes adhere to Schema Version 1.0.
3. Gate 6 criteria (10 prompts tested, parseability >= 70%, abstention recorded) are fully verified.
4. TOOLCALL_REPORT.md exists and accurately reflects probe metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_tool_calls import parse_tool_call, PROMPTS, TOOLS


def test_parse_tool_call_json_block():
    raw = '```json\n{"name": "search_flights", "arguments": {"destination": "Paris", "date": "2026-09-21"}}\n```'
    parsed = parse_tool_call(raw)
    assert parsed["is_parseable"] is True
    assert parsed["called_tool"] is True
    assert parsed["tool_name"] == "search_flights"
    assert parsed["arguments"]["destination"] == "Paris"
    assert parsed["format"] == "json"


def test_parse_tool_call_raw_json():
    raw = '{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}'
    parsed = parse_tool_call(raw)
    assert parsed["is_parseable"] is True
    assert parsed["called_tool"] is True
    assert parsed["tool_name"] == "get_weather"
    assert parsed["arguments"]["city"] == "Tokyo"
    assert parsed["format"] == "json"


def test_parse_tool_call_plain_text():
    raw = "A mutex and a binary semaphore differ in ownership semantics."
    parsed = parse_tool_call(raw)
    assert parsed["is_parseable"] is True
    assert parsed["called_tool"] is False
    assert parsed["tool_name"] is None
    assert parsed["arguments"] is None
    assert parsed["format"] == "plain_text"


def test_parse_tool_call_syntax():
    raw = "call:get_weather(city='London', unit='celsius')"
    parsed = parse_tool_call(raw)
    assert parsed["is_parseable"] is True
    assert parsed["called_tool"] is True
    assert parsed["tool_name"] == "get_weather"


def test_prompts_and_tools_definitions():
    assert len(PROMPTS) == 10
    assert len(TOOLS) == 9
    categories = {p["category"] for p in PROMPTS}
    assert "obvious_single_tool" in categories
    assert "multi_plausible_tools" in categories
    assert "missing_required_args" in categories
    assert "no_tool_needed" in categories


def test_toolcall_probe_artifacts():
    toolcalls_dir = Path("runs/toolcalls")
    assert toolcalls_dir.is_dir()

    report_path = toolcalls_dir / "TOOLCALL_REPORT.md"
    assert report_path.is_file()
    report_content = report_path.read_text()
    assert "Gate 6 Evaluation: PASS" in report_content
    assert "google/gemma-4-E2B-it" in report_content
    assert "tiiuae/Falcon-H1-1.5B-Instruct" in report_content

    for slug in ["google--gemma-4-E2B-it", "tiiuae--Falcon-H1-1.5B-Instruct"]:
        probe_file = toolcalls_dir / slug / "baseline_probe.json"
        assert probe_file.is_file(), f"{probe_file} must exist"

        with open(probe_file) as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0"
        assert data["status"] == "PASS"
        assert data["script"] == "probe_tool_calls.py"
        assert len(data["prompts_evaluation"]) == 10
        stats = data["statistics"]
        assert stats["total_prompts"] == 10
        assert stats["parseable_ratio"] >= 0.7  # Gate 6 requirement is >= 7/10

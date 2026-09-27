"""xLAM Function-Calling Dataset Loader & Schema Models (Prompt 7 / Stage 7).

Provides Pydantic models and helper utilities for loading, inspecting,
and partitioning the Salesforce/xlam-function-calling-60k dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class XLAMToolCall(BaseModel):
    """An individual expected tool call within an xLAM example."""

    model_config = ConfigDict(extra="ignore")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class XLAMToolDefinition(BaseModel):
    """An available tool schema provided to the model."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class XLAMExample(BaseModel):
    """A single record from Salesforce/xlam-function-calling-60k."""

    model_config = ConfigDict(extra="ignore")

    id: str | int
    query: str
    tools: list[dict[str, Any]] = Field(default_factory=list)
    answers: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def available_tool_names(self) -> set[str]:
        """Names of all tools provided in the prompt context."""
        names = set()
        for t in self.tools:
            if isinstance(t, dict) and "name" in t:
                names.add(t["name"])
        return names

    @property
    def called_tool_names(self) -> set[str]:
        """Names of all tools invoked in the ground truth answer."""
        names = set()
        for a in self.answers:
            if isinstance(a, dict) and "name" in a:
                names.add(a["name"])
        return names

    @property
    def all_referenced_tool_names(self) -> set[str]:
        """Union of available and called tool names."""
        return self.available_tool_names | self.called_tool_names

    @property
    def category(self) -> str:
        """Infer category based on available tools and ground-truth invocation."""
        called_cnt = len(self.called_tool_names)
        avail_cnt = len(self.available_tool_names)

        if called_cnt == 0:
            return "no_tool_needed"
        elif avail_cnt == 1 and called_cnt == 1:
            return "obvious_single_tool"
        elif avail_cnt > 1 and called_cnt >= 1:
            return "multi_plausible_tools"
        else:
            return "complex_tool_invocation"


def parse_xlam_record(raw: dict[str, Any], index: int | None = None) -> XLAMExample:
    """Parse raw JSON dict into standardized XLAMExample model."""
    record_id = raw.get("id", str(index) if index is not None else "unknown")

    # In xLAM 60k, 'tools' and 'answers' can be JSON-encoded strings or raw lists
    raw_tools = raw.get("tools", [])
    if isinstance(raw_tools, str):
        try:
            raw_tools = json.loads(raw_tools)
        except Exception:
            raw_tools = []

    raw_answers = raw.get("answers", [])
    if isinstance(raw_answers, str):
        try:
            raw_answers = json.loads(raw_answers)
        except Exception:
            raw_answers = []

    query = raw.get("query", "")
    return XLAMExample(
        id=str(record_id),
        query=query,
        tools=raw_tools if isinstance(raw_tools, list) else [],
        answers=raw_answers if isinstance(raw_answers, list) else [],
    )


def load_xlam_file(json_path: Path | str) -> list[XLAMExample]:
    """Load and parse full xLAM dataset from local JSON file."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"xLAM dataset file does not exist: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of examples in {path}, got {type(data)}")

    examples = [parse_xlam_record(item, idx) for idx, item in enumerate(data)]
    return examples

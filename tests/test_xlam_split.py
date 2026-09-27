"""Tests for xLAM tool-disjoint split manifest (Gate A Data Preparation / Task I012).

Validates:
1. Manifest data/manifests/xlam_pilot_v1.json exists and adheres to schema.
2. Intersection of Train tools and Validation tools is strictly EMPTY (Zero contamination).
3. Split IDs match their cryptographic SHA-256 digest.
4. Target sample counts (~800 train, ~200 val) are met.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _compute_sha256_of_ids(ids: list[str]) -> str:
    sorted_ids = sorted(ids)
    data = "\n".join(sorted_ids).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def test_xlam_manifest_disjointness_and_integrity():
    manifest_path = Path("data/manifests/xlam_pilot_v1.json")
    if not manifest_path.is_file():
        # If manifest is not built yet, skip or assert failure clearly
        raise FileNotFoundError(
            f"Manifest file {manifest_path} does not exist. Run scripts/build_xlam_split.py first."
        )

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "1.0"
    assert data["status"] == "PASS"
    assert data["split_policy"] == "tool_disjoint"

    tools = data["tools"]
    train_tools = set(tools["train_tool_names"])
    val_tools = set(tools["val_tool_names"])

    # CRITICAL GATE ASSERTION: Empty tool intersection
    intersection = train_tools.intersection(val_tools)
    assert len(intersection) == 0, (
        f"CRITICAL CONTAMINATION ERROR: Train and validation tool sets intersect! Overlapping tools: {intersection}"
    )
    assert tools["intersection_count"] == 0

    splits = data["splits"]
    train_split = splits["train"]
    val_split = splits["validation"]

    # Sample counts
    assert train_split["count"] >= 500, f"Expected >= 500 train samples, got {train_split['count']}"
    assert val_split["count"] >= 100, f"Expected >= 100 val samples, got {val_split['count']}"

    # Verify cryptographic ID hashes
    train_ids = train_split["ids"]
    val_ids = val_split["ids"]
    assert len(train_ids) == train_split["count"]
    assert len(val_ids) == val_split["count"]

    assert _compute_sha256_of_ids(train_ids) == train_split["sha256"]
    assert _compute_sha256_of_ids(val_ids) == val_split["sha256"]

    # Ensure no example ID appears in both splits
    id_intersection = set(train_ids).intersection(set(val_ids))
    assert len(id_intersection) == 0, f"Example ID overlap detected: {id_intersection}"

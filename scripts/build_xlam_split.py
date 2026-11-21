"""Build 3-way tool-disjoint, category-stratified xLAM split (Protocol v1 / IPD).

Supports both legacy 2-way pilot splits and the frozen 3-way IPD/Final protocol:
  1,000 train / 200 validation / 400 sealed unseen-tool test
with:
- Zero pairwise tool-name overlap across (train, validation, test),
- Category-stratified sampling eliminating train/val category skew,
- Compact-JSON prompt length filtering so zero examples suffer schema truncation,
- Pinned dataset revision and raw-file SHA-256 provenance (zero absolute path leaks).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloe.datasets.xlam import XLAMExample, load_xlam_file

DATASET_ID = "Salesforce/xlam-function-calling-60k"
DATASET_FILENAME = "xlam_function_calling_60k.json"
PINNED_DATASET_SHA = "45d1ebe7f73a99abfb2bf71a30caaaaf0570cf32"


def _sha256_of_ids(ids: list[str]) -> str:
    """Deterministic SHA-256 hex digest of a sorted list of string IDs."""
    sorted_ids = sorted(ids)
    data = "\n".join(sorted_ids).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compact_example_char_len(ex: XLAMExample) -> int:
    """Estimate compact JSON prompt+answer character length to prevent schema truncation."""
    tools_compact = json.dumps(ex.tools, separators=(",", ":"))
    ans_compact = json.dumps(ex.answers, separators=(",", ":"))
    return len(tools_compact) + len(ex.query) + len(ans_compact) + 260


def acquire_xlam_dataset(
    local_cache_dir: Path = Path("data/raw/xlam"),
    revision: str = PINNED_DATASET_SHA,
) -> tuple[Path, dict[str, Any]]:
    """Acquire xLAM dataset with pinned SHA or use verified local file."""
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_cache_dir / DATASET_FILENAME

    commit_sha = revision
    license_name = "cc-by-4.0"
    if not local_file.is_file():
        api = HfApi()
        try:
            info = api.dataset_info(DATASET_ID, revision=revision)
            commit_sha = info.sha or revision
        except Exception:
            commit_sha = revision

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except Exception:
                pass
        try:
            downloaded_path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=DATASET_FILENAME,
                repo_type="dataset",
                revision=commit_sha,
                local_dir=str(local_cache_dir),
                token=token,
            )
            local_file = Path(downloaded_path)
        except GatedRepoError as gre:
            raise RuntimeError(
                f"Dataset '{DATASET_ID}' requires HuggingFace authentication (export HF_TOKEN=...)."
            ) from gre

    file_size = local_file.stat().st_size
    file_sha256 = _sha256_of_file(local_file)
    try:
        rel_path = str(local_file.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        rel_path = f"data/raw/xlam/{DATASET_FILENAME}"

    meta = {
        "dataset_id": DATASET_ID,
        "revision_sha": commit_sha,
        "raw_file_sha256": file_sha256,
        "license": license_name,
        "relative_path": rel_path,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024**2), 2),
    }
    return local_file, meta


def _stratified_draw(
    candidates: list[str],
    id_to_cat: dict[str, str],
    target_n: int,
    target_ratios: dict[str, float],
    rng: random.Random,
) -> list[str]:
    """Draw `target_n` IDs matching `target_ratios` across categories as closely as possible."""
    if len(candidates) < target_n:
        raise ValueError(f"Insufficient candidates ({len(candidates)}) for target {target_n}")

    by_cat: dict[str, list[str]] = defaultdict(list)
    for cid in candidates:
        by_cat[id_to_cat[cid]].append(cid)
    for bucket in by_cat.values():
        rng.shuffle(bucket)

    selected: list[str] = []
    for cat, ratio in sorted(target_ratios.items()):
        quota = min(int(round(target_n * ratio)), len(by_cat.get(cat, [])))
        selected.extend(by_cat[cat][:quota])
        by_cat[cat] = by_cat[cat][quota:]

    # Fill remainder from remaining shuffled pool
    remaining_pool: list[str] = []
    for bucket in by_cat.values():
        remaining_pool.extend(bucket)
    rng.shuffle(remaining_pool)

    if len(selected) < target_n:
        needed = target_n - len(selected)
        selected.extend(remaining_pool[:needed])
    elif len(selected) > target_n:
        selected = selected[:target_n]

    return selected


def build_tool_disjoint_split(
    local_file: Path,
    meta: dict[str, Any],
    manifest_path: Path = Path("data/manifests/xlam_ipd_v1.json"),
    report_path: Path = Path("runs/dataset/XLAM_IPD_SPLIT_REPORT.md"),
    target_train: int = 1000,
    target_val: int = 200,
    target_test: int = 400,
    max_compact_chars: int = 3400,
    seed: int = 42,
) -> dict[str, Any]:
    examples = load_xlam_file(local_file)
    total_records = len(examples)
    meta["total_records"] = total_records

    # Filter out duplicate IDs and extreme length outliers (> max_compact_chars) upfront
    seen_ids: set[str] = set()
    filtered_examples: list[XLAMExample] = []
    overlength_dropped = 0
    for ex in examples:
        if ex.id in seen_ids:
            continue
        seen_ids.add(ex.id)
        if max_compact_chars > 0 and compact_example_char_len(ex) > max_compact_chars:
            overlength_dropped += 1
            continue
        filtered_examples.append(ex)

    tool_usage: Counter[str] = Counter()
    example_to_tools: dict[str, set[str]] = {}
    id_to_ex: dict[str, XLAMExample] = {}
    id_to_cat: dict[str, str] = {}

    for ex in filtered_examples:
        id_to_ex[ex.id] = ex
        id_to_cat[ex.id] = ex.category
        tools_in_ex = ex.all_referenced_tool_names
        example_to_tools[ex.id] = tools_in_ex
        for t in tools_in_ex:
            tool_usage[t] += 1

    all_tools = sorted(tool_usage.keys())
    total_unique_tools = len(all_tools)

    rng = random.Random(seed)
    shuffled_tools = list(all_tools)
    rng.shuffle(shuffled_tools)

    if target_test > 0:
        # 60% train tools / 15% val tools / 25% test tools
        val_tool_count = max(int(total_unique_tools * 0.15), 1)
        test_tool_count = max(int(total_unique_tools * 0.25), 1)
        val_tools = set(shuffled_tools[:val_tool_count])
        test_tools = set(shuffled_tools[val_tool_count : val_tool_count + test_tool_count])
        train_tools = set(shuffled_tools[val_tool_count + test_tool_count :])
    else:
        val_tool_count = max(int(total_unique_tools * 0.20), 1)
        val_tools = set(shuffled_tools[:val_tool_count])
        test_tools = set()
        train_tools = set(shuffled_tools[val_tool_count:])

    assert len(train_tools & val_tools) == 0
    assert len(train_tools & test_tools) == 0
    assert len(val_tools & test_tools) == 0

    cand_train: list[str] = []
    cand_val: list[str] = []
    cand_test: list[str] = []
    cand_no_tool: list[str] = []
    discarded_ids: list[str] = []

    for ex in filtered_examples:
        tset = example_to_tools[ex.id]
        if not tset:
            cand_no_tool.append(ex.id)
            continue
        if tset.issubset(train_tools):
            cand_train.append(ex.id)
        elif tset.issubset(val_tools):
            cand_val.append(ex.id)
        elif target_test > 0 and tset.issubset(test_tools):
            cand_test.append(ex.id)
        else:
            discarded_ids.append(ex.id)

    # Compute overall category ratios across tool-using filtered corpus for consistent stratification
    overall_cat_counts = Counter(id_to_cat[eid] for eid in (cand_train + cand_val + cand_test))
    denom = max(sum(overall_cat_counts.values()), 1)
    target_ratios = {cat: cnt / denom for cat, cnt in overall_cat_counts.items()}

    rng.shuffle(cand_no_tool)
    nt_train = min(int(target_train * 0.05), len(cand_no_tool) // 3)
    nt_val = min(int(target_val * 0.05), (len(cand_no_tool) - nt_train) // 2)
    nt_test = min(int(target_test * 0.05), len(cand_no_tool) - nt_train - nt_val) if target_test > 0 else 0

    final_train_ids = _stratified_draw(cand_train, id_to_cat, target_train - nt_train, target_ratios, rng)
    final_val_ids = _stratified_draw(cand_val, id_to_cat, target_val - nt_val, target_ratios, rng)
    final_test_ids = (
        _stratified_draw(cand_test, id_to_cat, target_test - nt_test, target_ratios, rng)
        if target_test > 0
        else []
    )

    if nt_train > 0:
        final_train_ids.extend(cand_no_tool[:nt_train])
    if nt_val > 0:
        final_val_ids.extend(cand_no_tool[nt_train : nt_train + nt_val])
    if nt_test > 0:
        final_test_ids.extend(cand_no_tool[nt_train + nt_val : nt_train + nt_val + nt_test])

    # Strict cardinality & zero-overlap assertions
    assert len(final_train_ids) == target_train
    assert len(final_val_ids) == target_val
    assert len(final_test_ids) == target_test
    assert len(set(final_train_ids) & set(final_val_ids)) == 0
    assert len(set(final_train_ids) & set(final_test_ids)) == 0
    assert len(set(final_val_ids) & set(final_test_ids)) == 0

    train_tools_active = {t for eid in final_train_ids for t in example_to_tools[eid]}
    val_tools_active = {t for eid in final_val_ids for t in example_to_tools[eid]}
    test_tools_active = {t for eid in final_test_ids for t in example_to_tools[eid]}

    assert len(train_tools_active & val_tools_active) == 0
    assert len(train_tools_active & test_tools_active) == 0
    assert len(val_tools_active & test_tools_active) == 0

    splits_dict: dict[str, Any] = {
        "train": {
            "count": len(final_train_ids),
            "sha256": _sha256_of_ids(final_train_ids),
            "categories": dict(Counter(id_to_cat[eid] for eid in final_train_ids)),
            "ids": sorted(final_train_ids),
        },
        "validation": {
            "count": len(final_val_ids),
            "sha256": _sha256_of_ids(final_val_ids),
            "categories": dict(Counter(id_to_cat[eid] for eid in final_val_ids)),
            "ids": sorted(final_val_ids),
        },
    }
    if target_test > 0:
        splits_dict["test"] = {
            "count": len(final_test_ids),
            "sha256": _sha256_of_ids(final_test_ids),
            "sealed": True,
            "categories": dict(Counter(id_to_cat[eid] for eid in final_test_ids)),
            "ids": sorted(final_test_ids),
        }

    manifest_data = {
        "schema_version": "2.0",
        "protocol_tag": "protocol-v1",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": {
            "id": meta["dataset_id"],
            "revision_sha": meta["revision_sha"],
            "raw_file_sha256": meta["raw_file_sha256"],
            "license": meta["license"],
            "relative_path": meta["relative_path"],
            "total_records_source": total_records,
            "overlength_filtered_count": overlength_dropped,
            "max_compact_chars": max_compact_chars,
            "total_unique_tools_source": total_unique_tools,
        },
        "split_policy": "three_way_tool_disjoint_stratified" if target_test > 0 else "tool_disjoint",
        "seed": seed,
        "tools": {
            "train_tool_count": len(train_tools_active),
            "val_tool_count": len(val_tools_active),
            "test_tool_count": len(test_tools_active),
            "intersection_count": 0,
            "pairwise_intersections": {
                "train_val": len(train_tools_active & val_tools_active),
                "train_test": len(train_tools_active & test_tools_active),
                "val_test": len(val_tools_active & test_tools_active),
            },
            "train_tool_names": sorted(train_tools_active),
            "val_tool_names": sorted(val_tools_active),
            "test_tool_names": sorted(test_tools_active),
        },
        "splits": splits_dict,
        "discarded": {
            "cross_partition_discard_count": len(discarded_ids),
            "overlength_filtered_count": overlength_dropped,
            "total_candidate_train": len(cand_train),
            "total_candidate_val": len(cand_val),
            "total_candidate_test": len(cand_test),
        },
        "status": "PASS",
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_split_report(manifest_data, report_path)
    return manifest_data


def generate_split_report(manifest: dict[str, Any], report_path: Path) -> None:
    ds = manifest["dataset"]
    tools = manifest["tools"]
    splits = manifest["splits"]
    disc = manifest["discarded"]

    lines = [
        "# ALOE Protocol v1: 3-Way Stratified Tool-Disjoint xLAM Split Report",
        "",
        "> **Verification: PASS — Zero pairwise tool overlap across Train (1,000), Validation (200), and Sealed Test (400).**",
        "",
        "## 1. Dataset Provenance & Integrity",
        f"- **Source Dataset**: `{ds['id']}`",
        f"- **Pinned Revision SHA**: `{ds['revision_sha']}`",
        f"- **Raw File SHA-256**: `{ds['raw_file_sha256']}`",
        f"- **License**: `{ds['license']}`",
        f"- **Total Source Records**: `{ds['total_records_source']:,}`",
        f"- **Overlength Filtered (`>{ds['max_compact_chars']}` chars)**: `{ds['overlength_filtered_count']:,}`",
        "",
        "## 2. Partition Summary",
        "",
        "| Partition | Sample Count | Unique Tools | ID List SHA-256 |",
        "|---|---:|---:|---|",
        f"| **Train** | **{splits['train']['count']}** | {tools['train_tool_count']} | `{splits['train']['sha256'][:16]}...` |",
        f"| **Validation** | **{splits['validation']['count']}** | {tools['val_tool_count']} | `{splits['validation']['sha256'][:16]}...` |",
    ]
    if "test" in splits:
        lines.append(
            f"| **Test (Sealed)** | **{splits['test']['count']}** | {tools['test_tool_count']} | `{splits['test']['sha256'][:16]}...` |"
        )
    lines.extend([
        f"| **Pairwise Tool Overlap** | — | **0 (train∩val=0, train∩test=0, val∩test=0)** | — |",
        f"| **Cross-Partition Discards** | {disc['cross_partition_discard_count']:,} | — | — |",
        "",
        "## 3. Stratified Category Proportions",
    ])
    for sname, sdata in splits.items():
        lines.append(f"\n### `{sname}` ({sdata['count']} items)")
        for cat, cnt in sorted(sdata["categories"].items()):
            lines.append(f"- `{cat}`: **{cnt}** ({cnt / max(sdata['count'], 1) * 100:.1f}%)")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--max-compact-chars", type=int, default=3400)
    parser.add_argument("--manifest", type=str, default="data/manifests/xlam_ipd_v1.json")
    parser.add_argument("--report", type=str, default="runs/dataset/XLAM_IPD_SPLIT_REPORT.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    local_file, meta = acquire_xlam_dataset(REPO_ROOT / "data/raw/xlam")
    manifest = build_tool_disjoint_split(
        local_file,
        meta,
        manifest_path=REPO_ROOT / args.manifest,
        report_path=REPO_ROOT / args.report,
        target_train=args.train_size,
        target_val=args.val_size,
        target_test=args.test_size,
        max_compact_chars=args.max_compact_chars,
        seed=args.seed,
    )
    print(
        f"Built {args.manifest}: train={manifest['splits']['train']['count']} "
        f"val={manifest['splits']['validation']['count']} "
        f"test={manifest['splits'].get('test', {}).get('count', 0)} "
        f"(overlap={manifest['tools']['intersection_count']})"
    )

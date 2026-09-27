"""Download and verify models with pinned revision SHAs (ALOE Stage 2 / R2).

Acquires pilot models (google/gemma-4-E2B-it and tiiuae/Falcon-H1-1.5B-Instruct),
pins revision commit SHAs, validates configs and tokenizers, writes acquisition.json
envelopes (Schema Version 1.0), and programmatically generates runs/models/REGISTRY.md.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from transformers import AutoConfig, AutoTokenizer

# Ensure repository root is on sys.path so aloe can be imported when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aloe.datasets.model_registry import ModelAcquisitionRecord, ModelRegistry

MODELS: dict[str, dict[str, Any]] = {
    "gemma-4-E2B-it": {
        "repo_id": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "precision": "bf16",
        "quantisation": "none",
        "cached_path": Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    },
    "gemma-4-E4B-it": {
        "repo_id": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "precision": "bf16",
        "quantisation": "none",
        "cached_path": None,
    },
    "falcon-h1-1.5b": {
        "repo_id": "tiiuae/Falcon-H1-1.5B-Instruct",
        "revision": "80ebc50d7799a440b96c93bb6686a3924a09b0cb",
        "precision": "bf16",
        "quantisation": "none",
        "cached_path": None,
    },
}


def _utc_now() -> str:
    """Return current UTC time formatted as ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def download_model(model_key: str, out_dir: Path) -> dict[str, Any]:
    """Acquire, verify, and register a single pilot model."""
    # Resolve alias / model key
    if model_key not in MODELS:
        # Check if user passed full repo_id
        matched_key = None
        for k, v in MODELS.items():
            if v["repo_id"] == model_key:
                matched_key = k
                break
        if matched_key is None:
            raise ValueError(
                f"Unknown model '{model_key}'. Valid choices: {list(MODELS.keys())} or 'all'"
            )
        model_key = matched_key

    info = MODELS[model_key]
    repo_id: str = info["repo_id"]
    pinned_revision: str = info["revision"]
    cached_path: Path | None = info.get("cached_path")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    print(f"[{_utc_now()}] Resolving repository metadata for {repo_id}...")
    api = HfApi(token=token)

    try:
        model_info = api.model_info(repo_id, revision=pinned_revision)
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as e:
        raise RuntimeError(
            f"Gate 2 Failure: Unable to access repository {repo_id}. "
            f"Exact URL: https://huggingface.co/{repo_id}. Error: {e}"
        ) from e
    except Exception as e:
        # For non-auth issues (e.g. transient network when cache exists)
        if cached_path is not None and cached_path.is_dir():
            print(f"[{_utc_now()}] Network error contacting Hub, but cache exists: {e}")
            model_info = None
        else:
            raise RuntimeError(
                f"Gate 2 Failure: Hub query failed for {repo_id}. "
                f"Exact URL: https://huggingface.co/{repo_id}. Error: {e}"
            ) from e

    # Determine local directory path
    if cached_path is not None and cached_path.is_dir():
        print(f"[{_utc_now()}] Found existing verified snapshot at {cached_path}")
        local_dir = cached_path.resolve()
        commit_sha = pinned_revision
    else:
        print(
            f"[{_utc_now()}] Downloading snapshot for {repo_id} (pinned revision: {pinned_revision})..."
        )
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                revision=pinned_revision,
                token=token,
            )
            local_dir = Path(local_path).resolve()
            commit_sha = pinned_revision
        except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as e:
            raise RuntimeError(
                f"Gate 2 Failure: Download failed for {repo_id}. "
                f"Exact URL: https://huggingface.co/{repo_id}. Error: {e}"
            ) from e

    print(f"[{_utc_now()}] Local snapshot path: {local_dir}")
    print(f"[{_utc_now()}] Pinned Commit SHA: {commit_sha}")

    # Inspect all files on disk
    files: list[dict[str, Any]] = []
    total_size = 0
    for p in sorted(local_dir.rglob("*")):
        if p.is_file() and not p.is_symlink():
            sz = p.stat().st_size
            files.append({
                "path": str(p.relative_to(local_dir)),
                "size_bytes": sz,
                "symlink": False,
            })
            total_size += sz
        elif p.is_file() and p.is_symlink():
            resolved = p.resolve()
            sz = resolved.stat().st_size if resolved.exists() else 0
            files.append({
                "path": str(p.relative_to(local_dir)),
                "size_bytes": sz,
                "symlink": True,
            })
            total_size += sz

    # Load AutoConfig and AutoTokenizer
    print(f"[{_utc_now()}] Loading config & tokenizer from {local_dir}...")
    cfg = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    cfg_dict = cfg.to_dict()

    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    has_chat_template = tok.chat_template is not None
    chat_template = tok.chat_template if has_chat_template else "NOT PRESENT"

    # Extract license terms
    card_dict: dict[str, Any] = {}
    if model_info is not None and hasattr(model_info, "card_data") and model_info.card_data:
        card_dict = model_info.card_data.to_dict()

    license_slug = card_dict.get("license") or "see_card"
    license_name = card_dict.get("license_name")
    license_link = card_dict.get("license_link")
    gated = getattr(model_info, "gated", False) if model_info is not None else False

    slug = repo_id.replace("/", "--")
    target_dir = out_dir / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    acquisition_data = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": _utc_now(),
        "script": "download_models.py",
        "model": {
            "id": repo_id,
            "revision": commit_sha,
            "precision": info["precision"],
            "quantisation": info["quantisation"],
            "local_snapshot_path": str(local_dir),
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024**3), 3),
            "files": files,
            "license": license_slug,
            "license_name": license_name,
            "license_link": license_link,
            "gated": gated,
            "has_chat_template": has_chat_template,
            "chat_template": chat_template,
            "config": cfg_dict,
        },
        "status": "PASS",
    }

    # Validate against Pydantic schema before saving
    ModelAcquisitionRecord.model_validate(acquisition_data)

    out_file = target_dir / "acquisition.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(acquisition_data, f, indent=2)

    print(f"[{_utc_now()}] Acquisition record saved to {out_file}")
    print(
        f"[{_utc_now()}] Model {repo_id}: {acquisition_data['model']['total_size_gb']} GB "
        f"across {len(files)} files."
    )
    return acquisition_data


def main() -> None:
    """Main CLI entrypoint for model acquisition."""
    parser = argparse.ArgumentParser(
        description="Download and verify models with pinned revision SHAs (ALOE Stage 2 / R2)."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *list(MODELS.keys()), *[v["repo_id"] for v in MODELS.values()]],
        help="Model to acquire ('all' or specific model key/id).",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/models",
        help="Target directory to write acquisition records and REGISTRY.md.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        target_models = list(MODELS.keys())
    else:
        target_models = [args.model]

    print(f"[{_utc_now()}] Starting acquisition for models: {target_models}")
    for m in target_models:
        download_model(m, out_dir)

    print(f"[{_utc_now()}] Generating REGISTRY.md from acquisition records...")
    registry = ModelRegistry(out_dir)
    report_file = out_dir / "REGISTRY.md"
    registry.generate_registry_markdown(report_file)
    print(f"[{_utc_now()}] REGISTRY.md generated at {report_file}")


if __name__ == "__main__":
    main()

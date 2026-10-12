"""Export a compact mini-pilot xLAM dataset for remote GPU execution."""
import gzip
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    manifest_path = root / "data" / "manifests" / "xlam_pilot_v1.json"
    raw_path = root / "data" / "raw" / "xlam" / "xlam_function_calling_60k.json"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    train_ids = set(str(i) for i in manifest["splits"]["train"]["ids"][:50])
    val_ids = set(str(i) for i in manifest["splits"]["validation"]["ids"][:20])

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    train_data = [ex for ex in raw if str(ex.get("id", "")) in train_ids]
    val_data = [ex for ex in raw if str(ex.get("id", "")) in val_ids]

    out_file = root / "data" / "manifests" / "xlam_mini_pilot.json"
    out_file.write_text(json.dumps({"train": train_data, "validation": val_data}, indent=2), encoding="utf-8")
    print(f"Exported {len(train_data)} train samples and {len(val_data)} validation samples to {out_file}")

if __name__ == "__main__":
    main()

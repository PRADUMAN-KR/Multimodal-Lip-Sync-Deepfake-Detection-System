#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd


# Directory you used as --preprocessed_dir when validating
PREPROCESSED_DIR = Path("validation_data/")

# Directory you used as --output_dir (contains predictions.csv, metrics.json, etc.)
RESULTS_DIR = Path("results_zarr")

# Set to True if you also want to inspect tensors in samples.zarr
LOAD_FROM_ZARR = False
# =====================================

predictions_path = RESULTS_DIR / "predictions.csv"
manifest_path = PREPROCESSED_DIR / "manifest.jsonl"
zarr_path = PREPROCESSED_DIR / "samples.zarr"


def main() -> None:
    # 1) Load predictions and find false positives
    df = pd.read_csv(predictions_path)

    # ground_truth: 0=real, 1=fake; predicted_label: 0=real, 1=fake
    # false positives = real (0) predicted as fake (1)
    fp_df = df[(df["ground_truth"] == 0) & (df["predicted_label"] == 1)]

    # (optional) uncomment to keep only high‑confidence errors:
    # fp_df = fp_df[
    #     (fp_df["confidence"] > 0.9) | (fp_df["manipulation_probability"] > 0.9)
    # ]

    fp_indices = fp_df["sample_idx"].astype(int).tolist()
    print(f"Found {len(fp_indices)} false positives")

    if not fp_indices:
        return

    # 2) Load manifest.jsonl
    manifest: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                manifest.append(json.loads(s))

    # 3) Optionally open Zarr store
    if LOAD_FROM_ZARR:
        try:
            import zarr  # type: ignore

            zroot = zarr.open_group(str(zarr_path), mode="r")
        except Exception as e:
            print(f"Could not open Zarr store at {zarr_path}: {e}")
            zroot = None
    else:
        zroot = None

    # 4) Print info for each FP (and optionally Zarr shapes)
    for idx in fp_indices:
        if idx < 0 or idx >= len(manifest):
            print(f"sample_idx {idx} is out of range for manifest of length {len(manifest)}")
            continue

        rec = manifest[idx]
        source_path = rec.get("source_path")
        key = rec.get("key")
        label_manifest = rec.get("label")

        row = fp_df[fp_df["sample_idx"] == idx].iloc[0]
        gt = int(row["ground_truth"])
        pred = int(row["predicted_label"])
        conf_real = float(row["confidence"])
        conf_fake = float(row["manipulation_probability"])

        msg = {
            "sample_idx": idx,
            "manifest_label": label_manifest,
            "metrics_ground_truth": gt,
            "metrics_predicted": pred,
            "confidence_real": conf_real,
            "confidence_fake": conf_fake,
            "source_path": source_path,  # original video path (if recorded)
            "zarr_key": key,             # group name in samples.zarr
        }

        # If Zarr is open, also show tensor shapes
        if zroot is not None and key is not None and str(key) in zroot:
            grp = zroot[str(key)]
            visual_shape = getattr(grp.get("visual"), "shape", None)
            audio_shape = getattr(grp.get("audio"), "shape", None)
            msg["visual_shape"] = visual_shape
            msg["audio_shape"] = audio_shape

        print(msg)


if __name__ == "__main__":
    main()
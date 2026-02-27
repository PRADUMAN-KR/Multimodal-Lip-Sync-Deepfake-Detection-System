#!/usr/bin/env python3
"""
Validate the full lip-sync detection pipeline on a dataset with real/ and fake/ folders.

Uses the same inference entrypoint as production (Predictor.predict_from_upload) so that
long-video, multi-face, mouth-motion checks, and verdict logic are all exercised.

Label convention (used everywhere: CSV, metrics, confusion matrix, ROC):
    - real (authentic)  -> 0
    - fake (manipulated)-> 1
    Ground truth: from folder (real/ -> 0, fake/ -> 1) or --label real|fake.
    Predicted: from pipeline verdict (real -> 0, fake -> 1; uncertain -> 0 if confidence>=0.5 else 1).
    Metrics: positive class = fake (1); precision/recall/F1 and ROC are for detecting fake.

Usage::

    python scripts/validate_pipeline.py --data_root /path/to/dataset --output_dir ./results
    python scripts/validate_pipeline.py --data_root ./data/eval --output_dir ./results --model weights/best_model.pth

Dataset layout::

    data_root/
        real/    <- videos with ground_truth=0 (real)
        fake/    <- videos with ground_truth=1 (fake)

Outputs: predictions.csv, metrics.json, confusion_matrix.png, roc_curve.png,
         high_confidence_errors.csv
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import UploadFile
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Run from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reduce log noise during batch run
logging.getLogger("app").setLevel(logging.WARNING)

from app.config import get_settings
from app.core.device import get_device
from app.inference.predictor import Predictor


# -----------------------------------------------------------------------------
# Pipeline runner (same entrypoint as API)
# -----------------------------------------------------------------------------

def _upload_file_from_path(path: Path) -> UploadFile:
    """Create an UploadFile from a local path so we can call predict_from_upload."""
    with open(path, "rb") as f:
        content = f.read()
    return UploadFile(filename=path.name, file=BytesIO(content))


def run_full_pipeline_sync(predictor: Predictor, video_path: Path) -> dict:
    """
    Run the full inference pipeline on a local video file.
    Uses the same code path as the API (predict_from_upload).
    """
    upload = _upload_file_from_path(video_path)
    return asyncio.run(predictor.predict_from_upload(upload))


# -----------------------------------------------------------------------------
# Path resolution (works from any cwd; supports absolute and relative paths)
# -----------------------------------------------------------------------------

def resolve_path(path: Path, must_exist: bool = False) -> Path:
    """
    Resolve data_root or output_dir so the script works from any working directory.
    - Absolute path: use as-is (resolved).
    - Relative path: try (project root / path) first, then (cwd / path).
      If must_exist=True, return the first that exists; else prefer project root.
    """
    if path.is_absolute():
        return path.resolve()
    from_project = (ROOT / path).resolve()
    from_cwd = (Path.cwd() / path).resolve()
    if must_exist:
        if from_project.exists():
            return from_project
        if from_cwd.exists():
            return from_cwd
        return from_project  # use for error message
    # For output_dir: prefer project root so ./results lands in project when possible
    return from_project if ROOT.exists() else from_cwd


# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------

def collect_videos(
    data_root: Path,
    real_dir: str = "real",
    fake_dir: str = "fake",
    real_only: bool = False,
    fake_only: bool = False,
    single_label: str | None = None,
) -> list[tuple[Path, int]]:
    """
    Collect (path, ground_truth) for videos.
    - single_label "real" or "fake": data_root is one folder of videos; all get that label.
    - real_only: only scan data_root/real_dir, all gt=0.
    - fake_only: only scan data_root/fake_dir, all gt=1.
    - else: scan both real_dir and fake_dir.
    ground_truth: 0 = real, 1 = fake.
    """
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Data root not found: {data_root}\n"
            "Use a path to a folder that contains 'real/' and 'fake/' (or --real_dir / --fake_dir).\n"
            "Or use --label real or --label fake for a single folder of videos."
        )

    extensions = {".mp4", ".mpg", ".mpeg", ".avi", ".mov", ".mkv", ".webm"}
    out: list[tuple[Path, int]] = []

    if single_label is not None:
        # Single folder: all videos get the same label
        gt = 0 if single_label.lower() == "real" else 1
        for path in data_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                out.append((path, gt))
        return sorted(out, key=lambda x: str(x[0]))

    if real_only:
        folder = data_root / real_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"Real folder not found: {folder} (--real_only)")
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                out.append((path, 0))
        return sorted(out, key=lambda x: str(x[0]))

    if fake_only:
        folder = data_root / fake_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"Fake folder not found: {folder} (--fake_only)")
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                out.append((path, 1))
        return sorted(out, key=lambda x: str(x[0]))

    # Both: real_dir and fake_dir
    for label_name, gt in [(real_dir, 0), (fake_dir, 1)]:
        folder = data_root / label_name
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                out.append((path, gt))
    return sorted(out, key=lambda x: str(x[0]))


# -----------------------------------------------------------------------------
# Inference and row extraction
# -----------------------------------------------------------------------------

def extract_row(
    video_path: Path,
    ground_truth: int,
    result: dict,
    inference_time_sec: float,
) -> dict:
    """Build a single row for predictions.csv."""
    verdict = result.get("verdict", "real")
    confidence = float(result.get("confidence", 0.0))
    manip_prob = float(result.get("manipulation_probability", 0.0))

    # Binary predicted label for metrics: real=0, fake=1. For uncertain, use threshold.
    if verdict == "real":
        predicted_label = 0
    elif verdict == "fake":
        predicted_label = 1
    else:
        predicted_label = 0 if confidence >= 0.5 else 1

    correct = 1 if predicted_label == ground_truth else 0

    return {
        "video_path": str(video_path),
        "ground_truth": ground_truth,
        "ground_truth_name": "real" if ground_truth == 0 else "fake",
        "predicted_label": predicted_label,
        "predicted_verdict": verdict,
        "confidence": confidence,
        "manipulation_probability": manip_prob,
        "correct": correct,
        "inference_time": round(inference_time_sec, 4),
        "selected_track_id": result.get("selected_track_id"),
        "speaking_tracks_count": result.get("speaking_tracks_count"),
        "temporal_drift": result.get("temporal_drift"),
        "video_duration_sec": result.get("video_duration_sec"),
    }


def is_high_confidence_error(row: dict) -> bool:
    """True if prediction was wrong and confidence in that prediction was > 0.9."""
    if row["correct"] == 1:
        return False
    conf_in_prediction = (
        row["confidence"] if row["predicted_label"] == 0 else row["manipulation_probability"]
    )
    return conf_in_prediction > 0.9


# -----------------------------------------------------------------------------
# Metrics and plots
# -----------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute accuracy, precision, recall, F1, FPR, FNR, and ROC AUC."""
    y_true = df["ground_truth"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()
    y_score = df["confidence"].to_numpy()  # P(real); higher = more real

    # Binary metrics (treat uncertain as threshold-based prediction)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # FPR = FP / (FP + TN), FNR = FN / (FN + TP). Force labels [0,1] for 2x2 matrix.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC AUC: y_score = P(real), so higher score = real. For AUC we need P(positive)
    # where positive = fake (1). So we use 1 - confidence as score for class 1.
    try:
        roc_auc = float(roc_auc_score(y_true, 1.0 - y_score))
    except ValueError:
        roc_auc = 0.0  # single class present

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "roc_auc": round(roc_auc, 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_samples": len(df),
        "num_real": int((y_true == 0).sum()),
        "num_fake": int((y_true == 1).sum()),
    }


def save_confusion_matrix_png(df: pd.DataFrame, output_path: Path) -> None:
    """Plot and save 2x2 confusion matrix."""
    y_true = df["ground_truth"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_roc_curve_png(df: pd.DataFrame, output_path: Path) -> None:
    """Plot and save ROC curve. Score for class 1 (fake) = 1 - confidence."""
    y_true = df["ground_truth"].to_numpy()
    y_score = 1.0 - df["confidence"].to_numpy()  # score for positive class = fake
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        fpr, tpr, auc = [0], [0], 0.0
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title("ROC Curve (positive class = fake)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Predictor setup (mirrors app/lifecycle.py)
# -----------------------------------------------------------------------------

def create_predictor(settings=None, model_path_override: Path | None = None) -> Predictor:
    """Build Predictor with the same config as production."""
    if settings is None:
        settings = get_settings()
    model_path = model_path_override or settings.model_path
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    device = get_device(settings.device)
    return Predictor(
        model_path=model_path,
        device=device,
        confidence_threshold=settings.confidence_threshold,
        use_torchscript=settings.use_torchscript,
        use_half_precision=settings.use_half_precision,
        uncertainty_margin=settings.uncertainty_margin,
        confidence_smoothing=settings.confidence_smoothing,
        trim_ratio=settings.trim_ratio,
        max_tracks=settings.max_tracks,
        refine_margin=settings.refine_margin,
        refine_top_k=settings.refine_top_k,
        chunk_size=settings.chunk_size,
        chunk_stride=settings.chunk_stride,
        long_video_threshold_sec=settings.long_video_threshold_sec,
        max_total_frames=settings.max_total_frames,
        confidence_margin=settings.confidence_margin,
        calibration_method=settings.calibration_method,
        calibration_temperature=settings.calibration_temperature,
        calibration_platt_a=settings.calibration_platt_a,
        calibration_platt_b=settings.calibration_platt_b,
        calibration_isotonic_path=settings.calibration_isotonic_path,
        mouth_motion_check=settings.mouth_motion_check,
        mouth_motion_low_threshold=settings.mouth_motion_low_threshold,
        mouth_motion_fake_penalty=settings.mouth_motion_fake_penalty,
        audio_energy_high_threshold=settings.audio_energy_high_threshold,
        audio_energy_low_threshold=settings.audio_energy_low_threshold,
        weak_real_gate=settings.weak_real_gate,
        weak_real_window_threshold=settings.weak_real_window_threshold,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate full lip-sync pipeline on real/ and fake/ dataset.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root folder containing real/ and fake/ subfolders with videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./results"),
        help="Directory for predictions.csv, metrics.json, plots (default: ./results).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model weights (default: from config).",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Limit number of videos (for quick runs).",
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default="real",
        help="Subfolder name for real/authentic videos (default: real).",
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        default="fake",
        help="Subfolder name for fake/manipulated videos (default: fake).",
    )
    parser.add_argument(
        "--real_only",
        action="store_true",
        help="Only run on videos in data_root/real_dir (all ground truth = real).",
    )
    parser.add_argument(
        "--fake_only",
        action="store_true",
        help="Only run on videos in data_root/fake_dir (all ground truth = fake).",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["real", "fake"],
        default=None,
        metavar="real|fake",
        help="Single folder of videos: data_root contains only real or only fake (no real/fake subdirs).",
    )
    args = parser.parse_args()

    if args.real_only and args.fake_only:
        parser.error("Use only one of --real_only, --fake_only, or --label.")
    if args.label and (args.real_only or args.fake_only):
        parser.error("Use only one of --real_only, --fake_only, or --label.")

    # Resolve paths: both absolute and relative work; relative tries project root then cwd
    data_root = resolve_path(args.data_root, must_exist=True)
    output_dir = resolve_path(args.output_dir, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect videos
    videos = collect_videos(
        data_root,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        real_only=args.real_only,
        fake_only=args.fake_only,
        single_label=args.label,
    )
    if not videos:
        subdirs = [p.name for p in data_root.iterdir() if p.is_dir()][:20]
        if args.label:
            print(f"No videos found under {data_root} (--label {args.label}).", file=sys.stderr)
        elif args.real_only or args.fake_only:
            d = args.real_dir if args.real_only else args.fake_dir
            print(f"No videos found under {data_root}/{d}.", file=sys.stderr)
        else:
            print(f"No videos found under {data_root}/{args.real_dir} or {data_root}/{args.fake_dir}.", file=sys.stderr)
        if subdirs and not args.label:
            print(f"Subfolders found: {subdirs}. Use --real_dir/--fake_dir, or --label real/fake for a single folder.", file=sys.stderr)
        elif not args.label:
            print("For a single folder of videos use: --label real or --label fake", file=sys.stderr)
        sys.exit(1)
    if args.n is not None:
        videos = videos[: args.n]
    print(f"Found {len(videos)} videos (real={sum(1 for _, g in videos if g == 0)}, fake={sum(1 for _, g in videos if g == 1)})")

    # Load predictor (full pipeline config)
    try:
        predictor = create_predictor(model_path_override=args.model)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    failed_paths: list[str] = []

    for video_path, ground_truth in tqdm(videos, desc="Inference", unit="video"):
        try:
            t0 = time.perf_counter()
            result = run_full_pipeline_sync(predictor, video_path)
            t1 = time.perf_counter()
            row = extract_row(video_path, ground_truth, result, t1 - t0)
            rows.append(row)
        except Exception as e:
            failed_paths.append(str(video_path))
            tqdm.write(f"FAILED {video_path}: {e}")
            continue

    predictor.close()

    if not rows:
        print("No successful predictions. Cannot compute metrics.", file=sys.stderr)
        sys.exit(1)

    # Build DataFrames
    df = pd.DataFrame(rows)
    df_high_conf_errors = df[df.apply(is_high_confidence_error, axis=1)]

    # Metrics
    metrics = compute_metrics(df)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # CSV
    df.to_csv(output_dir / "predictions.csv", index=False)
    df_high_conf_errors.to_csv(output_dir / "high_confidence_errors.csv", index=False)

    # Plots
    save_confusion_matrix_png(df, output_dir / "confusion_matrix.png")
    save_roc_curve_png(df, output_dir / "roc_curve.png")

    # Summary
    print("\n" + "=" * 60)
    print("  Validation results")
    print("=" * 60)
    print(f"  Total processed : {len(rows)}")
    print(f"  Failed          : {len(failed_paths)}")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  F1-score        : {metrics['f1_score']:.4f}")
    print(f"  FPR             : {metrics['false_positive_rate']:.4f}")
    print(f"  FNR             : {metrics['false_negative_rate']:.4f}")
    print(f"  ROC AUC         : {metrics['roc_auc']:.4f}")
    print(f"  Confusion matrix: TN={metrics['confusion_matrix']['tn']} FP={metrics['confusion_matrix']['fp']} "
          f"FN={metrics['confusion_matrix']['fn']} TP={metrics['confusion_matrix']['tp']}")
    print(f"  High-confidence errors: {len(df_high_conf_errors)} (saved to high_confidence_errors.csv)")
    print("=" * 60)
    print(f"\nOutputs written to: {output_dir}")
    print("  - predictions.csv")
    print("  - metrics.json")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - high_confidence_errors.csv")

    if failed_paths:
        fail_log = output_dir / "failed_videos.txt"
        with open(fail_log, "w") as f:
            f.write("\n".join(failed_paths))
        print(f"  - failed_videos.txt ({len(failed_paths)} entries)")


if __name__ == "__main__":
    main()

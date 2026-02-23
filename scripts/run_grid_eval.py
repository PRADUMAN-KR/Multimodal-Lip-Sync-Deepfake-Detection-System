#!/usr/bin/env python3
"""
Run lip-sync detection on GRID Sheffield .mpg files and compute
Precision, Recall, F1 and Accuracy when ground-truth labels are provided.

Usage::

    # Evaluate all .mpg files in a directory (assumes all videos are REAL)
    python3 scripts/run_grid_eval.py --dir data/grid_sheffield/s1

    # Limit to 20 files
    python3 scripts/run_grid_eval.py --dir data/grid_sheffield/s1 -n 20

    # Provide a labels CSV (columns: file_path, label) for fake-included eval
    python3 scripts/run_grid_eval.py --labels data/val_labels.csv

    # Save results to CSV
    python3 scripts/run_grid_eval.py --dir data/grid_sheffield/s1 -o results/grid_eval.csv

Labels CSV format::

    file_path,label
    data/grid_sheffield/s1/bbaf2n.mpg,real
    data/deepfakes/fake01.mp4,fake
"""

import argparse
import csv
import sys
from pathlib import Path

# Run from project root so app is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.core.device import get_device
from app.inference.predictor import Predictor
from app.utils.metrics import compute_metrics_at_threshold, find_best_threshold


def _load_labels_csv(csv_path: Path) -> dict:
    """Load {file_path: 'real'|'fake'} from a CSV with columns file_path,label."""
    mapping: dict = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = Path(row["file_path"].strip())
            if not fp.is_absolute():
                fp = ROOT / fp
            label = row["label"].strip().lower()
            mapping[str(fp)] = label
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate lip-sync model on a directory of videos."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory of .mpg files to evaluate (all treated as REAL unless --labels given).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="CSV file with columns: file_path, label (real/fake). "
             "Takes precedence over --dir.",
    )
    parser.add_argument(
        "-n", type=int, default=None, help="Limit number of files (default: all)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model weights (default: from config).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Write per-file results CSV here."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for P(REAL). Default: from config.",
    )
    args = parser.parse_args()

    if args.labels is None and args.dir is None:
        parser.error("Provide --dir or --labels.")

    # ── Build (file_path, label) list ─────────────────────────────────────────
    file_label_pairs: list[tuple[Path, str]] = []

    if args.labels is not None:
        labels_path = args.labels if args.labels.is_absolute() else ROOT / args.labels
        if not labels_path.is_file():
            print(f"Labels file not found: {labels_path}", file=sys.stderr)
            sys.exit(1)
        label_map = _load_labels_csv(labels_path)
        for fp_str, label in label_map.items():
            fp = Path(fp_str)
            if fp.is_file():
                file_label_pairs.append((fp, label))
            else:
                print(f"Warning: file not found, skipping: {fp}", file=sys.stderr)
    else:
        dir_path = args.dir if args.dir.is_absolute() else ROOT / args.dir
        if not dir_path.is_dir():
            print(f"Directory not found: {dir_path}", file=sys.stderr)
            sys.exit(1)
        mpg_files = sorted(dir_path.glob("*.mpg"))
        if not mpg_files:
            print(f"No .mpg files in {dir_path}", file=sys.stderr)
            sys.exit(1)
        if args.n is not None:
            mpg_files = mpg_files[: args.n]
        file_label_pairs = [(p, "real") for p in mpg_files]

    if args.n is not None and args.labels is not None:
        file_label_pairs = file_label_pairs[: args.n]

    # ── Load model ────────────────────────────────────────────────────────────
    settings = get_settings()
    model_path = args.model or settings.model_path
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        print(
            f"Model not found: {model_path}. Train a model or set --model.",
            file=sys.stderr,
        )
        sys.exit(1)

    threshold = args.threshold if args.threshold is not None else settings.confidence_threshold

    device = get_device(settings.device)
    predictor = Predictor(
        model_path=model_path,
        device=device,
        confidence_threshold=threshold,
        use_torchscript=settings.use_torchscript,
        use_half_precision=settings.use_half_precision,
        calibration_method=settings.calibration_method,
        calibration_temperature=settings.calibration_temperature,
        calibration_platt_a=settings.calibration_platt_a,
        calibration_platt_b=settings.calibration_platt_b,
        calibration_isotonic_path=settings.calibration_isotonic_path,
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    rows: list[dict] = []
    for i, (fp, true_label) in enumerate(file_label_pairs):
        try:
            out = predictor.predict_from_path(fp)
            row = {
                "file": fp.name,
                "true_label": true_label,
                "true_is_fake": true_label == "fake",
                "predicted_is_real": out["is_real"],
                "predicted_is_fake": out["is_fake"],
                "confidence": out["confidence"],
                "manipulation_probability": out["manipulation_probability"],
                "error": "",
            }
            rows.append(row)
            print(
                f"[{i+1}/{len(file_label_pairs)}] {fp.name}  "
                f"true={true_label}  predicted={'fake' if out['is_fake'] else 'real'}  "
                f"conf={out['confidence']:.4f}  manip={out['manipulation_probability']:.4f}"
            )
        except Exception as e:
            print(f"[{i+1}/{len(file_label_pairs)}] {fp.name}  ERROR: {e}", file=sys.stderr)
            rows.append({
                "file": fp.name,
                "true_label": true_label,
                "true_is_fake": true_label == "fake",
                "predicted_is_real": None,
                "predicted_is_fake": None,
                "confidence": None,
                "manipulation_probability": None,
                "error": str(e),
            })

    # ── Precision / Recall / F1 ───────────────────────────────────────────────
    valid = [r for r in rows if r["confidence"] is not None]
    if valid:
        y_true = [int(r["true_is_fake"]) for r in valid]
        scores = [float(r["confidence"]) for r in valid]

        metrics = compute_metrics_at_threshold(y_true, scores, threshold=threshold)

        print("\n" + "=" * 60)
        print(f"  Evaluation Results  (threshold={threshold:.3f})")
        print("=" * 60)
        print(f"  Total evaluated : {metrics['total']}")
        print(f"  Precision       : {metrics['precision']:.4f}")
        print(f"  Recall          : {metrics['recall']:.4f}")
        print(f"  F1 Score        : {metrics['f1']:.4f}")
        print(f"  Accuracy        : {metrics['accuracy']:.4f}")
        print(f"  TP={metrics['tp']}  TN={metrics['tn']}  "
              f"FP={metrics['fp']}  FN={metrics['fn']}")

        # Find the threshold that maximises F1
        if len(set(y_true)) > 1:  # only meaningful with both classes present
            best = find_best_threshold(y_true, scores, metric="f1")
            print(f"\n  Best F1 threshold : {best['best_threshold']:.3f}  "
                  f"(F1={best['f1']:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f})")
        print("=" * 60)
    else:
        print("\nNo valid predictions to compute metrics.", file=sys.stderr)

    # ── Write output CSV ──────────────────────────────────────────────────────
    if args.output:
        out_path = args.output if args.output.is_absolute() else ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "file", "true_label", "predicted_is_real", "predicted_is_fake",
            "confidence", "manipulation_probability", "error",
        ]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {out_path}")

    print(f"\nDone. Processed {len(rows)} files ({len(valid)} successful).")


if __name__ == "__main__":
    main()

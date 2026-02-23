#!/usr/bin/env python3
"""
Fit confidence calibration parameters on a labelled validation set and
print the settings to paste into app/config.py.

Supports three calibration methods:

temperature
    Divides the model logit by T before sigmoid.  T > 1 softens
    (reduces overconfidence).  Fit by minimising negative log-likelihood.

platt
    Affine transform of the logit: sigmoid(a * logit + b).
    Fit by minimising NLL with Nelder-Mead.

isotonic
    Monotone regression on raw probabilities (sklearn IsotonicRegression).
    Saves a .pkl file you point ``calibration_isotonic_path`` at.

Labels CSV format::

    file_path,label
    /abs/or/relative/path/video.mp4,real
    /abs/or/relative/path/fake.mp4,fake

Usage::

    python scripts/fit_calibrator.py --labels data/val_labels.csv --method temperature
    python scripts/fit_calibrator.py --labels data/val_labels.csv --method platt
    python scripts/fit_calibrator.py --labels data/val_labels.csv --method isotonic \\
        --output weights/isotonic_calibrator.pkl

After running, copy the printed ``calibration_*`` lines into ``app/config.py``.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.core.device import get_device
from app.inference.predictor import Predictor


def _load_labels(csv_path: Path) -> list[tuple[Path, int]]:
    """Return [(file_path, label)] where label 0=real, 1=fake."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = Path(row["file_path"].strip())
            if not fp.is_absolute():
                fp = ROOT / fp
            label = 0 if row["label"].strip().lower() == "real" else 1
            rows.append((fp, label))
    return rows


def _collect_logits(predictor: Predictor, rows: list[tuple[Path, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and collect (raw_confidence, label) pairs."""
    confidences, labels = [], []
    n = len(rows)
    for i, (fp, label) in enumerate(rows):
        if not fp.is_file():
            print(f"  [{i+1}/{n}] SKIP (not found): {fp}", file=sys.stderr)
            continue
        try:
            out = predictor.predict_from_path(fp)
            conf = float(out["confidence"])
            confidences.append(conf)
            labels.append(label)
            print(f"  [{i+1}/{n}] {fp.name}  conf={conf:.4f}  label={'fake' if label else 'real'}")
        except Exception as e:
            print(f"  [{i+1}/{n}] ERROR: {e}", file=sys.stderr)

    return np.array(confidences, dtype=np.float64), np.array(labels, dtype=int)


def _fit_temperature(confidences: np.ndarray, labels: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        T = max(T, 1e-3)
        logits = np.log(
            np.clip(confidences, 1e-7, 1 - 1e-7)
            / np.clip(1 - confidences, 1e-7, 1 - 1e-7)
        )
        p_real = 1.0 / (1.0 + np.exp(-logits / T))
        p_correct = np.where(labels == 0, p_real, 1.0 - p_real)
        return -float(np.mean(np.log(np.clip(p_correct, 1e-7, 1.0))))

    result = minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    return float(result.x)


def _fit_platt(confidences: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import minimize

    logits = np.log(
        np.clip(confidences, 1e-7, 1 - 1e-7)
        / np.clip(1 - confidences, 1e-7, 1 - 1e-7)
    )

    def nll(params: list) -> float:
        a, b = params
        p_real = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        p_correct = np.where(labels == 0, p_real, 1.0 - p_real)
        return -float(np.mean(np.log(np.clip(p_correct, 1e-7, 1.0))))

    result = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    return float(result.x[0]), float(result.x[1])


def _fit_isotonic(
    confidences: np.ndarray, labels: np.ndarray, out_path: Path
) -> None:
    from sklearn.isotonic import IsotonicRegression

    # Map P(REAL) confidences → true P(REAL) (1 - label)
    y_true_real = (1 - labels).astype(float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(confidences.reshape(-1, 1), y_true_real)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(ir, f)
    print(f"\nIsotonic calibrator saved → {out_path}")
    print(
        f"\nPaste into app/config.py:\n"
        f"    calibration_method = 'isotonic'\n"
        f"    calibration_isotonic_path = '{out_path}'"
    )


def main():
    parser = argparse.ArgumentParser(description="Fit confidence calibration parameters.")
    parser.add_argument(
        "--labels", type=Path, required=True,
        help="CSV with columns: file_path, label (real/fake)."
    )
    parser.add_argument(
        "--method",
        choices=["temperature", "platt", "isotonic"],
        default="temperature",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .pkl path for isotonic method (default: weights/isotonic_calibrator.pkl)."
    )
    args = parser.parse_args()

    labels_path = args.labels if args.labels.is_absolute() else ROOT / args.labels
    if not labels_path.is_file():
        print(f"Labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    rows = _load_labels(labels_path)
    if not rows:
        print("No rows loaded from labels CSV.", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    model_path = args.model or settings.model_path
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    device = get_device(settings.device)
    # Load predictor WITHOUT any existing calibration so we get raw logits.
    predictor = Predictor(
        model_path=model_path,
        device=device,
        confidence_threshold=settings.confidence_threshold,
        use_torchscript=False,
        use_half_precision=settings.use_half_precision,
        calibration_method="none",
    )

    print(f"Collecting predictions from {len(rows)} validation samples...")
    confidences, labels = _collect_logits(predictor, rows)

    if len(confidences) == 0:
        print("No predictions collected. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFitting {args.method} calibration on {len(confidences)} samples "
          f"(real={int(sum(labels == 0))}, fake={int(sum(labels == 1))})...")

    if args.method == "temperature":
        T = _fit_temperature(confidences, labels)
        print(f"\n  Optimal Temperature T = {T:.4f}")
        print(
            f"\nPaste into app/config.py:\n"
            f"    calibration_method = 'temperature'\n"
            f"    calibration_temperature = {T:.4f}"
        )

    elif args.method == "platt":
        a, b = _fit_platt(confidences, labels)
        print(f"\n  Platt: a = {a:.4f},  b = {b:.4f}")
        print(
            f"\nPaste into app/config.py:\n"
            f"    calibration_method = 'platt'\n"
            f"    calibration_platt_a = {a:.4f}\n"
            f"    calibration_platt_b = {b:.4f}"
        )

    elif args.method == "isotonic":
        out_path = args.output or ROOT / "weights" / "isotonic_calibrator.pkl"
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        _fit_isotonic(confidences, labels, out_path)


if __name__ == "__main__":
    main()

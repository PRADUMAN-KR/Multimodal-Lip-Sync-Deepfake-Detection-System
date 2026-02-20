#!/usr/bin/env python3
"""
Run lip-sync detection on GRID Sheffield .mpg files (e.g. data/grid_sheffield/s1).

Requires: trained weights at weights/best_model.pth (or set MODEL_PATH).
  python3 scripts/run_grid_eval.py
  python3 scripts/run_grid_eval.py --dir data/grid_sheffield/s1 -n 20
"""

import argparse
import sys
from pathlib import Path

# Run from project root so app is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.core.device import get_device
from app.inference.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="Run lip-sync on GRID .mpg files.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "data" / "grid_sheffield" / "s1",
        help="Directory containing .mpg files",
    )
    parser.add_argument("-n", type=int, default=None, help="Limit number of files (default: all)")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model weights (default: from config)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write results CSV here")
    args = parser.parse_args()

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

    settings = get_settings()
    model_path = args.model or settings.model_path
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        print(f"Model not found: {model_path}. Train a model or set --model.", file=sys.stderr)
        sys.exit(1)

    device = get_device(settings.device)
    predictor = Predictor(
        model_path=model_path,
        device=device,
        confidence_threshold=settings.confidence_threshold,
        use_torchscript=settings.use_torchscript,
        use_half_precision=settings.use_half_precision,
    )

    results = []
    for i, p in enumerate(mpg_files):
        try:
            out = predictor.predict_from_path(p)
            results.append((p.name, out["is_real"], out["confidence"], out["manipulation_probability"]))
            print(f"[{i+1}/{len(mpg_files)}] {p.name}  is_real={out['is_real']}  conf={out['confidence']:.4f}  manip_prob={out['manipulation_probability']:.4f}")
        except Exception as e:
            print(f"[{i+1}/{len(mpg_files)}] {p.name}  ERROR: {e}", file=sys.stderr)
            results.append((p.name, None, None, None))

    if args.output:
        out_path = args.output if args.output.is_absolute() else ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("file,is_real,confidence,manipulation_probability\n")
            for name, is_real, conf, manip_prob in results:
                r = str(is_real) if is_real is not None else ""
                c = f"{conf:.6f}" if conf is not None else ""
                m = f"{manip_prob:.6f}" if manip_prob is not None else ""
                f.write(f"{name},{r},{c},{m}\n")
        print(f"Wrote {out_path}")

    print(f"Done. Processed {len(results)} files.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download the full GRID audiovisual corpus in one go from Zenodo.

- 34 speakers (s1–s34; s21 has no video on Zenodo)
- Each: 1000 sentences, audio (25 kHz) + video (as .jpg frames in Zenodo version)
- Total ~16 GB. Saves to data/grid_corpus/

Usage:
  pip install zenodo_get   # one-time
  python3 scripts/download_grid_corpus.py

Or without zenodo_get: use --urls to print direct URLs and download manually / with curl.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ZENODO_RECORD = "3625687"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
# All files (no s21 video on Zenodo)
FILES = [
    "alignments.zip",
    "audio_25k.zip",
    "jasagrid.pdf",
] + [f"s{i}.zip" for i in list(range(1, 21)) + list(range(22, 35))]


def main():
    parser = argparse.ArgumentParser(description="Download GRID corpus from Zenodo.")
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: project_root/data/grid_corpus)",
    )
    parser.add_argument(
        "--urls",
        action="store_true",
        help="Only print download URLs (for curl/wget manual download)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not extract zip files after download",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = args.output_dir or root / "data" / "grid_corpus"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.urls:
        for f in FILES:
            print(f"{BASE_URL}/{f}?download=1")
        print("\n# Save to same directory, then unzip. Example:", file=sys.stderr)
        print(f"# cd {out_dir} && for f in *.zip; do unzip -o \"$f\"; done", file=sys.stderr)
        return

    # Use zenodo_get (one command for full record)
    if shutil.which("zenodo_get"):
        print(f"Downloading GRID corpus (record {ZENODO_RECORD}) to {out_dir}")
        print("This may take a while (~16 GB).")
        cmd = [
            "zenodo_get",
            ZENODO_RECORD,
            "-o", str(out_dir),
        ]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("zenodo_get failed. Try: pip install zenodo_get", file=sys.stderr)
            sys.exit(1)
        if not args.no_extract:
            _extract_zips(out_dir)
        print(f"Done. Data in {out_dir}")
        return

    # zenodo_get not installed
    print("Install zenodo_get for one-command download:", file=sys.stderr)
    print("  pip install zenodo_get", file=sys.stderr)
    print("Then run: python3 scripts/download_grid_corpus.py", file=sys.stderr)
    print("\nOr print URLs and download with curl:", file=sys.stderr)
    print("  python3 scripts/download_grid_corpus.py --urls", file=sys.stderr)
    sys.exit(1)


def _extract_zips(directory: Path) -> None:
    for z in directory.glob("*.zip"):
        print(f"Extracting {z.name} ...")
        subprocess.run(["unzip", "-o", str(z), "-d", str(directory)], check=False)


if __name__ == "__main__":
    main()

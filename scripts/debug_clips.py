#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import json

# Run from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training.dataset import LipSyncDataset


def extract_audio_energy(path: Path, sr: int = 16000) -> np.ndarray:
    audio, sr = librosa.load(str(path), sr=sr)

    frame = 512
    hop = 256

    energy = []
    for i in range(0, len(audio) - frame, hop):
        chunk = audio[i : i + frame]
        energy.append(np.mean(chunk**2))

    energy = np.array(energy, dtype=np.float32)
    if energy.size == 0:
        return energy

    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
    return energy


def extract_mouth_motion(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    prev = None
    motion: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is not None:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32))))
            motion.append(diff)

        prev = gray

    cap.release()

    motion = np.array(motion, dtype=np.float32)
    if motion.size == 0:
        return motion

    motion = (motion - motion.min()) / (motion.max() - motion.min() + 1e-6)
    return motion


def resolve_video_from_sample_idx(
    sample_idx: int,
    preprocessed_dir: Path,
) -> Path:
    """
    Given a sample_idx from predictions.csv, look up manifest.jsonl under
    `preprocessed_dir` and return the source_path as a Path.
    """
    manifest_path = preprocessed_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            manifest.append(json.loads(s))

    if sample_idx < 0 or sample_idx >= len(manifest):
        raise IndexError(f"sample_idx {sample_idx} out of range for manifest length {len(manifest)}")

    rec = manifest[sample_idx]
    source_path = rec.get("source_path")
    if not source_path:
        raise ValueError(f"Manifest entry {sample_idx} has no 'source_path': {rec}")

    return Path(source_path)


def _load_manifest_record(sample_idx: int, preprocessed_dir: Path) -> dict:
    manifest_path = preprocessed_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            manifest.append(json.loads(s))

    if sample_idx < 0 or sample_idx >= len(manifest):
        raise IndexError(f"sample_idx {sample_idx} out of range for manifest length {len(manifest)}")
    return manifest[sample_idx]


def export_exact_preprocessed_clip(
    sample_idx: int,
    preprocessed_dir: Path,
    storage_format: str = "zarr",
    video_frames: int = 32,
    audio_frames: int = 128,
    output_dir: Path = Path("debug_exact_clips"),
    fps: float = 15.0,
) -> Path:
    """
    Export the exact visual tensor used by preprocessed validation for sample_idx.
    This uses the same dataset path as validate_pipeline preprocessed mode:
    dataset.get_item(idx, train_mode_override=False).
    """
    record = _load_manifest_record(sample_idx, preprocessed_dir)
    dataset = LipSyncDataset(
        data_dir=preprocessed_dir,
        preprocessed_dir=preprocessed_dir,
        storage_format=storage_format,
        video_frames=video_frames,
        audio_frames=audio_frames,
        require_face_detection=False,
    )
    item = dataset.get_item(sample_idx, train_mode_override=False)
    if item is None:
        raise RuntimeError(f"Could not load preprocessed sample {sample_idx}")

    visual_t, audio_t, label_t = item
    visual = visual_t.detach().cpu().numpy()  # (3, T, H, W)
    audio = audio_t.detach().cpu().numpy()    # (1, 80, T_a)

    if visual.ndim != 4 or visual.shape[0] != 3:
        raise ValueError(f"Unexpected visual tensor shape: {visual.shape}")

    # Convert to uint8 frames (T, H, W, 3)
    frames = np.transpose(visual, (1, 2, 3, 0))
    if frames.max() <= 1.0:
        frames = frames * 255.0
    frames_u8 = np.clip(frames, 0, 255).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{sample_idx:05d}"
    out_video = output_dir / f"{stem}.mp4"
    out_visual = output_dir / f"{stem}_visual.npy"
    out_audio = output_dir / f"{stem}_audio.npy"
    out_meta = output_dir / f"{stem}_meta.json"

    h, w = int(frames_u8.shape[1]), int(frames_u8.shape[2])
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {out_video}")

    for frame in frames_u8:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    np.save(out_visual, visual, allow_pickle=False)
    np.save(out_audio, audio, allow_pickle=False)

    meta = {
        "sample_idx": sample_idx,
        "source_path": record.get("source_path"),
        "key": record.get("key"),
        "precompute_mode": record.get("precompute_mode"),
        "label_manifest": record.get("label"),
        "visual_shape": list(visual.shape),
        "audio_shape": list(audio.shape),
        "video_path": str(out_video),
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported exact preprocessed clip video: {out_video}")
    print(f"Saved tensors: {out_visual}, {out_audio}")
    print(f"Saved metadata: {out_meta}")
    return out_video


def plot_sync_debug(video_path: Path, title_suffix: str | None = None) -> None:
    print(f"Debugging video: {video_path}")
    audio_energy = extract_audio_energy(video_path)
    mouth_motion = extract_mouth_motion(video_path)

    if audio_energy.size == 0 or mouth_motion.size == 0:
        print("Empty audio or motion sequence; nothing to plot.")
        return

    min_len = min(len(audio_energy), len(mouth_motion))
    audio_energy = audio_energy[:min_len]
    mouth_motion = mouth_motion[:min_len]

    sync_error = np.abs(audio_energy - mouth_motion)

    plt.figure(figsize=(12, 6))
    plt.plot(audio_energy, label="audio energy")
    plt.plot(mouth_motion, label="mouth motion")
    plt.plot(sync_error, label="sync error")
    plt.legend()
    base_title = "Lip-Sync Debug Plot"
    if title_suffix:
        plt.title(f"{base_title} – {title_suffix}")
    else:
        plt.title(base_title)
    plt.tight_layout()
    # Save figure for headless/server environments
    out_dir = Path("debug_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug lip-sync on a specific clip (by video path or sample_idx)."
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        default=None,
        help="Path to a video file to debug directly.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="sample_idx from predictions.csv (preprocessed mode). "
        "Requires --preprocessed_dir.",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=Path,
        default=None,
        help="Preprocessed dataset dir (manifest.jsonl + samples.zarr/.npy/lmdb). "
        "Required if using --sample_idx.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Optional: results dir containing predictions.csv. Used only for context; "
        "not strictly required just to plot.",
    )
    parser.add_argument(
        "--export_exact_clip",
        action="store_true",
        help="When using --sample_idx + --preprocessed_dir, export the exact "
        "preprocessed clip used by validation to debug_exact_clips/.",
    )
    parser.add_argument(
        "--storage_format",
        type=str,
        choices=["zarr", "npy", "lmdb"],
        default="zarr",
        help="Storage format for preprocessed data when exporting exact clip.",
    )
    parser.add_argument(
        "--video_frames",
        type=int,
        default=32,
        help="video_frames used in preprocessed validation (default: 32).",
    )
    parser.add_argument(
        "--audio_frames",
        type=int,
        default=128,
        help="audio_frames used in preprocessed validation (default: 128).",
    )
    args = parser.parse_args()

    if args.video_path is None and args.sample_idx is None:
        parser.error("Provide either --video_path or --sample_idx.")

    if args.sample_idx is not None and args.preprocessed_dir is None:
        parser.error("--preprocessed_dir is required when using --sample_idx.")

    if args.sample_idx is not None:
        video_path = resolve_video_from_sample_idx(args.sample_idx, args.preprocessed_dir)
        title_suffix = f"sample_idx={args.sample_idx}"
    else:
        video_path = args.video_path
        title_suffix = video_path.name if video_path is not None else None

    if args.export_exact_clip:
        if args.sample_idx is None or args.preprocessed_dir is None:
            parser.error("--export_exact_clip requires --sample_idx and --preprocessed_dir.")
        export_exact_preprocessed_clip(
            sample_idx=args.sample_idx,
            preprocessed_dir=args.preprocessed_dir,
            storage_format=args.storage_format,
            video_frames=args.video_frames,
            audio_frames=args.audio_frames,
        )

    if video_path is None:
        parser.error("Could not resolve video path.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    plot_sync_debug(video_path, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
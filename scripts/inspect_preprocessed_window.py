#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_manifest(preprocessed_dir: Path) -> list[dict]:
    manifest_path = preprocessed_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    out: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    if not out:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return out


def get_full_visual_length(preprocessed_dir: Path, rec: dict, storage_format: str) -> int:
    mode = str(rec.get("precompute_mode", "fixed_clip"))
    if mode == "fixed_clip":
        # Already one model-ready sample; no windowing from full sequence.
        return int(rec.get("video_frames", 32))

    if storage_format == "zarr":
        try:
            import zarr  # type: ignore
        except ImportError as e:
            raise RuntimeError("zarr not installed. Install with: pip install zarr") from e
        key = str(rec.get("key", ""))
        root = zarr.open_group(str(preprocessed_dir / "samples.zarr"), mode="r")
        if key not in root:
            raise KeyError(f"Zarr key not found: {key}")
        visual_shape = tuple(root[key]["visual"].shape)
        # full_sequence visual is (T, H, W, C)
        if len(visual_shape) != 4:
            raise ValueError(f"Unexpected full visual shape for key {key}: {visual_shape}")
        return int(visual_shape[0])

    if storage_format == "npy":
        visual_rel = rec.get("visual_relpath")
        if not visual_rel:
            raise KeyError("visual_relpath missing in manifest record")
        visual_path = preprocessed_dir / str(visual_rel)
        arr = np.load(visual_path, mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(f"Unexpected full visual shape: {arr.shape}")
        return int(arr.shape[0])

    raise NotImplementedError(
        "LMDB full-sequence length lookup is not implemented in this helper. "
        "Use zarr or npy preprocessed storage."
    )


def compute_window(t_total: int, video_frames: int) -> tuple[int, int, int]:
    """
    Match dataset._sample_aligned_contiguous_clip(train_mode=False):
      - if t_total > video_frames: center window
      - else: start at 0 and tail-pad to video_frames
    Returns (start, end_exclusive, pad_tail_frames).
    """
    if t_total <= 0:
        raise ValueError("Empty visual sequence")
    if t_total > video_frames:
        start = int((t_total - video_frames) // 2)
        end = int(start + video_frames)
        pad = 0
    else:
        start = 0
        end = int(t_total)
        pad = int(video_frames - t_total)
    return start, end, pad


def export_source_segment_with_audio(
    source_path: Path,
    start_sec: float,
    duration_sec: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration_sec:.6f}",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def save_window_plot(report: dict, out_path: Path) -> None:
    total = int(report["full_sequence_total_frames"])
    start = int(report["window_start_frame"])
    end = int(report["window_end_frame_exclusive"])
    fps = float(report["target_fps"])
    duration_sec = float(report["window_duration_sec"])

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axhspan(0.35, 0.65, color="#d9d9d9", alpha=0.6, zorder=1, label="full sequence")
    ax.axvspan(start, end, color="#2ca02c", alpha=0.35, zorder=2, label="validation window")
    ax.plot([0, total], [0.5, 0.5], color="black", linewidth=1.2, zorder=3)

    ax.set_xlim(0, max(total, 1))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Preprocessed frame index")
    ax.set_title(
        f"sample_idx={report['sample_idx']} | window=[{start}, {end}) "
        f"| fps={fps:.2f} | duration={duration_sec:.2f}s"
    )

    # Add secondary x-axis in seconds for easier reading.
    def _f_to_s(x):
        return x / max(fps, 1e-6)

    def _s_to_f(x):
        return x * fps

    sec_ax = ax.secondary_xaxis("top", functions=(_f_to_s, _s_to_f))
    sec_ax.set_xlabel("Time (seconds)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_full_sequence_curves_zarr(
    preprocessed_dir: Path, rec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    For precompute_mode='full_sequence' + zarr storage:
      - visual_full: (T, H, W, C) → per-frame mouth motion (diff over time)
      - audio_full:  (80, T_audio) → per-frame audio energy aligned to visual frames
    Returns (motion_curve, audio_curve) both shape (T,).
    """
    try:
        import zarr  # type: ignore
    except ImportError as e:
        raise RuntimeError("zarr not installed. Install with: pip install zarr") from e

    key = str(rec.get("key", ""))
    root = zarr.open_group(str(preprocessed_dir / "samples.zarr"), mode="r")
    if key not in root:
        raise KeyError(f"Zarr key not found: {key}")
    grp = root[key]
    visual = grp["visual"][:]  # (T, H, W, C) uint8
    audio_full = grp["audio"][:]  # (80, T_audio) float32

    if visual.ndim != 4 or visual.shape[-1] != 3:
        raise ValueError(f"Unexpected full visual shape: {visual.shape}")
    if audio_full.ndim != 2 or audio_full.shape[0] != 80:
        raise ValueError(f"Unexpected full audio shape: {audio_full.shape}")

    # Mouth motion: mean abs diff between consecutive frames in lower half.
    frames = visual.astype(np.float32) / 255.0  # (T, H, W, C)
    frames_gray = frames.mean(axis=3)  # (T, H, W)
    H = int(frames_gray.shape[1])
    mouth = frames_gray[:, H // 2 :, :]  # (T, H/2, W)
    if mouth.shape[0] < 2:
        motion = np.zeros(mouth.shape[0], dtype=np.float32)
    else:
        diff = np.abs(np.diff(mouth, axis=0)).mean(axis=(1, 2))  # (T-1,)
        motion = np.concatenate([[diff[0]], diff], axis=0)  # align length T

    # Audio energy: mean mel power per frame, aligned via fps and mel_hz.
    t_total = int(visual.shape[0])
    mel_hz = float(rec.get("mel_hz", 100.0))  # 100 mel frames/sec by default
    fps = float(rec.get("target_fps", 15.0))
    T_audio = int(audio_full.shape[1])

    # Map each visual frame center time to mel index.
    times = np.arange(t_total, dtype=np.float32) / max(fps, 1e-6)
    mel_indices = np.clip(
        np.round(times * mel_hz).astype(np.int64), 0, max(T_audio - 1, 0)
    )
    mel_power = audio_full  # (80, T_audio)
    mel_energy_all = mel_power.mean(axis=0)  # (T_audio,)
    audio_curve = mel_energy_all[mel_indices]  # (T,)

    # Normalize both to [0,1] for plotting comparably.
    def _norm(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        x = x.astype(np.float32)
        mn = float(x.min())
        mx = float(x.max())
        if mx - mn < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    motion_n = _norm(motion)
    audio_n = _norm(audio_curve)
    return motion_n, audio_n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect exact frame window used by preprocessed validation for a sample_idx. "
            "Optionally export the corresponding source-video segment with audio."
        )
    )
    parser.add_argument("--sample_idx", type=int, required=True, help="sample_idx from predictions.csv")
    parser.add_argument(
        "--preprocessed_dir",
        type=Path,
        required=True,
        help="Preprocessed dir containing manifest.jsonl + samples.zarr/.npy",
    )
    parser.add_argument(
        "--storage_format",
        type=str,
        choices=["zarr", "npy", "lmdb"],
        default="zarr",
        help="Storage format of preprocessed tensors.",
    )
    parser.add_argument(
        "--video_frames",
        type=int,
        default=32,
        help="video_frames used by validation (default: 32).",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=None,
        help="Override FPS for frame->time mapping. Default: use manifest target_fps or 15.0.",
    )
    parser.add_argument(
        "--export_source_segment",
        action="store_true",
        help="Export matching source-video segment with audio using ffmpeg.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("debug_exact_clips"),
        help="Output directory for report and optional source segment export.",
    )
    parser.add_argument(
        "--plot_full_sequence",
        action="store_true",
        help="Also plot full-sequence mouth motion and audio energy curves with "
        "the validation window highlighted (zarr + full_sequence only).",
    )
    args = parser.parse_args()

    preprocessed_dir = args.preprocessed_dir.resolve()
    manifest = load_manifest(preprocessed_dir)
    if args.sample_idx < 0 or args.sample_idx >= len(manifest):
        raise IndexError(f"sample_idx {args.sample_idx} out of range [0, {len(manifest)-1}]")

    rec = manifest[args.sample_idx]
    mode = str(rec.get("precompute_mode", "fixed_clip"))
    source_path = Path(str(rec.get("source_path", "")))
    fps = float(args.target_fps) if args.target_fps is not None else float(rec.get("target_fps", 15.0))

    t_total = get_full_visual_length(preprocessed_dir, rec, args.storage_format)
    start_f, end_f, pad_tail = compute_window(t_total, args.video_frames)
    duration_frames = args.video_frames
    start_sec = start_f / max(fps, 1e-6)
    duration_sec = duration_frames / max(fps, 1e-6)
    end_sec = start_sec + duration_sec

    report = {
        "sample_idx": int(args.sample_idx),
        "precompute_mode": mode,
        "source_path": str(source_path),
        "key": rec.get("key"),
        "target_fps": fps,
        "full_sequence_total_frames": int(t_total),
        "validation_video_frames": int(args.video_frames),
        "window_start_frame": int(start_f),
        "window_end_frame_exclusive": int(end_f),
        "window_duration_frames": int(duration_frames),
        "pad_tail_frames": int(pad_tail),
        "window_start_sec": round(float(start_sec), 6),
        "window_end_sec": round(float(end_sec), 6),
        "window_duration_sec": round(float(duration_sec), 6),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"sample_{args.sample_idx:05d}_window.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved window report: {out_json}")

    out_plot = args.output_dir / f"sample_{args.sample_idx:05d}_window.png"
    save_window_plot(report, out_plot)
    print(f"Saved window plot: {out_plot}")

    if args.plot_full_sequence:
        if mode != "full_sequence" or args.storage_format != "zarr":
            print(
                "plot_full_sequence is only supported for precompute_mode=full_sequence "
                "with zarr storage_format.",
            )
        else:
            motion_curve, audio_curve = compute_full_sequence_curves_zarr(
                preprocessed_dir, rec
            )
            total = len(motion_curve)
            start = int(report["window_start_frame"])
            end = int(report["window_end_frame_exclusive"])
            fig, ax = plt.subplots(figsize=(12, 4))
            x = np.arange(total)
            ax.plot(x, audio_curve, label="audio energy (norm)", color="#1f77b4")
            ax.plot(x, motion_curve, label="mouth motion (norm)", color="#ff7f0e")
            ax.axvspan(start, end, color="#2ca02c", alpha=0.25, label="validation window")
            ax.set_xlabel("Preprocessed frame index")
            ax.set_ylabel("Normalized value")
            ax.set_title(
                f"sample_idx={report['sample_idx']} full-sequence curves "
                f"(window=[{start}, {end}))"
            )
            ax.legend(loc="upper right")
            fig.tight_layout()
            out_full_plot = (
                args.output_dir / f"sample_{args.sample_idx:05d}_full_sequence.png"
            )
            fig.savefig(out_full_plot, dpi=160)
            plt.close(fig)
            print(f"Saved full-sequence debug plot: {out_full_plot}")

    if args.export_source_segment:
        if not source_path.is_file():
            raise FileNotFoundError(f"source_path not found: {source_path}")
        out_video = args.output_dir / f"sample_{args.sample_idx:05d}_source_segment.mp4"
        export_source_segment_with_audio(
            source_path=source_path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            out_path=out_video,
        )
        print(f"Exported source segment with audio: {out_video}")


if __name__ == "__main__":
    main()

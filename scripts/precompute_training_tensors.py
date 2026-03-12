#!/usr/bin/env python3
"""
Offline precompute pipeline for faster training.

Raw video path:
  video decode -> MediaPipe -> mouth crop -> resize/normalize -> tensors

Fast training path:
  load precomputed tensors -> augment -> model

Shape contract (must match LipSyncModel.forward):
  - visual: (C, T, H, W) per sample → batched (B, 3, T, H, W)
           C=3, T=video_frames (default 32), H=W=96 (crop_size)
  - audio:  (1, F, T_mel) per sample → batched (B, 1, F, T_a)
           F=80 (n_mels), T_a=audio_frames (default 128)
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on path when run as: python scripts/precompute_training_tensors.py
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Prevent BLAS/OpenMP thread explosion when using many multiprocessing workers.
# Keep these defaults conservative; users can still override via environment.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm

from app.core.logger import get_logger
from app.preprocessing.face_detection import detect_and_crop_mouths
from app.preprocessing.audio import preprocess_audio
from app.preprocessing.video import dummy_mouth_crop, load_video_frames, preprocess_video
from app.training.dataset import discover_video_samples

logger = get_logger(__name__)


def _precompute_one(
    video_path: Path,
    label: int,
    sample_id: str,
    output_dir: Path,
    storage_format: str,
    precompute_mode: str,
    video_frames: int,
    audio_frames: int,
    strict_face_detection: bool,
    target_fps: float,
) -> tuple[dict[str, object], Any | None]:
    if precompute_mode == "fixed_clip":
        visual = preprocess_video(
            video_path,
            use_face_detection=True,
            max_faces=1,
            crop_size=(96, 96),
            max_frames=video_frames,
            strict_face_detection=strict_face_detection,
            target_fps=target_fps,
        ).astype(np.float32, copy=False)
        audio = preprocess_audio(video_path, target_frames=audio_frames).astype(
            np.float32, copy=False
        )

        # Match LipSyncModel: visual (B, 3, T, H, W), audio (B, 1, F, T_a)
        expected_visual = (3, video_frames, 96, 96)
        expected_audio = (1, 80, audio_frames)  # 80 = n_mels in preprocess_audio
        if visual.shape != expected_visual:
            raise ValueError(
                f"Visual shape {visual.shape} != expected {expected_visual}. "
                "Check video_frames and crop_size."
            )
        if audio.shape != expected_audio:
            raise ValueError(
                f"Audio shape {audio.shape} != expected {expected_audio}. "
                "Check audio_frames and n_mels in preprocess_audio."
            )
    else:
        frames = load_video_frames(
            video_path,
            load_all=True,
            max_total_frames=900,
            target_fps=target_fps,
        )
        try:
            visual = detect_and_crop_mouths(
                frames, crop_size=(96, 96), max_faces=1, select_strategy="longest"
            )
        except Exception:
            if strict_face_detection:
                raise
            visual = dummy_mouth_crop(frames, crop_size=(96, 96))
        # Store full sequence as uint8 to reduce size.
        visual = visual.astype(np.uint8, copy=False)  # (T, H, W, C)
        audio = preprocess_audio(video_path, target_frames=None).astype(
            np.float32, copy=False
        )  # (1, 80, T)
        if visual.ndim != 4 or visual.shape[-1] != 3:
            raise ValueError(f"Unexpected full visual shape: {visual.shape}")
        if audio.ndim != 3 or audio.shape[1] != 80:
            raise ValueError(f"Unexpected full audio shape: {audio.shape}")
        # Store full mel as (80, T_audio) per proposed pipeline.
        audio = audio[0]

    record: dict[str, object] = {
        "id": sample_id,
        "label": int(label),
        "source_path": str(video_path),
        "precompute_mode": precompute_mode,
        "target_fps": float(target_fps),
        "mel_hz": 100.0,  # sr/hop_length = 16000/160
    }

    if storage_format == "npy":
        v_dir = "visual_full" if precompute_mode == "full_sequence" else "visual"
        a_dir = "audio_full" if precompute_mode == "full_sequence" else "audio"
        visual_rel = Path(v_dir) / f"{sample_id}.npy"
        audio_rel = Path(a_dir) / f"{sample_id}.npy"
        visual_out = output_dir / visual_rel
        audio_out = output_dir / audio_rel
        visual_out.parent.mkdir(parents=True, exist_ok=True)
        audio_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(visual_out, visual, allow_pickle=False)
        np.save(audio_out, audio, allow_pickle=False)
        record["visual_relpath"] = str(visual_rel)
        record["audio_relpath"] = str(audio_rel)
        return record, None

    key = sample_id
    record["key"] = key

    if storage_format == "lmdb":
        buf = io.BytesIO()
        # Prefer uncompressed payload for faster preprocessing and reads.
        np.savez(buf, visual=visual, audio=audio)
        payload = buf.getvalue()
        return record, ("lmdb", key, payload)

    if storage_format == "zarr":
        return record, ("zarr", key, visual, audio)

    raise ValueError(f"Unsupported storage format: {storage_format}")


def _worker_job(
    job: tuple[
        int,
        str,
        int,
        str,
        str,
        int,
        int,
        bool,
        float,
    ]
) -> tuple[str, int, dict | None, Any | None, str | None]:
    """
    Worker wrapper for multiprocessing.

    Returns:
        (status, idx, record, storage_item, error)
        - status: "ok" or "err"
        - idx:    original sample index
    """
    (
        idx,
        video_path_str,
        label,
        output_dir_str,
        storage_format,
        precompute_mode,
        video_frames,
        audio_frames,
        strict_face_detection,
        target_fps,
    ) = job
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    sample_id = f"{idx:08d}"
    try:
        rec, storage_item = _precompute_one(
            video_path=video_path,
            label=label,
            sample_id=sample_id,
            output_dir=output_dir,
            storage_format=storage_format,
            precompute_mode=precompute_mode,
            video_frames=video_frames,
            audio_frames=audio_frames,
            strict_face_detection=strict_face_detection,
            target_fps=target_fps,
        )
        return "ok", idx, rec, storage_item, None
    except Exception as e:
        return (
            "err",
            idx,
            {"source_path": str(video_path), "label": int(label)},
            None,
            f"{type(e).__name__}: {e}",
        )


def _write_zarr_sample(zarr_root: Any, key: str, visual: np.ndarray, audio: np.ndarray) -> None:
    """
    Write one sample into Zarr in a way that works across zarr versions.
    """
    grp = zarr_root.require_group(key)
    for name, arr in (("visual", visual), ("audio", audio)):
        if name in grp:
            del grp[name]
        try:
            grp.create_array(name, data=arr, chunks=arr.shape)
        except TypeError:
            # Older/newer zarr variants may differ in accepted kwargs.
            grp.create_array(name, data=arr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute mouth-crop/audio tensors for training speedups."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing 0_real/ and 1_fake/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write precomputed tensors + manifest.jsonl",
    )
    parser.add_argument(
        "--storage-format",
        type=str,
        default="zarr",
        choices=["npy", "lmdb", "zarr"],
        help="Storage backend for precomputed tensors.",
    )
    parser.add_argument(
        "--precompute-mode",
        type=str,
        default="full_sequence",
        choices=["full_sequence", "fixed_clip"],
        help=(
            "full_sequence: store full mouth-frame sequence + full mel "
            "(recommended for random contiguous training clips). "
            "fixed_clip: store one fixed (3,T,H,W)+(1,80,T_a) sample per video."
        ),
    )
    parser.add_argument(
        "--video-frames", type=int, default=32, help="Target visual frames (T)"
    )
    parser.add_argument(
        "--audio-frames", type=int, default=128, help="Target mel frames (T_mel)"
    )
    parser.add_argument(
        "--strict-face-detection",
        action="store_true",
        help="Fail sample when face detection fails (no center-crop fallback).",
    )
    parser.add_argument(
        "--lmdb-map-size-gb",
        type=int,
        default=64,
        help="LMDB map size in GB when --storage-format lmdb.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes for preprocessing. "
            "0 or 1 = no multiprocessing; >1 uses multiprocessing.Pool."
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="FPS used when decoding frames before face/mouth processing.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_video_samples(data_dir)
    if not samples:
        raise ValueError(f"No videos found under {data_dir}")

    logger.info(
        "Precompute start: samples=%d, format=%s, output=%s",
        len(samples),
        args.storage_format,
        output_dir,
    )

    lmdb_env = None
    lmdb_txn = None
    zarr_root = None
    if args.storage_format == "lmdb":
        try:
            import lmdb  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "LMDB format requested but `lmdb` is not installed. "
                "Install with: pip install lmdb"
            ) from e
        lmdb_path = output_dir / "samples.lmdb"
        lmdb_env = lmdb.open(
            str(lmdb_path),
            map_size=int(args.lmdb_map_size_gb) * (1024**3),
            subdir=False,
            lock=True,
            map_async=True,
            readahead=False,
            meminit=False,
        )
        lmdb_txn = lmdb_env.begin(write=True)
    elif args.storage_format == "zarr":
        try:
            import zarr  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Zarr format requested but `zarr` is not installed. "
                "Install with: pip install zarr"
            ) from e
        zarr_path = output_dir / "samples.zarr"
        # If the Zarr store already exists, open in append mode so that we can
        # skip samples that have been precomputed in earlier runs.
        zarr_mode = "a" if zarr_path.exists() else "w"
        zarr_root = zarr.open_group(str(zarr_path), mode=zarr_mode)

    manifest_path = output_dir / "manifest.jsonl"
    skipped = 0
    written = 0

    # If re-running on a directory where some samples are already precomputed,
    # we can skip those to support incremental additions to 0_real/1_fake.
    def _already_precomputed(idx: int, video_path: Path, label: int) -> bool:
        sample_id = f"{idx:08d}"
        if args.storage_format == "zarr" and zarr_root is not None:
            # Groups are keyed by sample_id under the Zarr root.
            return sample_id in zarr_root
        if args.storage_format == "npy":
            v_dir = "visual_full" if args.precompute_mode == "full_sequence" else "visual"
            a_dir = "audio_full" if args.precompute_mode == "full_sequence" else "audio"
            visual_rel = Path(v_dir) / f"{sample_id}.npy"
            audio_rel = Path(a_dir) / f"{sample_id}.npy"
            visual_out = output_dir / visual_rel
            audio_out = output_dir / audio_rel
            return visual_out.exists() and audio_out.exists()
        # LMDB: currently always overwrite; no incremental skip.
        return False

    # Build worker jobs
    jobs: list[tuple[int, str, int, str, str, str, int, int, bool, float]] = []
    for idx, (video_path, label) in enumerate(samples):
        if _already_precomputed(idx, video_path, label):
            skipped += 1
            logger.info("Skipping already-precomputed sample idx=%d (%s)", idx, video_path)
            continue
        jobs.append(
            (
                idx,
                str(video_path),
                int(label),
                str(output_dir),
                args.storage_format,
                args.precompute_mode,
                int(args.video_frames),
                int(args.audio_frames),
                bool(args.strict_face_detection),
                float(args.target_fps),
            )
        )

    workers = int(args.workers or 0)
    use_pool = workers > 1
    if use_pool:
        logger.info("Using multiprocessing with %d workers", workers)
    else:
        logger.info("Using single-process preprocessing")

    with manifest_path.open("w", encoding="utf-8") as mf:
        if use_pool:
            with mp.Pool(processes=workers) as pool:
                for status, idx, rec, storage_item, err in tqdm(
                    pool.imap_unordered(_worker_job, jobs, chunksize=8),
                    total=len(jobs),
                    desc="Precomputing samples",
                ):
                    if status != "ok" or rec is None:
                        skipped += 1
                        logger.warning(
                            "Skipping idx=%d: %s",
                            idx,
                            err or "unknown error",
                        )
                        continue

                    if storage_item is not None:
                        if args.storage_format == "lmdb":
                            _, key, payload = storage_item
                            lmdb_txn.put(key.encode("utf-8"), payload)
                            if (written + 1) % 100 == 0:
                                lmdb_txn.commit()
                                lmdb_txn = lmdb_env.begin(write=True)
                        elif args.storage_format == "zarr":
                            _, key, visual, audio = storage_item
                            _write_zarr_sample(zarr_root, key, visual, audio)

                    mf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    written += 1
        else:
            for idx, (video_path, label) in enumerate(
                tqdm(samples, desc="Precomputing samples")
            ):
                if _already_precomputed(idx, video_path, label):
                    skipped += 1
                    logger.info(
                        "Skipping already-precomputed sample idx=%d (%s)",
                        idx,
                        video_path,
                    )
                    continue
                sample_id = f"{idx:08d}"
                try:
                    rec, storage_item = _precompute_one(
                        video_path=video_path,
                        label=label,
                        sample_id=sample_id,
                        output_dir=output_dir,
                        storage_format=args.storage_format,
                        video_frames=args.video_frames,
                        audio_frames=args.audio_frames,
                        strict_face_detection=args.strict_face_detection,
                        target_fps=args.target_fps,
                    )

                    if storage_item is not None:
                        if args.storage_format == "lmdb":
                            _, key, payload = storage_item
                            lmdb_txn.put(key.encode("utf-8"), payload)
                            if (written + 1) % 100 == 0:
                                lmdb_txn.commit()
                                lmdb_txn = lmdb_env.begin(write=True)
                        elif args.storage_format == "zarr":
                            _, key, visual, audio = storage_item
                            _write_zarr_sample(zarr_root, key, visual, audio)

                    mf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    written += 1
                except Exception as e:
                    skipped += 1
                    logger.warning(
                        "Skipping %s: %s: %s", video_path, type(e).__name__, e
                    )

    if lmdb_txn is not None:
        lmdb_txn.commit()
    if lmdb_env is not None:
        lmdb_env.sync()
        lmdb_env.close()

    meta = {
        "data_dir": str(data_dir),
        "storage_format": args.storage_format,
        "precompute_mode": args.precompute_mode,
        "target_fps": float(args.target_fps),
        "video_frames": int(args.video_frames),
        "audio_frames": int(args.audio_frames),
        "written": int(written),
        "skipped": int(skipped),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    logger.info(
        "Precompute complete: written=%d, skipped=%d, manifest=%s",
        written,
        skipped,
        manifest_path,
    )


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np 
from ..core.logger import get_logger
from .face_detection import detect_and_crop_mouths, detect_and_crop_mouth_tracks

logger = get_logger(__name__)

_AV_AVAILABLE = False
try:
    import av  
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False


def _load_video_frames_pyav(
    path: Path,
    target_fps: float,
    max_total_frames: int,
) -> np.ndarray:
    """
    Load video frames at target_fps using PyAV (FFmpeg bindings).
    Uses presentation timestamps (PTS) for accurate time-based sampling,
    handling VFR and unreliable OpenCV metadata correctly.
    """
    container = av.open(str(path))
    video_stream = container.streams.video[0]
    stream_time_base = float(video_stream.time_base)

    # Duration in seconds (PyAV uses proper container/stream metadata)
    duration_sec = None
    if video_stream.duration is not None:
        duration_sec = float(video_stream.duration * video_stream.time_base)
    elif container.duration is not None:
        # container.duration is in 1/AV_TIME_BASE units (microseconds)
        duration_sec = float(container.duration) / 1_000_000
    if duration_sec is None:
        duration_sec = 1e9  # assume very long if unknown

    expected_frames = int(duration_sec * target_fps)
    target_count = min(expected_frames, max_total_frames)
    target_times = [i / target_fps for i in range(target_count)]

    frames: List[np.ndarray] = []
    next_target_idx = 0
    half_interval = 0.5 / target_fps  # tolerance for matching

    for frame in container.decode(video=0):
        pts_sec = (
            frame.pts * stream_time_base
            if frame.pts is not None
            else (len(frames) / target_fps)  # fallback
        )

        img = frame.to_ndarray(format="rgb24")
        if img is None:
            continue

        # Assign this frame to all target timestamps it covers (first frame at or past each target)
        while (
            next_target_idx < len(target_times)
            and pts_sec >= target_times[next_target_idx] - half_interval
        ):
            frames.append(img.copy())
            next_target_idx += 1

        if next_target_idx >= target_count:
            break

    container.close()

    if not frames:
        raise ValueError(
            f"No valid frames decoded from video {path} "
            "(may be corrupt or unsupported codec). Try with OpenCV fallback."
        )

    return np.stack(frames, axis=0)



def get_video_info(path: Path) -> Tuple[float, int]:
    """
    Return (fps, total_frame_count) for a video file without decoding frames.
    Uses PyAV when available for more reliable metadata; falls back to OpenCV on failure.
    Returns (0.0, 0) if the file cannot be opened.

    NOTE: frame_count is derived from container/stream metadata, NOT by decoding.
    Some encoders write incorrect frame counts in headers (common with screen-record
    or re-encoded MP4s). If you see a duration mismatch between this and actual loaded
    frames, the container metadata is incorrect — the actual decoded frame coverage
    reported by load_video_frames (PyAV) is the ground truth.
    """
    if _AV_AVAILABLE:
        try:
            container = av.open(str(path))
            video_stream = container.streams.video[0]

            fps = 0.0
            if video_stream.average_rate is not None:
                fps = float(video_stream.average_rate)

            frame_count = 0
            frame_count_source = "unknown"
            if hasattr(video_stream, "frames") and video_stream.frames is not None and video_stream.frames > 0:
                frame_count = int(video_stream.frames)
                frame_count_source = "stream.frames (nb_frames header)"
            else:
                duration_sec = None
                if video_stream.duration is not None:
                    duration_sec = float(video_stream.duration * video_stream.time_base)
                    frame_count_source = "stream.duration × fps"
                elif container.duration is not None:
                    duration_sec = float(container.duration) / 1_000_000
                    frame_count_source = "container.duration × fps"
                if duration_sec is not None and fps > 0:
                    frame_count = int(duration_sec * fps)

            container.close()
            logger.debug(
                "get_video_info [PyAV]: %s → fps=%.2f, frames=%d (%s)",
                path.name, fps, frame_count, frame_count_source,
            )
            return fps, max(0, frame_count)
        except Exception as e:
            logger.warning(
                "get_video_info [PyAV] failed for %s: %s — falling back to OpenCV",
                path.name, e,
            )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    logger.debug(
        "get_video_info [OpenCV]: %s → fps=%.2f, frames=%d",
        path.name, fps, frame_count,
    )
    return fps, max(0, frame_count)


def load_video_frames(
    path: Path,
    max_frames: int = 32,
    load_all: bool = False,
    max_total_frames: int = 900,
    target_fps: Optional[float] = None,
) -> np.ndarray:
    """
    Load video frames as RGB numpy array (T, H, W, C).

    Args:
        path:             Video file path.
        max_frames:       Used only when ``load_all=False`` and ``target_fps=None``;
                          reads exactly this many frames from the START of the video
                          (original short-clip behaviour).
        load_all:         When True (and target_fps=None), reads the entire video up
                          to ``max_total_frames``. Ignored when target_fps is set
                          (full video is always loaded in that mode).
        max_total_frames: Hard safety cap on the number of OUTPUT frames to prevent
                          OOM on very long videos.
                          - target_fps=None  → caps raw frames  (900 ≈ 30 s @ 30 fps)
                          - target_fps=15    → caps resampled frames (900 ≈ 60 s @ 15 fps)
        target_fps:       When set, resample the video to this frame rate using PyAV
                          (FFmpeg).  PyAV handles duration, PTS, and codecs correctly,
                          so loaded frames are proportional to video duration regardless
                          of source FPS.  Requires PyAV: ``pip install av``.
                          Set to None to keep the original fixed-count behaviour.

    Raises ValueError if video cannot be decoded.
    """
    fps_mode = target_fps is not None
    if fps_mode:
        mode_desc = f"target_fps={target_fps}, output_cap={max_total_frames}"
        limit = max_total_frames
    else:
        limit = max_total_frames if load_all else max_frames
        mode_desc = f"full-clip cap={max_total_frames}" if load_all else f"first {max_frames} frames"

    logger.debug(
        "Loading video frames from %s (mode=%s, limit=%d frames)",
        path, mode_desc, limit,
    )

    try:
        if fps_mode:
            if not _AV_AVAILABLE:
                raise ValueError(
                    f"target_fps={target_fps} requires PyAV for FPS-based frame loading. "
                    "Install with: pip install av"
                )
            logger.debug("load_video_frames [PyAV]: %s @ target_fps=%.1f", path.name, target_fps)
            frames = _load_video_frames_pyav(path, target_fps, max_total_frames)
            actual_coverage = len(frames) / target_fps
            logger.info(
                "load_video_frames [PyAV]: %s → %d frames @ %.1ffps (coverage=%.2fs)",
                path.name, len(frames), target_fps, actual_coverage,
            )
            return frames

        logger.debug("load_video_frames [OpenCV]: %s", path.name)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if native_fps <= 0 or frame_count <= 0:
            cap.release()
            raise ValueError(
                f"Video has invalid properties (fps={native_fps}, frames={frame_count}): {path}"
            )

        if load_all and frame_count > max_total_frames:
            duration_sec = frame_count / max(1.0, native_fps)
            logger.warning(
                "Video has %d frames (%.1fs @ %.1ffps) but max_total_frames=%d. "
                "Only first %d frames will be processed. Increase max_total_frames in config if needed.",
                frame_count, duration_sec, native_fps, max_total_frames, max_total_frames,
            )

        frames = []
        consecutive_failures = 0
        max_consecutive_failures = 10

        if load_all:
            limit = max_total_frames
        else:
            limit = max_frames

        while len(frames) < limit:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                continue

            consecutive_failures = 0
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            except Exception as e:
                logger.warning("Failed to convert frame to RGB in %s: %s", path, e)
                continue

        cap.release()

        if not frames:
            raise ValueError(
                f"No valid frames read from video {path} "
                "(may be corrupt or unsupported codec)"
            )

        logger.info(
            "load_video_frames [OpenCV]: %s → %d frames (native_fps=%.1f)",
            path.name, len(frames), native_fps,
        )

        return np.stack(frames, axis=0)

    except Exception as e:
        if (
            "VideoCapture" in str(type(e).__name__)
            or "codec" in str(e).lower()
            or "h264" in str(e).lower()
        ):
            raise ValueError(
                f"Video decoding failed for {path.name}: {e}\n"
                "This video may be corrupt, use an unsupported codec, or be missing "
                "codec support.\nThe video will be skipped during training."
            ) from e
        raise


def dummy_mouth_crop(frames: np.ndarray, crop_size: Tuple[int, int] = (96, 96)) -> np.ndarray:
    """
    Placeholder mouth cropper.
    Currently crops the central region of each frame.
    Replace with a proper face / landmark-based mouth tracking in production.
    """
    T, H, W, C = frames.shape
    ch, cw = crop_size
    y1 = max(0, H // 2 - ch // 2)
    x1 = max(0, W // 2 - cw // 2)
    y2 = min(H, y1 + ch)
    x2 = min(W, x1 + cw)
    crops = frames[:, y1:y2, x1:x2, :]
    logger.debug(
        "Dummy mouth crop applied: input_shape=%s, crop_size=%s, output_shape=%s",
        frames.shape,
        crop_size,
        crops.shape,
    )
    return crops


def preprocess_video(
    path: Path,
    use_face_detection: bool = True,
    max_faces: int = 1,
    crop_size: Tuple[int, int] = (96, 96),
    max_frames: int = 32,
    strict_face_detection: bool = False,
    target_fps: float = 15.0,
    max_total_frames: int = 900,
) -> np.ndarray:
    """
    Full video preprocessing with production-grade face detection.

    Args:
        path:                Video file path.
        use_face_detection:  Use MediaPipe face detection (True) or dummy crop (False).
        max_faces:           Maximum faces to detect per frame.
        crop_size:           Output mouth crop size (H, W).
        max_frames:          Model input window size T.  The loaded frames are
                             uniformly sampled / padded to exactly this length.
        strict_face_detection: Raise instead of falling back to dummy crop.
        target_fps:          Resample source video to this frame rate before any
                             other processing.  Ensures time-consistent inputs
                             regardless of source FPS (default 15 fps).
        max_total_frames:    Safety cap on resampled frames loaded from disk.

    Returns:
        (C, T, H, W) float32 - preprocessed video ready for model, where T = max_frames.
        At target_fps=15, each frame represents 1/15 s of real time, so the model
        always sees the same temporal resolution regardless of the original video FPS.
    """
    frames = load_video_frames(
        path,
        load_all=True,
        max_total_frames=max_total_frames,
        target_fps=target_fps,
    )  # (T_full, H, W, C) – full video at target_fps

    if use_face_detection:
        try:
            crops = detect_and_crop_mouths(
                frames, crop_size=crop_size, max_faces=max_faces, select_strategy="longest"
            )  # (T, H, W, C)
            logger.debug("Face detection successful, cropped %d frames", len(crops))
        except Exception as e:
            if strict_face_detection:
                raise
            logger.warning("Face detection failed, falling back to dummy crop: %s", e)
            crops = dummy_mouth_crop(frames, crop_size=crop_size)  # (T, H, W, C)
    else:
        crops = dummy_mouth_crop(frames, crop_size=crop_size)  # (T, H, W, C)

    # Pad or temporally sample to max_frames (fixed T for batching)
    T_current = crops.shape[0]
    if T_current < max_frames:
        # Pad with last frame
        padding = np.repeat(crops[-1:], max_frames - T_current, axis=0)
        crops = np.concatenate([crops, padding], axis=0)
    elif T_current > max_frames:
        # Uniform temporal sampling (better than naive truncation)
        idx = np.linspace(0, T_current - 1, max_frames).astype(np.int64)
        crops = crops[idx]

    crops = crops.astype("float32") / 255.0
    crops = np.transpose(crops, (3, 0, 1, 2))  # (C, T, H, W)
    logger.debug("Video preprocessed to array with shape %s", crops.shape)
    return crops


def preprocess_video_tracks(
    path: Path,
    max_faces: int = 5,
    max_tracks: int = 5,
    crop_size: Tuple[int, int] = (96, 96),
    max_frames: int = 32,
    target_fps: float = 15.0,
    max_total_frames: int = 900,
) -> list[dict[str, Any]]:
    """
    Multi-subject preprocessing.

    Returns a list of tracked mouth clips, one per detected face track.
    The full video is loaded at ``target_fps`` so that clips from different
    source videos are time-consistent before being padded/sampled to ``max_frames``.

    Output:
      List of track dictionaries:
      - `track_id`: int
      - `clip`: `(C, T, H, W)` float32  (T = max_frames)
      - `hits`: int
      - `total_frames`: int  (resampled frame count at target_fps)
      - `stability`: float in [0, 1]
    """
    frames = load_video_frames(
        path,
        load_all=True,
        max_total_frames=max_total_frames,
        target_fps=target_fps,
    )  # (T_full, H, W, C) – full video at target_fps
    logger.info(
        "Preprocessing video tracks: %d frames @ %.1ffps, max_faces=%d, max_tracks=%d",
        len(frames), target_fps, max_faces, max_tracks,
    )
    tracks = detect_and_crop_mouth_tracks(
        frames,
        crop_size=crop_size,
        max_faces=max_faces,
        max_tracks=max_tracks,
    )  # list[{"track_id", "crops", "hits", "total_frames", "stability"}]
    logger.info(f"Face detection result: {len(tracks)} track(s) returned")

    out: list[dict[str, Any]] = []
    for tr in tracks:
        track_id = int(tr["track_id"])
        crops = tr["crops"]
        # Ensure fixed T
        T_current = crops.shape[0]
        if T_current < max_frames:
            padding = np.repeat(crops[-1:], max_frames - T_current, axis=0)
            crops = np.concatenate([crops, padding], axis=0)
        elif T_current > max_frames:
            idx = np.linspace(0, T_current - 1, max_frames).astype(np.int64)
            crops = crops[idx]

        clip = crops.astype("float32") / 255.0
        clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
        out.append(
            {
                "track_id": track_id,
                "clip": clip,
                "hits": int(tr.get("hits", 0)),
                "total_frames": int(tr.get("total_frames", max_frames)),
                "stability": float(tr.get("stability", 0.0)),
            }
        )

    return out


def preprocess_video_tracks_chunked(
    path: Path,
    chunk_size: int = 32,
    stride: int = 16,
    max_faces: int = 5,
    max_tracks: int = 3,
    crop_size: Tuple[int, int] = (96, 96),
    max_total_frames: int = 900,
    target_fps: float = 15.0,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """
    Full-video multi-track preprocessing for long clips.

    Loads **all** frames resampled to ``target_fps``, runs face detection and
    tracking across the full duration, then splits each track's mouth-crop
    sequence into overlapping ``chunk_size``-frame windows with step ``stride``.

    Using ``target_fps`` makes each chunk represent the same wall-clock duration
    regardless of source video FPS.  At the default target_fps=15:
      - chunk_size=32 frames  →  32/15 ≈ 2.1 s per model window
      - stride=16 frames      →  16/15 ≈ 1.1 s step between windows

    This replaces ``preprocess_video_tracks`` for inference on videos longer
    than a single model window (T=32).

    Args:
        path:             Video file.
        chunk_size:       Frames per model input window (must match model T, default 32).
        stride:           Step between consecutive windows (default 16 = 50 % overlap).
        max_faces:        Max faces detected per frame.
        max_tracks:       Max face tracks retained.
        crop_size:        Mouth crop spatial size.
        max_total_frames: Safety cap on resampled output frames (900 @ 15 fps ≈ 60 s).
        target_fps:       Resample video to this frame rate before chunking so all
                          chunks represent consistent real-time durations (default 15).

    Returns:
        ``(tracks, target_fps, total_frames)`` where each track dict contains:

        - ``track_id``           : int
        - ``chunks``             : list of ``(C, chunk_size, H, W)`` float32 arrays
        - ``chunk_starts``       : list of int – resampled frame index each chunk starts at
        - ``hits``               : matched frames across full clip
        - ``total_frames``       : total resampled frames loaded
        - ``stability``          : weighted stability score
        - ``consecutive_miss_max``: worst consecutive-miss streak
    """
    fps, _ = get_video_info(path)

    # Load full video resampled to target_fps for time-consistent chunking
    all_frames = load_video_frames(
        path,
        load_all=True,
        max_total_frames=max_total_frames,
        target_fps=target_fps,
    )  # (N, H, W, C) where N = duration_sec * target_fps
    total_frames = len(all_frames)

    logger.info(
        "Full-video chunked preprocessing: %d frames loaded "
        "(native=%.1ffps → target=%.1ffps, coverage=%.1fs), "
        "chunk_size=%d (%.2fs), stride=%d (%.2fs)",
        total_frames, fps, target_fps, total_frames / target_fps,
        chunk_size, chunk_size / target_fps,
        stride, stride / target_fps,
    )

    # Run face detection and tracking over all frames
    raw_tracks = detect_and_crop_mouth_tracks(
        all_frames,
        crop_size=crop_size,
        max_faces=max_faces,
        max_tracks=max_tracks,
    )
    logger.info("Full-video tracking: %d track(s) returned", len(raw_tracks))

    out: List[Dict[str, Any]] = []
    for tr in raw_tracks:
        crops = tr["crops"]  # (total_frames, H, W, C)
        N = crops.shape[0]

        # Build sliding-window chunks
        chunks: List[np.ndarray] = []
        chunk_starts: List[int] = []

        start = 0
        while start + chunk_size <= N:
            window = crops[start : start + chunk_size]  # (chunk_size, H, W, C)
            clip = window.astype("float32") / 255.0
            clip = np.transpose(clip, (3, 0, 1, 2))  # (C, chunk_size, H, W)
            chunks.append(clip)
            chunk_starts.append(int(start))
            start += stride

        # If the video is shorter than chunk_size, pad and keep as single chunk
        if not chunks:
            window = crops[:N]
            if N < chunk_size:
                padding = np.repeat(crops[-1:], chunk_size - N, axis=0)
                window = np.concatenate([window, padding], axis=0)
            clip = window.astype("float32") / 255.0
            clip = np.transpose(clip, (3, 0, 1, 2))
            chunks.append(clip)
            chunk_starts.append(0)

        logger.debug(
            "Track %d: %d chunk(s) from %d frames (chunk_size=%d, stride=%d)",
            tr["track_id"], len(chunks), N, chunk_size, stride,
        )

        out.append(
            {
                "track_id": int(tr["track_id"]),
                "chunks": chunks,
                "chunk_starts": chunk_starts,
                "hits": int(tr.get("hits", 0)),
                "total_frames": total_frames,
                "stability": float(tr.get("stability", 0.0)),
                "consecutive_miss_max": int(tr.get("consecutive_miss_max", 0)),
            }
        )

    return out, float(target_fps), total_frames

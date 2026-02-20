from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from time import perf_counter

import torch
import numpy as np
from fastapi import UploadFile

from ..core.logger import get_logger
from ..models.lip_sync_model import LipSyncModel
from ..preprocessing.video import (
    preprocess_video,
    preprocess_video_tracks,
    preprocess_video_tracks_chunked,
    get_video_info,
)
from ..preprocessing.audio import preprocess_audio, detect_voice_activity
from ..utils.file_manager import save_upload_to_temp, split_av_paths


logger = get_logger(__name__)


class Predictor:
    """
    High-level inference wrapper around LipSyncModel and preprocessing.

    Designed for production use: loads weights once at startup, fails fast
    if they are missing or incompatible, and exposes a narrow prediction API.
    """

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        confidence_threshold: float,
        use_torchscript: bool,
        use_half_precision: bool,
        uncertainty_margin: float = 0.05,
        confidence_smoothing: str = "median",
        trim_ratio: float = 0.1,
        max_tracks: int = 3,
        refine_margin: float = 0.08,
        refine_top_k: int = 2,
        chunk_size: int = 32,
        chunk_stride: int = 16,
        long_video_threshold_sec: float = 2.0,
        max_total_frames: int = 900,
    ):
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.use_half_precision = bool(use_half_precision and device.type == "cuda")
        self.uncertainty_margin = float(max(0.0, uncertainty_margin))
        allowed = {"none", "median", "trimmed_mean"}
        self.confidence_smoothing = (
            confidence_smoothing if confidence_smoothing in allowed else "median"
        )
        self.trim_ratio = float(min(max(trim_ratio, 0.0), 0.49))
        self.max_tracks = int(max(1, max_tracks))
        self.refine_margin = float(max(0.0, refine_margin))
        self.refine_top_k = int(max(1, refine_top_k))
        self.chunk_size = int(chunk_size)
        self.chunk_stride = int(chunk_stride)
        self.long_video_threshold_sec = float(long_video_threshold_sec)
        self.max_total_frames = int(max_total_frames)

        model = LipSyncModel()

        if not model_path.is_file():
            msg = f"Model weights not found at {model_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(
            "Initializing LipSyncModel on device=%s, model_path=%s, half_precision=%s, torchscript=%s",
            device,
            model_path,
            self.use_half_precision,
            use_torchscript,
        )
        logger.info(
            "Multi-track selection config: uncertainty_margin=%.3f, confidence_smoothing=%s, trim_ratio=%.2f, "
            "max_tracks=%d, refine_margin=%.3f, refine_top_k=%d, "
            "chunk_size=%d, chunk_stride=%d, long_video_threshold_sec=%.1f, max_total_frames=%d",
            self.uncertainty_margin,
            self.confidence_smoothing,
            self.trim_ratio,
            self.max_tracks,
            self.refine_margin,
            self.refine_top_k,
            self.chunk_size,
            self.chunk_stride,
            self.long_video_threshold_sec,
            self.max_total_frames,
        )
        logger.info("Loading lip-sync model weights from %s", model_path)
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]

        model.load_state_dict(state, strict=True)

        if self.use_half_precision:
            model.half()

        model.to(self.device)
        model.eval()

        # Optional TorchScript compilation for lower latency.
        if use_torchscript:
            try:
                logger.info("Compiling LipSyncModel with torch.jit.script for inference")
                model = torch.jit.script(model)
            except Exception:
                logger.exception("TorchScript compilation failed; falling back to eager model")

        self.model = model

    def _infer_confidence(self, visual_np: np.ndarray, audio_np: np.ndarray) -> float:
        """Run a single forward pass and return P(REAL) confidence."""
        visual_tensor = torch.from_numpy(visual_np).unsqueeze(0)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        if self.use_half_precision:
            visual_tensor = visual_tensor.half()
            audio_tensor = audio_tensor.half()

        visual_tensor = visual_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        logits = self.model(visual_tensor, audio_tensor)
        return float(torch.sigmoid(logits).item())

    def _robust_confidence(self, confidences: List[float]) -> float:
        if not confidences:
            return 0.5
        arr = np.asarray(confidences, dtype=np.float32)
        if self.confidence_smoothing == "none":
            return float(arr.mean())
        if self.confidence_smoothing == "median":
            return float(np.median(arr))
        # trimmed_mean
        n = int(arr.size)
        k = int(n * self.trim_ratio)
        if k <= 0 or (2 * k) >= n:
            return float(arr.mean())
        arr_sorted = np.sort(arr)
        return float(arr_sorted[k : n - k].mean())

    def _speech_weighted_confidence(
        self, confidences: List[float], speaking_scores: List[float], vad_weights: Optional[List[float]] = None
    ) -> float:
        """
        Aggregate confidences with higher weight on speech-active windows.
        Uses VAD when available, otherwise falls back to audio-visual alignment scores.
        Helps reduce false positives on long clips with many low-information windows.
        """
        if not confidences:
            return 0.5
        if len(confidences) != len(speaking_scores):
            return self._robust_confidence(confidences)

        conf = np.asarray(confidences, dtype=np.float32)
        speech = np.asarray(speaking_scores, dtype=np.float32)
        speech = np.clip(speech, 0.0, 1.0)
        
        # Use VAD weights if provided (more reliable), otherwise use alignment scores
        if vad_weights is not None and len(vad_weights) == len(confidences):
            vad_arr = np.asarray(vad_weights, dtype=np.float32)
            vad_arr = np.clip(vad_arr, 0.0, 1.0)
            # Blend VAD (70%) with alignment score (30%) for robustness
            combined_speech = 0.7 * vad_arr + 0.3 * speech
        else:
            combined_speech = speech
        
        # Keep a non-zero floor so non-speaking windows still contribute a bit.
        weights = np.clip(0.2 + 0.8 * combined_speech, 0.2, 1.0)
        denom = float(weights.sum())
        if denom <= 1e-8:
            return self._robust_confidence(confidences)
        return float(np.dot(conf, weights) / denom)

    def _temporal_smoothed_confidence(
        self, visual_np: np.ndarray, audio_np: np.ndarray
    ) -> tuple[float, List[float], List[Tuple[int, int]]]:
        """
        Multi-window confidence smoothing over time.
        Uses full clip and 3 overlapping windows when possible.
        """
        t_v = int(visual_np.shape[1])
        t_a = int(audio_np.shape[2])
        windows: List[tuple[np.ndarray, np.ndarray]] = [(visual_np, audio_np)]
        spans: List[Tuple[int, int]] = [(0, max(1, t_v))]

        # Use shorter windows for better speaker turn sensitivity.
        # For T_v=32 this yields 16-frame windows: [0:16], [8:24], [16:32].
        win_v = max(12, t_v // 2)
        win_a = max(48, t_a // 2)
        if t_v >= win_v and t_a >= win_a:
            v_starts = [0, max(0, (t_v - win_v) // 2), max(0, t_v - win_v)]
            for v_start in v_starts:
                v_end = min(t_v, v_start + win_v)
                a_start = int(round(v_start * (t_a / max(1, t_v))))
                a_end = min(t_a, a_start + win_a)
                # Keep window length consistent where possible
                if (v_end - v_start) >= 16 and (a_end - a_start) >= 64:
                    windows.append(
                        (
                            visual_np[:, v_start:v_end, :, :],
                            audio_np[:, :, a_start:a_end],
                        )
                    )
                    spans.append((v_start, v_end))

        confidences: List[float] = []
        for v_win, a_win in windows:
            confidences.append(self._infer_confidence(v_win, a_win))

        return self._robust_confidence(confidences), confidences, spans

    @staticmethod
    def _speaking_alignment_score(visual_np: np.ndarray, audio_np: np.ndarray) -> float:
        """
        Estimate speaking activity by aligning mouth-motion energy with audio energy.
        Returns a score in [0, 1].
        """
        # visual_np: (C, T, H, W), audio_np: (1, F, T)
        frames = visual_np.mean(axis=0)  # (T, H, W)
        if frames.shape[0] < 2:
            return 0.5
        motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))  # (T-1,)
        if motion.size == 0:
            return 0.5
        motion = np.concatenate([motion[:1], motion], axis=0)

        audio_energy = audio_np[0].mean(axis=0)  # (T_a,)
        if audio_energy.size < 2:
            return 0.5

        x_old = np.linspace(0.0, 1.0, num=motion.size)
        x_new = np.linspace(0.0, 1.0, num=audio_energy.size)
        motion_resampled = np.interp(x_new, x_old, motion)

        def _z(x: np.ndarray) -> np.ndarray:
            mu = float(x.mean())
            sigma = float(x.std())
            if sigma < 1e-6:
                return x * 0.0
            return (x - mu) / sigma

        m = _z(motion_resampled)
        a = _z(audio_energy)
        if float(np.abs(m).sum()) < 1e-6 or float(np.abs(a).sum()) < 1e-6:
            return 0.5
        corr = float(np.corrcoef(m, a)[0, 1])
        if np.isnan(corr):
            return 0.5
        return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))

    def _align_audio_chunk(
        self,
        audio_np_full: np.ndarray,
        v_start: int,
        total_v_frames: int,
        chunk_a_size: int = 128,
    ) -> np.ndarray:
        """
        Slice a chunk_a_size-wide audio window whose position aligns with
        video frames [v_start : v_start + chunk_size].

        audio_np_full: (1, F, T_full)
        Returns:       (1, F, chunk_a_size)
        """
        total_a = int(audio_np_full.shape[2])
        a_ratio = total_a / max(1, total_v_frames)
        a_start = int(round(v_start * a_ratio))
        a_end = a_start + chunk_a_size
        # Clamp to valid range
        if a_end > total_a:
            a_end = total_a
            a_start = max(0, a_end - chunk_a_size)
        chunk = audio_np_full[:, :, a_start:a_end]
        # Pad if edge is shorter than chunk_a_size
        if chunk.shape[2] < chunk_a_size:
            pad = np.repeat(chunk[:, :, -1:], chunk_a_size - chunk.shape[2], axis=2)
            chunk = np.concatenate([chunk, pad], axis=2)
        return chunk

    def _run_chunked_inference(
        self,
        chunks: List[np.ndarray],
        chunk_starts: List[int],
        audio_np_full: np.ndarray,
        total_v_frames: int,
    ) -> Tuple[float, List[float]]:
        """
        Run inference over every 32-frame chunk of a track and aggregate.

        Returns (aggregated_confidence, per_chunk_confidences).
        """
        chunk_confs: List[float] = []
        for visual_chunk, v_start in zip(chunks, chunk_starts):
            audio_chunk = self._align_audio_chunk(
                audio_np_full, v_start, total_v_frames
            )
            conf = self._infer_confidence(visual_chunk, audio_chunk)
            chunk_confs.append(conf)
            logger.debug(
                "  Chunk @frame %d: confidence=%.4f", v_start, conf
            )
        agg = self._robust_confidence(chunk_confs)
        logger.debug(
            "Chunked inference: %d chunks → aggregated_confidence=%.4f", len(chunk_confs), agg
        )
        return agg, chunk_confs

    async def _predict_long_video(
        self, video_path: Any, audio_path: Any, t_start: float
    ) -> Dict[str, Any]:
        """
        Full-clip inference for videos longer than long_video_threshold_sec.

        Strategy:
        1. Load ALL frames and run face tracking across the full clip.
        2. Split each track into overlapping chunk_size-frame windows.
        3. Run inference on every chunk; aggregate with robust confidence.
        4. Produce per-chunk window timeline (speaker turns visible across full clip).
        """
        t_pre_start = perf_counter()

        # Full audio (no target_frames truncation)
        audio_np_full = preprocess_audio(audio_path)   # (1, F, T_full)
        total_a_frames = int(audio_np_full.shape[2])

        # Detect voice activity for window weighting
        try:
            vad_mask, _ = detect_voice_activity(audio_path)  # audio_duration_sec unused
            logger.info(
                "VAD detected speech in %.1f%% of audio frames (%d/%d)",
                np.mean(vad_mask) * 100,
                np.sum(vad_mask),
                len(vad_mask)
            )
        except Exception as e:
            logger.warning("VAD detection failed, using fallback: %s", e)
            # Fallback: assume all frames contain speech
            vad_mask = np.ones(total_a_frames, dtype=bool)

        chunked_tracks, fps, total_v_frames = preprocess_video_tracks_chunked(
            video_path,
            chunk_size=self.chunk_size,
            stride=self.chunk_stride,
            max_faces=5,
            max_tracks=self.max_tracks,
            max_total_frames=self.max_total_frames,
        )
        t_pre_end = perf_counter()

        logger.info(
            "Long-video preprocessing: %.1fs video, %d frames, %d audio frames, "
            "%d track(s), %.1f ms",
            total_v_frames / max(1.0, fps),
            total_v_frames,
            total_a_frames,
            len(chunked_tracks),
            (t_pre_end - t_pre_start) * 1000.0,
        )

        if not chunked_tracks:
            logger.warning("No tracks found in long video; returning uncertain result.")
            return {
                "is_real": False,
                "is_fake": True,
                "confidence": 0.5,
                "manipulation_probability": 0.5,
                "tracks": None,
                "selected_track_id": None,
                "turn_taking_detected": False,
                "speaker_case": "no_face_detected",
                "speaking_tracks_count": 0,
                "speaking_real_count": 0,
                "speaking_fake_count": 0,
                "verdicts": {
                    "active_speaker_policy_is_fake": True,
                    "any_speaking_fake_policy_is_fake": False,
                    "all_speaking_fake_policy_is_fake": False,
                    "majority_speaking_fake_policy_is_fake": False,
                },
                "window_results": None,
                "speaker_timeline": None,
                "detail": "No face tracks detected in video.",
                "video_duration_sec": float(total_v_frames / max(1.0, fps)),
                "total_chunks_analyzed": 0,
            }

        # ── Run chunked inference per track ───────────────────────────────────
        t_inf_start = perf_counter()
        track_results: List[Dict[str, Any]] = []

        with torch.inference_mode():
            for tr in chunked_tracks:
                track_id = int(tr["track_id"])
                chunks = tr["chunks"]          # list of (C, T, H, W)
                chunk_starts = tr["chunk_starts"]
                stability = float(tr.get("stability", 0.0))
                hits = int(tr.get("hits", 0))
                consecutive_miss_max = int(tr.get("consecutive_miss_max", 0))

                agg_conf, chunk_confs = self._run_chunked_inference(
                    chunks, chunk_starts, audio_np_full, total_v_frames
                )

                # Speaking activity from first chunk (representative)
                speaking_score = self._speaking_alignment_score(
                    chunks[0], self._align_audio_chunk(audio_np_full, chunk_starts[0], total_v_frames)
                )
                selection_score = (
                    0.65 * agg_conf + 0.20 * stability + 0.15 * speaking_score
                )
                is_real = agg_conf >= self.confidence_threshold

                logger.info(
                    "Track %d: %d chunks, agg_conf=%.4f, stability=%.3f, "
                    "speaking=%.3f, selection_score=%.4f, is_real=%s",
                    track_id, len(chunks), agg_conf, stability,
                    speaking_score, selection_score, is_real,
                )

                track_results.append({
                    "track_id": track_id,
                    "is_real": is_real,
                    "is_fake": not is_real,
                    "confidence": float(agg_conf),
                    "raw_confidence": float(chunk_confs[0]) if chunk_confs else float(agg_conf),
                    "manipulation_probability": float(1.0 - agg_conf),
                    "stability": stability,
                    "hits": hits,
                    "total_frames": total_v_frames,
                    "speaking_activity": float(speaking_score),
                    "selection_score": float(selection_score),
                    "window_confidences": [float(c) for c in chunk_confs],
                    "window_spans": [
                        (int(s), int(s + self.chunk_size)) for s in chunk_starts
                    ],
                    "consecutive_miss_max": consecutive_miss_max,
                })

        t_inf_end = perf_counter()

        # ── Select best track ─────────────────────────────────────────────────
        sorted_tracks = sorted(
            track_results, key=lambda tr: tr["selection_score"], reverse=True
        )
        best_result = sorted_tracks[0]
        best_track_id = int(best_result["track_id"])
        selection_margin = (
            float(sorted_tracks[0]["selection_score"] - sorted_tracks[1]["selection_score"])
            if len(sorted_tracks) > 1 else 1.0
        )
        selection_uncertain = selection_margin < self.uncertainty_margin

        # ── Per-chunk timeline across full clip ───────────────────────────────
        # For each chunk index shared across tracks, pick the winner
        max_chunks = max(len(tr["window_confidences"]) for tr in sorted_tracks)
        total_chunks = sum(len(tr["window_confidences"]) for tr in sorted_tracks)
        track_chunks_map: Dict[int, List[np.ndarray]] = {
            int(tr["track_id"]): tr["chunks"] for tr in chunked_tracks
        }
        chunk_window_results: List[Dict[str, Any]] = []
        for c_idx in range(max_chunks):
            candidates = [
                tr for tr in sorted_tracks
                if len(tr["window_confidences"]) > c_idx
            ]
            if not candidates:
                continue
            win = max(
                candidates,
                key=lambda tr: (
                    0.75 * float(tr["window_confidences"][c_idx])
                    + 0.25 * float(tr.get("stability", 0.0))
                ),
            )
            v_start = int(win["window_spans"][c_idx][0])
            v_end = int(win["window_spans"][c_idx][1])
            win_conf = float(win["window_confidences"][c_idx])
            win_track_id = int(win["track_id"])
            win_speaking = float(win.get("speaking_activity", 0.5))
            win_chunks = track_chunks_map.get(win_track_id, [])
            if c_idx < len(win_chunks):
                try:
                    audio_chunk = self._align_audio_chunk(
                        audio_np_full, v_start, total_v_frames
                    )
                    win_speaking = float(
                        self._speaking_alignment_score(win_chunks[c_idx], audio_chunk)
                    )
                except Exception:
                    # Keep previously computed per-track speaking score if chunk scoring fails.
                    pass
            
            # Map VAD mask to this window's time range
            time_start_sec = float(v_start / max(1.0, fps))
            time_end_sec = float(v_end / max(1.0, fps))
            # VAD mask is in mel-spectrogram frame space (hop_length=160 @ 16kHz)
            # Map video time to mel frame indices
            mel_hop_ms = 160.0 / 16000.0 * 1000.0  # ~10ms per mel frame
            mel_start_idx = int(time_start_sec * 1000.0 / mel_hop_ms)
            mel_end_idx = int(time_end_sec * 1000.0 / mel_hop_ms)
            mel_start_idx = max(0, min(mel_start_idx, len(vad_mask) - 1))
            mel_end_idx = max(mel_start_idx + 1, min(mel_end_idx, len(vad_mask)))
            window_vad_coverage = float(np.mean(vad_mask[mel_start_idx:mel_end_idx])) if mel_end_idx > mel_start_idx else 0.5
            
            chunk_window_results.append({
                "window_index": int(c_idx),
                "frame_start": v_start,
                "frame_end": v_end,
                "time_start_sec": round(time_start_sec, 3),
                "time_end_sec": round(time_end_sec, 3),
                "selected_track_id": int(win["track_id"]),
                "confidence": win_conf,
                "speaking_activity": win_speaking,
                "vad_coverage": round(window_vad_coverage, 3),
                "is_real": bool(win_conf >= self.confidence_threshold),
                "is_fake": bool(win_conf < self.confidence_threshold),
            })

        # ── Compress into speaker timeline ────────────────────────────────────
        speaker_timeline: List[Dict[str, Any]] = []
        for wr in chunk_window_results:
            if (
                speaker_timeline
                and speaker_timeline[-1]["selected_track_id"] == wr["selected_track_id"]
                and wr["frame_start"] <= speaker_timeline[-1]["frame_end"]
            ):
                speaker_timeline[-1]["frame_end"] = max(
                    speaker_timeline[-1]["frame_end"], wr["frame_end"]
                )
                speaker_timeline[-1]["time_end_sec"] = wr["time_end_sec"]
            else:
                speaker_timeline.append({
                    "selected_track_id": wr["selected_track_id"],
                    "frame_start": wr["frame_start"],
                    "frame_end": wr["frame_end"],
                    "time_start_sec": wr["time_start_sec"],
                    "time_end_sec": wr["time_end_sec"],
                })

        unique_window_speakers = len({wr["selected_track_id"] for wr in chunk_window_results})
        turn_taking_detected = unique_window_speakers > 1

        # ── Aggregate final verdict ───────────────────────────────────────────
        all_chunk_confs = [float(c) for c in best_result["window_confidences"]]
        window_confs = (
            [float(wr["confidence"]) for wr in chunk_window_results]
            if chunk_window_results
            else all_chunk_confs
        )
        window_speaking = (
            [float(wr.get("speaking_activity", 0.5)) for wr in chunk_window_results]
            if chunk_window_results
            else [float(best_result.get("speaking_activity", 0.5))] * len(window_confs)
        )
        window_vad_weights = (
            [float(wr.get("vad_coverage", 0.5)) for wr in chunk_window_results]
            if chunk_window_results
            else None
        )
        window_median_confidence = self._robust_confidence(window_confs)
        weighted_window_confidence = self._speech_weighted_confidence(
            window_confs, window_speaking, vad_weights=window_vad_weights
        )
        # Blend robust median and speech-aware weighting.
        final_confidence = float(
            0.5 * window_median_confidence + 0.5 * weighted_window_confidence
        )

        conf_arr = np.asarray(window_confs, dtype=np.float32)
        speech_arr = np.asarray(window_speaking, dtype=np.float32)
        strong_real = int(
            np.sum(conf_arr >= max(self.confidence_threshold + 0.15, 0.65))
        )
        strong_fake = int(
            np.sum(conf_arr <= min(self.confidence_threshold - 0.15, 0.35))
        )
        mixed_window_signal = strong_real >= 2 and strong_fake >= 2

        speech_mask = speech_arr >= 0.45
        vote_src = conf_arr[speech_mask] if np.any(speech_mask) else conf_arr
        fake_vote_ratio = float(np.mean(vote_src < self.confidence_threshold)) if vote_src.size else 1.0
        strict_fake_evidence = fake_vote_ratio >= 0.70

        final_is_real = final_confidence >= self.confidence_threshold
        window_consensus_uncertain = False
        # Conservative guard against false positives on mixed-content long clips.
        if (not final_is_real) and mixed_window_signal and (not strict_fake_evidence):
            window_consensus_uncertain = True
            selection_uncertain = True
            final_confidence = float(max(final_confidence, self.confidence_threshold))
            final_is_real = True

        # ── Edge-case speaker classification ─────────────────────────────────
        speaking_tracks = [
            tr for tr in sorted_tracks
            if float(tr.get("speaking_activity", 0.0)) >= 0.50
            and float(tr.get("stability", 0.0)) >= 0.20
        ] or sorted_tracks[: min(2, len(sorted_tracks))]

        speaking_count = len(speaking_tracks)
        speaking_fake_count = sum(1 for tr in speaking_tracks if tr["is_fake"])
        speaking_real_count = speaking_count - speaking_fake_count

        if speaking_fake_count == 0:
            speaker_case = "all_speaking_real"
        elif speaking_real_count == 0:
            speaker_case = "all_speaking_fake"
        else:
            speaker_case = "mixed_real_and_fake"

        track_policy_verdicts = {
            "active_speaker_policy_is_fake": bool(best_result["is_fake"]),
            "any_speaking_fake_policy_is_fake": bool(speaking_fake_count > 0),
            "all_speaking_fake_policy_is_fake": bool(
                speaking_count > 0 and speaking_fake_count == speaking_count
            ),
            "majority_speaking_fake_policy_is_fake": bool(
                speaking_fake_count > speaking_real_count
            ),
        }
        conservative_override_applied = bool(window_consensus_uncertain and final_is_real)
        if conservative_override_applied:
            # Keep top-level verdicts aligned with conservative final decision.
            verdicts = {
                "active_speaker_policy_is_fake": False,
                "any_speaking_fake_policy_is_fake": False,
                "all_speaking_fake_policy_is_fake": False,
                "majority_speaking_fake_policy_is_fake": False,
            }
            speaker_case = "mixed_window_consensus_uncertain"
        else:
            verdicts = track_policy_verdicts

        t_end = perf_counter()
        logger.info(
            "Long-video inference done: tracks=%d, chunks_per_track_max=%d, "
            "total_model_passes=%d, final_conf=%.4f, weighted_conf=%.4f, "
            "fake_vote_ratio=%.2f, strict_fake=%s, is_real=%s, turn_taking=%s, "
            "total_ms=%.1f preproc_ms=%.1f infer_ms=%.1f",
            len(track_results),
            max_chunks,
            total_chunks,
            final_confidence,
            weighted_window_confidence,
            fake_vote_ratio,
            strict_fake_evidence,
            final_is_real,
            turn_taking_detected,
            (t_end - t_start) * 1000.0,
            (t_pre_end - t_pre_start) * 1000.0,
            (t_inf_end - t_inf_start) * 1000.0,
        )

        # ── Build detail message ──────────────────────────────────────────────
        if turn_taking_detected:
            spans_str = " → ".join(
                f"track_{seg['selected_track_id']} "
                f"({seg['time_start_sec']:.1f}s–{seg['time_end_sec']:.1f}s)"
                for seg in speaker_timeline
            )
            detail = (
                f"Long video ({total_v_frames/max(1.0,fps):.1f}s, {max_chunks} chunks analyzed). "
                f"Speaker turn-taking detected: {spans_str}. "
                f"Final verdict window-aggregated (confidence={final_confidence:.4f})."
            )
            selection_uncertain = False
        elif window_consensus_uncertain:
            detail = (
                f"Long video ({total_v_frames/max(1.0,fps):.1f}s, {max_chunks} chunks). "
                f"Window consensus is mixed (strong_real={strong_real}, strong_fake={strong_fake}, "
                f"fake_vote_ratio={fake_vote_ratio:.2f}). "
                f"Returning conservative REAL verdict (confidence={final_confidence:.4f})."
            )
        elif selection_uncertain:
            detail = (
                f"Long video ({total_v_frames/max(1.0,fps):.1f}s, {max_chunks} chunks). "
                f"Track selection uncertain (margin={selection_margin:.4f})."
            )
        else:
            detail = (
                f"Long video ({total_v_frames/max(1.0,fps):.1f}s). "
                f"Analyzed {max_chunks} chunk(s) across full clip. "
                f"Dominant speaker: track {best_track_id} "
                f"(confidence={final_confidence:.4f})."
            )

        result = {
            "is_real": final_is_real,
            "is_fake": not final_is_real,
            "confidence": float(final_confidence),
            "manipulation_probability": float(1.0 - final_confidence),
            "tracks": sorted_tracks,
            "selected_track_id": best_track_id,
            "selection_uncertain": selection_uncertain,
            "selection_margin": float(selection_margin),
            "turn_taking_detected": turn_taking_detected,
            "speaker_case": speaker_case,
            "speaking_tracks_count": speaking_count,
            "speaking_real_count": speaking_real_count,
            "speaking_fake_count": speaking_fake_count,
            "verdicts": verdicts,
            "track_policy_verdicts": track_policy_verdicts,
            "conservative_override_applied": conservative_override_applied,
            "window_results": chunk_window_results if chunk_window_results else None,
            "speaker_timeline": speaker_timeline if speaker_timeline else None,
            "video_duration_sec": float(total_v_frames / max(1.0, fps)),
            "total_chunks_analyzed": int(total_chunks),
            "chunks_per_track_max": int(max_chunks),
            "window_weighted_confidence": float(weighted_window_confidence),
            "window_fake_vote_ratio": float(fake_vote_ratio),
            "window_consensus_uncertain": bool(window_consensus_uncertain),
            "strict_fake_evidence": bool(strict_fake_evidence),
            "detail": detail,
        }
        return result

    async def predict_from_upload(self, upload: UploadFile) -> Dict[str, Any]:
        """
        Entry point for FastAPI: takes an uploaded container file, extracts
        frames / audio, runs the model, and returns a JSON-serializable result.
        
        Supports multi-face detection: if multiple faces are detected, returns
        results for each track with the best track selected for top-level decision.
        """
        t_start = perf_counter()
        tmp_video = save_upload_to_temp(upload, suffix=".mp4")
        video_path, audio_path = split_av_paths(tmp_video)

        try:
            # ── Detect video length and choose short vs long path ─────────────
            fps, total_frame_count = get_video_info(video_path)
            # Use chunked path when metadata says more than one model window (chunk_size).
            is_long_video = total_frame_count > self.chunk_size
            logger.debug(
                "Video path: %s (metadata frames=%d, fps=%.1f)",
                "LONG (chunked)" if is_long_video else "SHORT",
                total_frame_count,
                fps,
            )

            # ── Long video: full-clip tracking + sliding-window inference ─────
            if is_long_video:
                return await self._predict_long_video(
                    video_path, audio_path, t_start
                )

            # ── Short video: original single-pass path ────────────────────────
            try:
                t_pre_start = perf_counter()
                # Try multi-face detection first
                tracks = preprocess_video_tracks(
                    video_path, max_faces=5, max_tracks=self.max_tracks
                )  # list[track dict]
                audio_np = preprocess_audio(audio_path, target_frames=128)  # (1, F, T) T=128
                t_pre_end = perf_counter()
                logger.info(
                    "Preprocessing completed in %.3f ms (video+audio), detected %d face track(s)",
                    (t_pre_end - t_pre_start) * 1000.0,
                    len(tracks),
                )
                if tracks:
                    logger.info(
                        "Face tracks detected: %s",
                        [tr["track_id"] for tr in tracks],
                    )
            except Exception:
                logger.exception("Preprocessing failed")
                raise

            # Fallback to single-face processing if no tracks detected
            if not tracks:
                logger.warning("No face tracks detected, falling back to single-face processing")
                try:
                    t_pre_start = perf_counter()
                    visual_np = preprocess_video(video_path)  # (C, T, H, W) T=32
                    audio_np = preprocess_audio(audio_path, target_frames=128)  # (1, F, T) T=128
                    t_pre_end = perf_counter()
                    logger.info(
                        "Single-face preprocessing completed in %.3f ms (video+audio)",
                        (t_pre_end - t_pre_start) * 1000.0,
                    )
                except Exception:
                    logger.exception("Single-face preprocessing failed")
                    raise

                # Add batch dimension and move to device
                visual_tensor = torch.from_numpy(visual_np).unsqueeze(0)
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

                if self.use_half_precision:
                    visual_tensor = visual_tensor.half()
                    audio_tensor = audio_tensor.half()

                visual_tensor = visual_tensor.to(self.device)
                audio_tensor = audio_tensor.to(self.device)

                with torch.inference_mode():
                    t_inf_start = perf_counter()
                    logits = self.model(visual_tensor, audio_tensor)  # (1,) - logits for P(REAL)
                    t_inf_end = perf_counter()

                # Convert logits to probability using sigmoid
                prob_real = torch.sigmoid(logits).item()
                confidence = float(prob_real)  # P(REAL) - probability video is authentic
                manipulation_probability = float(1.0 - prob_real)  # P(FAKE) - probability video is manipulated
                
                # Determine if video is real (authentic) or fake (manipulated)
                is_real = confidence >= self.confidence_threshold
                is_fake = not is_real

                t_end = perf_counter()
                logger.info(
                    "Inference completed (single-face): is_real=%s, is_fake=%s, confidence=%.4f, manipulation_prob=%.4f, total_time_ms=%.3f, "
                    "preproc_ms≈%.3f, infer_ms=%.3f",
                    is_real,
                    is_fake,
                    confidence,
                    manipulation_probability,
                    (t_end - t_start) * 1000.0,
                    (t_pre_end - t_pre_start) * 1000.0 if 't_pre_end' in locals() else -1.0,
                    (t_inf_end - t_inf_start) * 1000.0,
                )

                result = {
                    "is_real": is_real,
                    "is_fake": is_fake,
                    "confidence": confidence,
                    "manipulation_probability": manipulation_probability,
                    "tracks": None,
                    "selected_track_id": None,
                }
                return result
            else:
                # Process each track
                track_results = []
                track_clip_map: Dict[int, np.ndarray] = {}
                best_track_id = None
                best_score = -1.0

                with torch.inference_mode():
                    t_inf_start = perf_counter()

                    # Phase 1 (fast path): one forward pass per track.
                    for tr in tracks:
                        track_id = int(tr["track_id"])
                        visual_np = tr["clip"]  # (C, T, H, W)
                        track_clip_map[track_id] = visual_np
                        stability = float(tr.get("stability", 0.0))
                        hits = int(tr.get("hits", 0))
                        total_frames = int(tr.get("total_frames", 0))
                        consecutive_miss_max = int(tr.get("consecutive_miss_max", 0))

                        raw_confidence = self._infer_confidence(visual_np, audio_np)
                        confidence = raw_confidence
                        manipulation_probability = float(1.0 - confidence)
                        speaking_score = self._speaking_alignment_score(visual_np, audio_np)
                        selection_score = (
                            0.65 * confidence + 0.20 * stability + 0.15 * speaking_score
                        )
                        is_real = confidence >= self.confidence_threshold
                        is_fake = not is_real
                        track_results.append(
                            {
                                "track_id": track_id,
                                "is_real": is_real,
                                "is_fake": is_fake,
                                "confidence": confidence,
                                "raw_confidence": raw_confidence,
                                "manipulation_probability": manipulation_probability,
                                "stability": stability,
                                "hits": hits,
                                "total_frames": total_frames,
                                "speaking_activity": speaking_score,
                                "selection_score": selection_score,
                                "window_confidences": [float(raw_confidence)],
                                "window_spans": [(0, int(visual_np.shape[1]))],
                                "consecutive_miss_max": consecutive_miss_max,
                            }
                        )

                    # Adaptive Phase 2 (refine only when competition is close).
                    quick_sorted = sorted(
                        track_results, key=lambda tr: tr["selection_score"], reverse=True
                    )
                    quick_margin = (
                        float(
                            quick_sorted[0]["selection_score"]
                            - quick_sorted[1]["selection_score"]
                        )
                        if len(quick_sorted) > 1
                        else 1.0
                    )
                    needs_refine = quick_margin < self.refine_margin
                    if needs_refine:
                        for tr in quick_sorted[: self.refine_top_k]:
                            tr_id = int(tr["track_id"])
                            visual_np = track_clip_map[tr_id]
                            smoothed_confidence, conf_samples, spans = (
                                self._temporal_smoothed_confidence(visual_np, audio_np)
                            )
                            tr["confidence"] = float(smoothed_confidence)
                            tr["manipulation_probability"] = float(
                                1.0 - tr["confidence"]
                            )
                            tr["is_real"] = bool(
                                tr["confidence"] >= self.confidence_threshold
                            )
                            tr["is_fake"] = not tr["is_real"]
                            tr["window_confidences"] = [float(v) for v in conf_samples]
                            tr["window_spans"] = [(int(s), int(e)) for s, e in spans]
                            tr["selection_score"] = (
                                0.55 * tr["confidence"]
                                + 0.25 * float(tr["stability"])
                                + 0.20 * float(tr["speaking_activity"])
                            )

                    # Final winner after optional refinement.
                    for tr in track_results:
                        if float(tr["selection_score"]) > best_score:
                            best_score = float(tr["selection_score"])
                            best_track_id = int(tr["track_id"])

                    t_inf_end = perf_counter()

                # Use best track for top-level decision
                best_result = next((tr for tr in track_results if tr["track_id"] == best_track_id), track_results[0])
                sorted_tracks = sorted(track_results, key=lambda tr: tr["selection_score"], reverse=True)
                selection_margin = (
                    float(sorted_tracks[0]["selection_score"] - sorted_tracks[1]["selection_score"])
                    if len(sorted_tracks) > 1
                    else 1.0
                )
                selection_uncertain = selection_margin < self.uncertainty_margin

                track_summary = ", ".join(
                    [
                        f"track_{tr['track_id']}=sel_{tr['selection_score']:.3f}/conf_{tr['confidence']:.3f}"
                        for tr in sorted_tracks
                    ]
                )
                logger.info(
                    "Track selection: selected track %s (selection_score=%.4f, confidence=%.4f), "
                    "margin=%.4f, uncertain=%s, refined=%s (quick_margin=%.4f, refine_margin=%.4f), tracks=%d, details=[%s]",
                    best_track_id,
                    best_result["selection_score"],
                    best_result["confidence"],
                    selection_margin,
                    selection_uncertain,
                    needs_refine,
                    quick_margin,
                    self.refine_margin,
                    len(track_results),
                    track_summary,
                )

                # Build per-window winning track summary for turn-taking conversations.
                max_windows = max((len(tr.get("window_confidences", [])) for tr in sorted_tracks), default=0)
                window_results: List[Dict[str, Any]] = []
                if max_windows > 1:
                    for w_idx in range(1, max_windows):  # Skip index 0 (full clip confidence)
                        candidates = [tr for tr in sorted_tracks if len(tr.get("window_confidences", [])) > w_idx]
                        if not candidates:
                            continue
                        # Blend per-window confidence with stability and per-window speaking activity.
                        def _window_score(tr: Dict[str, Any]) -> float:
                            start, end = tr.get("window_spans", [(0, 0)])[w_idx]
                            tr_id = int(tr["track_id"])
                            clip = track_clip_map[tr_id]
                            win_v = clip[:, int(start) : int(end), :, :]
                            t_a = int(audio_np.shape[2])
                            t_v = int(clip.shape[1])
                            a_start = int(round(int(start) * (t_a / max(1, t_v))))
                            a_end = int(round(int(end) * (t_a / max(1, t_v))))
                            a_start = max(0, min(a_start, t_a - 1))
                            a_end = max(a_start + 1, min(a_end, t_a))
                            win_a = audio_np[:, :, a_start:a_end]
                            win_speaking = self._speaking_alignment_score(win_v, win_a)
                            return (
                                0.75 * float(tr["window_confidences"][w_idx])
                                + 0.15 * float(tr.get("stability", 0.0))
                                + 0.10 * float(win_speaking)
                            )

                        win_best = max(
                            candidates,
                            key=_window_score,
                        )
                        start, end = win_best.get("window_spans", [(0, 0)])[w_idx]
                        win_conf = float(win_best["window_confidences"][w_idx])
                        window_results.append(
                            {
                                "window_index": int(w_idx - 1),
                                "frame_start": int(start),
                                "frame_end": int(end),
                                "selected_track_id": int(win_best["track_id"]),
                                "confidence": win_conf,
                                "is_real": bool(win_conf >= self.confidence_threshold),
                                "is_fake": bool(win_conf < self.confidence_threshold),
                            }
                        )

                # Compress consecutive same-speaker windows into timeline segments.
                speaker_timeline: List[Dict[str, Any]] = []
                for wr in window_results:
                    if (
                        speaker_timeline
                        and speaker_timeline[-1]["selected_track_id"] == wr["selected_track_id"]
                        and wr["frame_start"] <= speaker_timeline[-1]["frame_end"]
                    ):
                        speaker_timeline[-1]["frame_end"] = max(
                            speaker_timeline[-1]["frame_end"], wr["frame_end"]
                        )
                    else:
                        speaker_timeline.append(
                            {
                                "selected_track_id": wr["selected_track_id"],
                                "frame_start": wr["frame_start"],
                                "frame_end": wr["frame_end"],
                            }
                        )

                # Speaking-track edge-case summary (both real / mixed / both fake).
                speaking_tracks = [
                    tr
                    for tr in sorted_tracks
                    if float(tr.get("speaking_activity", 0.0)) >= 0.55
                    and float(tr.get("stability", 0.0)) >= 0.20
                ]
                if not speaking_tracks:
                    # Fallback: consider top tracks if speaking cue is weak.
                    speaking_tracks = sorted_tracks[: min(2, len(sorted_tracks))]

                speaking_count = len(speaking_tracks)
                speaking_fake_count = sum(1 for tr in speaking_tracks if tr["is_fake"])
                speaking_real_count = speaking_count - speaking_fake_count

                if speaking_fake_count == 0:
                    speaker_case = "all_speaking_real"
                elif speaking_real_count == 0:
                    speaker_case = "all_speaking_fake"
                else:
                    speaker_case = "mixed_real_and_fake"

                verdicts = {
                    "active_speaker_policy_is_fake": bool(best_result["is_fake"]),
                    "any_speaking_fake_policy_is_fake": bool(speaking_fake_count > 0),
                    "all_speaking_fake_policy_is_fake": bool(speaking_count > 0 and speaking_fake_count == speaking_count),
                    "majority_speaking_fake_policy_is_fake": bool(speaking_fake_count > speaking_real_count),
                }

                # Turn-taking aware aggregate over per-window selected speakers.
                if window_results:
                    window_conf = [float(wr["confidence"]) for wr in window_results]
                    window_agg_conf = self._robust_confidence(window_conf)
                    window_agg_is_real = window_agg_conf >= self.confidence_threshold
                    unique_window_speakers = len({wr["selected_track_id"] for wr in window_results})
                else:
                    window_agg_conf = float(best_result["confidence"])
                    window_agg_is_real = bool(best_result["is_real"])
                    unique_window_speakers = 1

                t_end = perf_counter()
                logger.info(
                    "Inference completed: %d tracks detected, best_track_id=%s, is_real=%s, is_fake=%s, confidence=%.4f, manipulation_prob=%.4f, total_time_ms=%.3f, "
                    "preproc_ms≈%.3f, infer_ms=%.3f",
                    len(track_results),
                    best_track_id,
                    best_result["is_real"],
                    best_result["is_fake"],
                    best_result["confidence"],
                    best_result["manipulation_probability"],
                    (t_end - t_start) * 1000.0,
                    (t_pre_end - t_pre_start) * 1000.0 if 't_pre_end' in locals() else -1.0,
                    (t_inf_end - t_inf_start) * 1000.0,
                )

                # Build response
                final_is_real = bool(best_result["is_real"])
                final_confidence = float(best_result["confidence"])
                # If multiple speakers are active across windows, prefer window aggregate.
                if unique_window_speakers > 1:
                    final_is_real = bool(window_agg_is_real)
                    final_confidence = float(window_agg_conf)

                result = {
                    "is_real": final_is_real,
                    "is_fake": (not final_is_real),
                    "confidence": final_confidence,
                    "manipulation_probability": float(1.0 - final_confidence),
                    "selection_uncertain": selection_uncertain,
                    "selection_margin": selection_margin,
                    "turn_taking_detected": bool(unique_window_speakers > 1),
                    "speaker_case": speaker_case,
                    "speaking_tracks_count": speaking_count,
                    "speaking_real_count": speaking_real_count,
                    "speaking_fake_count": speaking_fake_count,
                    "verdicts": verdicts,
                    "window_results": window_results if window_results else None,
                    "speaker_timeline": speaker_timeline if speaker_timeline else None,
                }

                # Add track information whenever tracks are detected
                if len(track_results) > 0:
                    result["tracks"] = sorted_tracks
                    result["selected_track_id"] = best_track_id
                else:
                    result["tracks"] = None
                    result["selected_track_id"] = None

                # Determine the best detail message.
                # Turn-taking takes priority: if windows already resolved who is
                # speaking when, the global score being "close" is expected and
                # should not be labelled as uncertain.
                turn_taking_detected = unique_window_speakers > 1
                if turn_taking_detected:
                    speaker_ids = [seg["selected_track_id"] for seg in speaker_timeline]
                    spans_str = " → ".join(
                        f"track_{seg['selected_track_id']} "
                        f"(frames {seg['frame_start']}-{seg['frame_end']})"
                        for seg in speaker_timeline
                    )
                    result["detail"] = (
                        f"Speaker turn-taking detected across {len(speaker_timeline)} segment(s): "
                        f"{spans_str}. "
                        f"Final verdict is window-aggregated (confidence={final_confidence:.4f})."
                    )
                    # Override selection_uncertain: turn-taking explains the close scores.
                    result["selection_uncertain"] = False
                elif selection_uncertain:
                    result["detail"] = (
                        f"Track selection uncertain: top-two selection scores are too close "
                        f"(margin={selection_margin:.4f}, threshold={self.uncertainty_margin:.4f}). "
                        f"Consider using a longer clip for more reliable results."
                    )

                return result
        finally:
            # Cleanup temporary file
            try:
                video_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                logger.warning("Failed to remove temporary video file %s", video_path)

    def predict_from_path(self, video_path: Path) -> Dict[str, Any]:
        """
        Run lip-sync detection on a video file on disk (e.g. GRID .mpg).
        Same logic as predict_from_upload but for a local path. Audio is
        extracted from the video when needed (e.g. .mpg, .mp4).
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        visual_np = preprocess_video(video_path)
        audio_np = preprocess_audio(video_path, target_frames=128)

        visual_tensor = torch.from_numpy(visual_np).unsqueeze(0)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        if self.use_half_precision:
            visual_tensor = visual_tensor.half()
            audio_tensor = audio_tensor.half()

        visual_tensor = visual_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)

        with torch.inference_mode():
            logits = self.model(visual_tensor, audio_tensor)  # (1,) - logits for P(REAL)

        # Convert logits to probability using sigmoid
        prob_real = torch.sigmoid(logits).item()
        confidence = float(prob_real)  # P(REAL) - probability video is authentic
        manipulation_probability = float(1.0 - prob_real)  # P(FAKE) - probability video is manipulated
        
        # Determine if video is real (authentic) or fake (manipulated)
        is_real = confidence >= self.confidence_threshold
        is_fake = not is_real

        return {
            "is_real": is_real,
            "is_fake": is_fake,
            "confidence": confidence,
            "manipulation_probability": manipulation_probability,
        }

    def close(self) -> None:
        # Placeholder for releasing resources if needed later.
        pass


from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

# Top-level verdict: use this instead of relying only on is_real / is_fake when uncertain.
VerdictType = Literal["real", "fake", "uncertain"]


class TrackResult(BaseModel):
    track_id: int
    is_real: bool
    is_fake: bool
    confidence: float
    manipulation_probability: float
    raw_confidence: Optional[float] = None
    stability: Optional[float] = None
    hits: Optional[int] = None
    total_frames: Optional[int] = None
    speaking_activity: Optional[float] = None
    selection_score: Optional[float] = None
    window_confidences: Optional[List[float]] = None
    consecutive_miss_max: Optional[int] = None
    """Worst consecutive-miss streak in face tracker. Higher = less reliable."""


class WindowResult(BaseModel):
    window_index: int
    frame_start: int
    frame_end: int
    time_start_sec: Optional[float] = None
    time_end_sec: Optional[float] = None
    selected_track_id: int
    confidence: float
    is_real: bool
    is_fake: bool


class SpeakerSegment(BaseModel):
    selected_track_id: int
    frame_start: int
    frame_end: int
    time_start_sec: Optional[float] = None
    time_end_sec: Optional[float] = None


class MouthMotionCheck(BaseModel):
    """Result of the mouth-motion vs audio-energy cross-check."""

    check_result: str
    """One of: 'likely_fake', 'uncertain', 'no_issue', 'no_data', 'disabled'."""
    audio_energy: Optional[float] = None
    """Mean mel-dB energy of the audio window (higher = louder, max = 0 dB)."""
    mouth_motion_energy: Optional[float] = None
    """Mean abs frame-diff in the lower face region (proxy for mouth motion)."""


class LipSyncResponse(BaseModel):
    """Response for lip-sync manipulation detection."""

    verdict: VerdictType
    """Primary result: 'real' | 'fake' | 'uncertain'. Prefer this over is_real/is_fake
    when handling edge cases (e.g. no face detected, low confidence overrides)."""
    is_real: bool
    """True if video is authentic (not AI-manipulated), False if fake or uncertain."""
    is_fake: bool
    """True if video is AI-manipulated (fake), False if authentic or uncertain.
    When verdict is 'uncertain', both is_real and is_fake are False."""
    confidence: float
    """Confidence that video is REAL (authentic). Range [0, 1]."""
    manipulation_probability: float
    """Probability that video was AI-manipulated. Range [0, 1]."""
    tracks: Optional[List[TrackResult]] = None
    """Optional per-face/track results when multiple faces are detected."""
    selected_track_id: Optional[int] = None
    """Track id used for top-level decision (when tracks are returned)."""
    selection_uncertain: Optional[bool] = None
    """True when top-2 track selection scores are too close and turn-taking did not resolve it."""
    selection_margin: Optional[float] = None
    """Difference between top-1 and top-2 track selection scores."""

    # ── Confidence Margin Rule ────────────────────────────────────────────────
    confidence_margin_uncertain: Optional[bool] = None
    """True when top-2 track raw confidence scores are within confidence_margin of each other."""
    confidence_gap: Optional[float] = None
    """Absolute difference between top-1 and top-2 track confidence scores."""

    turn_taking_detected: Optional[bool] = None
    """True when different face tracks are the active speaker in different time windows."""
    speaker_case: Optional[str] = None
    """One of: all_speaking_real, mixed_real_and_fake, all_speaking_fake."""
    speaking_tracks_count: Optional[int] = None
    speaking_real_count: Optional[int] = None
    speaking_fake_count: Optional[int] = None
    verdicts: Optional[Dict[str, bool]] = None
    """Alternative clip verdict policies for edge-case handling."""
    window_results: Optional[List[WindowResult]] = None
    """Per-window selected-speaker results for turn-taking clips."""
    speaker_timeline: Optional[List[SpeakerSegment]] = None
    """Compressed speaker timeline from consecutive selected windows."""
    video_duration_sec: Optional[float] = None
    """Duration of the analyzed video in seconds."""
    total_chunks_analyzed: Optional[int] = None
    """Number of 32-frame chunks analyzed (long video mode only)."""

    # ── Mouth Motion Energy Check ─────────────────────────────────────────────
    mouth_motion_check: Optional[MouthMotionCheck] = None
    """Cross-check of audio energy vs mouth region motion energy."""

    # ── Override guards ───────────────────────────────────────────────────────
    sparse_real_guard_applied: Optional[bool] = None
    """True when the sparse-real-signal guard lifted a very-low-confidence fake
    verdict to uncertain/real because at least one window showed a real-like signal."""
    mouth_motion_override_applied: Optional[bool] = None
    """True when the mouth-motion 'uncertain' override lifted a fake verdict to
    real/uncertain because audio was quiet AND mouth motion was near-zero."""
    override_reason: Optional[str] = None
    """Human-readable reason for any conservative override that was applied.
    One of: 'sparse_real_signal', 'mouth_motion_uncertain', 'window_consensus_mixed', or None."""

    # ── Temporal drift (partial-manipulation detection) ───────────────────────
    temporal_confidence_drop: Optional[bool] = None
    """True when the second half of the video has a mean confidence ≥ 0.20
    lower than the first half.  May indicate a spliced or partially-manipulated
    clip even when the overall verdict is REAL."""
    temporal_drift: Optional[float] = None
    """first_half_avg_confidence − second_half_avg_confidence. Positive means
    confidence degraded toward the end of the clip."""
    first_half_avg_confidence: Optional[float] = None
    second_half_avg_confidence: Optional[float] = None

    detail: Optional[str] = None
    """Optional detail message."""


# ── Batch Evaluation (Precision / Recall / F1) ───────────────────────────────


class EvaluationItem(BaseModel):
    """A single prediction + ground-truth pair for batch evaluation."""

    predicted_is_fake: bool
    true_is_fake: bool
    video_id: Optional[str] = None


class BatchEvaluateRequest(BaseModel):
    """Request body for the /api/metrics/evaluate endpoint."""

    evaluations: List[EvaluationItem]


class BatchEvaluateResponse(BaseModel):
    """Precision, Recall, F1 and related metrics computed over a batch."""

    precision: float
    """TP / (TP + FP)  – of all predicted fakes, how many were truly fake."""
    recall: float
    """TP / (TP + FN)  – of all true fakes, how many were caught."""
    f1: float
    """Harmonic mean of precision and recall."""
    accuracy: float
    """(TP + TN) / total."""
    tp: int
    tn: int
    fp: int
    fn: int
    total: int

from typing import Optional, List, Dict

from pydantic import BaseModel


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


class LipSyncResponse(BaseModel):
    """Response for lip-sync manipulation detection."""

    is_real: bool
    """True if video is authentic (not AI-manipulated), False if fake/manipulated."""
    is_fake: bool
    """True if video is AI-manipulated (fake), False if authentic."""
    confidence: float
    """Confidence that video is REAL (authentic). Range [0, 1]."""
    manipulation_probability: float
    """Probability that video was AI-manipulated. Range [0, 1]."""
    tracks: Optional[List[TrackResult]] = None
    """Optional per-face/track results when multiple faces are detected."""
    selected_track_id: Optional[int] = None
    """Track id used for top-level decision (when tracks are returned)."""
    selection_uncertain: Optional[bool] = None
    """True when top-2 track scores are too close and turn-taking did not resolve it."""
    selection_margin: Optional[float] = None
    """Difference between top-1 and top-2 track selection scores."""
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
    detail: Optional[str] = None
    """Optional detail message."""



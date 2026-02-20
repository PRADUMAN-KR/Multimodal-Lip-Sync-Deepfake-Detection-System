"""
Production-grade face detection and mouth cropping for lip-sync detection.

Supports:
- Multiple faces per frame
- Multiple angles (frontal, profile, etc.)
- Robust tracking across frames
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import sys

from ..core.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """
    Face detection and landmark extraction using MediaPipe Face Mesh.
    supports multiple faces and angles.
    """

    def __init__(
        self,
        max_num_faces: int = 5,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ) -> None:
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._face_mesh = None  # Lazy initialization
        self._face_detector = None  # BBox fallback when FaceMesh misses
        self._haar_cascade = None  # OpenCV frontal fallback
        self._haar_profile_cascade = None  # OpenCV profile fallback for angled faces

        # Lazy import to avoid hard-crashing on unsupported Python versions.
        # MediaPipe wheels typically support only a subset of Python versions.
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "MediaPipe is required for production-grade face detection but could not be imported.\n"
                f"- Python: {sys.version.split()[0]}\n"
                "Fix:\n"
                "- Use Python 3.10 / 3.11 / 3.12 (MediaPipe often does NOT support bleeding-edge versions).\n"
                "- Reinstall: `pip install --upgrade --force-reinstall mediapipe`\n"
                "- Or disable face detection (not recommended for training).\n"
                f"Original import error: {e}"
            ) from e

        self._mp = mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Your installed `mediapipe` does not expose `mediapipe.solutions`.\n"
                "This typically happens when running an unsupported Python version (e.g. Python 3.14) "
                "or a broken/partial MediaPipe install.\n"
                f"- Python: {sys.version.split()[0]}\n"
                "Fix (recommended):\n"
                "- Activate your Python 3.11/3.12 virtualenv for this project and reinstall mediapipe there.\n"
                "- Do NOT train with dummy crops.\n"
            )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        
        # Mouth landmark indices (MediaPipe Face Mesh)
        # Outer lips: 61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321
        # Inner lips: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324
        self.mouth_landmarks = [
            61,
            146,
            91,
            181,
            84,
            17,
            314,
            405,
            320,
            307,
            375,
            321,  # Outer
            78,
            95,
            88,
            178,
            87,
            14,
            317,
            402,
            318,
            324,  # Inner
        ]
    
    @property
    def face_mesh(self):
        """Lazy initialization of FaceMesh to manage resources better."""
        if self._face_mesh is None:
            try:
                self._face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
            except Exception as e:
                # MediaPipe can fail to initialize due to threading/resource issues
                raise RuntimeError(
                    f"Failed to initialize MediaPipe FaceMesh: {e}\n"
                    "This may be due to threading issues or resource limits.\n"
                    "Try: Restart Python, reduce batch size, or check system resources."
                ) from e
        return self._face_mesh

    @property
    def face_detector(self):
        """Lazy initialization of MediaPipe face detector (bbox-based fallback)."""
        if self._face_detector is None:
            try:
                self._face_detector = self.mp_face_detection.FaceDetection(
                    # model_selection=1 performs better on smaller / distant faces.
                    model_selection=1,
                    min_detection_confidence=max(0.10, self.min_detection_confidence),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize MediaPipe FaceDetection fallback: {e}"
                ) from e
        return self._face_detector

    @property
    def haar_cascade(self):
        """Lazy initialization of OpenCV Haar cascade (frontal) as last-resort fallback."""
        if self._haar_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")
            self._haar_cascade = cascade
        return self._haar_cascade

    @property
    def haar_profile_cascade(self):
        """Lazy initialization of OpenCV Haar cascade (profile) for angled face detection."""
        if self._haar_profile_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                # Profile cascade may not be available in all OpenCV installations
                logger.warning(
                    "Profile face cascade not found at %s. Profile face detection disabled.",
                    cascade_path
                )
                self._haar_profile_cascade = None
            else:
                self._haar_profile_cascade = cascade
        return self._haar_profile_cascade
    
    def __del__(self):
        """Cleanup FaceMesh resources on deletion."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False
    
    def close(self):
        """Explicitly close and cleanup MediaPipe resources."""
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._face_mesh = None
        if self._face_detector is not None:
            try:
                self._face_detector.close()
            except Exception:
                pass
            self._face_detector = None
        self._haar_cascade = None
        self._haar_profile_cascade = None

    def _face_bbox_to_mouth_bbox(
        self, fx1: int, fy1: int, fw: int, fh: int, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        """Convert a face box to an approximate mouth-region box."""
        x_min = fx1 + int(0.18 * fw)
        x_max = fx1 + int(0.82 * fw)
        y_min = fy1 + int(0.52 * fh)
        y_max = fy1 + int(0.96 * fh)

        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(x_min + 1, min(x_max, width))
        y_max = max(y_min + 1, min(y_max, height))
        return x_min, y_min, x_max, y_max

    def _detect_faces_haar_fallback(self, frame: np.ndarray) -> List[dict]:
        """
        OpenCV Haar cascade fallback (frontal + profile) for cases where MediaPipe finds nothing.
        Tries frontal cascade first, then profile cascade for angled faces.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        min_side = max(20, int(min(h, w) * 0.04))

        all_rects = []
        
        # Try frontal face cascade
        try:
            frontal_rects = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(min_side, min_side),
            )
            # Convert numpy arrays to tuples to avoid boolean ambiguity issues
            if len(frontal_rects) > 0:
                all_rects.extend([tuple(rect) for rect in frontal_rects])
        except Exception as e:
            logger.debug("Frontal Haar cascade failed: %s: %s", type(e).__name__, e)

        # Try profile face cascade for angled faces
        if self.haar_profile_cascade is not None:
            try:
                profile_rects = self.haar_profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=3,
                    minSize=(min_side, min_side),
                )
                # Convert numpy arrays to tuples to avoid boolean ambiguity issues
                if len(profile_rects) > 0:
                    all_rects.extend([tuple(rect) for rect in profile_rects])
            except Exception as e:
                logger.debug("Profile Haar cascade failed: %s: %s", type(e).__name__, e)

        if not all_rects:
            logger.debug(
                "FaceMesh miss → MediaPipe FaceDetection miss → OpenCV Haar cascade fallback: 0 faces detected"
            )
            return []

        # Remove duplicate detections (same face detected by both cascades)
        # Simple NMS: keep largest bbox if IoU > 0.5
        def _bbox_iou(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter_area = inter_x * inter_y
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - inter_area
            return inter_area / union if union > 0 else 0.0

        filtered_rects = []
        for rect in all_rects:
            is_duplicate = False
            for existing in filtered_rects:
                if _bbox_iou(rect, existing) > 0.5:
                    # Keep the larger one
                    if (rect[2] * rect[3]) > (existing[2] * existing[3]):
                        filtered_rects.remove(existing)
                        filtered_rects.append(rect)
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_rects.append(rect)

        faces: List[dict] = []
        for x, y, fw, fh in filtered_rects[: self.max_num_faces]:
            x_min, y_min, x_max, y_max = self._face_bbox_to_mouth_bbox(x, y, fw, fh, w, h)
            if (x_max - x_min) < 4 or (y_max - y_min) < 4:
                continue
            faces.append(
                {
                    "bbox": (x_min, y_min, x_max, y_max),
                    "landmarks": np.empty((0, 2), dtype=np.float32),
                    "mouth_landmarks": np.empty((0, 2), dtype=np.float32),
                    "_detector": "haar",  # Marker for detector tracking
                }
            )

        if faces:
            logger.debug(
                "FaceMesh miss → MediaPipe FaceDetection miss → OpenCV Haar cascade fallback: found %d face(s) (frontal+profile)",
                len(faces)
            )
        return faces

    def _detect_faces_bbox_fallback(self, frame: np.ndarray) -> List[dict]:
        """
        Fallback detector using MediaPipe FaceDetection bounding boxes.

        Converts full-face bbox into an approximate mouth-region bbox.
        """
        h, w = frame.shape[:2]
        try:
            results = self.face_detector.process(frame)
        except Exception as e:
            logger.warning(
                "MediaPipe FaceDetection fallback failed (trying Haar): %s: %s",
                type(e).__name__,
                e,
            )
            return self._detect_faces_haar_fallback(frame)

        if not results or not results.detections:
            logger.debug(
                "FaceMesh miss → MediaPipe FaceDetection fallback: 0 detections, trying Haar cascade"
            )
            return self._detect_faces_haar_fallback(frame)

        faces: List[dict] = []
        for det in results.detections[: self.max_num_faces]:
            rel = det.location_data.relative_bounding_box
            fx1 = int(rel.xmin * w)
            fy1 = int(rel.ymin * h)
            fw = int(rel.width * w)
            fh = int(rel.height * h)

            x_min, y_min, x_max, y_max = self._face_bbox_to_mouth_bbox(
                fx1, fy1, fw, fh, w, h
            )
            if (x_max - x_min) < 4 or (y_max - y_min) < 4:
                continue

            faces.append(
                {
                    "bbox": (x_min, y_min, x_max, y_max),
                    "landmarks": np.empty((0, 2), dtype=np.float32),
                    "mouth_landmarks": np.empty((0, 2), dtype=np.float32),
                    "_detector": "mp_bbox",  # Marker for detector tracking
                }
            )

        if faces:
            logger.debug(
                "FaceMesh miss → MediaPipe FaceDetection fallback: found %d face(s)",
                len(faces)
            )
            return faces
        logger.debug(
            "FaceMesh miss → MediaPipe FaceDetection fallback: 0 valid faces, trying Haar cascade"
        )
        return self._detect_faces_haar_fallback(frame)

    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """
        Detect faces and extract landmarks.

        Args:
            frame: RGB image (H, W, 3)

        Returns:
            List of face dicts with keys: 'bbox', 'landmarks', 'mouth_landmarks'
        """
        try:
            results = self.face_mesh.process(frame)
        except Exception as e:
            # MediaPipe can crash on certain inputs (threading issues, etc.)
            # Log and use bbox fallback instead of returning empty.
            logger.debug(
                "FaceMesh exception → using fallback chain: %s: %s",
                type(e).__name__,
                e,
            )
            return self._detect_faces_bbox_fallback(frame)
        
        faces = []

        if not results or not results.multi_face_landmarks:
            logger.debug("FaceMesh: 0 faces detected → using fallback chain")
            return self._detect_faces_bbox_fallback(frame)

        h, w = frame.shape[:2]

        for face_landmarks in results.multi_face_landmarks:
            # Extract all landmarks
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x * w, landmark.y * h])

            landmarks = np.array(landmarks)

            # Extract mouth landmarks
            mouth_pts = landmarks[self.mouth_landmarks]

            # Bounding box from mouth region (with padding)
            x_min = int(mouth_pts[:, 0].min()) - 20
            x_max = int(mouth_pts[:, 0].max()) + 20
            y_min = int(mouth_pts[:, 1].min()) - 20
            y_max = int(mouth_pts[:, 1].max()) + 20

            # Clamp to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            faces.append(
                {
                    "bbox": (x_min, y_min, x_max, y_max),
                    "landmarks": landmarks,
                    "mouth_landmarks": mouth_pts,
                    "_detector": "facemesh",  # Marker for detector tracking
                }
            )

        return faces if faces else self._detect_faces_bbox_fallback(frame)

    def crop_mouth_region(
        self, frame: np.ndarray, face_info: dict, crop_size: Tuple[int, int] = (96, 96)
    ) -> Optional[np.ndarray]:
        """
        Crop and align mouth region from face info.

        Args:
            frame: RGB image (H, W, 3)
            face_info: Face dict from detect_faces()
            crop_size: Output size (H, W)

        Returns:
            Cropped mouth region (H, W, 3) or None if invalid
        """
        x_min, y_min, x_max, y_max = face_info["bbox"]

        if x_max <= x_min or y_max <= y_min:
            return None

        # Crop mouth region
        mouth_crop = frame[y_min:y_max, x_min:x_max]

        if mouth_crop.size == 0:
            return None

        # Resize to target size
        mouth_crop = cv2.resize(mouth_crop, crop_size, interpolation=cv2.INTER_LINEAR)

        return mouth_crop


class MultiFaceTracker:
    """
    Tracks multiple faces across video frames using IoU-based matching.
    """

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self.tracks: List[List[dict]] = []  # List of tracks, each track is list of face_info

    def _iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, faces: List[dict]) -> List[List[dict]]:
        """
        Update tracks with new detections.

        Args:
            faces: List of face dicts from current frame

        Returns:
            Updated tracks (list of face sequences)
        """
        if not self.tracks:
            # Initialize tracks with first frame
            self.tracks = [[face] for face in faces]
            return self.tracks

        # Match faces to existing tracks
        matched_tracks = set()
        matched_faces = set()

        for track_idx, track in enumerate(self.tracks):
            if not track:
                continue
            last_face = track[-1]
            best_iou = 0.0
            best_face_idx = -1

            for face_idx, face in enumerate(faces):
                if face_idx in matched_faces:
                    continue
                iou = self._iou(last_face["bbox"], face["bbox"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_face_idx = face_idx

            if best_face_idx >= 0:
                track.append(faces[best_face_idx])
                matched_tracks.add(track_idx)
                matched_faces.add(best_face_idx)

        # Create new tracks for unmatched faces
        for face_idx, face in enumerate(faces):
            if face_idx not in matched_faces:
                self.tracks.append([face])

        # Remove tracks that haven't been updated (optional: can keep for a few frames)
        self.tracks = [track for idx, track in enumerate(self.tracks) if idx in matched_tracks or len(track) == 1]

        return self.tracks

    def get_longest_track(self) -> Optional[List[dict]]:
        """Get the longest track (most consistent face)."""
        if not self.tracks:
            return None
        return max(self.tracks, key=len)


def detect_and_crop_mouths(
    frames: np.ndarray,
    crop_size: Tuple[int, int] = (96, 96),
    max_faces: int = 1,
    select_strategy: str = "longest",
) -> np.ndarray:
    """
    Detect faces and crop mouth regions from video frames.

    Args:
        frames: Video frames (T, H, W, 3) RGB
        crop_size: Output crop size (H, W)
        max_faces: Maximum number of faces to process
        select_strategy: "longest" (track longest), "largest" (largest bbox), "first" (first detected)

    Returns:
        Mouth crops (T, H, W, 3) - if multiple faces, uses selected strategy
        
    Raises:
        RuntimeError: If face detection fails catastrophically (MediaPipe crash, etc.)
    """
    try:
        detector = FaceDetector(max_num_faces=max_faces)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize FaceDetector: {e}\n"
            "This usually means MediaPipe is broken. Fix MediaPipe installation."
        ) from e
    
    tracker = MultiFaceTracker()

    mouth_crops = []

    for frame_idx, frame in enumerate(frames):
        try:
            faces = detector.detect_faces(frame)
        except Exception as e:
            # MediaPipe can crash with pthread errors on certain frames
            # Log and use fallback instead of crashing
            logger.warning(
                f"MediaPipe crashed on frame {frame_idx} (using fallback crop): {type(e).__name__}: {e}"
            )
            # Fallback to center crop
            h, w = frame.shape[:2]
            ch, cw = crop_size
            y1 = max(0, h // 2 - ch // 2)
            x1 = max(0, w // 2 - cw // 2)
            y2 = min(h, y1 + ch)
            x2 = min(w, x1 + cw)
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, crop_size, interpolation=cv2.INTER_LINEAR)
            mouth_crops.append(crop)
            continue

        if not faces:
            # Fallback: center crop if no face detected
            h, w = frame.shape[:2]
            ch, cw = crop_size
            y1 = max(0, h // 2 - ch // 2)
            x1 = max(0, w // 2 - cw // 2)
            y2 = min(h, y1 + ch)
            x2 = min(w, x1 + cw)
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, crop_size, interpolation=cv2.INTER_LINEAR)
            mouth_crops.append(crop)
            continue

        # Update tracker
        tracks = tracker.update(faces)

        # Select face based on strategy
        if select_strategy == "longest":
            selected_track = tracker.get_longest_track()
            if selected_track:
                face_info = selected_track[-1]
            else:
                face_info = faces[0]
        elif select_strategy == "largest":
            # Select face with largest bounding box
            face_info = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        else:  # first
            face_info = faces[0]

        # Crop mouth
        crop = detector.crop_mouth_region(frame, face_info, crop_size)
        if crop is None:
            # Fallback
            h, w = frame.shape[:2]
            ch, cw = crop_size
            y1 = max(0, h // 2 - ch // 2)
            x1 = max(0, w // 2 - cw // 2)
            y2 = min(h, y1 + ch)
            x2 = min(w, x1 + cw)
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, crop_size, interpolation=cv2.INTER_LINEAR)

        mouth_crops.append(crop)

    return np.stack(mouth_crops, axis=0)


def detect_and_crop_mouth_tracks(
    frames: np.ndarray,
    crop_size: Tuple[int, int] = (96, 96),
    max_faces: int = 5,
    max_tracks: int = 5,
    iou_threshold: float = 0.25,
    iou_threshold_relaxed: float = 0.12,
    max_age: int = 3,
    min_stability: float = 0.35,
    min_detection_confidence: float = 0.3,
    min_tracking_confidence: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Detect and track multiple faces, returning mouth crops per track.

    Improvements over basic IoU tracker:
    - Velocity extrapolation: predicted bbox used when detection is missed.
    - Grace period (max_age): track stays alive for N frames without a match
      before being dropped, using a relaxed IoU threshold for re-association.
    - Backfilled interpolated crops for missed frames between two known detections.
    - Weighted stability score that penalises consecutive misses more than
      scattered ones.
    - min_stability filter: weak tracks are dropped before being returned.

    Returns:
        List of track dictionaries:
        - ``track_id``     : int
        - ``crops``        : (T, H, W, 3) RGB
        - ``hits``         : matched frames count
        - ``total_frames`` : total processed frames
        - ``stability``    : weighted stability score in [0, 1]
        - ``consecutive_miss_max``: worst consecutive-miss streak
    """
    detector = FaceDetector(
        max_num_faces=max_faces,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    def _center_crop(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        ch, cw = crop_size
        y1 = max(0, h // 2 - ch // 2)
        x1 = max(0, w // 2 - cw // 2)
        y2 = min(h, y1 + ch)
        x2 = min(w, x1 + cw)
        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, crop_size, interpolation=cv2.INTER_LINEAR)

    def _bbox_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop mouth region from a bbox (x_min, y_min, x_max, y_max)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(x1 + 1, min(x2, w)))
        y2 = int(max(y1 + 1, min(y2, h)))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return _center_crop(frame)
        return cv2.resize(crop, crop_size, interpolation=cv2.INTER_LINEAR)

    def _iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = max(1, (x1_max - x1_min) * (y1_max - y1_min))
        area2 = max(1, (x2_max - x2_min) * (y2_max - y2_min))
        union = area1 + area2 - inter_area
        return float(inter_area / union) if union > 0 else 0.0

    def _predict_bbox(
        bbox: Tuple[int, int, int, int],
        velocity: Tuple[float, float, float, float],
        steps: int = 1,
    ) -> Tuple[int, int, int, int]:
        """Extrapolate bbox forward using stored velocity."""
        x1, y1, x2, y2 = bbox   
        vx1, vy1, vx2, vy2 = velocity
        return (
            int(round(x1 + vx1 * steps)),
            int(round(y1 + vy1 * steps)),
            int(round(x2 + vx2 * steps)),
            int(round(y2 + vy2 * steps)),
        )

    def _lerp_bbox(
        b_from: Tuple[int, int, int, int],
        b_to: Tuple[int, int, int, int],
        t: float,
    ) -> Tuple[int, int, int, int]:
        """Linearly interpolate between two bboxes (t in [0, 1])."""
        return (
            int(round(b_from[0] + (b_to[0] - b_from[0]) * t)),
            int(round(b_from[1] + (b_to[1] - b_from[1]) * t)),
            int(round(b_from[2] + (b_to[2] - b_from[2]) * t)),
            int(round(b_from[3] + (b_to[3] - b_from[3]) * t)),
        )

    # ── Track state ───────────────────────────────────────────────────────────
    # Each track dict holds:
    #   id, last_bbox, prev_bbox, velocity, crops (list, one per frame),
    #   bbox_history (for backfill), hits, age, consecutive_miss,
    #   max_consecutive_miss, miss_frame_indices
    tracks: List[dict] = []
    next_id = 0
    total_frames = max(1, len(frames))

    logger.info(
        "Starting face detection and tracking: %d frames, "
        "max_faces=%d, max_tracks=%d, iou=%.2f, iou_relaxed=%.2f, max_age=%d",
        total_frames, max_faces, max_tracks,
        iou_threshold, iou_threshold_relaxed, max_age,
    )

    # Track detector usage for summary
    detector_stats = {
        "facemesh": 0,
        "mp_bbox": 0,
        "haar": 0,
        "none": 0,
    }

    for t_idx, frame in enumerate(frames):
        faces = detector.detect_faces(frame)
        
        # Track which detector was used based on _detector marker
        if not faces:
            detector_stats["none"] += 1
        elif faces:
            detector_name = faces[0].get("_detector", "facemesh")  # Default to facemesh if marker missing
            if detector_name in detector_stats:
                detector_stats[detector_name] += 1
            else:
                detector_stats["facemesh"] += 1  # Fallback

        if t_idx == 0 or (t_idx + 1) % 10 == 0:
            logger.debug("Frame %d/%d: Detected %d face(s)", t_idx + 1, total_frames, len(faces))

        # ── Precompute crops for this frame's faces ───────────────────────────
        face_items: List[Tuple[dict, np.ndarray]] = []
        for f in faces:
            crop = detector.crop_mouth_region(frame, f, crop_size=crop_size)
            if crop is None:
                crop = _center_crop(frame)
            face_items.append((f, crop))

        matched_faces: set = set()

        # ── Match existing tracks ─────────────────────────────────────────────
        for tr in tracks:
            if tr["age"] > max_age:
                # Already expired; skip matching (will be pruned below)
                continue

            # Use velocity-predicted bbox for matching when track is in grace period
            predicted_bbox = (
                _predict_bbox(tr["last_bbox"], tr["velocity"], steps=tr["age"] + 1)
                if tr["age"] > 0
                else tr["last_bbox"]
            )

            best_iou = 0.0
            best_idx = -1
            # First pass: standard threshold
            for i, (f, _crop) in enumerate(face_items):
                if i in matched_faces:
                    continue
                score = _iou(predicted_bbox, f["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            # Second pass: relax threshold for tracks in grace period
            effective_threshold = (
                iou_threshold_relaxed if tr["age"] > 0 else iou_threshold
            )

            if best_idx >= 0 and best_iou >= effective_threshold:
                f, crop = face_items[best_idx]
                matched_faces.add(best_idx)

                new_bbox: Tuple[int, int, int, int] = tuple(f["bbox"])  # type: ignore[assignment]

                # ── Backfill interpolated crops for missed frames ─────────────
                if tr["age"] > 0 and tr["crops"]:
                    last_known_bbox = tr["last_bbox"]
                    gap = tr["age"]  # how many frames were missed
                    for g in range(gap):
                        interp_bbox = _lerp_bbox(
                            last_known_bbox, new_bbox, (g + 1) / (gap + 1)
                        )
                        interp_crop = _bbox_crop(
                            frames[t_idx - gap + g], interp_bbox
                        )
                        # Overwrite the placeholder crop at that slot
                        fill_idx = len(tr["crops"]) - gap + g
                        if 0 <= fill_idx < len(tr["crops"]):
                            tr["crops"][fill_idx] = interp_crop
                            logger.debug(
                                "  Track %d: backfilled frame %d with interpolated bbox",
                                tr["id"], t_idx - gap + g,
                            )

                # ── Update velocity (exponential smoothing) ───────────────────
                old_bbox = tr["last_bbox"]
                raw_v = (
                    float(new_bbox[0] - old_bbox[0]),
                    float(new_bbox[1] - old_bbox[1]),
                    float(new_bbox[2] - old_bbox[2]),
                    float(new_bbox[3] - old_bbox[3]),
                )
                alpha = 0.4  # smoothing factor; higher → faster adaptation
                tr["velocity"] = tuple(
                    alpha * raw_v[k] + (1 - alpha) * tr["velocity"][k]
                    for k in range(4)
                )  # type: ignore[assignment]

                tr["last_bbox"] = new_bbox
                tr["crops"].append(crop)
                tr["hits"] += 1
                tr["age"] = 0
                tr["consecutive_miss"] = 0

                if t_idx == 0 or (t_idx + 1) % 10 == 0:
                    logger.debug(
                        "  Track %d: Matched face %d (IoU=%.3f), hits=%d",
                        tr["id"], best_idx, best_iou, tr["hits"],
                    )
            else:
                # ── Miss: append placeholder, increment age ───────────────────
                placeholder = (
                    _bbox_crop(frame, _predict_bbox(tr["last_bbox"], tr["velocity"]))
                    if tr["crops"]
                    else _center_crop(frame)
                )
                tr["crops"].append(placeholder)
                tr["age"] += 1
                tr["consecutive_miss"] += 1
                tr["max_consecutive_miss"] = max(
                    tr["max_consecutive_miss"], tr["consecutive_miss"]
                )
                tr["miss_frame_indices"].append(t_idx)

        # ── Prune expired tracks ──────────────────────────────────────────────
        tracks = [tr for tr in tracks if tr["age"] <= max_age]

        # ── Create new tracks for unmatched faces ─────────────────────────────
        for i, (f, crop) in enumerate(face_items):
            if i in matched_faces:
                continue
            if len(tracks) >= max_tracks:
                logger.debug(
                    "  Frame %d: Skipping unmatched face %d (max_tracks=%d reached)",
                    t_idx + 1, i, max_tracks,
                )
                continue
            tr = {
                "id": next_id,
                "last_bbox": tuple(f["bbox"]),
                "velocity": (0.0, 0.0, 0.0, 0.0),
                "crops": [_center_crop(frame)] * t_idx + [crop],
                "hits": 1,
                "age": 0,
                "consecutive_miss": 0,
                "max_consecutive_miss": 0,
                "miss_frame_indices": [],
            }
            logger.debug(
                "  Frame %d: Created new track %d for unmatched face %d",
                t_idx + 1, next_id, i,
            )
            next_id += 1
            tracks.append(tr)

    # ── Compute weighted stability and filter ─────────────────────────────────
    def _weighted_stability(tr: dict) -> float:
        """
        Stability score in [0, 1].

        Penalties applied:
        - Base: hits / total_frames
        - Consecutive-miss penalty: each extra consecutive miss beyond 1
          adds a multiplicative penalty.
        """
        base = float(tr["hits"]) / total_frames
        max_consec = int(tr.get("max_consecutive_miss", 0))
        if max_consec <= 1:
            return base
        # Each consecutive miss beyond 1 applies a 15% relative penalty.
        consec_penalty = min(0.5, (max_consec - 1) * 0.15)
        return float(base * (1.0 - consec_penalty))

    logger.info("Face tracking completed: %d tracks created", len(tracks))
    
    # Log detector usage summary
    total_detections = sum(detector_stats.values())
    if total_detections > 0:
        pct_facemesh = (detector_stats["facemesh"] / total_detections) * 100
        pct_mp_bbox = (detector_stats["mp_bbox"] / total_detections) * 100
        pct_haar = (detector_stats["haar"] / total_detections) * 100
        pct_none = (detector_stats["none"] / total_detections) * 100
        
        logger.info(
            "Detector usage summary (%d frames): FaceMesh=%.1f%% (%d), "
            "MP FaceDetection=%.1f%% (%d), Haar cascade=%.1f%% (%d), None=%.1f%% (%d)",
            total_detections,
            pct_facemesh, detector_stats["facemesh"],
            pct_mp_bbox, detector_stats["mp_bbox"],
            pct_haar, detector_stats["haar"],
            pct_none, detector_stats["none"],
        )

    # Score, filter, sort
    for tr in tracks:
        tr["w_stability"] = _weighted_stability(tr)

    viable = [tr for tr in tracks if tr["w_stability"] >= min_stability]
    if not viable:
        # Fallback: keep best track regardless of threshold
        viable = tracks
        logger.warning(
            "No tracks passed min_stability=%.2f; keeping %d best track(s)",
            min_stability, min(1, len(viable)),
        )

    tracks_sorted = sorted(viable, key=lambda tr: tr["w_stability"], reverse=True)[
        :max_tracks
    ]
    logger.info(
        "Selected %d track(s): ids=%s, hits=%s, weighted_stability=%s",
        len(tracks_sorted),
        [tr["id"] for tr in tracks_sorted],
        [tr["hits"] for tr in tracks_sorted],
        [f"{tr['w_stability']:.3f}" for tr in tracks_sorted],
    )

    # Retry with much lower confidence if first pass found nothing.
    # Helps with real-world clips (street interviews, distant faces, etc.)
    RETRY_DET_CONF = 0.15
    RETRY_TRK_CONF = 0.15
    if (
        not tracks_sorted
        and min_detection_confidence > RETRY_DET_CONF
    ):
        logger.warning(
            "Zero tracks with det_conf=%.2f, trk_conf=%.2f. "
            "Retrying with lowered thresholds (det=%.2f, trk=%.2f).",
            min_detection_confidence, min_tracking_confidence,
            RETRY_DET_CONF, RETRY_TRK_CONF,
        )
        return detect_and_crop_mouth_tracks(
            frames,
            crop_size=crop_size,
            max_faces=max_faces,
            max_tracks=max_tracks,
            iou_threshold=iou_threshold,
            iou_threshold_relaxed=iou_threshold_relaxed,
            max_age=max_age,
            min_stability=min_stability,
            min_detection_confidence=RETRY_DET_CONF,
            min_tracking_confidence=RETRY_TRK_CONF,
        )

    out: List[Dict[str, Any]] = []
    for tr in tracks_sorted:
        crops = np.stack(tr["crops"], axis=0)
        out.append(
            {
                "track_id": int(tr["id"]),
                "crops": crops,
                "hits": int(tr["hits"]),
                "total_frames": total_frames,
                "stability": float(tr["w_stability"]),
                "consecutive_miss_max": int(tr.get("max_consecutive_miss", 0)),
            }
        )
    return out

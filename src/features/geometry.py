"""
Branch A — Geometric Math features for the DMS pipeline (Stage 1).

This module implements:
    1. HeadPoseEstimator   -> 6-point solvePnP head pose (yaw, pitch, roll)
    2. eye_aspect_ratio     -> standard 6-point EAR formula
    3. AsymmetryEAR         -> |EAR_left - EAR_right|
    4. RollingMinMaxNormalizer -> per-driver, per-window [0,1] EAR normalization

Landmark indices follow MediaPipe FaceMesh (468-point model). The "left/right"
naming below is internally consistent with the 6 solvePnP anchors specified
in the project spec (Nose=1, Chin=152, L-Eye=33, R-Eye=263, L-Mouth=61,
R-Mouth=291). Because MediaPipe outputs are in image (camera) space, "left"
here means "appears on the left side of the image", not the driver's
anatomical left — verify this against your actual camera mounting before
trusting alert direction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Landmark index sets (MediaPipe FaceMesh, 468 points)
# ---------------------------------------------------------------------------

# 6 anchors used for solvePnP, per project spec.
POSE_ANCHOR_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

# 6-point EAR sets. Order matches the standard EAR formula:
# p1, p4 = horizontal corners ; p2, p3 (upper lid) ; p5, p6 (lower lid)
# Corner indices (33, 263) are kept consistent with POSE_ANCHOR_IDX above.
LEFT_EYE_EAR_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_IDX = (362, 385, 387, 263, 373, 380)

# Generic 3D face model (arbitrary unit, e.g. mm) for solvePnP.
# This is the widely used canonical 6-point model from the classic
# OpenCV head-pose-estimation approach (Nose, Chin, Eye corners, Mouth corners).
GENERIC_3D_FACE_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Head Pose Estimation (solvePnP)
# ---------------------------------------------------------------------------

@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    success: bool = True


class HeadPoseEstimator:
    """Estimate head pose (yaw, pitch, roll) from 6 facial landmarks via solvePnP.

    Camera intrinsics are approximated from frame size when no calibration is
    available (focal length ~= frame width, principal point at image center,
    zero lens distortion). This is the standard practical approximation used
    when a real camera calibration is not provided; accuracy is "good enough"
    for relative pose tracking but not metrically precise.
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        self.camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def estimate(self, landmarks_px: Dict[int, Tuple[float, float]]) -> HeadPose:
        """
        Args:
            landmarks_px: dict mapping MediaPipe landmark index -> (x_px, y_px)
                           for ALL 468 points (only the 6 anchors are used here).
        Returns:
            HeadPose with yaw/pitch/roll in degrees. success=False if any
            required anchor point is missing (e.g. due to occlusion).
        """
        try:
            image_points = np.array(
                [landmarks_px[idx] for idx in POSE_ANCHOR_IDX.values()],
                dtype=np.float64,
            )
        except KeyError:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, success=False)

        ok, rotation_vec, _translation_vec = cv2.solvePnP(
            GENERIC_3D_FACE_MODEL,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, success=False)

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_mat)
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll, success=True)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """Decompose a 3x3 rotation matrix into (pitch, yaw, roll) in degrees."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0
        return np.degrees(x), np.degrees(y), np.degrees(z)


# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR) + Asymmetry
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_points: Sequence[Tuple[float, float]]) -> float:
    """Standard 6-point EAR. `eye_points` order: p1..p6 as in LEFT/RIGHT_EYE_EAR_IDX."""
    p1, p2, p3, p4, p5, p6 = (np.array(p, dtype=np.float64) for p in eye_points)
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_both_ear(
    landmarks_px: Dict[int, Tuple[float, float]]
) -> Tuple[Optional[float], Optional[float]]:
    """Returns (ear_left, ear_right); None for an eye if any of its points is missing."""
    try:
        left_pts = [landmarks_px[i] for i in LEFT_EYE_EAR_IDX]
        ear_left = eye_aspect_ratio(left_pts)
    except KeyError:
        ear_left = None
    try:
        right_pts = [landmarks_px[i] for i in RIGHT_EYE_EAR_IDX]
        ear_right = eye_aspect_ratio(right_pts)
    except KeyError:
        ear_right = None
    return ear_left, ear_right


def ear_asymmetry(ear_left: Optional[float], ear_right: Optional[float]) -> Optional[float]:
    if ear_left is None or ear_right is None:
        return None
    return abs(ear_left - ear_right)


# ---------------------------------------------------------------------------
# Rolling Min-Max Normalizer (10s window, per-driver calibration)
# ---------------------------------------------------------------------------

class RollingMinMaxNormalizer:
    """Maintains a rolling window of raw values and rescales the latest value to [0, 1].

    Handles two edge cases that a naive implementation misses:
      - Cold start: window not yet full -> normalize against whatever history exists.
      - Degenerate range (max ~= min, e.g. eyes closed the whole window) -> avoid
        dividing by ~0 by falling back to a small epsilon-based "no information" output.
    """

    def __init__(self, window_seconds: float, fps: float, eps: float = 1e-3):
        self.window_size = max(1, int(round(window_seconds * fps)))
        self.eps = eps
        self.buffer: Deque[float] = deque(maxlen=self.window_size)

    def update_and_normalize(self, value: float) -> float:
        self.buffer.append(value)
        arr = np.array(self.buffer)
        vmin = np.percentile(arr, 5)  # Lọc bỏ 5% giá trị nhiễu thấp nhất
        vmax = np.percentile(arr, 95)
        value_range = vmax - vmin
        if value_range < self.eps:
            # Degenerate window (near-constant signal): no useful contrast to
            # normalize against. Return 0.5 (neutral) rather than blowing up.
            return 0.5
        return float(np.clip((value - vmin) / value_range, 0.0, 1.0))

    def is_warmed_up(self) -> bool:
        return len(self.buffer) >= self.window_size


# ---------------------------------------------------------------------------
# Orchestrator: produces the full Branch-A geometry feature vector per frame
# ---------------------------------------------------------------------------

@dataclass
class GeometryFeatures:
    yaw: float
    pitch: float
    roll: float
    ear_left: float
    ear_right: float
    ear_asymmetry: float
    ear_left_norm: float
    ear_right_norm: float
    valid: bool  # False if landmarks were unusable (e.g. low confidence frame)


class GeometryFeatureExtractor:
    """Stateful extractor: wraps HeadPoseEstimator + EAR + rolling normalizers.

    One instance should persist per video/driver session, since the rolling
    normalizers carry state across frames.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        window_seconds: float = 10.0,
        fps: float = 30.0,
    ):
        self.pose_estimator = HeadPoseEstimator(frame_width, frame_height)
        self.left_normalizer = RollingMinMaxNormalizer(window_seconds, fps)
        self.right_normalizer = RollingMinMaxNormalizer(window_seconds, fps)

    def extract(self, landmarks_px: Dict[int, Tuple[float, float]]) -> GeometryFeatures:
        pose = self.pose_estimator.estimate(landmarks_px)
        ear_left, ear_right = compute_both_ear(landmarks_px)

        if not pose.success or ear_left is None or ear_right is None:
            return GeometryFeatures(
                yaw=0.0, pitch=0.0, roll=0.0,
                ear_left=0.0, ear_right=0.0, ear_asymmetry=0.0,
                ear_left_norm=0.5, ear_right_norm=0.5,
                valid=False,
            )

        asymmetry = ear_asymmetry(ear_left, ear_right) or 0.0
        ear_left_norm = self.left_normalizer.update_and_normalize(ear_left)
        ear_right_norm = self.right_normalizer.update_and_normalize(ear_right)

        return GeometryFeatures(
            yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll,
            ear_left=ear_left, ear_right=ear_right, ear_asymmetry=asymmetry,
            ear_left_norm=ear_left_norm, ear_right_norm=ear_right_norm,
            valid=True,
        )

    def as_vector(self, feats: GeometryFeatures) -> np.ndarray:
        """Flatten to the geometry vector consumed by Stage 2 (FiLM generator)."""
        return np.array(
            [
                feats.yaw, feats.pitch, feats.roll,
                feats.ear_left_norm, feats.ear_right_norm, feats.ear_asymmetry,
            ],
            dtype=np.float32,
        )

"""
Branch B — Isotropic Patches for the DMS pipeline (Stage 1).

Implements:
    1. square_pad         -> letterbox an arbitrary-aspect-ratio crop into a
                              square via zero (black) padding, never via
                              stretching, so a CNN never sees a squashed eye.
    2. EyePatchExtractor   -> crops an eye region from a frame given
                              MediaPipe landmarks, with a configurable margin,
                              then applies square_pad + resize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from .geometry import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX


def square_pad(image: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Pad a H x W (x C) image with black borders so H == W.

    Pads symmetrically on the shorter axis. This avoids the "squashed eye"
    distortion that a naive cv2.resize(image, (S, S)) would introduce when
    the crop's aspect ratio is far from 1:1.
    """
    h, w = image.shape[:2]
    if h == w:
        return image

    side = max(h, w)
    pad_total_h = side - h
    pad_total_w = side - w
    top, bottom = pad_total_h // 2, pad_total_h - pad_total_h // 2
    left, right = pad_total_w // 2, pad_total_w - pad_total_w // 2

    border_value = fill_value if image.ndim == 2 else (fill_value,) * image.shape[2]
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value
    )


@dataclass
class EyePatchExtractor:
    """Crops a square, padded, resized eye patch from a frame.

    margin_ratio expands the tight landmark bounding box by this fraction of
    its own size on each side, so the patch includes a bit of context (brow,
    upper cheek) rather than just the bare eye outline.
    """

    target_size: int = 64
    margin_ratio: float = 0.35

    def _bbox_from_landmarks(
        self, points_px: Sequence[Tuple[float, float]], frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in points_px]
        ys = [p[1] for p in points_px]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        box_w, box_h = x_max - x_min, y_max - y_min
        margin_x, margin_y = box_w * self.margin_ratio, box_h * self.margin_ratio

        x_min = max(0, int(x_min - margin_x))
        y_min = max(0, int(y_min - margin_y))
        x_max = min(frame_w, int(x_max + margin_x))
        y_max = min(frame_h, int(y_max + margin_y))
        return x_min, y_min, x_max, y_max

    def extract(
        self,
        frame: np.ndarray,
        landmarks_px: Dict[int, Tuple[float, float]],
        eye: str,
    ) -> np.ndarray:
        """
        Args:
            frame: full BGR/RGB frame, shape (H, W, C).
            landmarks_px: dict of landmark idx -> (x_px, y_px) for the full face.
            eye: "left" or "right".
        Returns:
            Square, padded, resized patch of shape (target_size, target_size, C).
        """
        idx_set = LEFT_EYE_EAR_IDX if eye == "left" else RIGHT_EYE_EAR_IDX
        points_px = [landmarks_px[i] for i in idx_set]
        frame_h, frame_w = frame.shape[:2]
        x_min, y_min, x_max, y_max = self._bbox_from_landmarks(points_px, frame_w, frame_h)

        if x_max <= x_min or y_max <= y_min:
            # Degenerate box (shouldn't normally happen) -> blank patch.
            channels = frame.shape[2] if frame.ndim == 3 else 1
            return np.zeros((self.target_size, self.target_size, channels), dtype=frame.dtype)

        crop = frame[y_min:y_max, x_min:x_max]
        padded = square_pad(crop)
        resized = cv2.resize(padded, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        return resized

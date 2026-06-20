"""
Data Pipeline (part 1) — Offline preprocessing: raw frames + MediaPipe
landmarks -> one saved "window sample" file per window.

WHY OFFLINE, NOT LIVE: solvePnP/EAR/eye-cropping (Stage 1) are CPU-bound and
would otherwise be re-run on every single training epoch for the same
frames. Running them ONCE here and caching the result as plain tensors
lets the actual training loop be a fast, GPU-bound tensor-loading pipeline.

This module is agnostic to how you obtained the landmarks (MediaPipe
FaceMesh, a different detector, etc.) — it only needs, for every frame:
    - the raw frame (H, W, 3) uint8 image
    - a dict {landmark_idx: (x_px, y_px)} (same format Stage 1 expects)
    - a scalar tracking confidence in [0, 1]
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..features.geometry import GeometryFeatureExtractor
from ..features.patches import EyePatchExtractor


def build_window_sample(
    frames: List[np.ndarray],
    landmarks_per_frame: List[Dict[int, Tuple[float, float]]],
    confidences: List[float],
    label: int,
    geometry_extractor: GeometryFeatureExtractor,
    patch_extractor: EyePatchExtractor,
) -> dict:
    """Runs Stage 1 over every frame of one window and packs the result.

    Args:
        frames: list of T raw (H, W, 3) uint8 frames for this window.
        landmarks_per_frame: list of T landmark dicts (same length as frames).
        confidences: list of T tracking-confidence scalars in [0, 1].
        label: 0 (Alert) or 1 (Drowsy) for this window.
        geometry_extractor: a GeometryFeatureExtractor — reuse the SAME
            instance across consecutive windows from the same video, since
            its rolling min-max normalizer carries state across frames.
        patch_extractor: an EyePatchExtractor (stateless, can be shared freely).

    Returns:
        dict of tensors ready to be saved with torch.save():
            left_patches, right_patches: (T, 3, H, W) uint8
            geometry: (T, 6) float32
            confidence: (T,) float32
            label: 0-d int64 tensor
    """
    assert len(frames) == len(landmarks_per_frame) == len(confidences), \
        "frames/landmarks/confidences phải cùng độ dài T"

    left_patches, right_patches, geometry_vectors = [], [], []
    for frame, landmarks in zip(frames, landmarks_per_frame):
        geom_feats = geometry_extractor.extract(landmarks)
        geometry_vectors.append(geometry_extractor.as_vector(geom_feats))

        left = patch_extractor.extract(frame, landmarks, eye="left")
        right = patch_extractor.extract(frame, landmarks, eye="right")
        # HWC uint8 -> CHW uint8 (standard torch image layout)
        left_patches.append(np.transpose(left, (2, 0, 1)))
        right_patches.append(np.transpose(right, (2, 0, 1)))

    return {
        "left_patches": torch.from_numpy(np.stack(left_patches)).to(torch.uint8),
        "right_patches": torch.from_numpy(np.stack(right_patches)).to(torch.uint8),
        "geometry": torch.from_numpy(np.stack(geometry_vectors)).to(torch.float32),
        "confidence": torch.tensor(confidences, dtype=torch.float32),
        "label": torch.tensor(label, dtype=torch.int64),
    }


def save_window_sample(sample: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(sample, path)


def attach_xgb_oof_proba(sample_paths: List[str], oof_proba: np.ndarray) -> None:
    """Adds the precomputed XGBoost OOF probability to each saved window
    sample, in the same order as `sample_paths`. Call this once after
    running XGBoostBaseline.fit_oof() on the aggregated geometry features of
    the whole training set — see src/training/xgboost_baseline.py.
    """
    assert len(sample_paths) == len(oof_proba)
    for path, proba in zip(sample_paths, oof_proba):
        sample = torch.load(path, map_location="cpu")
        sample["xgb_oof_proba"] = torch.tensor(float(proba), dtype=torch.float32)
        torch.save(sample, path)

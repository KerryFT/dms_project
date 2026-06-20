"""
Stage 5 (part 1) — Aggregate per-frame geometry features into one fixed-size
tabular feature vector per window, for the XGBoost baseline.

XGBoost needs a fixed-length tabular input, but Stage 1 produces one
6-dim geometry vector PER FRAME. This module summarizes a window (sequence)
of geometry vectors into statistics that are known to be informative for
drowsiness from the classical PERCLOS/EAR literature: mean, std, min, max
per dimension, plus an explicit PERCLOS-style "fraction of frames with low
normalized EAR" feature (this is exactly the statistic that made the
0.5422 F1 baseline work — keep it, don't let XGBoost have to rediscover it
from raw mean/std alone).
"""

from __future__ import annotations

from typing import List

import numpy as np

# Must match the order produced by GeometryFeatureExtractor.as_vector():
# [yaw, pitch, roll, ear_left_norm, ear_right_norm, ear_asymmetry]
GEOMETRY_DIM_NAMES = ["yaw", "pitch", "roll", "ear_left_norm", "ear_right_norm", "ear_asymmetry"]

PERCLOS_THRESHOLD = 0.2  # normalized-EAR threshold below which an eye counts as "closed"


def aggregate_geometry_window(geometry_sequence: np.ndarray) -> np.ndarray:
    """
    Args:
        geometry_sequence: (T, 6) array of per-frame geometry vectors for one window.
    Returns:
        1D feature vector of length 6*4 + 1 = 25:
            [mean_i, std_i, min_i, max_i for i in 6 dims] + [perclos]
    """
    assert geometry_sequence.ndim == 2 and geometry_sequence.shape[1] == 6, \
        f"Kỳ vọng shape (T, 6), nhận được {geometry_sequence.shape}"

    mean = geometry_sequence.mean(axis=0)
    std = geometry_sequence.std(axis=0)
    vmin = geometry_sequence.min(axis=0)
    vmax = geometry_sequence.max(axis=0)

    ear_left_norm = geometry_sequence[:, 3]
    ear_right_norm = geometry_sequence[:, 4]
    eyes_closed = (ear_left_norm < PERCLOS_THRESHOLD) & (ear_right_norm < PERCLOS_THRESHOLD)
    perclos = eyes_closed.mean()

    return np.concatenate([mean, std, vmin, vmax, [perclos]]).astype(np.float32)


def feature_names() -> List[str]:
    names = []
    for stat in ["mean", "std", "min", "max"]:
        names += [f"{dim}_{stat}" for dim in GEOMETRY_DIM_NAMES]
    names.append("perclos")
    return names


def aggregate_batch(geometry_sequences: List[np.ndarray]) -> np.ndarray:
    """Convenience for a list of (T_i, 6) windows (T_i may vary) -> (N, 25)."""
    return np.stack([aggregate_geometry_window(seq) for seq in geometry_sequences])

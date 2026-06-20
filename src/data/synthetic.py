"""
FOR TESTING / DEMO ONLY — NOT real preprocessing.

Generates fake window sample .pt files with the exact schema
build_window_sample() produces, so the Dataset/DataLoader/training loop
can be exercised end-to-end in environments without real video + MediaPipe
landmarks (e.g. this sandbox). Do not use these files to report any real
accuracy numbers — the "signal" is synthetic and only exists to verify the
code path is wired correctly.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch


def generate_synthetic_window_dataset(
    output_dir: str,
    n_windows: int = 40,
    t_min: int = 15,
    t_max: int = 25,
    patch_size: int = 64,
    seed: int = 0,
) -> List[str]:
    """Creates n_windows fake .pt files in output_dir; returns their paths.

    Drowsy windows (label=1) get systematically lower ear_norm values and
    higher PERCLOS, so the synthetic data is at least directionally
    sensible (not pure noise) — useful for sanity-checking that a training
    loop's loss can go down at all.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []

    for i in range(n_windows):
        label = int(rng.integers(0, 2))
        t = int(rng.integers(t_min, t_max + 1))

        left_patches = rng.integers(0, 255, size=(t, 3, patch_size, patch_size), dtype=np.uint8)
        right_patches = rng.integers(0, 255, size=(t, 3, patch_size, patch_size), dtype=np.uint8)

        yaw = rng.normal(0, 10, size=t)
        pitch = rng.normal(0, 10, size=t)
        roll = rng.normal(0, 5, size=t)
        if label == 1:
            ear_left = np.clip(rng.normal(0.2, 0.1, size=t), 0, 1)
            ear_right = np.clip(rng.normal(0.2, 0.1, size=t), 0, 1)
        else:
            ear_left = np.clip(rng.normal(0.7, 0.1, size=t), 0, 1)
            ear_right = np.clip(rng.normal(0.7, 0.1, size=t), 0, 1)
        asymmetry = np.abs(ear_left - ear_right)
        geometry = np.stack([yaw, pitch, roll, ear_left, ear_right, asymmetry], axis=1).astype(np.float32)

        confidence = np.clip(rng.normal(0.9, 0.1, size=t), 0, 1).astype(np.float32)
        # Occasionally simulate a tracking dropout to exercise ConfidenceGate.
        if rng.random() < 0.3:
            drop_start = rng.integers(0, max(1, t - 3))
            confidence[drop_start: drop_start + 3] = rng.uniform(0, 0.2)

        sample = {
            "left_patches": torch.from_numpy(left_patches),
            "right_patches": torch.from_numpy(right_patches),
            "geometry": torch.from_numpy(geometry),
            "confidence": torch.from_numpy(confidence),
            "label": torch.tensor(label, dtype=torch.int64),
        }
        path = os.path.join(output_dir, f"window_{i:04d}.pt")
        torch.save(sample, path)
        paths.append(path)

    return paths

"""
Quick sanity test for Stage 1 (Branch A + Branch B) using synthetic landmarks.
This does NOT require MediaPipe to be installed — it fabricates a plausible
468-point face landmark dict so we can exercise solvePnP, EAR, rolling
normalization, and the square-pad eye patch logic end-to-end.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.features.geometry import GeometryFeatureExtractor
from src.features.patches import EyePatchExtractor, square_pad


def make_synthetic_landmarks(frame_w=640, frame_h=480, yaw_offset=0.0, seed=0):
    """Builds a dict of landmark_idx -> (x_px, y_px) covering all indices our
    code actually touches (pose anchors + both EAR eye sets), with small
    deterministic jitter so frames differ slightly from each other."""
    rng = np.random.default_rng(seed)
    cx, cy = frame_w / 2 + yaw_offset, frame_h / 2

    base = {
        1: (cx, cy - 20),  # nose tip
        152: (cx, cy + 80),  # chin
        33: (cx - 60, cy - 10),  # left eye corner
        263: (cx + 60, cy - 10),  # right eye corner
        61: (cx - 30, cy + 50),  # left mouth corner
        291: (cx + 30, cy + 50),  # right mouth corner
        # left eye EAR ring around 33
        160: (cx - 50, cy - 18), 158: (cx - 40, cy - 18),
        133: (cx - 30, cy - 10),
        153: (cx - 40, cy - 2), 144: (cx - 50, cy - 2),
        # right eye EAR ring around 263
        362: (cx + 30, cy - 10),
        385: (cx + 40, cy - 18), 387: (cx + 50, cy - 18),
        373: (cx + 50, cy - 2), 380: (cx + 40, cy - 2),
    }
    jitter = rng.normal(scale=0.5, size=(len(base), 2))
    return {idx: (x + jitter[i, 0], y + jitter[i, 1]) for i, (idx, (x, y)) in enumerate(base.items())}


def test_geometry_branch():
    extractor = GeometryFeatureExtractor(frame_width=640, frame_height=480, window_seconds=2.0, fps=10)
    print("--- Branch A: Geometry ---")
    for frame_idx in range(15):
        # Simulate a slight head turn over time to see yaw change.
        landmarks = make_synthetic_landmarks(yaw_offset=frame_idx * 2, seed=frame_idx)
        feats = extractor.extract(landmarks)
        vec = extractor.as_vector(feats)
        print(
            f"frame {frame_idx:02d} | yaw={feats.yaw:7.2f} pitch={feats.pitch:7.2f} "
            f"roll={feats.roll:7.2f} | EAR L/R={feats.ear_left:.3f}/{feats.ear_right:.3f} "
            f"asym={feats.ear_asymmetry:.3f} | norm L/R={feats.ear_left_norm:.2f}/{feats.ear_right_norm:.2f} "
            f"| vector={np.round(vec, 2)} | valid={feats.valid}"
        )
    assert vec.shape == (6,), "Geometry vector phải có đúng 6 chiều cho Stage 2 (FiLM)"
    print("OK: Branch A chạy không lỗi, vector đúng shape (6,)\n")


def test_missing_landmark_fallback():
    extractor = GeometryFeatureExtractor(frame_width=640, frame_height=480)
    landmarks = make_synthetic_landmarks()
    del landmarks[152]  # simulate occlusion losing the chin point
    feats = extractor.extract(landmarks)
    assert feats.valid is False
    print("OK: Mất landmark (occlusion) -> fallback an toàn, valid=False, không crash\n")


def test_degenerate_rolling_normalizer():
    from src.features.geometry import RollingMinMaxNormalizer
    norm = RollingMinMaxNormalizer(window_seconds=1.0, fps=5, eps=1e-3)
    # Constant value the whole window (eyes closed the entire 1s window).
    outputs = [norm.update_and_normalize(0.15) for _ in range(5)]
    assert all(abs(o - 0.5) < 1e-6 for o in outputs), "Window hằng số phải fallback về 0.5, không chia /0"
    print("OK: Window EAR hằng số (degenerate range) -> fallback 0.5, không lỗi chia 0\n")


def test_patch_branch():
    print("--- Branch B: Isotropic Patches ---")
    frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    landmarks = make_synthetic_landmarks()
    extractor = EyePatchExtractor(target_size=64, margin_ratio=0.4)

    left_patch = extractor.extract(frame, landmarks, eye="left")
    right_patch = extractor.extract(frame, landmarks, eye="right")
    print(f"Left patch shape:  {left_patch.shape}")
    print(f"Right patch shape: {right_patch.shape}")
    assert left_patch.shape == (64, 64, 3)
    assert right_patch.shape == (64, 64, 3)

    # Directly test square_pad on a deliberately non-square crop (40x20 case from spec).
    rect = (np.random.default_rng(1).random((20, 40, 3)) * 255).astype(np.uint8)
    squared = square_pad(rect)
    print(f"square_pad(40x20) -> {squared.shape} (must be square)")
    assert squared.shape[0] == squared.shape[1]
    print("OK: Branch B chạy không lỗi, patch luôn vuông, không bị méo (squash)\n")


if __name__ == "__main__":
    test_geometry_branch()
    test_missing_landmark_fallback()
    test_degenerate_rolling_normalizer()
    test_patch_branch()
    print("=== TẤT CẢ TEST STAGE 1 ĐỀU PASS ===")

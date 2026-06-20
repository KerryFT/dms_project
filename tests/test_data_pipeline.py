"""
Sanity tests for the Data Pipeline (preprocessing -> Dataset -> DataLoader).
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.features.geometry import GeometryFeatureExtractor
from src.features.patches import EyePatchExtractor
from src.data.preprocessing import build_window_sample, save_window_sample, attach_xgb_oof_proba
from src.data.dataset import DMSWindowDataset, collate_windows, normalize_patches
from src.data.synthetic import generate_synthetic_window_dataset
from src.training.train_loop import precompute_xgb_oof

TMP_DIR = "/tmp/dms_data_pipeline_test"


def _make_synthetic_landmarks(frame_w=640, frame_h=480, seed=0):
    rng = np.random.default_rng(seed)
    cx, cy = frame_w / 2, frame_h / 2
    base = {
        1: (cx, cy - 20), 152: (cx, cy + 80), 33: (cx - 60, cy - 10), 263: (cx + 60, cy - 10),
        61: (cx - 30, cy + 50), 291: (cx + 30, cy + 50),
        160: (cx - 50, cy - 18), 158: (cx - 40, cy - 18), 133: (cx - 30, cy - 10),
        153: (cx - 40, cy - 2), 144: (cx - 50, cy - 2),
        362: (cx + 30, cy - 10), 385: (cx + 40, cy - 18), 387: (cx + 50, cy - 18),
        373: (cx + 50, cy - 2), 380: (cx + 40, cy - 2),
    }
    jitter = rng.normal(scale=0.5, size=(len(base), 2))
    return {idx: (x + jitter[i, 0], y + jitter[i, 1]) for i, (idx, (x, y)) in enumerate(base.items())}


def test_build_window_sample_from_stage1():
    print("--- build_window_sample: nối Stage 1 (frame thô + landmark) -> tensor lưu được ---")
    frame_w, frame_h, t = 640, 480, 12
    frames = [(np.random.default_rng(i).random((frame_h, frame_w, 3)) * 255).astype(np.uint8) for i in range(t)]
    landmarks = [_make_synthetic_landmarks(frame_w, frame_h, seed=i) for i in range(t)]
    confidences = [0.9] * t

    geom_extractor = GeometryFeatureExtractor(frame_w, frame_h, window_seconds=2.0, fps=10)
    patch_extractor = EyePatchExtractor(target_size=64)

    sample = build_window_sample(frames, landmarks, confidences, label=1, geometry_extractor=geom_extractor, patch_extractor=patch_extractor)
    print(f"left_patches: {sample['left_patches'].shape} dtype={sample['left_patches'].dtype}")
    print(f"geometry: {sample['geometry'].shape}")
    assert sample["left_patches"].shape == (t, 3, 64, 64)
    assert sample["right_patches"].shape == (t, 3, 64, 64)
    assert sample["geometry"].shape == (t, 6)
    assert sample["confidence"].shape == (t,)
    assert sample["label"].item() == 1

    os.makedirs(TMP_DIR, exist_ok=True)
    path = os.path.join(TMP_DIR, "sample_roundtrip.pt")
    save_window_sample(sample, path)
    reloaded = torch.load(path)
    assert torch.equal(reloaded["geometry"], sample["geometry"])
    print("OK: build_window_sample đúng shape, save/load round-trip khớp dữ liệu\n")


def test_dataset_and_dataloader_padding():
    print("--- DMSWindowDataset + collate_windows: padding + valid_mask đúng ---")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    paths = generate_synthetic_window_dataset(TMP_DIR, n_windows=16, t_min=10, t_max=20, seed=1)

    true_lengths = [torch.load(p)["geometry"].shape[0] for p in paths]

    dataset = DMSWindowDataset(TMP_DIR)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_windows)
    batch = next(iter(loader))

    print(f"left_patches: {batch['left_patches'].shape}")
    print(f"valid_mask: {batch['valid_mask'].shape}, sum per sample: {batch['valid_mask'].sum(dim=1).tolist()}")
    max_t = batch["geometry"].shape[1]
    assert batch["left_patches"].shape == (4, max_t, 3, 64, 64)
    assert batch["geometry"].shape == (4, max_t, 6)
    assert batch["valid_mask"].shape == (4, max_t)

    for i in range(4):
        assert batch["valid_mask"][i].sum().item() == true_lengths[i], \
            f"valid_mask phải đúng độ dài thật của sample {i}"
        # Phần padding (sau true_length) phải toàn 0.
        if true_lengths[i] < max_t:
            assert torch.all(batch["geometry"][i, true_lengths[i]:] == 0)
    print("OK: padding đúng, valid_mask khớp độ dài thật từng sample, vùng pad toàn 0\n")

    # Normalize phải biến uint8 [0,255] thành float đã chuẩn hoá ImageNet (không còn trong [0,255]).
    assert batch["left_patches"].dtype == torch.float32
    assert batch["left_patches"].max().item() < 10.0  # đã normalize, không còn ở thang 0-255
    print("OK: ImageNet normalization được áp dụng trong Dataset, không phải trong file cache\n")


def test_attach_xgb_oof_and_precompute():
    print("--- precompute_xgb_oof: gắn xgb_oof_proba vào từng file, không leak ---")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    paths = generate_synthetic_window_dataset(TMP_DIR, n_windows=60, seed=2)

    baseline = precompute_xgb_oof(TMP_DIR, n_splits=5)
    assert baseline.final_model is not None

    for p in paths:
        sample = torch.load(p)
        assert "xgb_oof_proba" in sample
        assert 0.0 <= sample["xgb_oof_proba"].item() <= 1.0

    dataset_with_oof = DMSWindowDataset(TMP_DIR, require_xgb_oof=True)
    loader = DataLoader(dataset_with_oof, batch_size=8, shuffle=False, collate_fn=collate_windows)
    batch = next(iter(loader))
    assert not torch.isnan(batch["xgb_oof_proba"]).any()
    print(f"xgb_oof_proba mẫu đầu batch: {batch['xgb_oof_proba'][:4].tolist()}")
    print("OK: mọi sample đều có xgb_oof_proba hợp lệ trong [0,1], không còn NaN khi DataLoader yêu cầu bắt buộc\n")


if __name__ == "__main__":
    test_build_window_sample_from_stage1()
    test_dataset_and_dataloader_padding()
    test_attach_xgb_oof_and_precompute()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("=== TẤT CẢ TEST DATA PIPELINE ĐỀU PASS ===")

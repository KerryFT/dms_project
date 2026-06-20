"""
Data Pipeline (part 2) — torch Dataset + DataLoader collation for windows
saved by src/data/preprocessing.py.

Handles two things a naive Dataset would get wrong:
    1. Pretrained MobileNetV3 expects ImageNet-normalized float input, not
       raw uint8 — normalization happens HERE (once per sample, on the fly),
       not baked into the cached files, so you can swap normalization
       strategy without re-running the expensive Stage-1 preprocessing.
    2. Real videos don't all produce the same number of frames per window
       (variable fps, dropped frames, clips near a video's end). collate_fn
       pads every sequence in a batch to the batch's max length and returns
       a valid_mask so Stage 3's TemporalAttention can correctly ignore the
       padding (see TemporalAttention's valid_mask docstring).
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_patches(patches_uint8: torch.Tensor) -> torch.Tensor:
    """(T, 3, H, W) uint8 in [0,255] -> (T, 3, H, W) float, ImageNet-normalized."""
    x = patches_uint8.float() / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class DMSWindowDataset(Dataset):
    def __init__(self, sample_dir: str, require_xgb_oof: bool = False):
        """
        Args:
            sample_dir: directory containing *.pt window files written by
                        src/data/preprocessing.save_window_sample().
            require_xgb_oof: if True, raise immediately on a sample missing
                        "xgb_oof_proba" rather than failing silently later
                        in the residual-loss step of training.
        """
        self.sample_paths = sorted(glob.glob(os.path.join(sample_dir, "*.pt")))
        if len(self.sample_paths) == 0:
            raise FileNotFoundError(f"Không tìm thấy file .pt nào trong {sample_dir}")
        self.require_xgb_oof = require_xgb_oof

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> dict:
        sample = torch.load(self.sample_paths[idx], map_location="cpu")
        if self.require_xgb_oof and "xgb_oof_proba" not in sample:
            raise KeyError(
                f"{self.sample_paths[idx]} thiếu 'xgb_oof_proba'. "
                "Chạy XGBoostBaseline.fit_oof() + attach_xgb_oof_proba() trước."
            )
        return {
            "left_patches": normalize_patches(sample["left_patches"]),
            "right_patches": normalize_patches(sample["right_patches"]),
            "geometry": sample["geometry"],
            "confidence": sample["confidence"],
            "label": sample["label"],
            "xgb_oof_proba": sample.get("xgb_oof_proba", torch.tensor(float("nan"))),
        }


def collate_windows(batch: List[dict]) -> dict:
    """Pads every (T, ...) field to max_T in the batch; returns valid_mask (B, max_T)."""
    lengths = [b["geometry"].shape[0] for b in batch]
    max_t = max(lengths)
    batch_size = len(batch)

    def pad_seq(tensor: torch.Tensor, target_t: int) -> torch.Tensor:
        pad_amount = target_t - tensor.shape[0]
        if pad_amount == 0:
            return tensor
        pad_shape = (pad_amount, *tensor.shape[1:])
        return torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=0)

    left_patches = torch.stack([pad_seq(b["left_patches"], max_t) for b in batch])
    right_patches = torch.stack([pad_seq(b["right_patches"], max_t) for b in batch])
    geometry = torch.stack([pad_seq(b["geometry"], max_t) for b in batch])
    confidence = torch.stack([pad_seq(b["confidence"], max_t) for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    xgb_oof_proba = torch.stack([b["xgb_oof_proba"] for b in batch])

    valid_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    for i, length in enumerate(lengths):
        valid_mask[i, :length] = True

    return {
        "left_patches": left_patches,
        "right_patches": right_patches,
        "geometry": geometry,
        "confidence": confidence,
        "label": labels,
        "xgb_oof_proba": xgb_oof_proba,
        "valid_mask": valid_mask,
    }

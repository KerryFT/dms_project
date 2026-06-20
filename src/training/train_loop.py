"""
Training Loop — wires every stage together:

    Stage 1 (offline, already cached in the dataset)
        -> Stage 2+3+4 (DMSModel, trained jointly by backprop)
        -> Stage 5a: XGBoost on aggregated geometry (precompute_xgb_oof,
           fit ONCE before joint training, not part of the backprop loop)
        -> Stage 5b: ResidualFallbackHead (trained jointly with 2+3+4,
           using the frozen XGBoost OOF probability as a constant input)

Total loss per batch:
    L = drowsiness_loss(window_logits, label)             # Stage 3
      + lambda_triplet  * triplet_loss(embedding, label)   # Stage 4
      + lambda_residual * residual_bce(final_score, label) # Stage 5
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

from ..features.window_features import aggregate_batch
from .losses import DrowsinessLoss
from .triplet import TripletLoss
from .xgboost_baseline import XGBoostBaseline
from ..data.preprocessing import attach_xgb_oof_proba
from ..data.dataset import DMSWindowDataset


# ---------------------------------------------------------------------------
# Helper: load .pt file an toàn, tương thích mọi phiên bản PyTorch
# ---------------------------------------------------------------------------

def _safe_load(path: str) -> Optional[dict]:
    """
    Load một .pt file, trả về None nếu file bị hỏng (corrupted).

    Lý do KHÔNG dùng weights_only=True:
      - weights_only=True yêu cầu PyTorch >= 2.0 và gây lỗi
        'PytorchStreamReader failed locating file data.pkl' trên Kaggle
        (PyTorch 2.6+) với một số file được tạo bởi phiên bản cũ hơn.
      - Các file .pt trong project này do CHÚNG TA tạo ra (chỉ chứa
        tensor thuần), không có rủi ro bảo mật khi tắt weights_only.
    """
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"  [WARN] Bỏ qua file hỏng: {path} ({type(e).__name__}: {e})")
        return None


def _load_all_safe(paths: List[str]) -> Tuple[List[dict], List[str]]:
    """Load nhiều .pt file, tự động lọc bỏ file hỏng.
    Returns (valid_samples, valid_paths)."""
    valid_samples, valid_paths = [], []
    for p in paths:
        s = _safe_load(p)
        if s is not None:
            valid_samples.append(s)
            valid_paths.append(p)
    n_skip = len(paths) - len(valid_paths)
    if n_skip:
        print(f"  [WARN] Bỏ qua {n_skip}/{len(paths)} file hỏng.")
    return valid_samples, valid_paths


# ---------------------------------------------------------------------------
# Stage 5a: precompute XGBoost OOF (train/val) và final model (deploy)
# ---------------------------------------------------------------------------

def precompute_xgb_oof(
    sample_dir: str,
    n_splits: int = 5,
    save_model_path: Optional[str] = None,
) -> XGBoostBaseline:
    """
    Fit XGBoost qua OOF cross-validation, ghi xgb_oof_proba vào từng .pt file,
    fit final model trên toàn bộ set.
    """
    dataset = DMSWindowDataset(sample_dir, require_xgb_oof=False)
    samples, valid_paths = _load_all_safe(dataset.sample_paths)

    geometry_sequences = [s["geometry"].numpy() for s in samples]
    labels = np.array([s["label"].item() for s in samples])

    X = aggregate_batch(geometry_sequences)
    baseline = XGBoostBaseline(n_splits=n_splits)
    oof_proba = baseline.fit_oof(X, labels)
    attach_xgb_oof_proba(valid_paths, oof_proba)
    baseline.fit_final(X, labels)

    if save_model_path:
        baseline.save(save_model_path)
        print(f"XGBoost final model saved → {save_model_path}")

    return baseline


def compute_xgb_proba_for_set(
    sample_dir: str,
    xgb_model_path: str,
) -> np.ndarray:
    """
    Dùng XGBoost FINAL model để predict P(Drowsy) cho val hoặc test set.
    Ghi kết quả vào từng .pt file để DataLoader đọc được.
    """
    baseline = XGBoostBaseline.load(xgb_model_path)
    dataset = DMSWindowDataset(sample_dir, require_xgb_oof=False)
    samples, valid_paths = _load_all_safe(dataset.sample_paths)

    geometry_sequences = [s["geometry"].numpy() for s in samples]
    X = aggregate_batch(geometry_sequences)
    proba = baseline.predict_proba(X)
    attach_xgb_oof_proba(valid_paths, proba)
    print(f"XGBoost proba ghi vào {len(valid_paths)} files trong {sample_dir}")
    return proba


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    lambda_triplet: float = 0.2
    lambda_residual: float = 0.3
    use_residual: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 5.0


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    drowsiness_loss: DrowsinessLoss,
    triplet_loss: TripletLoss,
    config: TrainConfig,
) -> Dict[str, float]:
    model.train()
    totals = {"total": 0.0, "drowsiness": 0.0, "triplet": 0.0, "residual": 0.0}
    n_batches = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        labels = batch["label"]

        xgb_proba = batch["xgb_oof_proba"] if config.use_residual else None
        out = model(
            batch["left_patches"], batch["right_patches"], batch["geometry"],
            batch["confidence"], valid_mask=batch["valid_mask"], xgb_proba=xgb_proba,
        )

        loss_drowsy = drowsiness_loss(out["window_logits"], labels)
        loss_triplet = triplet_loss(out["triplet_embedding"], labels)
        loss = loss_drowsy + config.lambda_triplet * loss_triplet

        loss_residual = torch.tensor(0.0, device=device)
        if config.use_residual:
            final_score = out["final_score"].clamp(1e-6, 1 - 1e-6)
            loss_residual = F.binary_cross_entropy(final_score, labels.float())
            loss = loss + config.lambda_residual * loss_residual

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        totals["total"] += loss.item()
        totals["drowsiness"] += loss_drowsy.item()
        totals["triplet"] += loss_triplet.item()
        totals["residual"] += loss_residual.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Evaluate — 3 chế độ: xgb_only, dl_only, full_pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_residual: bool = True,
) -> Dict[str, float]:
    """
    Evaluate một lần, trả về metrics.

    Nếu use_residual=True: dùng final_score (XGBoost + ΔS) — chế độ deploy.
    Nếu use_residual=False: dùng window_logits (chỉ Stage 3) — chế độ ablation.

    Lưu ý: với test set, gọi compute_xgb_proba_for_set() trước khi evaluate
    để xgb_oof_proba trong batch là hợp lệ (không phải NaN).
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        labels = batch["label"]
        xgb_proba = batch["xgb_oof_proba"] if use_residual else None
        out = model(
            batch["left_patches"], batch["right_patches"], batch["geometry"],
            batch["confidence"], valid_mask=batch["valid_mask"], xgb_proba=xgb_proba,
        )
        if use_residual:
            preds = (out["final_score"] > 0.5).long()
        else:
            preds = out["window_logits"].argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        "accuracy":          accuracy_score(all_labels, all_preds),
        "precision_drowsy":  precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "recall_drowsy":     recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "f1_drowsy":         f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
    }


@torch.no_grad()
def evaluate_xgb_only(sample_dir: str, xgb_model_path: str) -> Dict[str, float]:
    """
    Baseline thuần: chỉ dùng XGBoost final model, không có DL.
    Dùng để in ra baseline F1 để so sánh với full pipeline.
    """
    baseline  = XGBoostBaseline.load(xgb_model_path)
    dataset   = DMSWindowDataset(sample_dir, require_xgb_oof=False)
    samples, _ = _load_all_safe(dataset.sample_paths)
    geom_seqs = [s["geometry"].numpy() for s in samples]
    labels    = np.array([s["label"].item() for s in samples])
    X = aggregate_batch(geom_seqs)
    proba = baseline.predict_proba(X)
    preds = (proba > 0.5).astype(int)
    return {
        "accuracy":         accuracy_score(labels, preds),
        "precision_drowsy": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_drowsy":    recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_drowsy":        f1_score(labels, preds, pos_label=1, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    class_weights: Optional[torch.Tensor] = None,
    config: Optional[TrainConfig] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """Orchestrates the full joint-training loop, keeping the best-F1 checkpoint."""
    config = config or TrainConfig()
    model.to(device)
    drowsiness_loss = DrowsinessLoss(class_weights=class_weights).to(device)
    triplet_loss    = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    history  = {"train_loss": [], "val_f1_drowsy": []}
    best_f1  = -1.0

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            drowsiness_loss, triplet_loss, config,
        )
        val_metrics = evaluate(model, val_loader, device, use_residual=config.use_residual)

        history["train_loss"].append(train_metrics["total"])
        history["val_f1_drowsy"].append(val_metrics["f1_drowsy"])

        if verbose:
            print(
                f"[Epoch {epoch+1:>3}/{num_epochs}] "
                f"loss={train_metrics['total']:.4f} "
                f"(cls={train_metrics['drowsiness']:.4f} "
                f"tri={train_metrics['triplet']:.4f} "
                f"res={train_metrics['residual']:.4f}) | "
                f"val_acc={val_metrics['accuracy']:.3f}  "
                f"val_f1={val_metrics['f1_drowsy']:.3f}  "
                f"val_rec={val_metrics['recall_drowsy']:.3f}"
            )

        if checkpoint_path and val_metrics["f1_drowsy"] > best_f1:
            best_f1 = val_metrics["f1_drowsy"]
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    return history

"""
Training Loop — wires every stage together:

    Stage 1 (offline, already cached in the dataset)
        -> Stage 2+3+4 (DMSModel, trained jointly by backprop)
        -> Stage 5a: XGBoost on aggregated geometry (precompute_xgb_oof,
           fit ONCE before joint training, not part of the backprop loop)
        -> Stage 5b: ResidualFallbackHead (trained jointly with 2+3+4,
           using the frozen XGBoost OOF probability as a constant input)

Total loss per batch:
    L = drowsiness_loss(window_logits, label)            # Stage 3
      + lambda_triplet  * triplet_loss(embedding, label)  # Stage 4
      + lambda_residual * residual_bce(final_score, label) # Stage 5
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

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
# Stage 5a: precompute XGBoost OOF probabilities for the WHOLE training set
# (run this ONCE, before joint training; never inside the training loop).
# ---------------------------------------------------------------------------

def precompute_xgb_oof(sample_dir: str, n_splits: int = 5) -> XGBoostBaseline:
    """Loads every cached window's geometry, fits XGBoost via OOF
    cross-validation, writes 'xgb_oof_proba' back into each .pt file, and
    also fits a final deployment model on the full set. Returns the fitted
    XGBoostBaseline (its .final_model is what you ship for inference).

    CHỈ gọi hàm này cho TRAIN SET. Đối với val/test, dùng
    attach_xgb_proba_for_eval() bên dưới — fit_oof()/fit_final() ở đây
    luôn dùng label của chính sample_dir để fit model MỚI, nên gọi hàm
    này trên val/test sẽ vô tình dùng label của chúng để huấn luyện một
    phần của pipeline (leakage), đặc biệt nghiêm trọng với test set.
    """
    dataset = DMSWindowDataset(sample_dir, require_xgb_oof=False)
    geometry_sequences = [torch.load(p)["geometry"].numpy() for p in dataset.sample_paths]
    labels = np.array([torch.load(p)["label"].item() for p in dataset.sample_paths])

    X = aggregate_batch(geometry_sequences)
    baseline = XGBoostBaseline(n_splits=n_splits)
    oof_proba = baseline.fit_oof(X, labels)
    attach_xgb_oof_proba(dataset.sample_paths, oof_proba)
    baseline.fit_final(X, labels)
    return baseline


def attach_xgb_proba_for_eval(sample_dir: str, baseline: XGBoostBaseline) -> None:
    """Sinh 'xgb_oof_proba' cho VAL hoặc TEST set bằng baseline ĐÃ fit
    trên train (baseline.final_model, qua fit_final) — chỉ predict, không
    fit lại gì cả. Đây là cách đúng để đưa XGBoost proba vào val/test mà
    không leak label của chính chúng vào pipeline trước khi đánh giá.

    Args:
        sample_dir: thư mục val/ hoặc test/ chứa các window .pt.
        baseline: object trả về từ precompute_xgb_oof(train_dir, ...),
            phải đã gọi fit_final() (precompute_xgb_oof tự làm việc này).
    """
    dataset = DMSWindowDataset(sample_dir, require_xgb_oof=False)
    geometry_sequences = [torch.load(p)["geometry"].numpy() for p in dataset.sample_paths]
    X = aggregate_batch(geometry_sequences)
    proba = baseline.predict_proba(X)
    attach_xgb_oof_proba(dataset.sample_paths, proba)


# ---------------------------------------------------------------------------
# Joint training of Stage 2+3+4+5b
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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_residual: bool = True,
) -> Dict[str, float]:
    """Computes accuracy/precision/recall/F1 for the Drowsy class (label=1).

    If use_residual, predictions come from Stage 5's final_score (the
    actual deployed decision); otherwise from Stage 3's window_logits alone
    — useful for comparing "with vs without the residual fallback" during
    ablation.
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

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_drowsy": precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "recall_drowsy": recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "f1_drowsy": f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
    }


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"]


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
    triplet_loss = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history = {"train_loss": [], "val_f1_drowsy": []}
    best_f1 = -1.0

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, drowsiness_loss, triplet_loss, config)
        val_metrics = evaluate(model, val_loader, device, use_residual=config.use_residual)

        history["train_loss"].append(train_metrics["total"])
        history["val_f1_drowsy"].append(val_metrics["f1_drowsy"])

        if verbose:
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"train_loss={train_metrics['total']:.4f} "
                f"(drowsy={train_metrics['drowsiness']:.4f} "
                f"triplet={train_metrics['triplet']:.4f} "
                f"residual={train_metrics['residual']:.4f}) | "
                f"val_acc={val_metrics['accuracy']:.3f} "
                f"val_f1_drowsy={val_metrics['f1_drowsy']:.3f} "
                f"val_recall_drowsy={val_metrics['recall_drowsy']:.3f}"
            )

        if checkpoint_path and val_metrics["f1_drowsy"] > best_f1:
            best_f1 = val_metrics["f1_drowsy"]
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    return history

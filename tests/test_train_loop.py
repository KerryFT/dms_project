"""
Sanity tests for the end-to-end Training Loop (DMSModel + train/evaluate).

Uses synthetic data where the IMAGE patches are pure random noise (no real
signal) but the GEOMETRY (EAR) carries a strong, clean signal by
construction. This deliberately isolates whether the training loop and the
geometry pathway (direct skip-connection + FiLM) are wired correctly,
WITHOUT requiring the CNN branch to learn anything from noise pixels in a
handful of epochs — that would need real eye images and is not what this
test is checking.
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.data.synthetic import generate_synthetic_window_dataset
from src.data.dataset import DMSWindowDataset, collate_windows
from src.models.dms_model import DMSModel
from src.training.train_loop import (
    precompute_xgb_oof, train, evaluate, TrainConfig,
    save_checkpoint, load_checkpoint,
)
from src.training.losses import compute_class_weights_from_counts

TMP_DIR = "/tmp/dms_train_loop_test"


def _build_small_model() -> DMSModel:
    # Kích thước nhỏ để test chạy nhanh trên CPU; production nên dùng default lớn hơn.
    return DMSModel(
        geometry_dim=6, film_hidden_dim=16, gru_hidden_dim=32,
        embed_dim=16, num_classes=2, pretrained_backbone=False,
    )


def test_dms_model_forward_matches_dataloader_batch():
    print("--- DMSModel.forward: khớp đúng với 1 batch thật từ DataLoader ---")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    generate_synthetic_window_dataset(TMP_DIR, n_windows=20, seed=3)
    precompute_xgb_oof(TMP_DIR, n_splits=4)

    dataset = DMSWindowDataset(TMP_DIR, require_xgb_oof=True)
    loader = DataLoader(dataset, batch_size=6, shuffle=False, collate_fn=collate_windows)
    batch = next(iter(loader))

    model = _build_small_model()
    out = model(
        batch["left_patches"], batch["right_patches"], batch["geometry"],
        batch["confidence"], valid_mask=batch["valid_mask"], xgb_proba=batch["xgb_oof_proba"],
    )

    print(f"window_logits: {out['window_logits'].shape}")
    print(f"triplet_embedding norm: {out['triplet_embedding'].norm(dim=-1)[:3].tolist()}")
    print(f"reliability_gate: {out['reliability_gate'].tolist()}")
    print(f"final_score: {out['final_score'].tolist()}")

    assert out["window_logits"].shape == (6, 2)
    assert out["triplet_embedding"].shape == (6, 16)
    assert torch.allclose(out["triplet_embedding"].norm(dim=-1), torch.ones(6), atol=1e-4)
    assert ((out["reliability_gate"] >= 0) & (out["reliability_gate"] <= 1)).all()
    assert ((out["final_score"] >= 0) & (out["final_score"] <= 1)).all()
    assert torch.isfinite(out["window_logits"]).all()
    print("OK: toàn bộ pipeline Stage 2->3->4->5 chạy thông suốt trên 1 batch thật\n")


def test_train_loop_runs_and_loss_is_finite():
    print("--- train(): chạy nhiều epoch, loss luôn finite, history đúng độ dài ---")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    train_paths = generate_synthetic_window_dataset(os.path.join(TMP_DIR, "train"), n_windows=48, seed=4)
    val_paths = generate_synthetic_window_dataset(os.path.join(TMP_DIR, "val"), n_windows=16, seed=5)

    precompute_xgb_oof(os.path.join(TMP_DIR, "train"), n_splits=5)
    precompute_xgb_oof(os.path.join(TMP_DIR, "val"), n_splits=4)

    train_ds = DMSWindowDataset(os.path.join(TMP_DIR, "train"), require_xgb_oof=True)
    val_ds = DMSWindowDataset(os.path.join(TMP_DIR, "val"), require_xgb_oof=True)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_windows)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_windows)

    labels = [torch.load(p)["label"].item() for p in train_paths]
    n_alert, n_drowsy = labels.count(0), labels.count(1)
    class_weights = compute_class_weights_from_counts([max(n_alert, 1), max(n_drowsy, 1)])
    print(f"Train label distribution: Alert={n_alert} Drowsy={n_drowsy} -> class_weights={class_weights.tolist()}")

    model = _build_small_model()
    device = torch.device("cpu")
    ckpt_path = os.path.join(TMP_DIR, "best_model.pt")

    history = train(
        model, train_loader, val_loader, device, num_epochs=6,
        class_weights=class_weights, config=TrainConfig(use_residual=True),
        checkpoint_path=ckpt_path, verbose=True,
    )

    assert len(history["train_loss"]) == 6
    assert len(history["val_f1_drowsy"]) == 6
    assert all(torch.isfinite(torch.tensor(v)) for v in history["train_loss"]), "Loss không được NaN/Inf"
    assert os.path.exists(ckpt_path), "Checkpoint tốt nhất phải được lưu lại"
    print(f"Train loss qua các epoch: {[round(v, 4) for v in history['train_loss']]}")
    print(f"Val F1 (Drowsy) qua các epoch: {[round(v, 4) for v in history['val_f1_drowsy']]}")
    print("OK: train loop chạy ổn định, không NaN/Inf, checkpoint được lưu\n")


def test_checkpoint_save_and_load_roundtrip():
    print("--- save_checkpoint / load_checkpoint: round-trip phải khớp tham số ---")
    model = _build_small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = os.path.join(TMP_DIR, "ckpt_roundtrip.pt")

    original_weight = model.embedding_head.proj[0].weight.clone()
    save_checkpoint(model, optimizer, epoch=3, path=path)

    new_model = _build_small_model()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    loaded_epoch = load_checkpoint(new_model, new_optimizer, path)

    assert loaded_epoch == 3
    assert torch.allclose(new_model.embedding_head.proj[0].weight, original_weight)
    print("OK: load lại đúng epoch và đúng trọng số đã lưu\n")


def test_evaluate_ablation_with_vs_without_residual():
    print("--- evaluate(): so sánh có/không dùng residual fallback (ablation) ---")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    generate_synthetic_window_dataset(TMP_DIR, n_windows=30, seed=6)
    precompute_xgb_oof(TMP_DIR, n_splits=5)
    dataset = DMSWindowDataset(TMP_DIR, require_xgb_oof=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_windows)

    model = _build_small_model()
    metrics_with_residual = evaluate(model, loader, torch.device("cpu"), use_residual=True)
    metrics_without_residual = evaluate(model, loader, torch.device("cpu"), use_residual=False)

    print(f"Có residual:    {metrics_with_residual}")
    print(f"Không residual: {metrics_without_residual}")
    for m in [metrics_with_residual, metrics_without_residual]:
        for v in m.values():
            assert 0.0 <= v <= 1.0
    print("OK: evaluate() hỗ trợ đúng ablation, mọi metric nằm trong [0,1]\n")


if __name__ == "__main__":
    test_dms_model_forward_matches_dataloader_batch()
    test_train_loop_runs_and_loss_is_finite()
    test_checkpoint_save_and_load_roundtrip()
    test_evaluate_ablation_with_vs_without_residual()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("=== TẤT CẢ TEST TRAINING LOOP ĐỀU PASS ===")

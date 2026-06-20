"""
Sanity tests for Stage 5 (XGBoost baseline + tanh-bounded, confidence-gated
residual fallback head).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from src.features.window_features import aggregate_geometry_window, feature_names, PERCLOS_THRESHOLD
from src.training.xgboost_baseline import XGBoostBaseline
from src.models.residual_fallback import ResidualFallbackHead, fuse_final_score, DELTA_MAX


def test_aggregate_geometry_window_correctness():
    print("--- aggregate_geometry_window: kiểm tra bằng tay ---")
    # 4 frame, 6 chiều geometry. Chiều ear_left_norm (idx 3), ear_right_norm (idx 4)
    # cố tình đặt 2/4 frame "nhắm mắt" (< PERCLOS_THRESHOLD) để kiểm PERCLOS.
    seq = np.array([
        [10.0, 0.0, 0.0, 0.1, 0.1, 0.0],   # nhắm mắt
        [10.0, 0.0, 0.0, 0.9, 0.9, 0.0],   # mở mắt
        [20.0, 0.0, 0.0, 0.1, 0.1, 0.0],   # nhắm mắt
        [20.0, 0.0, 0.0, 0.9, 0.9, 0.0],   # mở mắt
    ], dtype=np.float32)
    feats = aggregate_geometry_window(seq)
    names = feature_names()
    assert feats.shape == (25,)
    assert len(names) == 25

    yaw_mean = feats[names.index("yaw_mean")]
    assert abs(yaw_mean - 15.0) < 1e-4, "yaw_mean phải = (10+10+20+20)/4 = 15"

    perclos = feats[names.index("perclos")]
    print(f"PERCLOS tính được: {perclos} (kỳ vọng 0.5, ngưỡng={PERCLOS_THRESHOLD})")
    assert abs(perclos - 0.5) < 1e-4, "2/4 frame nhắm mắt -> PERCLOS phải = 0.5"
    print("OK: mean/PERCLOS khớp tính tay\n")


def test_xgboost_baseline_oof_and_final():
    print("--- XGBoostBaseline: fit_oof (không leak) + fit_final + predict_proba ---")
    rng = np.random.default_rng(0)
    n = 400
    # Feature 0 mang tín hiệu tách lớp rõ ràng, các feature còn lại là noise.
    y = rng.integers(0, 2, size=n)
    signal = y + rng.normal(0, 0.3, size=n)
    noise = rng.normal(0, 1, size=(n, 24))
    X = np.concatenate([signal[:, None], noise], axis=1).astype(np.float32)

    baseline = XGBoostBaseline(n_splits=5)
    oof_proba = baseline.fit_oof(X, y)
    assert oof_proba.shape == (n,)
    assert ((oof_proba >= 0) & (oof_proba <= 1)).all()

    oof_acc = ((oof_proba > 0.5).astype(int) == y).mean()
    print(f"OOF accuracy trên dữ liệu tách lớp rõ: {oof_acc:.3f} (kỳ vọng > 0.85)")
    assert oof_acc > 0.85

    X_train, X_test = X[:300], X[300:]
    y_train, y_test = y[:300], y[300:]
    baseline.fit_final(X_train, y_train)
    test_proba = baseline.predict_proba(X_test)
    test_acc = ((test_proba > 0.5).astype(int) == y_test).mean()
    print(f"Test accuracy (model fit_final, chưa từng thấy test set): {test_acc:.3f}")
    assert test_acc > 0.8

    # predict_proba phải báo lỗi rõ ràng nếu gọi trước fit_final.
    fresh = XGBoostBaseline()
    try:
        fresh.predict_proba(X_test)
        assert False, "Phải raise lỗi khi predict_proba() trước khi fit_final()"
    except RuntimeError:
        pass
    print("OK: OOF không leak (vẫn accuracy cao trên dữ liệu chưa thấy), predict_proba bảo vệ đúng thứ tự gọi\n")


def test_residual_head_bounded_even_with_extreme_input():
    print("--- ResidualFallbackHead: |delta_s| luôn <= 0.15 dù input cực lớn ---")
    torch.manual_seed(0)
    head = ResidualFallbackHead(input_dim=8, hidden_dim=16)
    extreme_features = torch.randn(50, 8) * 1000.0  # input cực lớn, cố tình phá
    reliability_gate = torch.ones(50)
    delta = head(extreme_features, reliability_gate)
    max_abs = delta.abs().max().item()
    print(f"max |delta_s| với input cực lớn: {max_abs:.4f} (kỳ vọng <= {DELTA_MAX})")
    assert max_abs <= DELTA_MAX + 1e-5
    print("OK: tanh chặn delta đúng biên ±0.15 bất kể input lớn cỡ nào\n")


def test_residual_head_zero_when_fully_unreliable():
    print("--- ResidualFallbackHead: reliability_gate=0 -> delta_s PHẢI = 0 (structural, không phải học) ---")
    torch.manual_seed(1)
    head = ResidualFallbackHead(input_dim=8, hidden_dim=16)
    features = torch.randn(20, 8) * 5.0
    reliability_gate_zero = torch.zeros(20)  # toàn bộ frame trong window đều bị mask
    delta = head(features, reliability_gate_zero)
    print(f"delta_s khi reliability_gate=0: max|delta|={delta.abs().max().item():.8f}")
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7), \
        "delta_s phải CHÍNH XÁC bằng 0 khi reliability_gate=0, bất kể MLP học gì"
    print("OK: 'CNN bị chói đèn -> ΔS=0' là guarantee cấu trúc, không phụ thuộc việc model học đúng hay không\n")


def test_fuse_final_score_clips_to_valid_probability():
    print("--- fuse_final_score: phải clip về [0,1] ---")
    xgb_proba = torch.tensor([0.95, 0.05, 0.5])
    delta_s = torch.tensor([0.15, -0.15, 0.10])
    fused = fuse_final_score(xgb_proba, delta_s)
    print(f"raw sum: {(xgb_proba + delta_s).tolist()} -> fused (clipped): {fused.tolist()}")
    assert torch.allclose(fused, torch.tensor([1.0, 0.0, 0.6]), atol=1e-5)
    print("OK: 0.95+0.15=1.10 bị clip về 1.0, 0.05-0.15=-0.10 bị clip về 0.0\n")


def test_residual_training_rescues_borderline_xgb_errors():
    print("--- Pipeline residual: train ResidualFallbackHead để sửa các case XGBoost sai BIÊN ---")
    print("Lưu ý quan trọng: với bound ±0.15, residual head CHỈ có thể sửa được các")
    print("case XGBoost sai GẦN ngưỡng 0.5 (borderline), KHÔNG thể lật được case XGBoost")
    print("tự tin sai nặng (ví dụ proba=0.9 cho nhãn thật=0) — đây là đánh đổi chủ đích")
    print("giữa an toàn (bound rủi ro) và khả năng sửa lỗi, không phải hạn chế của code.\n")

    torch.manual_seed(0)
    n_total = 300
    y = torch.randint(0, 2, (n_total,)).float()
    confusable_mask = torch.rand(n_total) < 0.2  # ~20% case XGBoost sai biên

    # XGBoost OOF proba: case bình thường tự tin ĐÚNG; case confusable sai NHẸ qua biên 0.5.
    oof_proba = torch.where(
        confusable_mask,
        torch.where(y == 1, torch.full_like(y, 0.42), torch.full_like(y, 0.58)),
        torch.where(y == 1, torch.full_like(y, 0.9), torch.full_like(y, 0.1)),
    )

    # DL feature: tín hiệu mạnh + đáng tin CHỈ hữu ích cho case confusable; case
    # bình thường chỉ có noise (mô phỏng: CNN không có thông tin gì thêm ở đó).
    dl_features = torch.randn(n_total, 8) * 0.3
    strong_signal = (2 * y - 1) + torch.randn(n_total) * 0.05
    dl_features[:, 0] = torch.where(confusable_mask, strong_signal, torch.randn(n_total) * 0.5)
    reliability_gate = torch.ones(n_total)

    head = ResidualFallbackHead(input_dim=8, hidden_dim=16)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)

    for _ in range(200):
        optimizer.zero_grad()
        delta = head(dl_features, reliability_gate)
        fused = fuse_final_score(oof_proba, delta).clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(fused, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_delta = head(dl_features, reliability_gate)
        final_fused = fuse_final_score(oof_proba, final_delta)

    acc_xgb_only = ((oof_proba > 0.5).float() == y).float().mean().item()
    acc_fused = ((final_fused > 0.5).float() == y).float().mean().item()
    confusable_acc_xgb = ((oof_proba[confusable_mask] > 0.5).float() == y[confusable_mask]).float().mean().item()
    confusable_acc_fused = ((final_fused[confusable_mask] > 0.5).float() == y[confusable_mask]).float().mean().item()

    print(f"Accuracy toàn bộ — XGBoost-only: {acc_xgb_only:.3f} | Sau residual: {acc_fused:.3f}")
    print(f"Accuracy riêng subset confusable — XGBoost-only: {confusable_acc_xgb:.3f} | "
          f"Sau residual: {confusable_acc_fused:.3f}")

    assert acc_fused > acc_xgb_only, "Residual head phải cải thiện accuracy tổng thể trên dữ liệu test này"
    assert confusable_acc_fused > confusable_acc_xgb, "Cải thiện phải đến từ việc sửa đúng subset borderline"
    print("OK: residual head học đúng việc 'chỉ sửa khi cần, im lặng khi XGBoost đã tự tin đúng'\n")


if __name__ == "__main__":
    test_aggregate_geometry_window_correctness()
    test_xgboost_baseline_oof_and_final()
    test_residual_head_bounded_even_with_extreme_input()
    test_residual_head_zero_when_fully_unreliable()
    test_fuse_final_score_clips_to_valid_probability()
    test_residual_training_rescues_borderline_xgb_errors()
    print("=== TẤT CẢ TEST STAGE 5 ĐỀU PASS ===")

"""
Sanity tests for Stage 3 (Gated GRU + Temporal Attention + class-weighted loss).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.models.temporal import ConfidenceGate, TemporalAttention, GatedTemporalModel
from src.training.losses import DrowsinessLoss, compute_class_weights_from_counts, CLASS_ALERT, CLASS_DROWSY


def test_confidence_gate_replaces_low_confidence_frames():
    print("--- ConfidenceGate: thay frame confidence thấp bằng [MASK] token ---")
    torch.manual_seed(0)
    gate = ConfidenceGate(embed_dim=8, confidence_threshold=0.4)
    emb = torch.randn(2, 5, 8)
    # frame 2 ở batch 0 và frame 0,4 ở batch 1 có confidence thấp.
    confidence = torch.tensor([
        [0.9, 0.8, 0.1, 0.7, 0.6],
        [0.2, 0.9, 0.9, 0.9, 0.3],
    ])
    gated, mask = gate(emb, confidence)

    expected_mask = confidence < 0.4
    assert torch.equal(mask, expected_mask)

    for b in range(2):
        for t in range(5):
            if expected_mask[b, t]:
                assert torch.allclose(gated[b, t], gate.mask_token), \
                    f"Frame ({b},{t}) confidence thấp phải bị thay bằng mask_token"
            else:
                assert torch.allclose(gated[b, t], emb[b, t]), \
                    f"Frame ({b},{t}) confidence cao phải giữ nguyên embedding gốc"
    print("OK: đúng frame bị mask, đúng frame giữ nguyên\n")


def test_mask_token_receives_gradient():
    print("--- ConfidenceGate: mask_token phải học được (nhận gradient) ---")
    gate = ConfidenceGate(embed_dim=8, confidence_threshold=0.4)
    emb = torch.randn(2, 5, 8)
    confidence = torch.tensor([[0.9, 0.1, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9]])
    gated, _ = gate(emb, confidence)
    gated.sum().backward()
    assert gate.mask_token.grad is not None and torch.any(gate.mask_token.grad != 0)
    print("OK: mask_token nhận gradient khi có ít nhất 1 frame bị mask\n")


def test_temporal_attention_weights_sum_to_one():
    print("--- TemporalAttention: trọng số attention phải sum = 1 theo thời gian ---")
    torch.manual_seed(0)
    attn = TemporalAttention(hidden_dim=16, attn_dim=8)
    hidden_states = torch.randn(3, 10, 16)
    context, weights = attn(hidden_states)
    assert context.shape == (3, 16)
    assert weights.shape == (3, 10)
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)
    print(f"OK: attention weights sum theo batch = {sums.tolist()}\n")


def test_temporal_attention_respects_valid_mask():
    print("--- TemporalAttention: valid_mask (padding) phải bị loại khỏi attention ---")
    torch.manual_seed(0)
    attn = TemporalAttention(hidden_dim=16, attn_dim=8)
    hidden_states = torch.randn(1, 6, 16)
    valid_mask = torch.tensor([[True, True, True, False, False, False]])
    _, weights = attn(hidden_states, valid_mask)
    padded_weight_mass = weights[0, 3:].sum().item()
    print(f"Tổng trọng số dồn vào phần padding: {padded_weight_mass:.6f}")
    assert padded_weight_mass < 1e-5, "Padding (valid_mask=False) phải gần như không nhận trọng số attention"
    print("OK: attention bỏ qua đúng phần padding\n")


def test_gated_temporal_model_end_to_end():
    print("--- GatedTemporalModel: forward pass end-to-end ---")
    torch.manual_seed(0)
    B, T, D, num_classes = 4, 20, 64, 2
    model = GatedTemporalModel(input_dim=D, hidden_dim=32, num_classes=num_classes)
    embeddings = torch.randn(B, T, D)
    confidence = torch.rand(B, T)

    out = model(embeddings, confidence)
    print(f"window_logits: {out['window_logits'].shape} (expect ({B}, {num_classes}))")
    print(f"frame_logits:  {out['frame_logits'].shape} (expect ({B}, {T}, {num_classes}))")
    print(f"attn_weights:  {out['attn_weights'].shape}, sum per-sample: "
          f"{out['attn_weights'].sum(dim=1).tolist()}")
    assert out["window_logits"].shape == (B, num_classes)
    assert out["frame_logits"].shape == (B, T, num_classes)
    assert out["attn_weights"].shape == (B, T)
    assert out["gate_mask"].shape == (B, T)
    assert torch.isfinite(out["window_logits"]).all()
    print("OK: shapes đúng, không có NaN/Inf\n")


def test_long_occlusion_2_seconds_does_not_break():
    print("--- GatedTemporalModel: chịu được occlusion dài ~2 giây liên tục ---")
    torch.manual_seed(0)
    fps = 10
    occlusion_seconds = 2.0
    occlusion_frames = int(fps * occlusion_seconds)
    T = occlusion_frames + 20  # vài giây trước/sau đoạn bị che

    model = GatedTemporalModel(input_dim=32, hidden_dim=16, num_classes=2, confidence_threshold=0.4)
    embeddings = torch.randn(1, T, 32)
    confidence = torch.ones(1, T)
    occlusion_start = 10
    confidence[0, occlusion_start: occlusion_start + occlusion_frames] = 0.05  # mất tracking 2s

    out = model(embeddings, confidence)
    n_masked = out["gate_mask"].sum().item()
    print(f"Số frame bị mask: {n_masked} (kỳ vọng {occlusion_frames})")
    assert n_masked == occlusion_frames
    assert torch.isfinite(out["window_logits"]).all() and torch.isfinite(out["frame_logits"]).all()
    print("OK: model vẫn chạy ổn định qua occlusion dài, không NaN/crash.")
    print("    Lưu ý: test này CHỈ xác nhận tính ổn định kỹ thuật, KHÔNG xác nhận")
    print("    độ chính xác dự đoán trong lúc occlusion — điều đó phải đánh giá")
    print("    bằng dữ liệu thật sau khi train.\n")


def test_drowsiness_loss_weighting_ratio():
    print("--- DrowsinessLoss: trọng số lớp phải tỉ lệ đúng theo class_weights ---")
    weights = torch.tensor([1.0, 5.0])  # Alert=1.0, Drowsy=5.0
    loss_fn = DrowsinessLoss(class_weights=weights, focal_gamma=0.0)

    # Hai case ĐỐI XỨNG về độ "sai" (cùng magnitude lệch logit, cùng kiểu sai),
    # chỉ khác nhãn thật -> base (unweighted) CE phải bằng nhau, chỉ trọng số khác.
    logits_predicts_drowsy = torch.tensor([[-2.0, 2.0]])  # model đoán Drowsy
    logits_predicts_alert = torch.tensor([[2.0, -2.0]])   # model đoán Alert

    # Case A: nhãn thật Alert, nhưng model đoán sai thành Drowsy.
    loss_true_alert_wrong = loss_fn(logits_predicts_drowsy, torch.tensor([CLASS_ALERT]))
    # Case B: nhãn thật Drowsy, nhưng model đoán sai thành Alert (đối xứng với case A).
    loss_true_drowsy_wrong = loss_fn(logits_predicts_alert, torch.tensor([CLASS_DROWSY]))

    ratio = (loss_true_drowsy_wrong / loss_true_alert_wrong).item()
    print(f"Loss khi nhãn thật=Alert, đoán sai thành Drowsy: {loss_true_alert_wrong.item():.4f}")
    print(f"Loss khi nhãn thật=Drowsy, đoán sai thành Alert (đối xứng): {loss_true_drowsy_wrong.item():.4f}")
    print(f"Tỉ lệ: {ratio:.3f} (kỳ vọng ~5.0)")
    assert abs(ratio - 5.0) < 1e-3
    print("OK: với cùng mức độ sai, model bị phạt nặng hơn đúng 5x khi nhãn thật là Drowsy\n")


def test_focal_loss_suppresses_easy_examples_more():
    print("--- DrowsinessLoss (focal): mẫu dễ (đã đúng, tự tin) phải bị giảm trọng số ---")
    ce_loss = DrowsinessLoss(class_weights=None, focal_gamma=0.0)
    focal_loss = DrowsinessLoss(class_weights=None, focal_gamma=2.0)

    easy_logits = torch.tensor([[5.0, -5.0]])  # rất tự tin và ĐÚNG
    hard_logits = torch.tensor([[0.1, -0.1]])  # gần như random, SAI lệch nhẹ
    target = torch.tensor([CLASS_ALERT])

    ce_easy, ce_hard = ce_loss(easy_logits, target), ce_loss(hard_logits, target)
    focal_easy, focal_hard = focal_loss(easy_logits, target), focal_loss(hard_logits, target)

    ce_ratio = (ce_hard / ce_easy).item()
    focal_ratio = (focal_hard / focal_easy).item()
    print(f"CE: easy={ce_easy.item():.5f} hard={ce_hard.item():.5f} ratio={ce_ratio:.2f}")
    print(f"Focal: easy={focal_easy.item():.5f} hard={focal_hard.item():.5f} ratio={focal_ratio:.2f}")
    assert focal_ratio > ce_ratio, "Focal loss phải khuếch đại khoảng cách easy-vs-hard nhiều hơn CE thường"
    print("OK: focal loss đè trọng số mẫu dễ xuống mạnh hơn CE thường, đúng kỳ vọng\n")


def test_compute_class_weights_from_counts():
    print("--- compute_class_weights_from_counts ---")
    weights = compute_class_weights_from_counts([8000, 2000])
    ratio = (weights[CLASS_DROWSY] / weights[CLASS_ALERT]).item()
    print(f"counts=[8000 Alert, 2000 Drowsy] -> weights={weights.tolist()}, ratio={ratio:.2f}")
    assert abs(ratio - 4.0) < 1e-4
    print("OK: lớp hiếm hơn 4x được trọng số cao hơn đúng 4x\n")


if __name__ == "__main__":
    test_confidence_gate_replaces_low_confidence_frames()
    test_mask_token_receives_gradient()
    test_temporal_attention_weights_sum_to_one()
    test_temporal_attention_respects_valid_mask()
    test_gated_temporal_model_end_to_end()
    test_long_occlusion_2_seconds_does_not_break()
    test_drowsiness_loss_weighting_ratio()
    test_focal_loss_suppresses_easy_examples_more()
    test_compute_class_weights_from_counts()
    print("=== TẤT CẢ TEST STAGE 3 ĐỀU PASS ===")

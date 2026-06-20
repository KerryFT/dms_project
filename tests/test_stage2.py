"""
Sanity tests for Stage 2 (FiLM-modulated MobileNetV3-Small).

Checks:
    1. Forward pass shapes are correct for both single-frame and sequence APIs.
    2. Gradients flow back into the FiLM generator (i.e. it's actually
       trainable, not a dead branch).
    3. Identity-init: at initialization, FiLM should be near-identity, so
       changing the geometry vector should barely change the output BEFORE
       training — but it should change MORE after a few optimizer steps
       (proves the generator is actually learning to use geometry, not
       just passing gradients into a no-op).
    4. Different geometry vectors -> different outputs once gamma/beta are
       no longer exactly identity (sanity that FiLM is wired correctly,
       not silently bypassed).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.models.backbone import FiLMEyeEncoder
from src.models.film import FiLMGenerator, FiLMLayer


def test_film_generator_identity_init():
    print("--- FiLMGenerator: identity init ---")
    gen = FiLMGenerator(geometry_dim=6, hidden_dim=16, channel_dims=[24, 40, 96])
    geom = torch.randn(4, 6)
    outputs = gen(geom)
    for (gamma, beta), c in zip(outputs, [24, 40, 96]):
        assert gamma.shape == (4, c) and beta.shape == (4, c)
        assert torch.allclose(gamma, torch.ones_like(gamma)), "gamma phải = 1 lúc init"
        assert torch.allclose(beta, torch.zeros_like(beta)), "beta phải = 0 lúc init"
    print("OK: gamma=1, beta=0 khi mới init -> FiLM khởi đầu là identity transform\n")


def test_film_layer_application():
    print("--- FiLMLayer: apply gamma/beta lên feature map ---")
    layer = FiLMLayer()
    x = torch.randn(2, 8, 5, 5)
    gamma = torch.full((2, 8), 2.0)
    beta = torch.full((2, 8), 1.0)
    out = layer(x, gamma, beta)
    expected = x * 2.0 + 1.0
    assert torch.allclose(out, expected, atol=1e-5)
    print("OK: out = gamma * x + beta đúng công thức FiLM\n")


def test_eye_encoder_forward_and_grad():
    print("--- FiLMEyeEncoder: forward + gradient flow (single frame) ---")
    torch.manual_seed(0)
    encoder = FiLMEyeEncoder(geometry_dim=6, film_hidden_dim=16, pretrained=False)

    left = torch.randn(3, 3, 64, 64, requires_grad=False)
    right = torch.randn(3, 3, 64, 64, requires_grad=False)
    geom = torch.randn(3, 6, requires_grad=True)

    emb = encoder(left, right, geom)
    print(f"Embedding shape: {emb.shape}  (expect (3, {encoder.output_dim}))")
    assert emb.shape == (3, encoder.output_dim)

    loss = emb.sum()
    loss.backward()

    # Bước 0: vì head zero-init, dL/dh = W_head^T @ dL/dy = 0 -> geom.grad = 0
    # (KHÔNG phải lỗi). Nhưng W_head tự nó vẫn nhận gradient riêng
    # (dL/dW_head = dL/dy ⊗ h ≠ 0) nên vẫn học được ngay từ bước này.
    head0 = encoder.backbone.film_generator.heads[0]
    assert head0.weight.grad is not None and torch.any(head0.weight.grad != 0), \
        "Head layer phải nhận gradient riêng dù input của nó (geom) chưa nhận được"
    assert geom.grad is None or torch.all(geom.grad == 0), \
        "Đúng như dự đoán: ở bước 0, geom.grad PHẢI bằng 0 vì W_head=0 (one-step-delay)"
    print("Bước 0: geom.grad = 0 (đúng dự đoán do zero-init), nhưng head.weight.grad != 0 -> head vẫn học được")

    # Cập nhật 1 bước để W_head thoát khỏi 0, sau đó gradient mới thực sự
    # chảy xuyên qua được tới geometry vector.
    with torch.no_grad():
        for p in [head0.weight, head0.bias]:
            p += 0.01 * torch.randn_like(p)

    geom2 = torch.randn(3, 6, requires_grad=True)
    emb2 = encoder(left, right, geom2)
    emb2.sum().backward()
    assert geom2.grad is not None and torch.any(geom2.grad != 0), \
        "Sau khi W_head thoát khỏi 0, gradient phải chảy được tới geometry vector"
    print("Sau 1 bước cập nhật: geom.grad != 0 -> mạch gradient đã thông từ Stage 1 tới Stage 2\n")


def test_sequence_api():
    print("--- FiLMEyeEncoder: encode_sequence (B, T, ...) ---")
    torch.manual_seed(0)
    encoder = FiLMEyeEncoder(geometry_dim=6, film_hidden_dim=16, pretrained=False)
    B, T = 2, 5
    left_seq = torch.randn(B, T, 3, 64, 64)
    right_seq = torch.randn(B, T, 3, 64, 64)
    geom_seq = torch.randn(B, T, 6)

    emb_seq = encoder.encode_sequence(left_seq, right_seq, geom_seq)
    print(f"Sequence embedding shape: {emb_seq.shape}  (expect ({B}, {T}, {encoder.output_dim}))")
    assert emb_seq.shape == (B, T, encoder.output_dim)
    print("OK: encode_sequence trả đúng shape (B, T, embed_dim) để đưa vào GRU ở Stage 3\n")


def test_geometry_actually_matters_after_training():
    print("--- FiLM thực sự học, không phải nhánh chết ---")
    torch.manual_seed(0)
    encoder = FiLMEyeEncoder(geometry_dim=6, film_hidden_dim=16, pretrained=False)
    left = torch.randn(4, 3, 64, 64)
    right = torch.randn(4, 3, 64, 64)
    geom_a = torch.zeros(4, 6)
    geom_b = torch.tensor([[45.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)  # large yaw

    with torch.no_grad():
        out_a_before = encoder(left, right, geom_a)
        out_b_before = encoder(left, right, geom_b)
    diff_before = (out_a_before - out_b_before).abs().mean().item()
    print(f"Chênh lệch output trước train (gamma=1,beta=0 cho mọi geometry): {diff_before:.6f}")
    assert diff_before < 1e-4, "Lúc init, FiLM gần như identity nên 2 geometry khác nhau không nên đổi output"

    # Vài bước gradient descent ngẫu nhiên để phá vỡ identity init.
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    for _ in range(20):
        optimizer.zero_grad()
        out = encoder(left, right, geom_b)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out_a_after = encoder(left, right, geom_a)
        out_b_after = encoder(left, right, geom_b)
    diff_after = (out_a_after - out_b_after).abs().mean().item()
    print(f"Chênh lệch output sau 20 bước train: {diff_after:.6f}")
    assert diff_after > diff_before, "Sau khi train, FiLM phải tạo ra khác biệt rõ hơn giữa 2 geometry khác nhau"
    print("OK: FiLM bắt đầu ở gần-identity, và thực sự học để phân biệt geometry sau khi train\n")


if __name__ == "__main__":
    test_film_generator_identity_init()
    test_film_layer_application()
    test_eye_encoder_forward_and_grad()
    test_sequence_api()
    test_geometry_actually_matters_after_training()
    print("=== TẤT CẢ TEST STAGE 2 ĐỀU PASS ===")

"""
Sanity tests for Stage 4 (EmbeddingHead + semi-hard TripletLoss).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.models.embedding import EmbeddingHead
from src.training.triplet import pairwise_squared_distances, semi_hard_triplet_indices, TripletLoss
from src.training.losses import CLASS_ALERT, CLASS_DROWSY


def test_embedding_head_l2_normalized():
    print("--- EmbeddingHead: output phải nằm trên unit hypersphere ---")
    head = EmbeddingHead(input_dim=32, embed_dim=16)
    context = torch.randn(5, 32)
    emb = head(context)
    norms = emb.norm(p=2, dim=-1)
    print(f"Norms: {norms.tolist()}")
    assert emb.shape == (5, 16)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)
    print("OK: mọi embedding đều có norm = 1\n")


def test_pairwise_squared_distances_correctness():
    print("--- pairwise_squared_distances: kiểm tra bằng tay ---")
    embeddings = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    dist = pairwise_squared_distances(embeddings)
    print(dist)
    assert torch.isclose(dist[0, 1], torch.tensor(25.0), atol=1e-4)
    assert torch.isclose(dist[0, 2], torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(dist[1, 2], torch.tensor(25.0), atol=1e-4)
    print("OK: khoảng cách khớp tính tay (3,4 cách gốc 5 -> bình phương = 25)\n")


def test_semi_hard_mining_picks_correct_negative():
    print("--- semi_hard_triplet_indices: phải chọn đúng 'semi-hard', không phải closest tuyệt đối ---")
    # index 0 = Anchor (Drowsy), 1 = Positive (Drowsy), 2/3/4 = negatives (Alert)
    embeddings = torch.tensor([
        [10.0, 0.0],  # 0: anchor (Drowsy)
        [10.0, 3.0],  # 1: positive (Drowsy)   -> d_ap = 9
        [10.0, 1.0],  # 2: negative "quá khó"  -> d=1   (gần anchor HƠN positive -> phải bị loại)
        [10.0, 4.0],  # 3: negative semi-hard  -> d=16  (xa hơn positive, nhỏ nhất trong số đó)
        [10.0, 5.0],  # 4: negative dễ hơn     -> d=25
    ])
    labels = torch.tensor([CLASS_DROWSY, CLASS_DROWSY, CLASS_ALERT, CLASS_ALERT, CLASS_ALERT])
    dist = pairwise_squared_distances(embeddings)
    triplets = semi_hard_triplet_indices(dist, labels)

    chosen = [t for t in triplets if t[0] == 0 and t[1] == 1]
    assert len(chosen) == 1, "Phải có đúng 1 triplet (anchor=0, positive=1)"
    _, _, neg_idx = chosen[0]
    print(f"Negative được chọn cho (anchor=0, positive=1): index {neg_idx} (kỳ vọng index 3, d=16)")
    assert neg_idx == 3, "Phải chọn negative SEMI-HARD (d=16), không phải closest tuyệt đối (index 2, d=1)"
    print("OK: mining đúng — loại bỏ negative quá gần (outlier), chọn semi-hard thật\n")


def test_degenerate_single_class_batch_returns_zero():
    print("--- TripletLoss: batch chỉ có 1 lớp -> phải trả về 0, không crash ---")
    loss_fn = TripletLoss(margin=0.3)
    embeddings = torch.randn(5, 16, requires_grad=True)
    labels = torch.full((5,), CLASS_DROWSY)  # toàn bộ đều là Drowsy, không có Alert
    loss = loss_fn(embeddings, labels)
    print(f"Loss: {loss.item()}")
    assert loss.item() == 0.0
    loss.backward()  # không được raise lỗi (graph vẫn phải connected)
    print("OK: không có Negative hợp lệ -> loss = 0, backward() vẫn an toàn\n")


def test_triplet_loss_value_matches_formula():
    print("--- TripletLoss: giá trị loss phải khớp công thức margin ---")
    loss_fn = TripletLoss(margin=0.3)

    # Lưu ý: với 2 điểm Drowsy, mining tạo triplet theo CẢ 2 CHIỀU (điểm nào
    # cũng có thể làm anchor), nên loss cuối là TRUNG BÌNH của 2 chiều đó —
    # không chỉ tính 1 chiều. Tính tay đầy đủ cả 2 chiều dưới đây.

    # Case A: negative đã đủ xa cả 2 phía.
    embeddings = torch.tensor([
        [10.0, 0.0], [10.0, 3.0], [10.0, 4.0],
    ], requires_grad=True)
    labels = torch.tensor([CLASS_DROWSY, CLASS_DROWSY, CLASS_ALERT])
    # Chiều (anchor=0,pos=1): d_ap=9, d_an=dist(0,2)=16 -> relu(9-16+0.3)=0
    # Chiều (anchor=1,pos=0): d_ap=9, d_an=dist(1,2)=1  -> relu(9-1+0.3)=8.3 (negative gần hơn d_ap -> fallback closest)
    expected = (max(0.0, 9 - 16 + 0.3) + max(0.0, 9 - 1 + 0.3)) / 2
    loss = loss_fn(embeddings, labels)
    print(f"Loss case A: {loss.item():.4f} (kỳ vọng {expected:.4f})")
    assert abs(loss.item() - expected) < 1e-3

    # Case B: negative kéo gần lại hơn nữa ở cả 2 phía -> loss phải TĂNG so với case A.
    embeddings2 = torch.tensor([
        [10.0, 0.0], [10.0, 3.0], [10.0, 1.0],
    ], requires_grad=True)
    labels2 = torch.tensor([CLASS_DROWSY, CLASS_DROWSY, CLASS_ALERT])
    # Chiều (anchor=0,pos=1): d_ap=9, d_an=dist(0,2)=1 -> relu(9-1+0.3)=8.3
    # Chiều (anchor=1,pos=0): d_ap=9, d_an=dist(1,2)=4 -> relu(9-4+0.3)=5.3
    expected2 = (max(0.0, 9 - 1 + 0.3) + max(0.0, 9 - 4 + 0.3)) / 2
    loss2 = loss_fn(embeddings2, labels2)
    print(f"Loss case B (negative gần hơn): {loss2.item():.4f} (kỳ vọng {expected2:.4f})")
    assert abs(loss2.item() - expected2) < 1e-3
    assert loss2.item() > loss.item(), "Negative càng gần Anchor/Positive thì loss phải càng lớn"
    print("OK: loss khớp đúng công thức relu(d_ap - d_an + margin), tính đúng cả 2 chiều anchor\n")


def test_triplet_loss_decreases_with_optimization():
    print("--- TripletLoss: phải tối ưu được (loss giảm theo training) ---")
    torch.manual_seed(0)
    raw = torch.nn.Parameter(torch.randn(20, 8) * 0.1)  # 2 cụm khởi đầu trộn lẫn
    labels = torch.tensor([CLASS_DROWSY] * 10 + [CLASS_ALERT] * 10)
    loss_fn = TripletLoss(margin=0.5)
    optimizer = torch.optim.Adam([raw], lr=0.05)

    losses = []
    for step in range(60):
        optimizer.zero_grad()
        emb = torch.nn.functional.normalize(raw, p=2, dim=-1)
        loss = loss_fn(emb, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Loss đầu: {losses[0]:.4f} -> Loss cuối: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss phải giảm sau quá trình tối ưu"
    print("OK: loss giảm dần, 2 cụm Drowsy/Alert thực sự tách xa nhau hơn trong embedding space\n")


if __name__ == "__main__":
    test_embedding_head_l2_normalized()
    test_pairwise_squared_distances_correctness()
    test_semi_hard_mining_picks_correct_negative()
    test_degenerate_single_class_batch_returns_zero()
    test_triplet_loss_value_matches_formula()
    test_triplet_loss_decreases_with_optimization()
    print("=== TẤT CẢ TEST STAGE 4 ĐỀU PASS ===")

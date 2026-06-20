"""
Stage 4 (part 1) — Embedding head.

Projects the Stage 3 pooled context vector (output of TemporalAttention)
into a lower-dimensional embedding space, L2-normalized onto the unit
hypersphere. Normalizing is important for triplet loss: without it, the
network can "cheat" by simply scaling all embeddings up, which inflates
every pairwise distance and can trivially satisfy the margin without
learning any real structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, embed_dim),
        )

    def forward(self, context_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_vector: (B, input_dim) — e.g. Stage 3's attention context.
        Returns:
            (B, embed_dim) L2-normalized embedding.
        """
        emb = self.proj(context_vector)
        return F.normalize(emb, p=2, dim=-1)

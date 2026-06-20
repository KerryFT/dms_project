"""
Stage 2 — Feature-wise Linear Modulation (FiLM).

Lets the geometry branch ("the math") tell the CNN branch ("the eyes") what
to expect, so e.g. a large yaw angle can explain away an EAR drop that would
otherwise look like a false drowsy alarm.

Design notes (read before changing channel_dims):
    - FiLMGenerator is ONE shared 2-layer MLP trunk + a separate linear head
      per injection point. This keeps the "2-layer MLP" spec while sharing
      the geometry representation across all injection sites (more
      parameter-efficient than one independent MLP per site, and arguably
      more coherent: the same head-pose context should inform every scale).
    - Heads are zero-initialized so that, at the start of training,
      gamma = 1 + 0 = 1 and beta = 0, i.e. FiLM starts as an identity
      transform. Without this, randomly-initialized gamma/beta would
      immediately corrupt the (often ImageNet-pretrained) CNN features and
      destabilize early training — a classic conditioning-layer pitfall.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """Shared trunk + per-injection-point heads, mapping geometry -> (gamma, beta) lists."""

    def __init__(self, geometry_dim: int, hidden_dim: int, channel_dims: Sequence[int]):
        super().__init__()
        self.channel_dims = list(channel_dims)
        self.trunk = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 2 * c) for c in self.channel_dims]
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, geometry_vector: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            geometry_vector: (B, geometry_dim) — output of Stage 1's
                              GeometryFeatureExtractor.as_vector(), batched.
        Returns:
            List (one per injection point) of (gamma, beta), each (B, C_i).
            gamma is already offset by +1 (identity-friendly), beta is raw.
        """
        h = self.trunk(geometry_vector)
        outputs = []
        for head, c in zip(self.heads, self.channel_dims):
            gamma_beta = head(h)
            gamma_raw, beta = gamma_beta[:, :c], gamma_beta[:, c:]
            gamma = 1.0 + gamma_raw
            outputs.append((gamma, beta))
        return outputs


class FiLMLayer(nn.Module):
    """Applies a pre-computed (gamma, beta) pair to a 4D feature map.

    out[b, c, h, w] = gamma[b, c] * x[b, c, h, w] + beta[b, c]
    """

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta

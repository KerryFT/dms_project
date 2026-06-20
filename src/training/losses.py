"""
Class-weighted (and optionally focal) loss for the Drowsy/Alert imbalance.

This is intentionally a separate module from src/models/temporal.py's
TemporalAttention — despite the spec's "Class-Weight Attention" naming, this
has nothing to do with attention. It is a loss-reweighting technique that
penalizes missing a Drowsy frame harder than missing an Alert frame, which
directly trades some precision for higher recall on the class that matters
most for safety.

Unifies vanilla class-weighted cross-entropy (gamma=0) and focal loss
(gamma>0) under one formula so you can switch between them with one flag:

    L_i = alpha_{y_i} * (1 - p_{y_i})^gamma * (-log p_{y_i})

gamma=0 reduces this exactly to standard weighted cross-entropy.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_ALERT = 0
CLASS_DROWSY = 1


def compute_class_weights_from_counts(counts: Sequence[int]) -> torch.Tensor:
    """Standard inverse-frequency class weighting: w_c = N / (num_classes * count_c).

    Example: counts=[8000, 2000] (8000 Alert frames, 2000 Drowsy frames) gives
    weights that make the rarer Drowsy class count ~4x more per mistake.
    """
    counts_t = torch.tensor(counts, dtype=torch.float32)
    n_total = counts_t.sum()
    num_classes = len(counts)
    weights = n_total / (num_classes * counts_t)
    return weights


class DrowsinessLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, focal_gamma: float = 0.0):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach())
        else:
            self.class_weights = None
        self.focal_gamma = focal_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) or (B, T, num_classes) raw scores.
            targets: (B,) or (B, T) integer class indices.
        Returns:
            scalar loss (mean over all elements).
        """
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1)

        ce_unweighted = F.cross_entropy(logits, targets, weight=None, reduction="none")  # -log p_t
        pt = torch.exp(-ce_unweighted)

        if self.focal_gamma > 0:
            modulating = (1.0 - pt) ** self.focal_gamma
        else:
            modulating = torch.ones_like(pt)

        if self.class_weights is not None:
            alpha_t = self.class_weights.to(logits.device)[targets]
        else:
            alpha_t = torch.ones_like(pt)

        per_sample_loss = alpha_t * modulating * ce_unweighted
        return per_sample_loss.mean()

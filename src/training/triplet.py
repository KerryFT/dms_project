"""
Stage 4 (part 2) — Contrastive training via Triplet Loss.

IMPORTANT design decision (binary-label dataset):
The original spec's triplet was (Anchor: Low Vigilant, Positive: Drowsy,
Negative: Alert) — but Low Vigilant is not an annotated class in the
current dataset (only Alert/Drowsy exist). Anchor and Positive must be the
SAME class for a triplet to be well-posed; using two different classes as
anchor/positive isn't standard triplet construction.

Resolution used here: Anchor = a Drowsy sample, Positive = a DIFFERENT
Drowsy sample, Negative = an Alert sample. This still achieves the stated
goal — "tired" embeddings cluster together, far from "alert" — using the
labels that actually exist, with no unvalidated pseudo-labeling.

Future extension (not implemented here, needs validation first): if you
later want the ordinal "Low Vigilant" nuance, a continuous proxy (e.g. a
PERCLOS-derived score, or Stage 5's XGBoost probability) could define
"borderline" anchors for harder triplet mining — but that proxy would
itself need to be validated against human-rated samples before trusting it
as a training signal for a publication.

Mining strategy: online semi-hard negative mining within each batch
(Schroff et al., FaceNet 2015) — for each (anchor, positive) pair, pick the
negative that is farther than the positive but by the SMALLEST margin
(hardest negative that still respects the ordering). This avoids both
trivially-easy negatives (no learning signal) and outlier-driven
hardest-negative collapse, and converges better than random triplet sampling.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def pairwise_squared_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Returns (N, N) matrix of squared L2 distances between all embedding pairs."""
    sq_norms = (embeddings ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    dist = sq_norms + sq_norms.t() - 2.0 * embeddings @ embeddings.t()
    return dist.clamp(min=0.0)  # guard against tiny negative values from float error


def semi_hard_triplet_indices(
    dist_matrix: torch.Tensor, labels: torch.Tensor
) -> List[Tuple[int, int, int]]:
    """For every valid (anchor, positive) same-class pair in the batch, find
    a semi-hard negative. Falls back to the closest negative overall if no
    strictly-semi-hard candidate exists (rather than dropping the pair).

    Returns a list of (anchor_idx, positive_idx, negative_idx) python ints.
    """
    n = labels.shape[0]
    triplets: List[Tuple[int, int, int]] = []

    for i in range(n):
        same_class = (labels == labels[i])
        same_class[i] = False  # exclude self as its own positive
        diff_class = ~same_class
        diff_class[i] = False  # exclude self (redundant, but explicit)

        positive_idxs = same_class.nonzero(as_tuple=True)[0]
        negative_idxs = diff_class.nonzero(as_tuple=True)[0]
        if positive_idxs.numel() == 0 or negative_idxs.numel() == 0:
            continue  # no valid triplet possible for this anchor in this batch

        neg_dists = dist_matrix[i, negative_idxs]  # (num_negatives,)

        for j in positive_idxs.tolist():
            d_ap = dist_matrix[i, j]
            semi_hard_mask = neg_dists > d_ap
            if semi_hard_mask.any():
                candidate_dists = neg_dists.clone()
                candidate_dists[~semi_hard_mask] = float("inf")
                best_local = torch.argmin(candidate_dists)
            else:
                # No negative is farther than this positive: fall back to the
                # single closest negative overall (hardest available signal).
                best_local = torch.argmin(neg_dists)
            k = negative_idxs[best_local].item()
            triplets.append((i, j, k))

    return triplets


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D) L2-normalized embeddings (e.g. EmbeddingHead output).
            labels: (N,) integer class labels (0=Alert, 1=Drowsy).
        Returns:
            scalar loss. Returns 0.0 (with grad) if the batch contains only
            one class (no valid triplets exist) — this can legitimately
            happen with small/unbalanced batches, and should NOT crash training.
        """
        dist_matrix = pairwise_squared_distances(embeddings)
        triplets = semi_hard_triplet_indices(dist_matrix.detach(), labels)

        if len(triplets) == 0:
            return embeddings.sum() * 0.0  # zero, but keeps autograd graph connected

        anchor_idx = torch.tensor([t[0] for t in triplets], device=embeddings.device)
        positive_idx = torch.tensor([t[1] for t in triplets], device=embeddings.device)
        negative_idx = torch.tensor([t[2] for t in triplets], device=embeddings.device)

        d_ap = dist_matrix[anchor_idx, positive_idx]
        d_an = dist_matrix[anchor_idx, negative_idx]
        losses = torch.relu(d_ap - d_an + self.margin)
        return losses.mean()

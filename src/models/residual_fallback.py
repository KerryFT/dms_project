"""
Stage 5 (part 3) — Residual Fallback head: FinalScore = XGBoost + ΔS.

Beyond the spec's tanh(±0.15) bound, this adds ONE more safety mechanism:
ΔS is additionally multiplied by a reliability gate derived from Stage 3's
gate_mask (the fraction of frames in the window that had usable tracking).
This makes "if the CNN is blinded, ΔS -> 0" a STRUCTURAL guarantee rather
than something the network merely has to learn and hopefully generalizes —
if every frame in the window was masked, reliability_gate = 0 and ΔS is
exactly 0 regardless of what the MLP produces, by construction.

IMPORTANT SCOPE NOTE on "never worse than baseline":
The tanh(±0.15) bound limits the WORST-CASE damage any single prediction
can take (XGBoost's score can move by at most 0.15), but it does NOT
mathematically guarantee the aggregate F1-score across a dataset stays
>= the 0.5422 baseline — that depends on whether, on net, the learned
corrections point the right direction more often than the wrong direction.
That has to be verified empirically on a held-out validation set. If
validation shows the residual head hurts F1, the practical deployment
decision is to ship XGBoost-only (or shrink the bound further) rather than
assume the architecture alone guarantees safety.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DELTA_MAX = 0.15


class ResidualFallbackHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, delta_max: float = DELTA_MAX):
        super().__init__()
        self.delta_max = delta_max
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, dl_features: torch.Tensor, reliability_gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dl_features: (B, input_dim) — e.g. Stage 3's pooled context vector,
                         or Stage 4's embedding, or a concat of both.
            reliability_gate: (B,) in [0, 1] — fraction of frames in the window
                         that were NOT confidence-gated (1 - mean(gate_mask)).
                         0 means "every frame was unreliable" -> ΔS forced to 0.
        Returns:
            delta_s: (B,) bounded in [-delta_max, delta_max] * reliability_gate.
        """
        raw = self.mlp(dl_features).squeeze(-1)  # (B,)
        delta_s = self.delta_max * torch.tanh(raw) * reliability_gate
        return delta_s


def fuse_final_score(xgb_proba: torch.Tensor, delta_s: torch.Tensor) -> torch.Tensor:
    """FinalScore = clip(XGBoost(Geometry) + ΔS(DeepLearning), 0, 1).

    Clipping is necessary because XGBoost's probability is already in
    [0, 1] and adding a +/-0.15 delta can push the raw sum slightly outside
    that range (e.g. 0.95 + 0.15 = 1.10), which would no longer be a valid
    probability.
    """
    return torch.clamp(xgb_proba + delta_s, 0.0, 1.0)

"""
Stage 3 — Gated GRU + Temporal Attention.

Handles the messy reality of driving: occlusion, motion blur, or MediaPipe
losing tracking mid-sequence.

This module deliberately separates two mechanisms that the spec bundled
under one name ("Class-Weight Attention"), because they are NOT the same
thing and conflating them would be a real terminology problem in a paper:

    1. ConfidenceGate      -> input-side masking (replace unreliable frames
                               with a learned [MASK] token; the GRU's hidden
                               state carries memory through the gap).
    2. TemporalAttention   -> an actual attention mechanism: a learned,
                               softmax-normalized weighting over the GRU's
                               hidden states across time, producing one
                               pooled context vector per window/sequence.

The class-weighted LOSS (penalize missing "Drowsy" harder) is a separate
concern and lives in src/training/losses.py — it has nothing to do with
attention and shouldn't be named as if it did.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConfidenceGate(nn.Module):
    """Replaces per-frame embeddings with a learned [MASK] token when the
    upstream tracking confidence (e.g. MediaPipe detection/presence score)
    drops below a threshold.

    The mask token is a trainable parameter (not a fixed zero vector) so the
    model can learn a representation for "no reliable input" that's most
    useful to the GRU, rather than forcing it to interpret an arbitrary
    zero vector that might collide with a legitimate feature value.
    """

    def __init__(self, embed_dim: int, confidence_threshold: float = 0.4):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, embeddings: torch.Tensor, confidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, T, D) per-frame fused embeddings (Stage 1+2 output).
            confidence: (B, T) tracking confidence in [0, 1] per frame.
        Returns:
            gated_embeddings: (B, T, D) with low-confidence frames replaced.
            mask: (B, T) bool, True where a frame WAS masked (for diagnostics/logging).
        """
        mask = confidence < self.confidence_threshold  # (B, T)
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)  # (B, T, D)
        token_expanded = self.mask_token.view(1, 1, -1).expand_as(embeddings)
        gated = torch.where(mask_expanded, token_expanded, embeddings)
        return gated, mask


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention pooling over GRU hidden states.

    score_t = v^T * tanh(W * h_t + b)
    alpha   = softmax_t(score_t)
    context = sum_t alpha_t * h_t

    This is a genuine attention mechanism (learned weighting that sums to 1
    across the time axis), distinct from class-weighted loss.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, H) GRU outputs for every timestep.
            valid_mask: optional (B, T) bool, True for timesteps that should
                        be ATTENDABLE. Use this if you want attention to ignore
                        padding frames (e.g. variable-length sequences padded
                        to a common T). Masked-but-gated frames (ConfidenceGate
                        output) are still valid GRU outputs (informed by
                        memory) and should normally stay attendable — only
                        pass valid_mask for real padding, not for confidence
                        gating.
        Returns:
            context: (B, H) pooled representation.
            weights: (B, T) attention weights (sum to 1 along T).
        """
        scores = self.score(torch.tanh(self.proj(hidden_states))).squeeze(-1)  # (B, T)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.einsum("bt,bth->bh", weights, hidden_states)
        return context, weights


class GatedTemporalModel(nn.Module):
    """Stage 3 end-to-end: ConfidenceGate -> GRU -> TemporalAttention -> classifier.

    Also exposes per-frame logits (cheap: one more linear layer on the GRU
    outputs) in case your label granularity is per-frame rather than
    per-window — use whichever output matches your actual annotations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        confidence_threshold: float = 0.4,
        attn_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gate = ConfidenceGate(input_dim, confidence_threshold)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim, attn_dim)
        self.window_classifier = nn.Linear(hidden_dim, num_classes)
        self.frame_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
        confidence: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            embeddings: (B, T, D) fused per-frame embeddings from Stage 1+2.
            confidence: (B, T) per-frame tracking confidence in [0, 1].
            valid_mask: optional (B, T) bool for real sequence padding (not
                        confidence gating — see TemporalAttention docstring).
        Returns dict with:
            window_logits: (B, num_classes) — sequence/window-level prediction.
            frame_logits:  (B, T, num_classes) — per-frame prediction.
            attn_weights:  (B, T) — where the window decision focused.
            gate_mask:     (B, T) bool — which frames were treated as unreliable.
        """
        gated, gate_mask = self.gate(embeddings, confidence)
        hidden_states, _ = self.gru(gated)  # (B, T, hidden_dim)
        context, attn_weights = self.attention(hidden_states, valid_mask)

        return {
            "context": context,                      
            "window_logits": self.window_classifier(context),
            "frame_logits": self.frame_classifier(hidden_states),
            "attn_weights": attn_weights,
            "gate_mask": gate_mask,
        }

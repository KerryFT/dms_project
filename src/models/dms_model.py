"""
End-to-end DMS model: Stage 2 (FiLM eye encoder) -> Stage 3 (gated GRU +
temporal attention) -> Stage 4 (embedding head) -> Stage 5's differentiable
residual head.

Stage 1 (geometry extraction) and Stage 5's XGBoost are NOT inside this
module on purpose:
    - Stage 1 runs on raw frames + MediaPipe landmarks, on CPU, typically
      as an offline preprocessing step (see src/data/preprocessing.py) — by
      the time a batch reaches this model, geometry vectors are already
      plain tensors.
    - XGBoost is not a torch.nn.Module and isn't trained by backprop; it's
      fit once (or refreshed periodically) on aggregated geometry features
      via src/training/xgboost_baseline.py, and its output probability is
      fed into this model's forward() as `xgb_proba` — a frozen constant
      from this model's point of view.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbone import FiLMEyeEncoder
from .temporal import GatedTemporalModel
from .embedding import EmbeddingHead
from .residual_fallback import ResidualFallbackHead, fuse_final_score


class DMSModel(nn.Module):
    def __init__(
        self,
        geometry_dim: int = 6,
        film_hidden_dim: int = 32,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        embed_dim: int = 64,
        num_classes: int = 2,
        confidence_threshold: float = 0.4,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.eye_encoder = FiLMEyeEncoder(
            geometry_dim=geometry_dim,
            film_hidden_dim=film_hidden_dim,
            pretrained=pretrained_backbone,
        )
        # Geometry vector reaches the GRU via TWO complementary paths:
        #   (1) FiLM, which contextually reshapes the CNN's feature maps
        #       (handles pose compensation, as in the original spec).
        #   (2) A direct skip-connection concatenation right here, so the
        #       classifier/GRU isn't solely dependent on the indirect
        #       FiLM-mediated pathway for an already-informative signal —
        #       this is a common, well-justified pattern in multi-modal
        #       fusion (e.g. conditioning features are often both used to
        #       modulate AND concatenated directly) and noticeably speeds
        #       up convergence / stabilizes gradients versus FiLM-only.
        temporal_input_dim = self.eye_encoder.output_dim + geometry_dim
        self.temporal_model = GatedTemporalModel(
            input_dim=temporal_input_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_layers,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
        )
        self.embedding_head = EmbeddingHead(input_dim=gru_hidden_dim, embed_dim=embed_dim)
        self.residual_head = ResidualFallbackHead(input_dim=gru_hidden_dim, hidden_dim=32)

    def forward(
        self,
        left_patches: torch.Tensor,
        right_patches: torch.Tensor,
        geometry_vectors: torch.Tensor,
        confidence: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        xgb_proba: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            left_patches, right_patches: (B, T, 3, H, W) normalized eye crops.
            geometry_vectors: (B, T, geometry_dim) — Stage 1 output.
            confidence: (B, T) per-frame tracking confidence in [0, 1].
            valid_mask: optional (B, T) bool, True for REAL (non-padding)
                        timesteps — only needed for variable-length sequences.
            xgb_proba: optional (B,) frozen XGBoost P(Drowsy) per window
                       (from src/training/xgboost_baseline.py). If provided,
                       the residual fallback score is also computed.
        Returns dict with at least:
            window_logits, frame_logits, attn_weights, gate_mask, context
                (from Stage 3), triplet_embedding (Stage 4),
                reliability_gate (derived from gate_mask).
            If xgb_proba is given, also: delta_s, final_score (Stage 5).
        """
        fused_eye = self.eye_encoder.encode_sequence(
            left_patches, right_patches, geometry_vectors
        )  # (B, T, D_cnn)
        fused_embeddings = torch.cat([fused_eye, geometry_vectors], dim=-1)  # (B, T, D_cnn + geom_dim)

        out = self.temporal_model(fused_embeddings, confidence, valid_mask)
        context = out["context"]  # (B, hidden_dim)

        out["triplet_embedding"] = self.embedding_head(context)

        # Reliability = fraction of frames in the window that were NOT gated.
        # If valid_mask is given, only count real (non-padding) timesteps.
        gate_mask = out["gate_mask"].float()
        if valid_mask is not None:
            valid_f = valid_mask.float()
            n_valid = valid_f.sum(dim=1).clamp(min=1.0)
            frac_gated = (gate_mask * valid_f).sum(dim=1) / n_valid
        else:
            frac_gated = gate_mask.mean(dim=1)
        out["reliability_gate"] = 1.0 - frac_gated

        if xgb_proba is not None:
            delta_s = self.residual_head(context, out["reliability_gate"])
            out["delta_s"] = delta_s
            out["final_score"] = fuse_final_score(xgb_proba, delta_s)

        return out

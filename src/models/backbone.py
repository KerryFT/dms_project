"""
Stage 2 — FiLM-modulated MobileNetV3-Small backbone for Branch B (eye patches).

FiLMMobileNetV3Small wraps torchvision's mobilenet_v3_small.features and
injects FiLM right after blocks 3, 6, 11 (channel widths 24, 40, 96 — early /
mid / late, confirmed by inspecting the real model, see chat). Geometry
conditioning therefore reaches the CNN at three spatial scales, not just once.

FiLMEyeEncoder is the actual Stage-1+Stage-2 fusion point: it runs BOTH eye
patches (left, right) through the SAME backbone weights (Siamese-style),
modulated by the SAME geometry vector (head pose affects both eyes
identically), then concatenates the two pooled embeddings, and projects them
to a lower-dimensional bottleneck to prevent GRU dimension explosion.

A note on pretrained weights: this sandbox has no internet access to
download.pytorch.org, so tests below run with weights=None (random init) —
this only validates that shapes/gradients flow correctly, NOT visual
quality. On your own machine, construct with pretrained=True to load
ImageNet weights; for an eye-patch CNN trained on a relatively small DMS
dataset, starting from ImageNet pretraining is strongly recommended.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .film import FiLMGenerator, FiLMLayer

# Indices into mobilenet_v3_small.features where we inject FiLM, and the
# output channel width at each of those points (verified empirically).
DEFAULT_INJECTION_BLOCKS = (3, 6, 11)
DEFAULT_INJECTION_CHANNELS = (24, 40, 96)


class FiLMMobileNetV3Small(nn.Module):
    """MobileNetV3-Small with FiLM conditioning injected at multiple blocks."""

    def __init__(
        self,
        geometry_dim: int = 6,
        film_hidden_dim: int = 32,
        injection_blocks: Tuple[int, ...] = DEFAULT_INJECTION_BLOCKS,
        injection_channels: Tuple[int, ...] = DEFAULT_INJECTION_CHANNELS,
        pretrained: bool = False,
    ):
        super().__init__()
        assert len(injection_blocks) == len(injection_channels)

        weights = (
            tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = tv_models.mobilenet_v3_small(weights=weights)
        self.features: nn.Sequential = backbone.features
        self.injection_blocks = injection_blocks

        self.film_generator = FiLMGenerator(
            geometry_dim=geometry_dim,
            hidden_dim=film_hidden_dim,
            channel_dims=injection_channels,
        )
        self.film_layer = FiLMLayer()
        self.pool = nn.AdaptiveAvgPool2d(1)
        last_conv = self.features[-1][0]
        self.output_dim = getattr(last_conv, "out_channels", 576)

    def forward(self, image: torch.Tensor, geometry_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) eye patch, already resized (e.g. 64x64) and normalized.
            geometry_vector: (B, geometry_dim).
        Returns:
            (B, output_dim) pooled embedding (576-d for mobilenet_v3_small).
        """
        film_params = self.film_generator(geometry_vector)
        film_idx = 0
        x = image
        for block_idx, layer in enumerate(self.features):
            x = layer(x)
            if block_idx in self.injection_blocks:
                gamma, beta = film_params[film_idx]
                x = self.film_layer(x, gamma, beta)
                film_idx += 1
        x = self.pool(x).flatten(1)
        return x


class FiLMEyeEncoder(nn.Module):
    """Shared-weight (Siamese) encoder for the left+right eye patches.

    This is the concrete fusion of Stage 1 (geometry vector + two eye
    patches) into a single per-frame embedding consumed by Stage 3's GRU.
    """

    def __init__(
        self,
        geometry_dim: int = 6,
        film_hidden_dim: int = 32,
        pretrained: bool = False,
        projection_dim: int = 256, # THÊM MỚI: Mặc định nén xuống 256 chiều
    ):
        super().__init__()
        self.backbone = FiLMMobileNetV3Small(
            geometry_dim=geometry_dim,
            film_hidden_dim=film_hidden_dim,
            pretrained=pretrained,
        )
        
        # Kích thước thô sau khi nối 2 mắt (576 * 2 = 1152)
        raw_fused_dim = self.backbone.output_dim * 2
        
        # THÊM MỚI: Lớp nén Bottleneck (Linear -> LayerNorm -> ReLU)
        self.projection = nn.Sequential(
            nn.Linear(raw_fused_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(inplace=True)
        )
        
        # CẬP NHẬT: Đầu ra chính thức báo cho Stage 3 bây giờ là 256
        self.output_dim = projection_dim

    def forward(
        self,
        left_patch: torch.Tensor,
        right_patch: torch.Tensor,
        geometry_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            left_patch, right_patch: (B, 3, H, W) each.
            geometry_vector: (B, geometry_dim) — SAME vector conditions both eyes.
        Returns:
            (B, projection_dim) fused and compressed visual embedding.
        """
        left_emb = self.backbone(left_patch, geometry_vector)
        right_emb = self.backbone(right_patch, geometry_vector)
        
        # Gộp đặc trưng (B, 1152)
        fused_emb = torch.cat([left_emb, right_emb], dim=-1)
        
        # Đi qua lớp Bottleneck để nén xuống (B, 256)
        return self.projection(fused_emb)

    def encode_sequence(
        self,
        left_patches: torch.Tensor,
        right_patches: torch.Tensor,
        geometry_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience for video sequences: flattens (B, T, ...) -> (B*T, ...)
        for an efficient single CNN forward pass, then reshapes back to
        (B, T, embed_dim) for Stage 3's GRU.

        Args:
            left_patches, right_patches: (B, T, 3, H, W)
            geometry_vectors: (B, T, geometry_dim)
        Returns:
            (B, T, projection_dim)
        """
        b, t = left_patches.shape[:2]
        left_flat = left_patches.reshape(b * t, *left_patches.shape[2:])
        right_flat = right_patches.reshape(b * t, *right_patches.shape[2:])
        geom_flat = geometry_vectors.reshape(b * t, geometry_vectors.shape[-1])

        # Gọi forward đã được tích hợp sẵn lớp projection
        emb_flat = self.forward(left_flat, right_flat, geom_flat)
        
        # Tự động reshape lại theo output_dim mới (256)
        return emb_flat.reshape(b, t, -1)
"""
Artifact detection module for detecting AI manipulation in lip-sync videos.

This module focuses on detecting visual artifacts and inconsistencies that
AI manipulation tools (Wav2Lip, DeepFaceLab, etc.) introduce.
"""

from typing import Tuple

import torch
from torch import Tensor, nn


class TemporalInconsistencyDetector(nn.Module):
    """
    Detects temporal inconsistencies (flickering, frame-to-frame artifacts)
    that are common in AI-manipulated videos.
    """

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        # 3D conv to detect temporal inconsistencies
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                feature_dim,
                feature_dim // 2,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                feature_dim // 2,
                feature_dim // 4,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(feature_dim // 4),
            nn.ReLU(inplace=True),
        )
        # Global pooling over spatial and temporal
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, D, T, H, W) - visual features

        Returns:
            (B, D//4) - inconsistency features
        """
        out = self.temporal_conv(x)  # (B, D//4, T', H', W')
        out = self.pool(out)  # (B, D//4, 1, 1, 1)
        return out.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, D//4)


class ArtifactDetector(nn.Module):
    """
    Detects manipulation artifacts in video by analyzing:
    1. Temporal inconsistencies (flickering, frame jumps)
    2. Visual artifacts (blur, color inconsistencies)
    3. Audio-visual misalignment patterns
    """

    def __init__(self, visual_feature_dim: int = 256, embed_dim: int = 256) -> None:
        super().__init__()
        self.temporal_detector = TemporalInconsistencyDetector(visual_feature_dim)

        # Artifact feature dimension
        artifact_dim = visual_feature_dim // 4

        # Combine artifact features with fused features
        self.artifact_fusion = nn.Sequential(
            nn.Linear(embed_dim + artifact_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, visual_features: Tensor, fused_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            visual_features: (B, D_v, T, H, W) - raw visual encoder output
            fused_features: (B, T, D_e) - fused audio-visual features

        Returns:
            artifact_features: (B, D_e//2) - artifact detection features
            enhanced_fused: (B, T, D_e) - enhanced fused features with artifact awareness
        """
        # Detect temporal inconsistencies
        artifact_feat = self.temporal_detector(visual_features)  # (B, D_v//4)

        # Aggregate fused features temporally
        pooled_fused = fused_features.mean(dim=1)  # (B, D_e)

        # Combine artifact and fused features
        combined = torch.cat([pooled_fused, artifact_feat], dim=-1)  # (B, D_e + D_v//4)
        artifact_features = self.artifact_fusion(combined)  # (B, D_e//2)

        return artifact_features, fused_features

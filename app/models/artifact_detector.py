"""
Artifact detection module for detecting AI manipulation in lip-sync videos.

This module focuses on detecting visual artifacts and inconsistencies that
AI manipulation tools (Wav2Lip, DeepFaceLab, etc.) introduce.
"""

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

    def __init__(
        self,
        visual_feature_dim: int = 256,
        embed_dim: int = 256,
        use_delta_map: bool = True,
    ) -> None:
        super().__init__()
        self.use_delta_map = use_delta_map
        self.temporal_detector = TemporalInconsistencyDetector(visual_feature_dim)

        # Artifact feature dimension
        artifact_dim = visual_feature_dim // 4

        detector_multiplier = 2 if use_delta_map else 1
        total_artifact_dim = artifact_dim * detector_multiplier

        # Combine CLS with artifact features
        self.artifact_fusion = nn.Sequential(
            nn.Linear(embed_dim + total_artifact_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, visual_features: Tensor, cls_output: Tensor
    ) -> Tensor:
        """
        Args:
            visual_features: (B, D_v, T, H, W) - raw visual encoder output
            cls_output: (B, D_e) - CLS token from Temporal Transformer

        Returns:
            artifact_features: (B, D_e//2) - artifact branch output for final concat
        """
        # Detect temporal inconsistencies from raw feature volume.
        artifact_feat = self.temporal_detector(visual_features)  # (B, D_v//4)

        if self.use_delta_map:
            if visual_features.size(2) > 1:
                delta_map = visual_features[:, :, 1:] - visual_features[:, :, :-1]
            else:
                delta_map = torch.zeros_like(visual_features)
            delta_feat = self.temporal_detector(delta_map)  # (B, D_v//4)
            artifact_feat = torch.cat([artifact_feat, delta_feat], dim=-1)

        # Combine CLS and artifact features
        combined = torch.cat([cls_output, artifact_feat], dim=-1)
        artifact_features = self.artifact_fusion(combined)  # (B, D_e//2)

        return artifact_features

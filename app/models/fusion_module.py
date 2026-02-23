from typing import Tuple

import torch
from torch import Tensor, nn


class FeatureProjection(nn.Module):
    """
    Projects visual and audio encodings to a shared embedding dimension.

    Expected inputs:
        visual_feat: (B, D_v, T_v)
        audio_feat:  (B, D_a, T_a)

    Outputs:
        visual_emb: (B, T_v, D_e)
        audio_emb:  (B, T_a, D_e)
    """

    def __init__(self, visual_dim: int, audio_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)

    def forward(
        self,
        visual_feat: Tensor,
        audio_feat: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if visual_feat.dim() != 3 or audio_feat.dim() != 3:
            raise ValueError(
                "FeatureProjection expects visual_feat and audio_feat of shape (B, D, T)"
            )

        # (B, D, T) -> (B, T, D)
        v = visual_feat.transpose(1, 2)
        a = audio_feat.transpose(1, 2)

        v = self.visual_proj(v)
        a = self.audio_proj(a)
        return v, a


class FusionModule(nn.Module):
    """
    Time‑wise fusion of projected visual and audio embeddings.

    If temporal lengths differ, the audio sequence is linearly interpolated
    to match the visual sequence length (simple but effective for lip‑sync).

    Inputs:
        visual_emb: (B, T_v, D_e)
        audio_emb:  (B, T_a, D_e)

    Output:
        fused:      (B, T_v, D_e)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, visual_emb: Tensor, audio_emb: Tensor) -> Tensor:
        if visual_emb.dim() != 3 or audio_emb.dim() != 3:
            raise ValueError(
                "FusionModule expects visual_emb and audio_emb of shape (B, T, D_e)"
            )

        b_v, t_v, d_v = visual_emb.shape
        b_a, t_a, d_a = audio_emb.shape
        if b_v != b_a or d_v != d_a:
            raise ValueError(
                "visual_emb and audio_emb must have the same batch size and feature dim"
            )

        if t_v != t_a:
            # Interpolate audio along the temporal dimension to match visual length.
            audio_emb = torch.nn.functional.interpolate(
                audio_emb.transpose(1, 2),  # (B, D_e, T_a)
                size=t_v,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)  # (B, T_v, D_e)

        x = torch.cat([visual_emb, audio_emb], dim=-1)  # (B, T_v, 2D_e)
        x = self.fc(x)
        return x


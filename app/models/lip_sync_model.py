from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn

from .artifact_detector import ArtifactDetector
from .audio_encoder import AudioEncoder
from .classifier import ClassificationHead
from .fusion_module import CrossModalAttention, FeatureProjection
from .temporal import TemporalTransformer
from .visual_encoder import VisualEncoder


class LipSyncModel(nn.Module):
    """
    End‑to‑end audio‑visual lip‑sync detection model with AI manipulation detection.

    Architecture:
    - VisualEncoder + AudioEncoder (unchanged)
    - FeatureProjection → CrossModalAttention (replaces concat fusion)
    - TemporalTransformer with CLS token (replaces global avg pool)
    - ArtifactDetector branch (CLS + visual feature map → artifact features)
    - Final concat: CLS (256) + artifact (128) → ClassificationHead
    """

    def __init__(
        self,
        visual_feature_dim: int = 256,
        audio_feature_dim: int = 256,
        embed_dim: int = 256,
        detect_artifacts: bool = True,
        cross_modal_heads: int = 8,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        temporal_pre_conv: bool = True,
        use_delta_artifact: bool = True,
    ) -> None:
        super().__init__()
        self.detect_artifacts = detect_artifacts

        # Encoders
        self.visual_encoder = VisualEncoder(feature_dim=visual_feature_dim)
        self.audio_encoder = AudioEncoder(feature_dim=audio_feature_dim)

        # Feature projection + cross-modal attention
        self.projection = FeatureProjection(
            visual_dim=visual_feature_dim,
            audio_dim=audio_feature_dim,
            embed_dim=embed_dim,
        )
        self.cross_modal = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=cross_modal_heads,
        )

        # Temporal transformer (replaces global avg pool)
        self.temporal = TemporalTransformer(
            embed_dim=embed_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            pre_conv=temporal_pre_conv,
        )

        # Artifact detection
        if detect_artifacts:
            self.artifact_detector = ArtifactDetector(
                visual_feature_dim=visual_feature_dim,
                embed_dim=embed_dim,
                use_delta_map=use_delta_artifact,
            )
            classifier_input_dim = embed_dim + embed_dim // 2  # 256 + 128
        else:
            self.artifact_detector = None
            classifier_input_dim = embed_dim

        self.classifier = ClassificationHead(
            input_dim=classifier_input_dim, hidden_dim=128
        )

    def forward(
        self,
        visual: Tensor,
        audio: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Args:
            visual: Tensor `(B, 3, T_v, H, W)` – mouth‑crop video clip.
            audio:  Tensor `(B, 1, F, T_a)` – log Mel‑spectrogram.

        Returns:
            Tensor `(B,)` – **logits** for P(REAL). Apply `torch.sigmoid` to get probability.
        """
        # Encode modalities
        if self.detect_artifacts and self.artifact_detector is not None:
            v_feat, v_map = self.visual_encoder(visual, return_map=True)
        else:
            v_feat = self.visual_encoder(visual)
            v_map = None
        a_feat = self.audio_encoder(audio)

        # Project to shared embedding
        v_emb, a_emb = self.projection(v_feat, a_feat)

        # Cross-modal attention (replaces concat fusion)
        fused = self.cross_modal(v_emb, a_emb)  # (B, T, D_e)

        # Temporal transformer → CLS output
        cls_output = self.temporal(fused)  # (B, D_e)

        # Artifact branch + final concat
        if self.detect_artifacts and self.artifact_detector is not None:
            if v_map is None:
                raise RuntimeError("Artifact detection enabled but visual feature map is missing.")
            artifact_feat = self.artifact_detector(v_map, cls_output)  # (B, 128)
            combined = torch.cat([cls_output, artifact_feat], dim=-1)  # (B, 384)
        else:
            combined = cls_output

        logits = self.classifier(combined)
        if not return_aux:
            return logits

        aux: Dict[str, Tensor] = {
            "visual_tokens": v_emb,
            "audio_tokens": a_emb,
            "fused_tokens": fused,
            "cls_output": cls_output,
        }
        return logits, aux

    @torch.no_grad()
    def predict(self, visual: Tensor, audio: Tensor) -> Tensor:
        """
        Convenience wrapper around `forward` that ensures eval mode and
        disables gradient tracking.
        """
        self.eval()
        return self.forward(visual, audio)


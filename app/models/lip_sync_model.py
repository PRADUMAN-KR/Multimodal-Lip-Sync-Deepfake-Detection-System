from typing import Tuple

import torch
from torch import Tensor, nn

from .artifact_detector import ArtifactDetector
from .audio_encoder import AudioEncoder
from .classifier import ClassificationHead
from .fusion_module import FeatureProjection, FusionModule
from .temporal import TemporalAggregation
from .visual_encoder import VisualEncoder


class LipSyncModel(nn.Module):
    """
    End‑to‑end audio‑visual lip‑sync detection model with AI manipulation detection.

    The model detects:
    1. Audio-visual sync (whether lips match audio)
    2. AI manipulation artifacts (whether video was modified by tools like Wav2Lip)

    It encodes visual mouth crops and audio spectrograms, detects temporal
    inconsistencies and manipulation artifacts, fuses modalities, and predicts
    both sync quality and manipulation probability.
    """

    def __init__(
        self,
        visual_feature_dim: int = 256,
        audio_feature_dim: int = 256,
        embed_dim: int = 256,
        detect_artifacts: bool = True,
    ) -> None:
        super().__init__()
        self.detect_artifacts = detect_artifacts

        # Encoders
        self.visual_encoder = VisualEncoder(feature_dim=visual_feature_dim)
        self.audio_encoder = AudioEncoder(feature_dim=audio_feature_dim)

        # Cross‑modal projection + fusion
        self.projection = FeatureProjection(
            visual_dim=visual_feature_dim,
            audio_dim=audio_feature_dim,
            embed_dim=embed_dim,
        )
        self.fusion = FusionModule(embed_dim=embed_dim, hidden_dim=256)

        # Artifact detection (for AI manipulation detection)
        if detect_artifacts:
            self.artifact_detector = ArtifactDetector(
                visual_feature_dim=visual_feature_dim, embed_dim=embed_dim
            )
            # Classifier uses both sync features and artifact features
            classifier_input_dim = embed_dim + embed_dim // 2
        else:
            self.artifact_detector = None
            classifier_input_dim = embed_dim

        # Temporal aggregation + classification
        self.temporal = TemporalAggregation()
        self.classifier = ClassificationHead(
            input_dim=classifier_input_dim, hidden_dim=128
        )

    def forward(self, visual: Tensor, audio: Tensor) -> Tensor:
        """
        Args:
            visual: Tensor `(B, 3, T_v, H, W)` – mouth‑crop video clip.
            audio:  Tensor `(B, 1, F, T_a)` – log Mel‑spectrogram.

        Returns:
            Tensor `(B,)` – **logits** for P(REAL). Apply `torch.sigmoid` to get probability.
        """
        # Encode modalities
        if self.detect_artifacts and self.artifact_detector is not None:
            v_feat, v_map = self.visual_encoder(visual, return_map=True)  # (B, D_v, T_v'), (B, D_v, T', H', W')
        else:
            v_feat = self.visual_encoder(visual)  # (B, D_v, T_v')
            v_map = None
        a_feat = self.audio_encoder(audio)  # (B, D_a, T_a')

        # Project to shared embedding, shape (B, T, D_e)
        v_emb, a_emb = self.projection(v_feat, a_feat)

        # Time‑wise fusion
        fused = self.fusion(v_emb, a_emb)  # (B, T, D_e)

        # Detect manipulation artifacts
        if self.detect_artifacts and self.artifact_detector is not None:
            # Detect temporal inconsistencies and artifacts
            if v_map is None:
                raise RuntimeError("Artifact detection enabled but visual feature map is missing.")
            artifact_feat, _ = self.artifact_detector(v_map, fused)  # (B, D_e//2)
            pooled = self.temporal(fused)  # (B, D_e)
            # Combine sync features and artifact features
            combined = torch.cat([pooled, artifact_feat], dim=-1)  # (B, D_e + D_e//2)
        else:
            pooled = self.temporal(fused)  # (B, D_e)
            combined = pooled

        # Classify: output is logits for P(REAL)
        logits = self.classifier(combined)  # (B,)
        return logits

    @torch.no_grad()
    def predict(self, visual: Tensor, audio: Tensor) -> Tensor:
        """
        Convenience wrapper around `forward` that ensures eval mode and
        disables gradient tracking.
        """
        self.eval()
        return self.forward(visual, audio)


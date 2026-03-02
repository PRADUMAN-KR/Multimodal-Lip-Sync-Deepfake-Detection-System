from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_modal_contrastive_loss(
    visual_tokens: Tensor,
    audio_tokens: Tensor,
    labels: Tensor,
    temperature: float = 0.07,
    fake_margin: float = 0.10,
) -> Tensor:
    """
    Contrastive objective for projected audio/visual token sequences.

    - Real samples (label=1): diagonal pair should be most similar (InfoNCE).
    - Fake samples (label=0): diagonal pair should be lower than hardest negative
      by a margin.
    """
    if visual_tokens.dim() != 3 or audio_tokens.dim() != 3:
        raise ValueError("visual_tokens and audio_tokens must be shaped (B, T, D)")
    if labels.dim() != 1:
        labels = labels.view(-1)

    v = F.normalize(visual_tokens.mean(dim=1), dim=-1)  # (B, D)
    a = F.normalize(audio_tokens.mean(dim=1), dim=-1)   # (B, D)

    sim = (v @ a.transpose(0, 1)) / max(temperature, 1e-6)  # (B, B)
    bsz = sim.size(0)
    target = torch.arange(bsz, device=sim.device)

    real_mask = labels >= 0.5
    fake_mask = ~real_mask
    losses = []

    if real_mask.any():
        real_idx = real_mask.nonzero(as_tuple=False).squeeze(1)
        losses.append(F.cross_entropy(sim[real_idx], target[real_idx]))
        losses.append(F.cross_entropy(sim.transpose(0, 1)[real_idx], target[real_idx]))

    if fake_mask.any() and bsz > 1:
        diag = sim.diag()
        off_diag = sim.masked_fill(
            torch.eye(bsz, device=sim.device, dtype=torch.bool), float("-inf")
        )
        hardest_neg_row = off_diag.max(dim=1).values
        hardest_neg_col = off_diag.max(dim=0).values
        fake_idx = fake_mask.nonzero(as_tuple=False).squeeze(1)
        row_loss = F.relu(diag[fake_idx] - hardest_neg_row[fake_idx] + fake_margin).mean()
        col_loss = F.relu(diag[fake_idx] - hardest_neg_col[fake_idx] + fake_margin).mean()
        losses.append(0.5 * (row_loss + col_loss))

    if not losses:
        return sim.new_zeros(())
    return torch.stack(losses).mean()

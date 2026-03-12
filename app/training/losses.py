from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


def sync_contrastive_loss(
    visual_tokens: Tensor,
    audio_tokens: Tensor,
    audio_tokens_negatives: List[Tensor],
    real_mask: Tensor | None = None,
    temperature: float = 0.07,
) -> Tensor:
    """
    Temporal alignment contrastive: (video, correct_audio) vs (video, shifted_audio).
    Pulls in-sync pairs together and pushes shifted-audio pairs apart.
    Apply only on REAL (label=1) samples; pass real_mask to mask fake samples.

    Args:
        visual_tokens: (B, T, D)
        audio_tokens: (B, T, D) — correct (aligned) audio
        audio_tokens_negatives: list of (B, T, D) — audio shifted by ±5, ±10, ±15 frames
        real_mask: (B,) boolean, True = real (in-sync) sample. If None, use all.
        temperature: softmax temperature
    """
    if not audio_tokens_negatives:
        return visual_tokens.new_zeros(())

    if real_mask is not None and not real_mask.any():
        return visual_tokens.new_zeros(())

    if real_mask is not None:
        visual_tokens = visual_tokens[real_mask]
        audio_tokens = audio_tokens[real_mask]
        audio_tokens_negatives = [a[real_mask] for a in audio_tokens_negatives]

    v = F.normalize(visual_tokens.mean(dim=1), dim=-1)  # (B, D)
    a = F.normalize(audio_tokens.mean(dim=1), dim=-1)   # (B, D)
    pos_sim = (v * a).sum(dim=-1) / max(temperature, 1e-6)  # (B,)

    neg_sims = []
    for a_neg in audio_tokens_negatives:
        a_neg_flat = F.normalize(a_neg.mean(dim=1), dim=-1)  # (B, D)
        neg_sims.append((v * a_neg_flat).sum(dim=-1) / max(temperature, 1e-6))  # (B,)
    neg_sim = torch.stack(neg_sims, dim=1)  # (B, N_neg)

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N_neg)
    target = logits.new_zeros(logits.size(0), dtype=torch.long)
    return F.cross_entropy(logits, target)


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

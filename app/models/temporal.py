from typing import Optional

from torch import Tensor, nn


class TemporalAggregation(nn.Module):
    """
    Temporal aggregation for fused audio‑visual features.

    Default behavior is simple global average pooling over the time axis,
    which is robust and efficient. The API is intentionally kept small
    but can be extended later to support attention‑based pooling while
    remaining backward compatible.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Tensor of shape `(B, T, D)` – sequence of fused features.
            lengths: Optional 1D tensor `(B,)` with valid lengths per
                sequence (before padding). If provided, padded positions
                (t >= lengths[b]) are ignored in the mean.

        Returns:
            Tensor of shape `(B, D)`.
        """
        if x.dim() != 3:
            raise ValueError(
                f"TemporalAggregation expected input of shape (B, T, D), got {tuple(x.shape)}"
            )

        if lengths is None:
            return x.mean(dim=1)

        # Masked temporal mean for variable‑length sequences.
        if lengths.dim() != 1 or lengths.size(0) != x.size(0):
            raise ValueError(
                "lengths must be 1D with shape (B,) and match batch size of x"
            )

        device = x.device
        max_t = x.size(1)
        # Shape: (B, T)
        time_indices = (
            x.new_tensor(range(max_t), dtype=lengths.dtype, device=device)
            .unsqueeze(0)
            .expand(x.size(0), -1)
        )
        # mask[b, t] = True if t < lengths[b]
        mask = time_indices < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # (B, T, 1)

        x_masked = x * mask  # zeros out padded steps
        # Avoid division by zero by clamping lengths
        denom = lengths.clamp_min(1).to(x.dtype).unsqueeze(-1)
        return x_masked.sum(dim=1) / denom


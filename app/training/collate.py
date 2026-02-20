from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor


BatchItem = Optional[Tuple[Tensor, Tensor, Tensor]]


def safe_collate(batch: List[BatchItem]) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """
    Collate that **skips** samples returned as None (e.g. failed preprocessing).

    This prevents DataLoader from crashing mid-epoch when a few videos are corrupt,
    have no detectable face, etc.
    
    Returns None if all samples in the batch failed (caller should skip this batch).
    """
    original_size = len(batch)
    batch = [b for b in batch if b is not None]
    
    if not batch:
        # All samples failed - return None to signal batch should be skipped
        # The training loop will catch this and continue
        return None

    if len(batch) < original_size:
        # Some samples were skipped, but we have at least one valid sample
        pass  # Continue with valid samples

    visuals, audios, labels = zip(*batch)
    return torch.stack(list(visuals), dim=0), torch.stack(list(audios), dim=0), torch.stack(list(labels), dim=0)


from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """
    Select the best available device, with explicit support for:
      - CUDA GPUs (Linux/Windows)
      - Apple Silicon GPUs via Metal (`mps`)
      - CPU as a safe fallback

    `preferred` can be one of: "cuda", "mps", "cpu" (case‑insensitive).
    If the preferred device is not available, we gracefully fall back to
    the best supported device.
    """
    preferred = (preferred or "").lower()

    # Explicit preference handling
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")

    # Auto‑select: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

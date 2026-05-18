"""
Combined attention mask for training-time use.

At training time, a batched padded sequence needs four constraints fused into
one additive 4D mask:

    allowed(q, k) = causal(q, k)
                 AND (sink(k) OR window(q, k))
                 AND not_pad(k)

If padding is dropped from this fusion the model silently learns to attend to
pad tokens. If causality is dropped it learns to peek at the future. Both
constraints have to compose with the sparse pattern.

The window can be randomized per call so the model sees varied sparsity across
training steps and learns robustness across windows it will be evaluated at.
"""

from __future__ import annotations

import random

import torch


def build_training_mask(
    padding_mask: torch.Tensor,
    *,
    n_sinks: int,
    window: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Combined 4D additive mask.

    Args:
        padding_mask: shape (B, S). 1 = real token, 0 = pad.
        n_sinks: leading tokens kept attendable from every query.
        window: recent tokens (including self) kept attendable per query.
        dtype: output dtype.

    Returns:
        Tensor of shape (B, 1, S, S). Zero where attention is allowed,
        -inf elsewhere. Broadcasts over heads.
    """
    if n_sinks < 0:
        raise ValueError(f"n_sinks must be >= 0, got {n_sinks}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    batch, seq_len = padding_mask.shape
    device = padding_mask.device

    q = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
    k = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
    pattern = (k <= q) & ((k < n_sinks) | ((q - k) < window))  # (S, S)

    # broadcast pattern to batch; intersect with key-side padding
    not_pad_k = padding_mask.to(torch.bool).unsqueeze(1)  # (B, 1, S)
    allow = pattern.unsqueeze(0) & not_pad_k  # (B, S, S)

    mask = torch.zeros(batch, seq_len, seq_len, dtype=dtype, device=device)
    mask.masked_fill_(~allow, float("-inf"))
    return mask.unsqueeze(1)  # (B, 1, S, S)


def sample_window(choices: tuple[int, ...] = (64, 128, 256), rng: random.Random | None = None) -> int:
    """Pick one window size for the current training step.

    Default discrete choices match the Phase 1 sweep's interesting range.
    Pass a seeded Random for reproducibility.
    """
    r = rng or random
    return r.choice(choices)

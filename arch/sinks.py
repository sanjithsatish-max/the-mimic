"""
Attention-sink masking utility.

Implements StreamingLLM-style sparse attention (Xiao et al. 2024,
https://arxiv.org/abs/2309.17453) as a 4D additive attention mask. Each
query position attends to:
  - the first ``n_sinks`` tokens (anchoring), and
  - the previous ``window`` tokens including itself (foveal focus).

Pure function. No model modification, no KV cache surgery. Designed for
a single forward pass to measure the quality impact of the sparse pattern
before any cache or training work.
"""

from __future__ import annotations

import torch


def sink_window_mask(
    seq_len: int,
    n_sinks: int,
    window: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a (1, 1, seq_len, seq_len) additive attention mask.

    Allowed positions get 0; blocked positions get -inf. The mask is always
    causal. The returned shape broadcasts over (batch, num_heads, q, k).

    Args:
        seq_len: number of tokens.
        n_sinks: number of leading tokens kept attendable from every query.
            Set to 0 to disable sinks (pure sliding window).
        window: number of recent tokens (including self) kept attendable.
            Must be >= 1.

    Raises:
        ValueError: on invalid n_sinks or window.
    """
    if n_sinks < 0:
        raise ValueError(f"n_sinks must be >= 0, got {n_sinks}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    q = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
    k = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)

    allow = (k <= q) & ((k < n_sinks) | ((q - k) < window))

    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask.masked_fill_(~allow, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)

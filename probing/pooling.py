"""Pool token-level hidden states into one vector per sentence per layer."""

from __future__ import annotations

import torch

from probing.layer_extractor import BatchHiddenStates


def pool_cls(hidden_states: torch.Tensor) -> torch.Tensor:
    """Take the [CLS] token representation. hidden_states: (batch, seq_len, hidden)."""
    return hidden_states[:, 0, :]


def pool_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over real tokens only, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def pool_max(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Max-pool over real tokens only, masking padding to -inf so it's never selected."""
    mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq_len, 1)
    masked = hidden_states.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1).values


POOLING_FNS = {
    "cls": lambda h, mask: pool_cls(h),
    "mean": pool_mean,
    "max": pool_max,
}


def pool_batch(batch: BatchHiddenStates, strategy: str) -> list[torch.Tensor]:
    """Apply a pooling strategy to every layer in a batch.

    Returns one (batch_size, hidden) tensor per layer, in the same
    layer order as batch.hidden_states (index 0 = embedding layer).
    """
    pool_fn = POOLING_FNS[strategy]
    return [pool_fn(layer_hidden, batch.attention_mask) for layer_hidden in batch.hidden_states]

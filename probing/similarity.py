"""Cosine similarity and Spearman correlation utilities for embedding evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from scipy.stats import spearmanr


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity between two (batch, hidden) tensors -> (batch,)."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def spearman_rho(predicted: Sequence[float], gold: Sequence[float]) -> float:
    """Spearman rank correlation between predicted similarity scores and gold labels."""
    correlation, _p_value = spearmanr(predicted, gold)
    return float(correlation)

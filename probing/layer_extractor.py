"""Extract per-layer hidden states from BERT-family encoder models."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_name: str, device: torch.device | None = None
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a tokenizer + encoder configured to emit hidden states from every layer."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()
    return tokenizer, model


@dataclass
class BatchHiddenStates:
    """Per-layer hidden states for one batch, plus the mask needed to pool them."""

    hidden_states: tuple[torch.Tensor, ...]  # (batch, seq_len, hidden) per layer; index 0 = embedding layer
    attention_mask: torch.Tensor  # (batch, seq_len)


def extract_hidden_states(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    sentences: list[str],
    device: torch.device | None = None,
    batch_size: int = 16,
    max_length: int = 128,
) -> Iterator[BatchHiddenStates]:
    """Yield per-layer hidden states batch by batch.

    Streamed rather than collected into one tensor: token-level hidden
    states across every layer for a full dataset don't fit comfortably
    in RAM on a CPU-only machine. pooling.py is expected to reduce each
    batch as it's yielded rather than materializing everything at once.
    """
    device = device or get_device()
    model.eval()

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        yield BatchHiddenStates(
            hidden_states=tuple(h.cpu() for h in output.hidden_states),
            attention_mask=encoded["attention_mask"].cpu(),
        )

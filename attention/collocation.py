"""Extract attention weights between word pairs, for collocation analysis."""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from probing.layer_extractor import get_device


def load_attention_model(
    model_name: str, device: torch.device | None = None
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a tokenizer + encoder configured to emit attention weights from every layer/head."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def find_word_token_span(tokenizer: PreTrainedTokenizerBase, sentence: str, word: str) -> list[int]:
    """Return token indices (including the [CLS]/[SEP] offset) covering `word`'s first occurrence."""
    char_start = sentence.index(word)
    char_end = char_start + len(word)

    encoded = tokenizer(sentence, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    return [
        idx
        for idx, (start, end) in enumerate(offsets)
        if start < char_end and end > char_start and not (start == 0 and end == 0)
    ]


def word_pair_attention(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    sentence: str,
    word_a: str,
    word_b: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a (num_layers, num_heads) tensor of attention weight between word_a and word_b.

    Each cell averages attention mass in both directions (a->b and b->a) and
    across every wordpiece-token pair spanning each word, since BERT's
    tokenizer can split either word into multiple subword tokens.
    """
    device = device or get_device()
    span_a = find_word_token_span(tokenizer, sentence, word_a)
    span_b = find_word_token_span(tokenizer, sentence, word_b)

    encoded = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoded)

    attentions = torch.stack(output.attentions).squeeze(1)  # (num_layers, num_heads, seq_len, seq_len)

    a_to_b = attentions[:, :, span_a, :][:, :, :, span_b].mean(dim=(-1, -2))
    b_to_a = attentions[:, :, span_b, :][:, :, :, span_a].mean(dim=(-1, -2))
    return (a_to_b + b_to_a) / 2

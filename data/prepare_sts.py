"""Load the STS-B benchmark via Hugging Face `datasets`."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

DATASET_NAME = "sentence-transformers/stsb"


@dataclass
class STSBSplit:
    sentence1: list[str]
    sentence2: list[str]
    scores: list[float]  # human similarity judgments, normalized to [0, 1]


def load_stsb(split: str = "test") -> STSBSplit:
    """Load one split of STS-B (train / validation / test)."""
    dataset = load_dataset(DATASET_NAME, split=split)
    return STSBSplit(
        sentence1=dataset["sentence1"],
        sentence2=dataset["sentence2"],
        scores=dataset["score"],
    )

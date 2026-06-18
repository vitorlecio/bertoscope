"""Load chunks + eval queries from the sibling hf_rag_agent project, reused for Part 4/5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HF_RAG_AGENT_DATA = Path(r"c:\Users\elvira\Desktop\data_science\hf_rag_agent\data")


@dataclass
class RagChunk:
    chunk_id: str
    page_path: str
    page_title: str
    heading: str
    content: str
    token_count: int


@dataclass
class EvalItem:
    query: str
    relevant_chunk_ids: list[str]


def load_chunks() -> list[RagChunk]:
    with open(HF_RAG_AGENT_DATA / "chunks.json", encoding="utf-8") as f:
        return [RagChunk(**c) for c in json.load(f)]


def load_eval_items() -> list[EvalItem]:
    with open(HF_RAG_AGENT_DATA / "eval_set.json", encoding="utf-8") as f:
        return [EvalItem(**item) for item in json.load(f)]

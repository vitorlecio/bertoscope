"""Classify RAG chunks as code-containing or pure prose."""

from __future__ import annotations


def is_code_chunk(content: str) -> bool:
    """A chunk counts as 'code' if it contains a fenced code block.

    Mirrors the heuristic already used by hf_rag_agent's eval runner
    (`_is_code_item`), so genre labels are comparable across both projects.
    """
    return "```" in content


def genre_label(content: str) -> str:
    return "code" if is_code_chunk(content) else "prose"

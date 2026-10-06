"""Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document


def doc_key(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.id or str(hash(doc.page_content))


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[tuple[Document, float]]:
    """score(d) = sum_i w_i / (k + rank_i(d)), ranks from 1. Documents are identified by
    ``chunk_id`` (fallback ``id``); duplicates inside one list count once; ties are broken
    by first appearance (deterministic)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    weights = list(weights) if weights is not None else [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    order: dict[str, int] = {}
    for lst, w in zip(ranked_lists, weights):
        seen: set[str] = set()
        for rank, doc in enumerate(lst, start=1):
            key = doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            if key not in docs:
                docs[key] = doc
                order[key] = len(order)
    fused = sorted(scores, key=lambda key: (-scores[key], order[key]))
    if top_n is not None:
        fused = fused[:top_n]
    return [(docs[key], scores[key]) for key in fused]

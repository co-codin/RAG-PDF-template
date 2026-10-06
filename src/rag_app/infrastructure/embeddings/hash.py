"""Deterministic, offline feature-hashing embeddings (tests, CI, demos).

Captures lexical overlap only - not semantic similarity. No network, no model download.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from rag_app.domain.text import tokenize


class HashEmbeddings(Embeddings):
    """Signed feature hashing of unigrams + bigrams, L2-normalised.
    Deterministic across processes (blake2b, not Python's randomised ``hash()``)."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        n = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return n % self.dim, (1.0 if (n >> 63) & 1 else -1.0)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        for feat in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
            idx, sign = self._bucket(feat)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0  # avoid zero vectors (cosine undefined)
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

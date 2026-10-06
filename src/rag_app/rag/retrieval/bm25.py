"""In-memory BM25 index with revision-based cache invalidation.

Limitation: the index lives in *each* worker process (memory x workers) and is rebuilt
from the full corpus when the store revision changes. Workers notice writes made by
other workers/replicas within ``refresh_s`` because the revision is stored in the
vector store itself. Fine for ~100k chunks; beyond that use a search engine
(OpenSearch/Elasticsearch) or native hybrid search (pgvector + tsvector, Qdrant, ...).
"""

from __future__ import annotations

import logging
import threading
import time

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_app.domain.text import tokenize
from rag_app.infrastructure.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, store: VectorStore, refresh_s: float = 2.0) -> None:
        self.store = store
        self.refresh_s = refresh_s
        self._lock = threading.Lock()
        self._revision: str | None = None
        self._checked_at = 0.0
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []
        self._token_sets: list[frozenset[str]] = []

    def invalidate(self) -> None:
        """Force a revision check on next search (call after local writes)."""
        self._checked_at = 0.0

    def _ensure_fresh(self) -> None:
        now = time.monotonic()
        if self._revision is not None and now - self._checked_at < self.refresh_s:
            return
        with self._lock:
            if self._revision is not None and time.monotonic() - self._checked_at < self.refresh_s:
                return
            revision = self.store.revision()
            self._checked_at = time.monotonic()
            if revision == self._revision:
                return
            docs = self.store.all_chunks()
            corpus = [tokenize(d.page_content) for d in docs]
            self._bm25 = BM25Okapi(corpus) if docs else None
            self._docs = docs
            self._token_sets = [frozenset(t) for t in corpus]
            self._revision = revision
            logger.info("bm25_index_built", extra={"chunks": len(docs), "revision": revision})

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        self._ensure_fresh()
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        q = set(tokens)
        # Lexical hits = chunks sharing a term. (Okapi IDF can be negative for terms in
        # >50% of a tiny corpus, so don't filter on score > 0.)
        hits = [i for i in range(len(scores)) if q & self._token_sets[i]]
        hits.sort(key=lambda i: (-scores[i], i))
        return [(self._docs[i], float(scores[i])) for i in hits[:k]]

"""Hybrid retriever: vector + BM25 candidates fused with RRF."""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from rag_app.infrastructure.vectorstore import VectorStore
from rag_app.rag.retrieval.bm25 import BM25Index
from rag_app.rag.retrieval.rrf import doc_key, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, store: VectorStore, bm25: BM25Index | None = None, vector_k: int = 12,
                 bm25_k: int = 12, rrf_k: int = 60, top_n: int = 6) -> None:
        self.store = store
        self.bm25 = bm25 or BM25Index(store)
        self.vector_k, self.bm25_k, self.rrf_k, self.top_n = vector_k, bm25_k, rrf_k, top_n

    def retrieve(self, query: str, top_n: int | None = None) -> list[Document]:
        vec = self.store.similarity_search(query, self.vector_k)
        lex = self.bm25.search(query, self.bm25_k)
        vec_rank = {doc_key(d): (i, s) for i, (d, s) in enumerate(vec, start=1)}
        lex_rank = {doc_key(d): (i, s) for i, (d, s) in enumerate(lex, start=1)}
        fused = reciprocal_rank_fusion([[d for d, _ in vec], [d for d, _ in lex]],
                                       k=self.rrf_k, top_n=top_n or self.top_n)
        out = []
        for doc, score in fused:
            key = doc_key(doc)
            meta = {**doc.metadata, "rrf_score": round(score, 6)}
            if key in vec_rank:
                meta["vector_rank"], meta["vector_score"] = vec_rank[key][0], round(vec_rank[key][1], 4)
            if key in lex_rank:
                meta["bm25_rank"], meta["bm25_score"] = lex_rank[key][0], round(lex_rank[key][1], 4)
            out.append(Document(page_content=doc.page_content, metadata=meta, id=key))
        logger.info("hybrid_retrieve", extra={
            "query": query[:200], "vector_hits": len(vec), "bm25_hits": len(lex), "fused": len(out)})
        return out

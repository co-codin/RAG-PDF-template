"""Knowledge-base management use-cases."""

from __future__ import annotations

from rag_app.core.errors import NotFoundError
from rag_app.domain.models import DocumentInfo
from rag_app.infrastructure.vectorstore import VectorStore
from rag_app.rag.retrieval import BM25Index


class DocumentService:
    def __init__(self, store: VectorStore, bm25: BM25Index | None = None) -> None:
        self.store = store
        self.bm25 = bm25

    def list(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    def delete(self, source: str) -> int:
        deleted = self.store.delete_source(source)
        if deleted == 0:
            raise NotFoundError(f"No document with source '{source}'")
        if self.bm25:
            self.bm25.invalidate()
        return deleted

    def chunk_count(self) -> int:
        return self.store.count()

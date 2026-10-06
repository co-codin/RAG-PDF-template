"""Vector store interface. Implement this Protocol to add a backend (pgvector, Qdrant, ...)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document

from rag_app.domain.models import DocumentInfo


@runtime_checkable
class VectorStore(Protocol):
    def revision(self) -> str:
        """Opaque token that changes on every write and is visible to *all* processes
        sharing the store. Derived indexes (BM25) use it for cache invalidation."""

    def sync_source(self, source: str, chunks: list[Document]) -> dict[str, int]:
        """Make the stored chunks of ``source`` equal to ``chunks`` (add new, drop stale).
        Returns counts: added, skipped_existing, removed_stale."""

    def delete_source(self, source: str) -> int: ...

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """(doc, relevance in [0, 1]) best first."""

    def all_chunks(self) -> list[Document]: ...

    def list_documents(self) -> list[DocumentInfo]: ...

    def count(self) -> int: ...

    def healthcheck(self) -> bool: ...

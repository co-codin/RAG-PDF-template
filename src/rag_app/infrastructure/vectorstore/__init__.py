"""Vector store backends behind the ``VectorStore`` Protocol. Selected by
VECTOR_STORE_BACKEND (+ CHROMA_HOST for Chroma server mode)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag_app.core.config import Settings
from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.embeddings import embedding_signature
from rag_app.infrastructure.vectorstore.base import VectorStore

__all__ = ["VectorStore", "build_vector_store"]


def build_vector_store(settings: Settings, embeddings: Embeddings) -> VectorStore:
    if settings.vector_store_backend == "chroma":
        from rag_app.infrastructure.vectorstore.chroma import ChromaVectorStore

        return ChromaVectorStore(settings, embeddings, signature=embedding_signature(settings))
    # Add e.g. "pgvector" here: implement VectorStore in pgvector.py and branch on the setting.
    raise ConfigurationError(f"Unknown VECTOR_STORE_BACKEND: {settings.vector_store_backend}")

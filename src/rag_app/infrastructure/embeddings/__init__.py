"""Embeddings providers. Interface: ``langchain_core.embeddings.Embeddings``
(embed_documents / embed_query). Selected by EMBEDDINGS_PROVIDER."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag_app.core.config import Settings
from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.embeddings.hash import HashEmbeddings

__all__ = ["Embeddings", "HashEmbeddings", "build_embeddings", "embedding_signature"]


def embedding_signature(settings: Settings) -> str:
    """Identifies the vector space; stored on the collection to prevent mixing models."""
    s = settings
    return {
        "openai": f"openai:{s.openai_embedding_model}",
        "ollama": f"ollama:{s.ollama_embedding_model}",
        "hash": f"hash:{s.hash_embedding_dim}",
    }[s.embeddings_provider]


def build_embeddings(settings: Settings) -> Embeddings:
    match settings.embeddings_provider:
        case "hash":
            return HashEmbeddings(dim=settings.hash_embedding_dim)
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            if settings.openai_api_key is None:
                raise ConfigurationError("EMBEDDINGS_PROVIDER=openai requires OPENAI_API_KEY")
            return OpenAIEmbeddings(model=settings.openai_embedding_model,
                                    api_key=settings.openai_api_key)
        case "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=settings.ollama_embedding_model,
                                    base_url=settings.ollama_base_url)
    raise ConfigurationError(f"Unknown EMBEDDINGS_PROVIDER: {settings.embeddings_provider}")

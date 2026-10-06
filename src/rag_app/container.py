"""Composition root: the only place that knows which concrete backends are used.

Built once per process (FastAPI lifespan, CLI scripts, tests). Tests inject fakes via
the keyword overrides instead of monkeypatching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag_app.core.config import Settings
from rag_app.infrastructure.checkpointer import CheckpointerHandle, build_checkpointer
from rag_app.infrastructure.embeddings import build_embeddings
from rag_app.infrastructure.jobs import JobStore, build_job_store
from rag_app.infrastructure.llm import build_chat_model
from rag_app.infrastructure.vectorstore import VectorStore, build_vector_store
from rag_app.rag.graph import RAGPipeline, build_graph
from rag_app.rag.retrieval import BM25Index, HybridRetriever
from rag_app.services import DocumentService, IngestionService, QueryService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    vector_store: VectorStore
    checkpointer: CheckpointerHandle
    job_store: JobStore
    retriever: HybridRetriever
    pipeline: RAGPipeline
    ingestion: IngestionService
    query: QueryService
    documents: DocumentService

    def close(self) -> None:
        self.checkpointer.close()
        close = getattr(self.job_store, "close", None)
        if callable(close):
            close()


def build_container(settings: Settings, *, embeddings: Embeddings | None = None,
                    llm: BaseChatModel | None = None, vector_store: VectorStore | None = None,
                    job_store: JobStore | None = None) -> Container:
    settings.ensure_dirs()
    store = vector_store or build_vector_store(settings, embeddings or build_embeddings(settings))
    bm25 = BM25Index(store, refresh_s=settings.bm25_refresh_s)
    retriever = HybridRetriever(store, bm25, vector_k=settings.vector_k, bm25_k=settings.bm25_k,
                                rrf_k=settings.rrf_k, top_n=settings.retrieval_k)
    chat = llm or build_chat_model(settings)
    checkpointer = build_checkpointer(settings)
    pipeline = RAGPipeline(build_graph(settings, retriever, chat, checkpointer.saver))
    jobs = job_store or build_job_store(settings)
    logger.info("container_built", extra={
        "vector_store": f"{settings.vector_store_backend}:{settings.chroma_mode}",
        "checkpointer": checkpointer.backend, "job_store": settings.job_store_backend,
        "embeddings": settings.embeddings_provider, "llm": settings.llm_provider})
    return Container(
        settings=settings, vector_store=store, checkpointer=checkpointer, job_store=jobs,
        retriever=retriever, pipeline=pipeline,
        ingestion=IngestionService(settings, store, jobs, bm25),
        query=QueryService(pipeline),
        documents=DocumentService(store, bm25),
    )

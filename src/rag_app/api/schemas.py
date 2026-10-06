"""HTTP request/response models (domain models are reused where the shape is the same)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_app.domain.models import ChatMessage, DocumentInfo, IngestResult, Job, QueryResult

SESSION_ID_PATTERN = r"^[A-Za-z0-9_.:\-]{1,128}$"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN,
                                   description="Conversation/thread id. Omit to start a new session.")


class QueryResponse(QueryResult):
    session_id: str
    latency_ms: float


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048)


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str
    accepted: list[str]
    rejected: list[IngestResult] = Field(default_factory=list)


class JobResponse(Job):
    pass


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class DeleteResponse(BaseModel):
    source: str
    deleted_chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    llm_provider: str
    llm_model: str
    embeddings_provider: str
    vector_store: str
    vector_store_ok: bool
    checkpointer: str
    job_store: str
    chunks: int


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]

"""Pydantic models shared across layers (services return these; the API serialises them)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SourceRef(BaseModel):
    """A retrieved chunk passed to the LLM, numbered as cited in the answer ([ref])."""
    ref: int
    source: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    title: str | None = None
    score: float | None = None
    cited: bool = False
    snippet: str = ""


class QueryResult(BaseModel):
    answer: str
    sources: list[SourceRef]
    query: str  # standalone query actually used for retrieval


class IngestResult(BaseModel):
    source: str
    status: Literal["ok", "empty", "error"]
    chunks: int = 0
    added: int = 0
    skipped_existing: int = 0
    removed_stale: int = 0
    duplicates_in_doc: int = 0
    elapsed_ms: float | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    pages: int | None = None
    type: str | None = None
    title: str | None = None
    ingested_at: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    partial = "partial"  # some inputs failed
    failed = "failed"


class Job(BaseModel):
    id: str
    kind: Literal["files", "url"]
    status: JobStatus = JobStatus.queued
    inputs: list[str] = Field(default_factory=list)
    results: list[IngestResult] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None

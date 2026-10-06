"""Ingestion use-cases: synchronous ingest (CLI, tests) and background jobs (API)."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document

from rag_app.core.config import Settings
from rag_app.core.errors import IngestionError
from rag_app.domain.models import IngestResult, Job, JobStatus, utcnow
from rag_app.infrastructure.jobs import JobStore
from rag_app.infrastructure.vectorstore import VectorStore
from rag_app.rag.chunking import chunk_documents
from rag_app.rag.loaders import load_file_bytes, load_url, validate_extension, validate_url
from rag_app.rag.retrieval import BM25Index

logger = logging.getLogger(__name__)

FilePayload = tuple[str, bytes]  # (filename, content)


class IngestionService:
    def __init__(self, settings: Settings, store: VectorStore, jobs: JobStore,
                 bm25: BM25Index | None = None) -> None:
        self.settings = settings
        self.store = store
        self.jobs = jobs
        self.bm25 = bm25

    # -- validation (cheap, done before enqueueing) ----------------------------
    @staticmethod
    def safe_filename(name: str | None) -> str:
        return (name or "upload").replace("\\", "/").split("/")[-1] or "upload"

    def validate_upload(self, filename: str, size: int) -> None:
        validate_extension(filename, self.settings.allowed_extensions)
        if size > self.settings.max_upload_mb * 1024 * 1024:
            raise IngestionError(f"File exceeds {self.settings.max_upload_mb} MB limit")
        if size == 0:
            raise IngestionError("File is empty")

    def validate_url(self, url: str) -> None:
        validate_url(url, allow_private=self.settings.allow_private_urls)

    # -- synchronous ingestion --------------------------------------------------
    def _ingest_docs(self, source: str, docs: list[Document]) -> IngestResult:
        t0 = time.perf_counter()
        result = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        if not result.chunks:
            return IngestResult(source=source, status="empty")
        stats = self.store.sync_source(source, result.chunks)
        if self.bm25:
            self.bm25.invalidate()
        out = IngestResult(source=source, status="ok", chunks=len(result.chunks),
                           duplicates_in_doc=result.duplicates_skipped,
                           elapsed_ms=round((time.perf_counter() - t0) * 1000, 1), **stats)
        logger.info("ingested", extra=out.model_dump())
        return out

    def ingest_bytes(self, filename: str, data: bytes) -> IngestResult:
        return self._ingest_docs(filename, load_file_bytes(data, filename, self.settings.allowed_extensions))

    def ingest_path(self, path: str | Path) -> IngestResult:
        p = Path(path)
        return self.ingest_bytes(p.name, p.read_bytes())

    def ingest_url(self, url: str) -> IngestResult:
        docs = load_url(url, timeout_s=self.settings.url_fetch_timeout_s,
                        user_agent=self.settings.user_agent, allow_private=self.settings.allow_private_urls)
        return self._ingest_docs(url, docs)

    # -- background jobs --------------------------------------------------------
    def create_job(self, kind: str, inputs: list[str], rejected: list[IngestResult] | None = None) -> Job:
        return self.jobs.create(Job(id=uuid.uuid4().hex, kind=kind, inputs=inputs, results=rejected or []))

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def _run(self, job_id: str, work: list[tuple[str, Callable[[], IngestResult]]]) -> None:
        job = self.jobs.get(job_id)
        if job is None:  # pragma: no cover - evicted
            logger.error("job_missing", extra={"job_id": job_id})
            return
        job.status, job.started_at = JobStatus.running, utcnow()
        self.jobs.save(job)
        for source, fn in work:
            try:
                job.results.append(fn())
            except IngestionError as exc:
                job.results.append(IngestResult(source=source, status="error", error=str(exc)))
            except Exception as exc:  # keep the worker alive; record the failure
                logger.exception("ingest_job_item_failed", extra={"job_id": job_id, "source": source})
                job.results.append(IngestResult(source=source, status="error", error=type(exc).__name__))
            self.jobs.save(job)  # progress is visible while the job runs
        errors = sum(r.status == "error" for r in job.results)
        job.status = (JobStatus.succeeded if errors == 0
                      else JobStatus.failed if errors == len(job.results) else JobStatus.partial)
        job.finished_at = utcnow()
        self.jobs.save(job)
        logger.info("ingest_job_finished", extra={"job_id": job_id, "status": job.status.value,
                                                  "items": len(job.results), "errors": errors})

    def run_files_job(self, job_id: str, files: list[FilePayload]) -> None:
        self._run(job_id, [(name, (lambda n=name, d=data: self.ingest_bytes(n, d))) for name, data in files])

    def run_url_job(self, job_id: str, url: str) -> None:
        self._run(job_id, [(url, lambda: self.ingest_url(url))])

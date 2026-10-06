"""Background-job stores. Selected by JOB_STORE_BACKEND."""

from __future__ import annotations

from rag_app.core.config import Settings
from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.jobs.base import JobStore
from rag_app.infrastructure.jobs.memory import InMemoryJobStore

__all__ = ["InMemoryJobStore", "JobStore", "build_job_store"]


def build_job_store(settings: Settings) -> JobStore:
    match settings.job_store_backend:
        case "memory":
            return InMemoryJobStore(max_jobs=settings.max_jobs_in_memory)
        case "sqlite":
            from rag_app.infrastructure.jobs.sqlite import SqliteJobStore

            return SqliteJobStore(settings.resolved_jobs_db)
    # A Redis implementation would live in redis.py (GET/SET job JSON with a TTL).
    raise ConfigurationError(f"Unknown JOB_STORE_BACKEND: {settings.job_store_backend}")

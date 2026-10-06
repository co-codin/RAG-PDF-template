"""Job store interface for background ingestion. Swap for Redis (or a real queue such
as Celery/RQ/Arq/SQS whose result backend implements this) in multi-host deployments."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_app.domain.models import Job


@runtime_checkable
class JobStore(Protocol):
    def create(self, job: Job) -> Job: ...

    def get(self, job_id: str) -> Job | None: ...

    def save(self, job: Job) -> None: ...

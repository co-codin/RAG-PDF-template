"""In-process job store (default). Bounded; per-process, so with several workers a
status poll may hit a worker that doesn't know the job - use JOB_STORE_BACKEND=sqlite
(one host) or Redis (many hosts) then."""

from __future__ import annotations

import threading
from collections import OrderedDict

from rag_app.domain.models import Job


class InMemoryJobStore:
    def __init__(self, max_jobs: int = 1000) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_jobs

    def create(self, job: Job) -> Job:
        self.save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)
            self._jobs.move_to_end(job.id)
            while len(self._jobs) > self._max:
                self._jobs.popitem(last=False)

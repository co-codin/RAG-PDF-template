"""SQLite job store: shared by all workers on one host (same DATA_DIR)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from rag_app.domain.models import Job


class SqliteJobStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL, "
            "updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        self._conn.commit()
        self._lock = threading.Lock()

    def create(self, job: Job) -> Job:
        self.save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.model_validate_json(row[0]) if row else None

    def save(self, job: Job) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, data) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET "
                "data = excluded.data, updated_at = CURRENT_TIMESTAMP", (job.id, job.model_dump_json()))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

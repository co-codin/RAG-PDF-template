"""SQLite checkpointer (default). WAL mode lets several workers on one host share
the file; for multiple hosts/replicas use the Postgres backend."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from rag_app.infrastructure.checkpointer.base import CheckpointerHandle


def build_sqlite_checkpointer(path: Path) -> CheckpointerHandle:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    saver = SqliteSaver(conn)
    saver.setup()
    return CheckpointerHandle(saver=saver, backend="sqlite", _close=conn.close)

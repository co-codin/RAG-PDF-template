"""Postgres checkpointer (multi-replica). Optional dependency:

    uv sync --extra postgres        # langgraph-checkpoint-postgres + psycopg[binary,pool]
    CHECKPOINTER_BACKEND=postgres POSTGRES_URL=postgresql://user:pass@host:5432/db
"""

from __future__ import annotations

from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.checkpointer.base import CheckpointerHandle


def build_postgres_checkpointer(url: str, max_pool_size: int = 10) -> CheckpointerHandle:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ConfigurationError(
            "CHECKPOINTER_BACKEND=postgres needs the optional extra: `uv sync --extra postgres`") from exc
    pool = ConnectionPool(conninfo=url, max_size=max_pool_size, open=True,
                          kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    saver = PostgresSaver(pool)
    saver.setup()  # idempotent migrations
    return CheckpointerHandle(saver=saver, backend="postgres", _close=pool.close)

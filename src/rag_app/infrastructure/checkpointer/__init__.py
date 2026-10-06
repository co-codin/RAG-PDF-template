"""Conversation-memory checkpointers. Selected by CHECKPOINTER_BACKEND."""

from __future__ import annotations

from rag_app.core.config import Settings
from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.checkpointer.base import CheckpointerHandle

__all__ = ["CheckpointerHandle", "build_checkpointer"]


def build_checkpointer(settings: Settings) -> CheckpointerHandle:
    match settings.checkpointer_backend:
        case "sqlite":
            from rag_app.infrastructure.checkpointer.sqlite import build_sqlite_checkpointer

            return build_sqlite_checkpointer(settings.resolved_checkpoint_db)
        case "memory":  # tests / ephemeral demos; lost on restart, not shared
            from langgraph.checkpoint.memory import InMemorySaver

            return CheckpointerHandle(saver=InMemorySaver(), backend="memory")
        case "postgres":
            if settings.postgres_url is None:
                raise ConfigurationError("CHECKPOINTER_BACKEND=postgres requires POSTGRES_URL")
            from rag_app.infrastructure.checkpointer.postgres import build_postgres_checkpointer

            return build_postgres_checkpointer(settings.postgres_url.get_secret_value())
    raise ConfigurationError(f"Unknown CHECKPOINTER_BACKEND: {settings.checkpointer_backend}")

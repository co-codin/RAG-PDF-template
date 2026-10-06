"""Checkpointer interface. The contract is LangGraph's ``BaseCheckpointSaver``; the
handle adds lifecycle (close) because backends own connections/pools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class CheckpointerHandle:
    saver: BaseCheckpointSaver
    backend: str
    _close: Callable[[], None] = field(default=lambda: None, repr=False)

    def close(self) -> None:
        self._close()

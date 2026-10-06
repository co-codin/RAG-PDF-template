"""contextualize: rewrite a follow-up into a standalone retrieval query using history."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from rag_app.infrastructure.llm import message_text
from rag_app.rag.graph.state import RAGState
from rag_app.rag.prompts import REWRITE_HUMAN, REWRITE_SYSTEM

logger = logging.getLogger(__name__)


def history_text(messages: list[AnyMessage], limit: int) -> str:
    return "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {message_text(m)}"
                     for m in messages[-limit:])


def make_contextualize_node(llm: BaseChatModel, enabled: bool = True, history_limit: int = 10):
    def contextualize(state: RAGState) -> dict:
        question = state["question"]
        reset = {"documents": [], "answer": "", "sources": []}  # per-turn fields
        history = state.get("messages") or []
        if not history or not enabled:
            return {"query": question, **reset}
        try:
            resp = llm.invoke([
                SystemMessage(content=REWRITE_SYSTEM),
                HumanMessage(content=REWRITE_HUMAN.format(
                    history=history_text(history, history_limit), question=question)),
            ])
            query = message_text(resp).strip() or question
        except Exception:
            logger.warning("query_rewrite_failed", exc_info=True)
            query = question
        logger.info("query_rewritten", extra={"question": question[:200], "query": query[:200]})
        return {"query": query, **reset}

    return contextualize

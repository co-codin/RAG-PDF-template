"""grade: drop irrelevant chunks; route to generate or no_answer."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from rag_app.rag.grading import GraderMode, grade_documents
from rag_app.rag.graph.state import RAGState, dict_to_doc, doc_to_dict

logger = logging.getLogger(__name__)


def make_grade_node(mode: GraderMode, llm: BaseChatModel | None):
    def grade(state: RAGState) -> dict:
        docs = [dict_to_doc(d) for d in state.get("documents", [])]
        kept = grade_documents(mode, state.get("query") or state["question"], docs, llm)
        logger.info("graded", extra={"mode": mode, "retrieved": len(docs), "kept": len(kept)})
        return {"documents": [doc_to_dict(d) for d in kept]}

    return grade


def make_route_after_grade(min_relevant_docs: int = 1):
    def route(state: RAGState) -> str:
        return "generate" if len(state.get("documents", [])) >= min_relevant_docs else "no_answer"

    return route

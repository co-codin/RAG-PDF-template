"""Relevance grading of retrieved chunks (the "filter" step before generation)."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag_app.domain.text import tokenize
from rag_app.rag.prompts import GRADE_HUMAN, GRADE_SYSTEM, format_context

logger = logging.getLogger(__name__)

GraderMode = Literal["llm", "heuristic", "none"]


class GradeResult(BaseModel):
    relevant: list[int] = Field(default_factory=list, description="Numbers of the relevant passages")


def heuristic_grade(question: str, docs: list[Document], min_vector_score: float = 0.35) -> list[Document]:
    """Keep chunks sharing a query term or with a high vector similarity. No LLM call."""
    q = set(tokenize(question))
    return [d for d in docs
            if q & set(tokenize(d.page_content)) or d.metadata.get("vector_score", 0.0) >= min_vector_score]


def llm_grade(llm: BaseChatModel, question: str, docs: list[Document]) -> list[Document]:
    """One structured-output call for all passages (cheaper than one call per chunk)."""
    result = llm.with_structured_output(GradeResult).invoke([
        SystemMessage(content=GRADE_SYSTEM),
        HumanMessage(content=GRADE_HUMAN.format(question=question,
                                                context=format_context(docs, max_chars_per_doc=1200))),
    ])
    keep = {i for i in result.relevant if 1 <= i <= len(docs)}
    return [d for i, d in enumerate(docs, start=1) if i in keep]


def grade_documents(mode: GraderMode, question: str, docs: list[Document],
                    llm: BaseChatModel | None = None) -> list[Document]:
    if not docs or mode == "none":
        return docs
    if mode == "llm" and llm is not None:
        try:
            return llm_grade(llm, question, docs)
        except Exception:
            logger.warning("llm_grader_failed_falling_back_to_heuristic", exc_info=True)
    return heuristic_grade(question, docs)

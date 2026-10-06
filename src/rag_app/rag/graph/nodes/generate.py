"""generate: grounded answer with [n] citations; no_answer: fixed "I don't know"."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_app.infrastructure.llm import message_text
from rag_app.rag.graph.state import RAGState, dict_to_doc
from rag_app.rag.prompts import ANSWER_HUMAN, ANSWER_SYSTEM, NO_ANSWER, format_context


def build_sources(docs: list[Document], answer: str) -> list[dict[str, Any]]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return [{
        "ref": i, "source": d.metadata.get("source"), "page": d.metadata.get("page"),
        "chunk_id": d.metadata.get("chunk_id") or d.id, "chunk_index": d.metadata.get("chunk_index"),
        "title": d.metadata.get("title"), "score": d.metadata.get("rrf_score"),
        "cited": i in cited, "snippet": d.page_content[:300],
    } for i, d in enumerate(docs, start=1)]


def make_generate_node(llm: BaseChatModel, history_limit: int = 10):
    def generate(state: RAGState) -> dict:
        docs = [dict_to_doc(d) for d in state.get("documents", [])]
        question = state["question"]
        history = (state.get("messages") or [])[-history_limit:]
        resp = llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM), *history,
            HumanMessage(content=ANSWER_HUMAN.format(context=format_context(docs), question=question)),
        ])
        answer = message_text(resp).strip()
        return {"answer": answer, "sources": build_sources(docs, answer),
                "messages": [HumanMessage(content=question), AIMessage(content=answer)]}

    return generate


def no_answer_node(state: RAGState) -> dict:
    return {"answer": NO_ANSWER, "sources": [],
            "messages": [HumanMessage(content=state["question"]), AIMessage(content=NO_ANSWER)]}

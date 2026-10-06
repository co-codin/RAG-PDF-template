"""Question answering use-cases: blocking query, token streaming, session history."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk

from rag_app.domain.models import ChatMessage, QueryResult, SourceRef
from rag_app.infrastructure.llm import message_text
from rag_app.rag.graph import RAGPipeline


class QueryService:
    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def query(self, question: str, session_id: str) -> QueryResult:
        state = self.pipeline.invoke(question, session_id)
        return QueryResult(answer=state.get("answer", ""), query=state.get("query", question),
                           sources=[SourceRef(**s) for s in state.get("sources", [])])

    def stream(self, question: str, session_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (event, payload): status*, token*, sources, done."""
        final: dict[str, Any] = {}
        streamed = False
        for mode, chunk in self.pipeline.stream(question, session_id):
            if mode == "messages":
                msg, meta = chunk
                # Only token chunks of the answer node (skip rewriter/grader calls and the
                # final messages written to state).
                if meta.get("langgraph_node") == "generate" and isinstance(msg, AIMessageChunk):
                    if text := message_text(msg):
                        streamed = True
                        yield "token", {"text": text}
                continue
            for node, update in (chunk or {}).items():
                update = update or {}
                if node == "contextualize":
                    yield "status", {"stage": "retrieving", "query": update.get("query", question)}
                elif node == "grade":
                    n = len(update.get("documents", []))
                    yield "status", {"stage": "generating" if n else "no_relevant_context", "relevant_chunks": n}
                elif node in ("generate", "no_answer"):
                    final = update
        answer = final.get("answer", "")
        if answer and not streamed:  # no_answer path / non-streaming model
            yield "token", {"text": answer}
        yield "sources", {"sources": final.get("sources", [])}
        yield "done", {"answer": answer, "session_id": session_id}

    def history(self, session_id: str) -> list[ChatMessage]:
        return self.pipeline.history(session_id)

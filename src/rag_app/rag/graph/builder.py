"""Wires the nodes into a LangGraph StateGraph:

    contextualize -> retrieve -> grade --(relevant)--> generate  -> END
                                      \\--(none)------> no_answer -> END

Conversation memory: the checkpointer persists state per ``thread_id`` (= session_id).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_app.core.config import Settings
from rag_app.domain.models import ChatMessage
from rag_app.infrastructure.llm import message_text
from rag_app.rag.graph.nodes import (
    make_contextualize_node, make_generate_node, make_grade_node, make_retrieve_node,
    make_route_after_grade, no_answer_node,
)
from rag_app.rag.graph.state import RAGState
from rag_app.rag.retrieval import HybridRetriever


def build_graph(settings: Settings, retriever: HybridRetriever, llm: BaseChatModel,
                checkpointer: BaseCheckpointSaver | None) -> CompiledStateGraph:
    g = StateGraph(RAGState)
    g.add_node("contextualize", make_contextualize_node(llm, settings.query_rewrite, settings.history_max_messages))
    g.add_node("retrieve", make_retrieve_node(retriever))
    g.add_node("grade", make_grade_node(settings.grader_mode, llm))
    g.add_node("generate", make_generate_node(llm, settings.history_max_messages))
    g.add_node("no_answer", no_answer_node)
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", make_route_after_grade(settings.min_relevant_docs), ["generate", "no_answer"])
    g.add_edge("generate", END)
    g.add_edge("no_answer", END)
    return g.compile(checkpointer=checkpointer)


class RAGPipeline:
    """Thin facade over the compiled graph used by the services."""

    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph

    @staticmethod
    def config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def invoke(self, question: str, session_id: str) -> RAGState:
        return self.graph.invoke({"question": question}, self.config(session_id))

    def stream(self, question: str, session_id: str):
        return self.graph.stream({"question": question}, self.config(session_id),
                                 stream_mode=["updates", "messages"])

    def history(self, session_id: str) -> list[ChatMessage]:
        snap = self.graph.get_state(self.config(session_id))
        msgs = (snap.values or {}).get("messages", []) if snap else []
        return [ChatMessage(role="user" if isinstance(m, HumanMessage) else "assistant",
                            content=message_text(m)) for m in msgs]

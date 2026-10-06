"""Graph state. Documents are stored as plain dicts so checkpoints stay serialisable."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # persisted chat history (Q/A only)
    question: str  # current user question
    query: str  # standalone retrieval query
    documents: list[dict[str, Any]]  # retrieved -> graded chunks
    answer: str
    sources: list[dict[str, Any]]


def doc_to_dict(doc: Document) -> dict[str, Any]:
    return {"id": doc.id, "page_content": doc.page_content, "metadata": dict(doc.metadata)}


def dict_to_doc(d: dict[str, Any]) -> Document:
    return Document(page_content=d["page_content"], metadata=d.get("metadata", {}), id=d.get("id"))

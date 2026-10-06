"""retrieve: hybrid (BM25 + vector, RRF) search for the standalone query."""

from __future__ import annotations

from rag_app.rag.graph.state import RAGState, doc_to_dict
from rag_app.rag.retrieval import HybridRetriever


def make_retrieve_node(retriever: HybridRetriever):
    def retrieve(state: RAGState) -> dict:
        docs = retriever.retrieve(state.get("query") or state["question"])
        return {"documents": [doc_to_dict(d) for d in docs]}

    return retrieve

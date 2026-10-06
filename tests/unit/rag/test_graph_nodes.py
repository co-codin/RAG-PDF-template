from langchain_core.messages import AIMessage, HumanMessage

from rag_app.infrastructure.llm import FakeChatModel
from rag_app.rag.graph.nodes import (
    build_sources, make_contextualize_node, make_generate_node, make_route_after_grade, no_answer_node,
)
from rag_app.rag.graph.state import dict_to_doc, doc_to_dict
from rag_app.rag.prompts import NO_ANSWER
from langchain_core.documents import Document


def test_contextualize_without_history_uses_question_and_resets_turn_fields():
    out = make_contextualize_node(FakeChatModel())({"question": "Q?", "messages": []})
    assert out == {"query": "Q?", "documents": [], "answer": "", "sources": []}


def test_contextualize_with_history_calls_rewriter():
    state = {"question": "And its speed?", "messages": [HumanMessage("Atlas-7 payload?"), AIMessage("18 kg [1]")]}
    assert make_contextualize_node(FakeChatModel())(state)["query"] == "And its speed?"


def test_route_after_grade():
    route = make_route_after_grade(1)
    assert route({"documents": []}) == "no_answer"
    assert route({"documents": [{"page_content": "x"}]}) == "generate"


def test_generate_cites_and_appends_messages():
    doc = Document(page_content="The Atlas-7 top speed is 2.4 meters per second.",
                   metadata={"source": "a.md", "chunk_id": "c1", "rrf_score": 0.03}, id="c1")
    out = make_generate_node(FakeChatModel())({"question": "Atlas-7 top speed?", "documents": [doc_to_dict(doc)]})
    assert "[1]" in out["answer"] and out["sources"][0]["cited"] is True
    assert [type(m) for m in out["messages"]] == [HumanMessage, AIMessage]


def test_no_answer_node():
    out = no_answer_node({"question": "?"})
    assert out["answer"] == NO_ANSWER and out["sources"] == []


def test_build_sources_marks_only_cited_refs():
    docs = [Document(page_content="a", metadata={"source": "s"}), Document(page_content="b", metadata={"source": "s"})]
    assert [s["cited"] for s in build_sources(docs, "x [2]")] == [False, True]


def test_doc_dict_roundtrip():
    d = Document(page_content="t", metadata={"k": 1}, id="i")
    assert dict_to_doc(doc_to_dict(d)) == d

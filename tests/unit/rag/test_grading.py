from langchain_core.documents import Document

from rag_app.infrastructure.llm import FakeChatModel
from rag_app.rag.grading import grade_documents, heuristic_grade


def docs():
    return [Document(page_content="The Atlas-7 battery lasts 11 hours.", metadata={"vector_score": 0.1}),
            Document(page_content="Employees get 25 days of PTO.", metadata={"vector_score": 0.1}),
            Document(page_content="Unrelated but semantically close.", metadata={"vector_score": 0.9})]


def test_heuristic_keeps_term_overlap_or_high_vector_score():
    kept = heuristic_grade("battery life", docs())
    assert [d.page_content for d in kept] == ["The Atlas-7 battery lasts 11 hours.", "Unrelated but semantically close."]


def test_none_mode_keeps_everything():
    assert len(grade_documents("none", "x", docs())) == 3


def test_llm_mode_falls_back_to_heuristic_when_structured_output_unsupported():
    kept = grade_documents("llm", "battery", docs(), llm=FakeChatModel())
    assert kept[0].page_content.startswith("The Atlas-7")

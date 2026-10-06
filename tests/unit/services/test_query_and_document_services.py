import pytest

from rag_app.core.errors import NotFoundError
from tests.support import SAMPLE_DOC


@pytest.fixture
def loaded(container):
    container.ingestion.ingest_bytes(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes())
    return container


def test_query_returns_domain_model_with_citations(loaded):
    res = loaded.query.query("What is the top speed of the Atlas-7?", "s1")
    assert "2.4 meters per second" in res.answer and any(s.cited for s in res.sources)


def test_stream_event_order(loaded):
    events = list(loaded.query.stream("How many PTO days do employees get?", "s2"))
    names = [e for e, _ in events]
    assert names[0] == "status" and names[-2:] == ["sources", "done"] and "token" in names
    assert "".join(p["text"] for e, p in events if e == "token").strip() == events[-1][1]["answer"]


def test_history_per_session(loaded):
    loaded.query.query("Who founded Northwind?", "a")
    loaded.query.query("Where is it based?", "a")
    assert [m.role for m in loaded.query.history("a")] == ["user", "assistant", "user", "assistant"]
    assert loaded.query.history("b") == []


def test_document_service_list_and_delete(loaded):
    assert loaded.documents.list()[0].source == SAMPLE_DOC.name
    assert loaded.documents.delete(SAMPLE_DOC.name) > 0
    with pytest.raises(NotFoundError):
        loaded.documents.delete(SAMPLE_DOC.name)
    assert "don't know" in loaded.query.query("Atlas-7 battery life?", "c").answer.lower()

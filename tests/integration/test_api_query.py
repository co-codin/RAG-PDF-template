import pytest
from fastapi.testclient import TestClient

from rag_app.main import create_app
from tests.support import SAMPLE_DOC, ingest_files, make_settings, parse_sse


def test_query_answers_with_citations(ingested_client):
    r = ingested_client.post("/query", json={"question": "How long does the Atlas-7 battery last?", "session_id": "s1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "11 hours" in body["answer"] and "[" in body["answer"]
    assert body["session_id"] == "s1" and body["latency_ms"] >= 0
    assert body["sources"] and all(s["source"] == SAMPLE_DOC.name for s in body["sources"])
    assert any(s["cited"] for s in body["sources"])


def test_query_without_relevant_context_says_unknown(ingested_client):
    body = ingested_client.post("/query", json={"question": "What is the capital of Mongolia?"}).json()
    assert "don't know" in body["answer"].lower() and body["sources"] == []
    assert len(body["session_id"]) == 32  # auto-generated


def test_query_on_empty_index_says_unknown(client):
    assert "don't know" in client.post("/query", json={"question": "Anything?"}).json()["answer"].lower()


def test_conversation_memory_per_session(ingested_client):
    c = ingested_client
    c.post("/query", json={"question": "What is the payload of the Atlas-7?", "session_id": "alice"})
    c.post("/query", json={"question": "And its top speed?", "session_id": "alice"})
    c.post("/query", json={"question": "How many PTO days do employees get?", "session_id": "bob"})
    alice = c.get("/sessions/alice/history").json()["messages"]
    bob = c.get("/sessions/bob/history").json()["messages"]
    assert [m["role"] for m in alice] == ["user", "assistant", "user", "assistant"]
    assert alice[0]["content"] == "What is the payload of the Atlas-7?"
    assert len(bob) == 2 and "PTO" in bob[0]["content"]


def test_invalid_session_id_in_path(client):
    assert client.get("/sessions/bad id!/history").status_code == 422


def test_memory_and_index_persist_across_restarts(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        ingest_files(c, [(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown")])
        c.post("/query", json={"question": "Who founded Northwind?", "session_id": "persist"})
    with TestClient(create_app(settings)) as c:
        assert len(c.get("/sessions/persist/history").json()["messages"]) == 2
        assert c.get("/health").json()["chunks"] > 0


def test_query_stream_sse(ingested_client):
    with ingested_client.stream("POST", "/query/stream",
                                json={"question": "What is the Atlas-7 top speed?", "session_id": "st"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse("".join(r.iter_text()))
    names = [e for e, _ in events]
    assert names[0] == "meta" and names[-1] == "done" and {"token", "sources", "status"} <= set(names)
    done = events[-1][1]
    assert "".join(p["text"] for e, p in events if e == "token").strip() == done["answer"]
    assert "2.4 meters per second" in done["answer"]
    assert any(s["cited"] for s in next(p for e, p in events if e == "sources")["sources"])
    assert len(ingested_client.get("/sessions/st/history").json()["messages"]) == 2


def test_query_stream_no_context(client):
    with client.stream("POST", "/query/stream", json={"question": "Unknown topic?"}) as r:
        assert "don't know" in parse_sse("".join(r.iter_text()))[-1][1]["answer"].lower()


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "ok", "session_id": "bad id!"}])
def test_query_validation(client, payload):
    r = client.post("/query", json=payload)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"

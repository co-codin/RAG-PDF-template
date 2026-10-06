import httpx
from fastapi.testclient import TestClient

from rag_app.main import create_app
from tests.support import SAMPLE_DOC, make_settings


def test_list_and_delete(ingested_client):
    c = ingested_client
    assert c.get("/documents").json()["documents"][0]["source"] == SAMPLE_DOC.name
    r = c.delete(f"/documents/{SAMPLE_DOC.name}")
    assert r.status_code == 200 and r.json()["deleted_chunks"] > 0
    assert c.get("/documents").json() == {"documents": [], "total_chunks": 0}
    assert c.delete(f"/documents/{SAMPLE_DOC.name}").status_code == 404
    # BM25 + vector indexes no longer return the deleted content.
    body = c.post("/query", json={"question": "How long does the Atlas-7 battery last?"}).json()
    assert "don't know" in body["answer"].lower()


def test_delete_url_source_with_slashes(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_app.rag.loaders.web.httpx.get", lambda url, **kw: httpx.Response(
        200, text="<html><body><p>Some page text about robots.</p></body></html>",
        headers={"content-type": "text/html"}, request=httpx.Request("GET", url)))
    with TestClient(create_app(make_settings(tmp_path, allow_private_urls=True))) as c:
        c.post("/ingest/url", json={"url": "https://docs.example.com/a/b"})
        r = c.delete("/documents/https://docs.example.com/a/b")
        assert r.status_code == 200 and r.json()["deleted_chunks"] == 1

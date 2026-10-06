def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["chunks"] == 0 and body["vector_store_ok"] is True
    assert body["vector_store"] == "chroma:embedded" and body["checkpointer"] == "sqlite"
    assert body["embeddings_provider"] == "hash" and body["llm_provider"] == "fake"
    assert "X-Request-ID" in r.headers and "X-Process-Time-Ms" in r.headers


def test_request_id_is_propagated(client):
    assert client.get("/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"


def test_openapi_lists_all_routes(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/health", "/ingest/files", "/ingest/url", "/ingest/jobs/{job_id}", "/query", "/query/stream",
            "/documents", "/documents/{source}", "/sessions/{session_id}/history"} <= paths

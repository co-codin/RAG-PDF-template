from fastapi.testclient import TestClient

from rag_app.main import create_app
from tests.support import make_settings


def test_api_key_auth(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, api_key="s3cret"))) as c:
        assert c.get("/health").status_code == 200  # public
        assert c.get("/documents").status_code == 401
        assert c.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.post("/query", json={"question": "hi"}).status_code == 401
        assert c.get("/ingest/jobs/x").status_code == 401


def test_unhandled_errors_return_json_500_with_request_id(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as c:
        def boom(*a, **k):
            raise RuntimeError("kaboom")
        app.state.container.query.query = boom
        r = c.post("/query", json={"question": "x"})
        assert r.status_code == 500
        assert r.json()["error"] == "internal_error" and "kaboom" not in r.text
        assert r.json()["request_id"] == r.headers["X-Request-ID"] != "-"

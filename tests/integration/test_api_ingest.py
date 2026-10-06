import httpx
from fastapi.testclient import TestClient

from rag_app.main import create_app
from tests.support import SAMPLE_DOC, ingest_files, make_settings, minimal_pdf


def test_files_job_and_idempotent_reingest(client):
    r = client.post("/ingest/files", files=[("files", (SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown"))])
    assert r.status_code == 202
    accepted = r.json()
    assert accepted["accepted"] == [SAMPLE_DOC.name] and accepted["status"] == "queued"
    assert accepted["status_url"] == f"/ingest/jobs/{accepted['job_id']}"
    job = client.get(accepted["status_url"]).json()
    assert job["status"] == "succeeded" and job["started_at"] and job["finished_at"]
    res1 = job["results"][0]
    assert res1["status"] == "ok" and res1["added"] == res1["chunks"] > 3

    res2 = ingest_files(client, [(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown")])["results"][0]
    assert res2["added"] == 0 and res2["skipped_existing"] == res1["chunks"]
    assert client.get("/documents").json()["total_chunks"] == res1["chunks"]


def test_reingest_changed_document_removes_stale_chunks(client):
    ingest_files(client, [("notes.txt", b"Version one says the sky is green.", "text/plain")])
    res = ingest_files(client, [("notes.txt", b"Version two says the sky is blue.", "text/plain")])["results"][0]
    assert res["added"] == 1 and res["removed_stale"] == 1
    assert client.get("/documents").json()["total_chunks"] == 1


def test_pdf_upload(client):
    job = ingest_files(client, [("spec.pdf", minimal_pdf(["Lidar stops it in 0.5 s.", "Warranty 24 months."]),
                                 "application/pdf")])
    assert job["results"][0]["chunks"] == 2
    doc = client.get("/documents").json()["documents"][0]
    assert (doc["source"], doc["pages"], doc["type"]) == ("spec.pdf", 2, "pdf")


def test_unsupported_type_rejected_synchronously(client):
    r = client.post("/ingest/files", files=[("files", ("malware.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]


def test_partial_upload_reports_rejected_and_job_is_partial(client):
    r = client.post("/ingest/files", files=[("files", ("ok.txt", b"Valid text about robots.", "text/plain")),
                                            ("files", ("bad.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] == ["ok.txt"] and body["rejected"][0]["source"] == "bad.exe"
    job = client.get(body["status_url"]).json()
    assert job["status"] == "partial"
    assert {x["source"]: x["status"] for x in job["results"]} == {"ok.txt": "ok", "bad.exe": "error"}


def test_corrupt_file_fails_inside_job(client):
    job = ingest_files(client, [("broken.pdf", b"not a pdf", "application/pdf")])
    assert job["status"] == "failed" and "Could not parse PDF" in job["results"][0]["error"]


def test_url_job(tmp_path, monkeypatch):
    html = ("<html><head><title>Robot FAQ</title></head><body><main>"
            "<p>The DockStation Pro recharges the Atlas-7 in 95 minutes.</p></main></body></html>")
    monkeypatch.setattr("rag_app.rag.loaders.web.httpx.get", lambda url, **kw: httpx.Response(
        200, text=html, headers={"content-type": "text/html"}, request=httpx.Request("GET", url)))
    with TestClient(create_app(make_settings(tmp_path, allow_private_urls=True))) as c:
        r = c.post("/ingest/url", json={"url": "https://docs.example.com/faq"})
        assert r.status_code == 202
        job = c.get(r.json()["status_url"]).json()
        assert job["status"] == "succeeded" and job["kind"] == "url"
        doc = c.get("/documents").json()["documents"][0]
        assert doc["title"] == "Robot FAQ" and doc["type"] == "web"


def test_url_ssrf_blocked_before_enqueue(client):
    r = client.post("/ingest/url", json={"url": "http://127.0.0.1:8000/health"})
    assert r.status_code == 400 and r.json()["error"] == "ingestion_error"


def test_unknown_job_404(client):
    r = client.get("/ingest/jobs/doesnotexist")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_upload_size_limit(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, max_upload_mb=1))) as c:
        r = c.post("/ingest/files", files=[("files", ("big.txt", b"a " * (600 * 1024), "text/plain"))])
        assert r.status_code == 400 and "limit" in r.json()["detail"]


def test_sqlite_job_store_backend(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, job_store_backend="sqlite"))) as c:
        job = ingest_files(c, [("a.txt", b"Shared job store for multiple workers.", "text/plain")])
        assert job["status"] == "succeeded"
    assert (tmp_path / "storage" / "jobs.sqlite").exists()

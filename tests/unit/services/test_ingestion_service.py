import pytest

from rag_app.core.errors import IngestionError
from rag_app.domain.models import JobStatus
from tests.support import SAMPLE_DOC


def test_ingest_bytes_is_idempotent(container, memory_store):
    svc = container.ingestion
    first = svc.ingest_bytes(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes())
    assert first.status == "ok" and first.added == first.chunks > 3
    second = svc.ingest_bytes(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes())
    assert second.added == 0 and second.skipped_existing == first.chunks
    assert memory_store.count() == first.chunks


def test_reingest_changed_source_removes_stale(container):
    svc = container.ingestion
    svc.ingest_bytes("n.txt", b"Version one says the sky is green.")
    res = svc.ingest_bytes("n.txt", b"Version two says the sky is blue.")
    assert (res.added, res.removed_stale) == (1, 1)


def test_validate_upload(container):
    svc = container.ingestion
    with pytest.raises(IngestionError):
        svc.validate_upload("x.exe", 10)
    with pytest.raises(IngestionError):
        svc.validate_upload("x.txt", 0)
    with pytest.raises(IngestionError):
        svc.validate_upload("x.txt", 26 * 1024 * 1024)
    svc.validate_upload("x.txt", 10)


def test_files_job_lifecycle_partial(container):
    svc = container.ingestion
    job = svc.create_job("files", ["ok.txt", "bad.pdf"])
    assert svc.get_job(job.id).status == JobStatus.queued
    svc.run_files_job(job.id, [("ok.txt", b"Robots are great."), ("bad.pdf", b"not a pdf")])
    done = svc.get_job(job.id)
    assert done.status == JobStatus.partial
    assert [r.status for r in done.results] == ["ok", "error"]
    assert done.started_at and done.finished_at and done.finished_at >= done.started_at


def test_url_job_failure_is_recorded(container):
    svc = container.ingestion
    job = svc.create_job("url", ["http://127.0.0.1/"])
    svc.run_url_job(job.id, "http://127.0.0.1/")
    done = svc.get_job(job.id)
    assert done.status == JobStatus.failed and "non-public" in done.results[0].error

import pytest

from rag_app.domain.models import IngestResult, Job, JobStatus
from rag_app.infrastructure.jobs import InMemoryJobStore
from rag_app.infrastructure.jobs.sqlite import SqliteJobStore


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    s = InMemoryJobStore(max_jobs=3) if request.param == "memory" else SqliteJobStore(tmp_path / "jobs.sqlite")
    yield s
    getattr(s, "close", lambda: None)()


def test_create_get_save_roundtrip(store):
    job = store.create(Job(id="j1", kind="files", inputs=["a.md"]))
    assert store.get("j1").status == JobStatus.queued
    job.status = JobStatus.succeeded
    job.results.append(IngestResult(source="a.md", status="ok", chunks=2))
    store.save(job)
    got = store.get("j1")
    assert got.status == JobStatus.succeeded and got.results[0].chunks == 2
    assert store.get("missing") is None


def test_returned_jobs_are_copies(store):
    store.create(Job(id="j2", kind="url"))
    store.get("j2").status = JobStatus.failed
    assert store.get("j2").status == JobStatus.queued


def test_memory_store_is_bounded():
    s = InMemoryJobStore(max_jobs=2)
    for i in range(3):
        s.create(Job(id=f"j{i}", kind="url"))
    assert s.get("j0") is None and s.get("j2") is not None


def test_sqlite_store_shared_between_instances(tmp_path):
    a, b = SqliteJobStore(tmp_path / "j.sqlite"), SqliteJobStore(tmp_path / "j.sqlite")
    a.create(Job(id="shared", kind="files"))
    assert b.get("shared") is not None  # e.g. two uvicorn workers on one host
    a.close(), b.close()

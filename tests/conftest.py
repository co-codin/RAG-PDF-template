"""Fixtures: a fully offline app (embedded Chroma in tmp dir, hash embeddings, fake LLM)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_app.container import build_container
from rag_app.main import create_app
from tests.support import SAMPLE_DOC, InMemoryVectorStore, ingest_files, make_settings


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def ingested_client(client):
    job = ingest_files(client, [(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown")])
    assert job["status"] == "succeeded", job
    return client


@pytest.fixture
def memory_store():
    return InMemoryVectorStore()


@pytest.fixture
def container(tmp_path, memory_store):
    """Container wired with the in-memory vector store + in-memory checkpointer."""
    c = build_container(make_settings(tmp_path, checkpointer_backend="memory"), vector_store=memory_store)
    yield c
    c.close()

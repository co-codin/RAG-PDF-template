import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.checkpointer import build_checkpointer
from rag_app.infrastructure.jobs import InMemoryJobStore, build_job_store
from rag_app.infrastructure.jobs.sqlite import SqliteJobStore
from rag_app.infrastructure.llm import FakeChatModel, build_chat_model
from rag_app.infrastructure.vectorstore import VectorStore, build_vector_store
from rag_app.infrastructure.embeddings import HashEmbeddings
from tests.support import InMemoryVectorStore, make_settings


def test_checkpointer_backends(tmp_path):
    h = build_checkpointer(make_settings(tmp_path))
    assert isinstance(h.saver, SqliteSaver) and h.backend == "sqlite"
    h.close()
    assert isinstance(build_checkpointer(make_settings(tmp_path, checkpointer_backend="memory")).saver, InMemorySaver)


def test_postgres_requires_url(tmp_path):
    with pytest.raises(ConfigurationError):
        build_checkpointer(make_settings(tmp_path, checkpointer_backend="postgres"))


def test_job_store_backends(tmp_path):
    assert isinstance(build_job_store(make_settings(tmp_path)), InMemoryJobStore)
    store = build_job_store(make_settings(tmp_path, job_store_backend="sqlite"))
    assert isinstance(store, SqliteJobStore)
    store.close()


def test_llm_factory(tmp_path):
    assert isinstance(build_chat_model(make_settings(tmp_path)), FakeChatModel)


def test_vector_store_protocol_conformance(tmp_path):
    assert isinstance(InMemoryVectorStore(), VectorStore)
    s = make_settings(tmp_path)
    assert isinstance(build_vector_store(s, HashEmbeddings(s.hash_embedding_dim)), VectorStore)

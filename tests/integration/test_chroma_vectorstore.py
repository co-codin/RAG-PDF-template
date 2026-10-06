"""Embedded Chroma backend against a real (temporary) persistent directory."""

import pytest
from langchain_core.documents import Document

from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.embeddings import HashEmbeddings
from rag_app.infrastructure.vectorstore.chroma import ChromaVectorStore
from rag_app.rag.chunking import chunk_documents
from rag_app.rag.retrieval import BM25Index
from tests.support import make_settings


def chunks(text, source="a.txt"):
    return chunk_documents([Document(page_content=text, metadata={"source": source})], 200, 20).chunks


def make(settings, sig="hash:64"):
    return ChromaVectorStore(settings, HashEmbeddings(64), signature=sig)


def test_sync_search_list_delete_and_revision(tmp_path):
    store = make(make_settings(tmp_path))
    r0 = store.revision()
    assert store.sync_source("a.txt", chunks("The Atlas-7 battery lasts 11 hours.")) == {
        "added": 1, "skipped_existing": 0, "removed_stale": 0}
    r1 = store.revision()
    assert r1 != r0
    assert store.sync_source("a.txt", chunks("The Atlas-7 battery lasts 11 hours."))["skipped_existing"] == 1
    assert store.revision() == r1, "no-op sync must not invalidate caches"
    doc, score = store.similarity_search("battery hours", 1)[0]
    assert "11 hours" in doc.page_content and 0 < score <= 1 and doc.id == doc.metadata["chunk_id"]
    assert store.list_documents()[0].source == "a.txt"
    assert store.delete_source("a.txt") == 1 and store.count() == 0 and store.revision() != r1


def test_cross_instance_bm25_invalidation(tmp_path):
    """Two store instances on the same collection simulate two workers: a write through
    one is picked up by the other's BM25 index via the shared revision."""
    s = make_settings(tmp_path)
    worker_a, worker_b = make(s), make(s)
    bm25_b = BM25Index(worker_b, refresh_s=0)
    assert bm25_b.search("zebra", 3) == []
    worker_a.sync_source("z.txt", chunks("A zebra crossing near the office.", "z.txt"))
    assert bm25_b.search("zebra", 3)


def test_embedding_signature_mismatch_refused(tmp_path):
    s = make_settings(tmp_path)
    make(s, "hash:64").sync_source("a.txt", chunks("hello world content"))
    with pytest.raises(ConfigurationError):
        make(s, "openai:text-embedding-3-small")

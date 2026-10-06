from langchain_core.documents import Document

from rag_app.rag.chunking import chunk_documents
from rag_app.rag.loaders import load_path
from rag_app.rag.retrieval import BM25Index, HybridRetriever
from tests.support import SAMPLE_DOC, InMemoryVectorStore


def _store_with_sample():
    store = InMemoryVectorStore()
    store.sync_source(SAMPLE_DOC.name, chunk_documents(load_path(SAMPLE_DOC), 400, 50).chunks)
    return store


def test_hybrid_ranks_relevant_chunk_first():
    store = _store_with_sample()
    docs = HybridRetriever(store, BM25Index(store, refresh_s=0)).retrieve("warranty for batteries charge cycles")
    assert docs and "1,500 charge cycles" in docs[0].page_content
    assert {"rrf_score", "chunk_id", "source"} <= docs[0].metadata.keys()
    assert "bm25_rank" in docs[0].metadata or "vector_rank" in docs[0].metadata


def test_bm25_rebuilds_when_revision_changes():
    store = InMemoryVectorStore()
    bm25 = BM25Index(store, refresh_s=0)
    assert bm25.search("zebra", 5) == []
    doc = Document(page_content="The zebra crossing is near the Rotterdam office.", metadata={"source": "z.txt"})
    store.sync_source("z.txt", chunk_documents([doc], 200, 20).chunks)
    hits = bm25.search("zebra", 5)
    assert hits and "zebra" in hits[0][0].page_content


def test_bm25_respects_refresh_interval_until_invalidated():
    store = _store_with_sample()
    bm25 = BM25Index(store, refresh_s=3600)
    assert bm25.search("payload", 3)
    store.delete_source(SAMPLE_DOC.name)
    assert bm25.search("payload", 3), "cached until the refresh interval elapses"
    bm25.invalidate()
    assert bm25.search("payload", 3) == []

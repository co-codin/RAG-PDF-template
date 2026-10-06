import pytest
from langchain_core.documents import Document

from rag_app.rag.chunking import chunk_documents


def _doc(text, **meta):
    return Document(page_content=text, metadata={"source": "test.txt", **meta})


def test_chunks_respect_size_and_carry_metadata():
    text = " ".join(f"Sentence number {i} talks about robots and batteries." for i in range(200))
    res = chunk_documents([_doc(text, page=3)], chunk_size=300, chunk_overlap=50)
    assert len(res.chunks) > 5
    for i, c in enumerate(res.chunks):
        assert len(c.page_content) <= 300
        assert c.metadata["source"] == "test.txt" and c.metadata["page"] == 3
        assert c.metadata["chunk_index"] == i
        assert c.metadata["chunk_id"] == c.metadata["content_hash"][:32] == c.id
        assert "ingested_at" in c.metadata and "start_index" in c.metadata


def test_chunk_overlap_present():
    chunks = chunk_documents([_doc(" ".join(f"word{i}" for i in range(400)))], 200, 60).chunks
    assert all(w in chunks[1].page_content for w in chunks[0].page_content.split()[-3:])


def test_dedupe_by_content_hash():
    para = "The Atlas-7 robot has an 18 kg payload."
    docs = [_doc(para, page=1), _doc(para, page=2), _doc("  the atlas-7 ROBOT has an 18 kg   payload. ", page=3)]
    res = chunk_documents(docs, 200, 20)
    assert len(res.chunks) == 1 and res.duplicates_skipped == 2


def test_chunk_ids_are_deterministic():
    docs = [_doc("alpha beta gamma " * 50)]
    assert [c.id for c in chunk_documents(docs, 120, 20).chunks] == [c.id for c in chunk_documents(docs, 120, 20).chunks]


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_documents([_doc("x")], chunk_size=100, chunk_overlap=100)

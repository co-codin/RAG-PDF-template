"""Recursive chunking + content-hash de-duplication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.domain.text import content_hash


@dataclass
class ChunkingResult:
    chunks: list[Document]
    duplicates_skipped: int


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> ChunkingResult:
    """Split page-level documents into chunks with metadata: ``source``, ``page`` (if
    known), ``chunk_index`` (per source), ``chunk_id`` (content-hash prefix, used as the
    vector-store id), ``content_hash``, ``start_index``, ``ingested_at``.
    Chunks with identical normalised content are emitted once."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""])
    seen: set[str] = set()
    per_source: dict[str, int] = {}
    out: list[Document] = []
    dupes = 0
    ingested_at = datetime.now(timezone.utc).isoformat()
    for chunk in splitter.split_documents(docs):
        text = chunk.page_content.strip()
        if not text:
            continue
        h = content_hash(text)
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        source = chunk.metadata.get("source", "unknown")
        idx = per_source.get(source, 0)
        per_source[source] = idx + 1
        meta = {k: v for k, v in chunk.metadata.items() if v is not None}
        meta.update({"chunk_index": idx, "chunk_id": h[:32], "content_hash": h, "ingested_at": ingested_at})
        out.append(Document(page_content=text, metadata=meta, id=h[:32]))
    return ChunkingResult(chunks=out, duplicates_skipped=dupes)

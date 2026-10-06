"""Shared test helpers and fakes (imported by conftest.py and tests)."""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

from rag_app.core.config import Settings
from rag_app.domain.models import DocumentInfo
from rag_app.infrastructure.embeddings import HashEmbeddings

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOC = ROOT / "data" / "northwind_handbook.md"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    """Offline settings: hash embeddings, fake LLM, heuristic grader, temp storage."""
    base = dict(
        _env_file=None,  # never read the developer's .env in tests
        environment="test", data_dir=tmp_path / "storage",
        embeddings_provider="hash", llm_provider="fake", grader_mode="heuristic",
        chunk_size=400, chunk_overlap=50, log_level="WARNING",
        api_key=None, anthropic_api_key=None, openai_api_key=None, chroma_host=None,
        bm25_refresh_s=0.0,
    )
    base.update(overrides)
    return Settings(**base)


class InMemoryVectorStore:
    """Pure-Python implementation of the VectorStore Protocol - proves backends are
    swappable and keeps service unit tests free of Chroma."""

    def __init__(self, dim: int = 256) -> None:
        self.emb = HashEmbeddings(dim)
        self.docs: dict[str, tuple[Document, list[float]]] = {}
        self._rev = "0"

    def _bump(self) -> None:
        self._rev = uuid.uuid4().hex

    def revision(self) -> str:
        return self._rev

    def sync_source(self, source, chunks):
        old = {i for i, (d, _) in self.docs.items() if d.metadata.get("source") == source}
        new_ids = {c.metadata["chunk_id"] for c in chunks}
        stale = old - new_ids
        for i in stale:
            del self.docs[i]
        added = skipped = 0
        for c in chunks:
            cid = c.metadata["chunk_id"]
            if cid in self.docs:
                skipped += 1
            else:
                self.docs[cid] = (c, self.emb.embed_query(c.page_content))
                added += 1
        if stale or added:
            self._bump()
        return {"added": added, "skipped_existing": skipped, "removed_stale": len(stale)}

    def delete_source(self, source):
        ids = [i for i, (d, _) in self.docs.items() if d.metadata.get("source") == source]
        for i in ids:
            del self.docs[i]
        if ids:
            self._bump()
        return len(ids)

    def similarity_search(self, query, k):
        q = self.emb.embed_query(query)
        scored = [(d, sum(a * b for a, b in zip(q, v))) for d, v in self.docs.values()]
        scored.sort(key=lambda x: -x[1])
        return [(Document(page_content=d.page_content, metadata=dict(d.metadata), id=d.id), max(0.0, s))
                for d, s in scored[:k]]

    def all_chunks(self):
        return [d for d, _ in self.docs.values()]

    def list_documents(self):
        agg = defaultdict(int)
        for d, _ in self.docs.values():
            agg[d.metadata["source"]] += 1
        return [DocumentInfo(source=s, chunks=n) for s, n in sorted(agg.items())]

    def count(self):
        return len(self.docs)

    def healthcheck(self):
        return True


def minimal_pdf(pages: list[str]) -> bytes:
    """Tiny valid PDF with one text line per page (no external deps)."""
    objs: list[bytes] = []
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 "
                    f"{font_id} 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode())
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def ingest_files(client, files: list[tuple[str, bytes, str]]) -> dict:
    """POST /ingest/files and return the finished job (TestClient runs background tasks
    before returning, so one GET is enough)."""
    r = client.post("/ingest/files", files=[("files", f) for f in files])
    assert r.status_code == 202, r.text
    job = client.get(r.json()["status_url"])
    assert job.status_code == 200, job.text
    return job.json()


def cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

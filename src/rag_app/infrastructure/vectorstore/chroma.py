"""Chroma backend.

* embedded (default): ``PersistentClient`` on a local dir - one process only.
* server: ``HttpClient`` when CHROMA_HOST is set - safe for many workers/replicas.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_app.core.config import Settings
from rag_app.core.errors import ConfigurationError
from rag_app.domain.models import DocumentInfo

logger = logging.getLogger(__name__)


def build_chroma_client(settings: Settings) -> chromadb.ClientAPI:
    chroma_settings = ChromaSettings(anonymized_telemetry=False)
    if settings.chroma_mode == "embedded":
        settings.resolved_chroma_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(settings.resolved_chroma_dir), settings=chroma_settings)
    headers = {}
    if settings.chroma_auth_token is not None:
        headers["Authorization"] = f"Bearer {settings.chroma_auth_token.get_secret_value()}"
    last_exc: Exception | None = None
    for attempt in range(1, settings.chroma_connect_retries + 1):  # server may still be booting
        try:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                                         ssl=settings.chroma_ssl, headers=headers or None,
                                         settings=chroma_settings)
            client.heartbeat()
            logger.info("chroma_connected", extra={"host": settings.chroma_host, "port": settings.chroma_port})
            return client
        except Exception as exc:  # connection refused, DNS, ...
            last_exc = exc
            logger.warning("chroma_connect_retry", extra={"attempt": attempt, "error": str(exc)[:200]})
            time.sleep(min(2.0, 0.25 * attempt))
    raise ConfigurationError(f"Cannot reach Chroma at {settings.chroma_host}:{settings.chroma_port}: {last_exc}")


class ChromaVectorStore:
    """Chunk ids are content hashes, so re-adding identical content is a no-op."""

    def __init__(self, settings: Settings, embeddings: Embeddings, signature: str,
                 client: chromadb.ClientAPI | None = None) -> None:
        self.settings = settings
        self.signature = signature
        self._lock = threading.RLock()  # in-process write serialisation
        self.client = client or build_chroma_client(settings)
        self.store = Chroma(
            client=self.client,
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            collection_configuration={"hnsw": {"space": "cosine"}},
            collection_metadata={"embeddings": signature},
        )
        existing = self._metadata().get("embeddings")
        if existing and existing != signature:
            raise ConfigurationError(
                f"Collection '{settings.collection_name}' was built with embeddings '{existing}' "
                f"but the current provider is '{signature}'. Use a different COLLECTION_NAME "
                "or re-ingest into a fresh store.")

    # -- revision (cross-process cache invalidation) ---------------------------
    def _metadata(self) -> dict[str, Any]:
        return dict(self.client.get_collection(self.settings.collection_name).metadata or {})

    def _bump_revision(self) -> None:
        self.store._collection.modify(metadata={"embeddings": self.signature, "revision": uuid.uuid4().hex})

    def revision(self) -> str:
        return str(self._metadata().get("revision", "0"))

    def healthcheck(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False

    # -- writes ---------------------------------------------------------------
    def _add_chunks(self, chunks: list[Document]) -> tuple[int, int]:
        ids = [c.metadata["chunk_id"] for c in chunks]
        existing = set(self.store.get(ids=ids, include=[])["ids"]) if ids else set()
        new = [c for c in chunks if c.metadata["chunk_id"] not in existing]
        if new:
            self.store.add_documents(new, ids=[c.metadata["chunk_id"] for c in new])
        return len(new), len(chunks) - len(new)

    def sync_source(self, source: str, chunks: list[Document]) -> dict[str, int]:
        with self._lock:
            old_ids = set(self.store.get(where={"source": source}, include=[])["ids"])
            stale = sorted(old_ids - {c.metadata["chunk_id"] for c in chunks})
            if stale:
                self.store.delete(ids=stale)
            added, skipped = self._add_chunks(chunks)
            if stale or added:
                self._bump_revision()
            return {"added": added, "skipped_existing": skipped, "removed_stale": len(stale)}

    def delete_source(self, source: str) -> int:
        with self._lock:
            ids = self.store.get(where={"source": source}, include=[])["ids"]
            if ids:
                self.store.delete(ids=ids)
                self._bump_revision()
            return len(ids)

    # -- reads ----------------------------------------------------------------
    def count(self) -> int:
        return self.store._collection.count()

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        n = self.count()
        if n == 0:
            return []
        out = []
        for doc, distance in self.store.similarity_search_with_score(query, k=min(k, n)):
            doc.id = doc.metadata.get("chunk_id", doc.id)
            out.append((doc, max(0.0, 1.0 - float(distance))))  # cosine distance -> similarity
        return out

    def all_chunks(self) -> list[Document]:
        data = self.store.get(include=["documents", "metadatas"])
        return [Document(page_content=t, metadata=m or {}, id=i)
                for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])]

    def list_documents(self) -> list[DocumentInfo]:
        agg: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"chunks": 0, "pages": set(), "type": None, "title": None, "ingested_at": None})
        for meta in self.store.get(include=["metadatas"])["metadatas"]:
            meta = meta or {}
            e = agg[meta.get("source", "unknown")]
            e["chunks"] += 1
            if "page" in meta:
                e["pages"].add(meta["page"])
            e["type"] = e["type"] or meta.get("type")
            e["title"] = e["title"] or meta.get("title")
            ts = meta.get("ingested_at")
            if ts and (e["ingested_at"] is None or ts > e["ingested_at"]):
                e["ingested_at"] = ts
        return [DocumentInfo(source=src, chunks=e["chunks"], pages=len(e["pages"]) or None,
                             type=e["type"], title=e["title"], ingested_at=e["ingested_at"])
                for src, e in sorted(agg.items())]

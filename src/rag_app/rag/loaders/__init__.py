"""Loaders turn raw inputs into page-level Documents (metadata: source, page?, type).

Add a format by writing ``load_<fmt>_bytes(data, source)`` and registering its
extension in ``LOADERS``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document

from rag_app.core.errors import IngestionError
from rag_app.rag.loaders.pdf import load_pdf_bytes
from rag_app.rag.loaders.text import load_markdown_bytes, load_text_bytes
from rag_app.rag.loaders.web import html_to_documents, load_url, validate_url

Loader = Callable[[bytes, str], list[Document]]

LOADERS: dict[str, Loader] = {
    ".pdf": load_pdf_bytes,
    ".txt": load_text_bytes,
    ".md": load_markdown_bytes,
    ".markdown": load_markdown_bytes,
}

__all__ = ["LOADERS", "html_to_documents", "load_file_bytes", "load_path", "load_pdf_bytes",
           "load_text_bytes", "load_url", "validate_extension", "validate_url"]


def validate_extension(filename: str, allowed_extensions: list[str] | None = None) -> str:
    ext = Path(filename).suffix.lower()
    allowed = allowed_extensions if allowed_extensions is not None else list(LOADERS)
    if ext not in allowed or ext not in LOADERS:
        raise IngestionError(f"Unsupported file type '{ext}' for '{filename}'. Allowed: {', '.join(allowed)}")
    return ext


def load_file_bytes(data: bytes, filename: str, allowed_extensions: list[str] | None = None) -> list[Document]:
    return LOADERS[validate_extension(filename, allowed_extensions)](data, filename)


def load_path(path: str | Path) -> list[Document]:
    p = Path(path)
    return load_file_bytes(p.read_bytes(), p.name)

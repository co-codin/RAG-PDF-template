"""PDF loader (pypdf): one Document per non-empty page, with 1-based ``page``."""

from __future__ import annotations

import io

from langchain_core.documents import Document
from pypdf import PdfReader

from rag_app.core.errors import IngestionError
from rag_app.domain.text import normalize_text


def load_pdf_bytes(data: bytes, source: str) -> list[Document]:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = list(reader.pages)
    except Exception as exc:  # pypdf raises a variety of errors
        raise IngestionError(f"Could not parse PDF '{source}': {exc}") from exc
    docs = []
    for i, page in enumerate(pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            docs.append(Document(page_content=text, metadata={"source": source, "page": i, "type": "pdf"}))
    return docs

"""Plain text / Markdown loader."""

from __future__ import annotations

from langchain_core.documents import Document

from rag_app.domain.text import normalize_text


def load_text_bytes(data: bytes, source: str, kind: str = "text") -> list[Document]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    text = normalize_text(text)
    return [Document(page_content=text, metadata={"source": source, "type": kind})] if text else []


def load_markdown_bytes(data: bytes, source: str) -> list[Document]:
    return load_text_bytes(data, source, kind="markdown")

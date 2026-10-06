"""Web page loader (httpx + BeautifulSoup) with a basic SSRF guard."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from rag_app.core.errors import IngestionError
from rag_app.domain.text import normalize_text
from rag_app.rag.loaders.pdf import load_pdf_bytes
from rag_app.rag.loaders.text import load_text_bytes


def html_to_documents(html: str, url: str) -> list[Document]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = normalize_text(main.get_text(separator="\n"))
    if not text:
        return []
    meta = {"source": url, "type": "web"}
    if title:
        meta["title"] = title
    return [Document(page_content=text, metadata=meta)]


def assert_public_host(host: str) -> None:
    """Refuse hosts resolving to private/loopback/link-local/reserved addresses."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise IngestionError(f"Cannot resolve host '{host}': {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise IngestionError(f"Refusing to fetch non-public address {ip} for host '{host}'")


def validate_url(url: str, allow_private: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestionError(f"Invalid URL: {url!r} (only http/https are supported)")
    if not allow_private:
        assert_public_host(parsed.hostname)


def load_url(url: str, timeout_s: float = 20.0, user_agent: str = "rag-production",
             allow_private: bool = False) -> list[Document]:
    validate_url(url, allow_private)
    try:
        resp = httpx.get(url, timeout=timeout_s, follow_redirects=True, headers={"User-Agent": user_agent})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise IngestionError(f"Failed to fetch {url}: {exc}") from exc
    ctype = resp.headers.get("content-type", "")
    if "pdf" in ctype:
        return load_pdf_bytes(resp.content, url)
    if "html" in ctype or "xml" in ctype or not ctype:
        return html_to_documents(resp.text, url)
    if ctype.startswith("text/"):
        return load_text_bytes(resp.content, url, kind="web")
    raise IngestionError(f"Unsupported content-type '{ctype}' at {url}")

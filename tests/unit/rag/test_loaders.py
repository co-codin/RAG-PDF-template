import pytest

from rag_app.core.errors import IngestionError
from rag_app.rag.chunking import chunk_documents
from rag_app.rag.loaders import LOADERS, html_to_documents, load_file_bytes, load_pdf_bytes, load_url
from tests.support import SAMPLE_DOC, minimal_pdf


def test_registry_covers_default_extensions():
    assert {".pdf", ".txt", ".md"} <= set(LOADERS)


def test_load_markdown_file():
    docs = load_file_bytes(SAMPLE_DOC.read_bytes(), SAMPLE_DOC.name, [".md", ".txt", ".pdf"])
    assert len(docs) == 1 and docs[0].metadata == {"source": SAMPLE_DOC.name, "type": "markdown"}


def test_load_pdf_pages_have_page_numbers():
    docs = load_pdf_bytes(minimal_pdf(["First page about lidar safety.", "Second page about warranty."]), "d.pdf")
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "lidar" in docs[0].page_content and "warranty" in docs[1].page_content
    assert [c.metadata["page"] for c in chunk_documents(docs, 200, 20).chunks] == [1, 2]


def test_unsupported_extension_rejected():
    with pytest.raises(IngestionError):
        load_file_bytes(b"data", "evil.exe", [".pdf", ".txt", ".md"])


def test_extension_must_also_be_allowed_by_config():
    with pytest.raises(IngestionError):
        load_file_bytes(b"x", "a.pdf", [".txt"])


def test_bad_pdf_rejected():
    with pytest.raises(IngestionError):
        load_pdf_bytes(b"not a pdf", "broken.pdf")


def test_html_to_documents_strips_boilerplate():
    html = ("<html><head><title>Docs</title><script>var x=1;</script></head><body>"
            "<nav>Menu</nav><main><h1>Hello</h1><p>Real content here.</p></main></body></html>")
    doc = html_to_documents(html, "https://example.com/x")[0]
    assert doc.metadata["title"] == "Docs" and "Real content here." in doc.page_content
    assert "var x" not in doc.page_content and "Menu" not in doc.page_content


@pytest.mark.parametrize("url", ["ftp://example.com/a", "not-a-url", "http://127.0.0.1/admin",
                                 "http://localhost:8000/", "http://169.254.169.254/latest/meta-data"])
def test_load_url_rejects_invalid_or_private(url):
    with pytest.raises(IngestionError):
        load_url(url)

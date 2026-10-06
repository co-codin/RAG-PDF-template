from rag_app.domain.text import content_hash, normalize_text, tokenize


def test_content_hash_normalises_whitespace_and_case():
    assert content_hash("Hello   World") == content_hash("hello world\n")


def test_normalize_text():
    assert normalize_text("a \t b\r\n\n\n\nc\x00") == "a b\n\nc"


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("How long does the battery last? It lasts 11 hours") == ["long", "battery", "last", "last", "11", "hour"]

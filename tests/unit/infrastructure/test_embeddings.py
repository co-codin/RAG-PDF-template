import math

import pytest

from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.embeddings import HashEmbeddings, build_embeddings, embedding_signature
from tests.support import cos, make_settings


def test_hash_embeddings_deterministic_and_normalised():
    v1 = HashEmbeddings(256).embed_query("battery life of the Atlas-7")
    assert v1 == HashEmbeddings(256).embed_documents(["battery life of the Atlas-7"])[0]
    assert len(v1) == 256 and math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-9)


def test_hash_embeddings_capture_lexical_similarity():
    e = HashEmbeddings(512)
    q = e.embed_query("how long does the battery last")
    assert cos(q, e.embed_query("the battery lasts 11 hours")) > cos(q, e.embed_query("employees get 25 PTO days"))


def test_empty_text_not_zero_vector():
    assert any(HashEmbeddings(8).embed_query(""))


def test_factory_and_signature(tmp_path):
    s = make_settings(tmp_path, hash_embedding_dim=64)
    assert isinstance(build_embeddings(s), HashEmbeddings) and embedding_signature(s) == "hash:64"


def test_openai_requires_key(tmp_path):
    with pytest.raises(ConfigurationError):
        build_embeddings(make_settings(tmp_path, embeddings_provider="openai"))

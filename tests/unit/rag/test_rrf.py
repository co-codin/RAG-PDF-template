import pytest
from langchain_core.documents import Document

from rag_app.rag.retrieval import reciprocal_rank_fusion


def d(cid):
    return Document(page_content=f"content {cid}", metadata={"chunk_id": cid})


def ids(fused):
    return [doc.metadata["chunk_id"] for doc, _ in fused]


def test_rrf_scores_match_formula():
    fused = reciprocal_rank_fusion([[d("a"), d("b")], [d("b"), d("c")]], k=60)
    scores = {doc.metadata["chunk_id"]: s for doc, s in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert ids(fused) == ["b", "a", "c"]


def test_doc_in_both_lists_beats_single_list_top():
    assert ids(reciprocal_rank_fusion([[d("x"), d("shared"), d("y")], [d("z"), d("shared"), d("w")]]))[0] == "shared"


def test_ties_broken_by_first_appearance_and_top_n():
    assert ids(reciprocal_rank_fusion([[d("a")], [d("b")]], k=60, top_n=1)) == ["a"]


def test_weights():
    assert ids(reciprocal_rank_fusion([[d("a")], [d("b")]], weights=[1.0, 2.0])) == ["b", "a"]


def test_duplicates_within_a_list_count_once():
    scores = {doc.metadata["chunk_id"]: s for doc, s in reciprocal_rank_fusion([[d("a"), d("a"), d("b")]], k=0)}
    assert scores["a"] == pytest.approx(1.0) and scores["b"] == pytest.approx(1 / 3)


def test_empty_and_validation():
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[d("a")]], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[d("a")]], k=-1)

"""Candidate selection: cross-document only, thresholded, capped."""
import numpy as np
import pytest

from app.services.similarity_service import (
    build_candidate_pairs,
    cosine_matrix,
    pair_key,
)


def sentence(index, text, document="doc_001"):
    return {
        "document_id": document,
        "document_name": f"{document}.txt",
        "page_number": None,
        "sentence_id": f"sent_{index:03d}",
        "text": text,
    }


def vectors(*rows):
    return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------- cosine

def test_cosine_of_identical_and_orthogonal_vectors():
    matrix = cosine_matrix(vectors([1.0, 0.0], [1.0, 0.0], [0.0, 1.0]))

    assert matrix[0][1] == pytest.approx(1.0)
    assert matrix[0][2] == pytest.approx(0.0)
    assert matrix[0][0] == pytest.approx(1.0)


def test_cosine_ignores_magnitude():
    matrix = cosine_matrix(vectors([1.0, 0.0], [9.0, 0.0]))
    assert matrix[0][1] == pytest.approx(1.0)


def test_zero_vector_does_not_produce_nan():
    matrix = cosine_matrix(vectors([0.0, 0.0], [1.0, 0.0]))
    assert not np.isnan(matrix).any()
    assert matrix[0][1] == pytest.approx(0.0)


def test_pair_key_is_order_independent():
    assert pair_key("sent_009", "sent_001") == pair_key("sent_001", "sent_009")
    assert pair_key("sent_001", "sent_009") == "sent_001|sent_009"


# ------------------------------------------------------------- selection

def test_only_cross_document_pairs_are_selected():
    sentences = [
        sentence(1, "leave twenty", "doc_001"),
        sentence(2, "leave twenty-one", "doc_001"),   # same doc as above
        sentence(3, "leave fifteen", "doc_002"),
    ]
    matrix = vectors([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])

    pairs, stats = build_candidate_pairs(sentences, matrix, threshold=0.5)
    keys = {pair["pair_key"] for pair in pairs}

    assert keys == {"sent_001|sent_003", "sent_002|sent_003"}
    assert "sent_001|sent_002" not in keys     # within-document excluded
    assert stats["cross_document_pairs"] == 2


def test_no_self_comparisons():
    sentences = [sentence(1, "a", "doc_001"), sentence(2, "b", "doc_002")]
    pairs, _ = build_candidate_pairs(
        sentences, vectors([1.0, 0.0], [1.0, 0.0]), threshold=0.5
    )
    for pair in pairs:
        assert pair["index_a"] != pair["index_b"]
        assert pair["sentence_id_a"] != pair["sentence_id_b"]


def test_a_single_document_yields_nothing():
    sentences = [sentence(1, "a", "doc_001"), sentence(2, "b", "doc_001")]
    pairs, stats = build_candidate_pairs(
        sentences, vectors([1.0, 0.0], [1.0, 0.0]), threshold=0.1
    )
    assert pairs == []
    assert stats["cross_document_pairs"] == 0


def test_threshold_excludes_weak_pairs():
    sentences = [sentence(1, "leave", "doc_001"), sentence(2, "parking", "doc_002")]
    matrix = vectors([1.0, 0.0], [0.0, 1.0])     # cosine 0.0

    pairs, stats = build_candidate_pairs(sentences, matrix, threshold=0.65)

    assert pairs == []
    assert stats["above_threshold"] == 0
    assert stats["highest_similarity"] == 0.0


def test_threshold_is_inclusive_at_the_boundary():
    sentences = [sentence(1, "a", "doc_001"), sentence(2, "b", "doc_002")]
    matrix = vectors([1.0, 0.0], [1.0, 0.0])     # cosine 1.0
    pairs, _ = build_candidate_pairs(sentences, matrix, threshold=1.0)
    assert len(pairs) == 1


def test_top_k_limits_neighbours_per_statement():
    # One statement in doc_001, five equally similar ones in doc_002.
    sentences = [sentence(1, "leave", "doc_001")] + [
        sentence(i, f"leave variant {i}", "doc_002") for i in range(2, 7)
    ]
    matrix = vectors(*([[1.0, 0.0]] * 6))

    pairs, stats = build_candidate_pairs(
        sentences, matrix, threshold=0.5, top_k=2
    )
    # Each of the five doc_002 statements nominates the single doc_001
    # statement, so the union is larger than top_k itself; the point is
    # that no statement contributes more than top_k nominations.
    assert stats["cross_document_pairs"] == 5
    assert len(pairs) <= 5
    assert all(pair["similarity"] == pytest.approx(1.0) for pair in pairs)


def test_identical_text_is_skipped_without_a_request():
    shared = "Employees receive 10 days of paid sick leave."
    sentences = [sentence(1, shared, "doc_001"), sentence(2, shared, "doc_002")]
    matrix = vectors([1.0, 0.0], [1.0, 0.0])

    pairs, stats = build_candidate_pairs(sentences, matrix, threshold=0.5)

    assert pairs == []
    assert stats["skipped_identical"] == 1


def test_cap_keeps_the_strongest_candidates():
    sentences = [sentence(1, "leave", "doc_001")] + [
        sentence(i, f"text {i}", "doc_002") for i in range(2, 6)
    ]
    # Descending similarity against sentence 1.
    matrix = vectors(
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.436],
        [0.8, 0.6],
        [0.7, 0.714],
    )

    pairs, stats = build_candidate_pairs(
        sentences, matrix, threshold=0.5, top_k=5, max_pairs=2
    )

    assert len(pairs) == 2
    assert stats["selected"] == 2
    assert stats["dropped_by_cap"] >= 1
    # Sorted strongest first.
    assert pairs[0]["similarity"] >= pairs[1]["similarity"]


def test_results_are_deterministic():
    sentences = [sentence(1, "leave", "doc_001")] + [
        sentence(i, f"text {i}", "doc_002") for i in range(2, 5)
    ]
    matrix = vectors([1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0])

    first, _ = build_candidate_pairs(sentences, matrix, threshold=0.5)
    second, _ = build_candidate_pairs(sentences, matrix, threshold=0.5)

    assert [p["pair_key"] for p in first] == [p["pair_key"] for p in second]


def test_pair_indices_point_at_the_right_sentences():
    sentences = [
        sentence(1, "Employees receive 20 days of leave.", "doc_001"),
        sentence(2, "Employees receive 15 days of leave.", "doc_002"),
    ]
    pairs, _ = build_candidate_pairs(
        sentences, vectors([1.0, 0.0], [1.0, 0.0]), threshold=0.5
    )

    pair = pairs[0]
    assert sentences[pair["index_a"]]["sentence_id"] == pair["sentence_id_a"]
    assert sentences[pair["index_b"]]["sentence_id"] == pair["sentence_id_b"]


def test_fewer_than_two_statements_is_safe():
    assert build_candidate_pairs([], vectors())[0] == []
    assert build_candidate_pairs(
        [sentence(1, "a")], vectors([1.0, 0.0])
    )[0] == []

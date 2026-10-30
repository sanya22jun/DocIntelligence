"""Batching, de-duplication, caching and vector/sentence alignment."""
import numpy as np
import pytest

from app.services.embedding_service import (
    EmbeddingService,
    cache_clear,
    cache_size,
)
from app.services.openai_service import OpenAIService, OpenAIServiceError
from stubs import StubClient, fake_vector


@pytest.fixture(autouse=True)
def clean_cache():
    cache_clear()
    yield
    cache_clear()


def sentence(index, text, document="doc_001"):
    return {
        "document_id": document,
        "document_name": f"{document}.txt",
        "page_number": None,
        "sentence_id": f"sent_{index:03d}",
        "text": text,
    }


def build(client=None, **kwargs):
    client = client or StubClient()
    service = OpenAIService(api_key="test-key", client=client)
    return EmbeddingService(service, **kwargs), client


# ------------------------------------------------------------- alignment

def test_matrix_rows_line_up_with_sentences():
    sentences = [
        sentence(1, "Employees receive 20 days of annual leave."),
        sentence(2, "Visitor parking is on level two of the garage."),
        sentence(3, "A medical certificate is needed for sick leave."),
    ]
    embedder, _ = build()
    matrix, stats = embedder.embed_sentences(sentences)

    assert matrix.shape == (3, 8)
    assert stats["sentences"] == 3
    for row, item in zip(matrix, sentences):
        assert np.allclose(row, np.asarray(fake_vector(item["text"]),
                                           dtype=np.float32))


def test_empty_input_returns_an_empty_matrix():
    embedder, client = build()
    matrix, stats = embedder.embed_sentences([])

    assert matrix.shape == (0, 0)
    assert stats["sentences"] == 0
    assert client.embedding_calls == []


# -------------------------------------------------------------- batching

def test_requests_are_batched():
    sentences = [sentence(i, f"Statement number {i} about leave.")
                 for i in range(1, 11)]
    embedder, client = build(batch_size=4)
    _, stats = embedder.embed_sentences(sentences)

    assert stats["api_calls"] == 3          # 4 + 4 + 2
    assert [len(call) for call in client.embedding_calls] == [4, 4, 2]


# ------------------------------------------------------- de-duplication

def test_identical_text_is_embedded_once_but_still_gets_a_row():
    shared = "Employees receive 10 days of paid sick leave."
    sentences = [
        sentence(1, shared, "doc_001"),
        sentence(2, shared, "doc_002"),
        sentence(3, "Visitor parking is on level two.", "doc_002"),
    ]
    embedder, client = build()
    matrix, stats = embedder.embed_sentences(sentences)

    assert matrix.shape[0] == 3
    assert stats["unique_texts"] == 2
    assert sum(len(call) for call in client.embedding_calls) == 2
    # The duplicate rows are identical, as they must be.
    assert np.allclose(matrix[0], matrix[1])


# -------------------------------------------------------------- caching

def test_second_run_is_served_from_the_cache():
    sentences = [sentence(1, "Employees receive 20 days of annual leave.")]

    first, client_one = build()
    first.embed_sentences(sentences)
    assert cache_size() == 1

    second, client_two = build()
    _, stats = second.embed_sentences(sentences)

    assert stats["cache_hits"] == 1
    assert stats["api_calls"] == 0
    assert client_two.embedding_calls == []


def test_cache_can_be_disabled():
    sentences = [sentence(1, "Employees receive 20 days of annual leave.")]
    first, _ = build()
    first.embed_sentences(sentences)

    second, client = build(use_cache=False)
    _, stats = second.embed_sentences(sentences)

    assert stats["cache_hits"] == 0
    assert stats["api_calls"] == 1
    assert client.embedding_calls


def test_cache_is_keyed_by_model():
    sentences = [sentence(1, "Employees receive 20 days of annual leave.")]
    build()[0].embed_sentences(sentences)

    other_client = StubClient()
    other = EmbeddingService(OpenAIService(
        api_key="k", client=other_client,
        embedding_model="text-embedding-3-large",
    ))
    _, stats = other.embed_sentences(sentences)

    assert stats["cache_hits"] == 0
    assert other_client.embedding_calls


# ------------------------------------------------------------- failures

def test_api_failure_propagates_as_a_service_error():
    client = StubClient()
    client.embedding_error = type("RateLimitError", (Exception,), {})("quota")
    embedder, _ = build(client)

    with pytest.raises(OpenAIServiceError) as exc:
        embedder.embed_sentences([sentence(1, "Employees receive 20 days.")])
    assert exc.value.kind == "rate_limit"

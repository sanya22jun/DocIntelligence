"""The full engine, end to end, against a stub client."""
import pytest

from app.services import analysis_pipeline
from app.services.analysis_pipeline import AnalysisRejected, run_analysis
from app.services.embedding_service import cache_clear
from app.services.openai_service import OpenAIService, OpenAIServiceError
from config import TestConfig
from stubs import StubClient


@pytest.fixture(autouse=True)
def clean_cache():
    cache_clear()
    yield
    cache_clear()


def sentence(index, text, document, page=None):
    return {
        "document_id": document,
        "document_name": f"{document}.txt",
        "page_number": page,
        "sentence_id": f"sent_{index:03d}",
        "text": text,
    }


def record(sentences, documents=2):
    return {
        "summary": {
            "documents_processed": documents,
            "total_sentences": len(sentences),
        },
        "documents": [],
        "sentences": sentences,
        "errors": [],
    }


CORPUS = [
    sentence(1, "Employees are entitled to 20 days of annual leave.", "doc_001", 1),
    sentence(2, "Employees may take 10 days of paid sick leave.", "doc_001", 1),
    sentence(3, "Employees receive 15 days of annual leave.", "doc_002", 1),
    sentence(4, "The company provides 10 days of paid sick leave.", "doc_002"),
    sentence(5, "Visitor parking is on the second level of the garage.", "doc_003"),
]


def service(client=None):
    return OpenAIService(api_key="test-key", client=client or StubClient())


# --------------------------------------------------------------- guards

def test_rejects_fewer_than_two_statements():
    with pytest.raises(AnalysisRejected, match="at least two statements|two statements"):
        run_analysis(record([CORPUS[0]]), TestConfig, service())


def test_rejects_a_single_document():
    only_one = [CORPUS[0], CORPUS[1]]
    with pytest.raises(AnalysisRejected, match="two different documents"):
        run_analysis(record(only_one), TestConfig, service())


def test_rejects_a_corpus_above_the_statement_cap():
    class Capped(TestConfig):
        MAX_SENTENCES_PER_ANALYSIS = 3

    with pytest.raises(AnalysisRejected, match="above the"):
        run_analysis(record(CORPUS), Capped, service())


def test_guards_run_before_any_api_call():
    client = StubClient()
    with pytest.raises(AnalysisRejected):
        run_analysis(record([CORPUS[0]]), TestConfig, service(client))
    assert client.embedding_calls == []
    assert client.reasoning_calls == []


# ------------------------------------------------------------ happy path

def test_full_pipeline_produces_a_report():
    report = run_analysis(record(CORPUS, documents=3), TestConfig, service())

    assert report["models"]["embedding"] == "text-embedding-3-small"
    assert report["models"]["reasoning"] == "gpt-4.1-mini"
    assert report["analysed_at"]

    overview = report["overview"]
    assert overview["documents_analyzed"] == 3
    assert overview["statements_extracted"] == 5
    assert overview["candidate_pairs_analyzed"] == len(report["results"])
    assert overview["contradictions_detected"] >= 1
    assert overview["pairs_unclassified"] == 0

    stages = report["stages"]
    assert stages["embedding"]["sentences"] == 5
    assert stages["candidate_selection"]["selected"] >= 1
    assert stages["reasoning"]["requests"] >= 1


def test_report_identifies_the_planted_numerical_conflict():
    report = run_analysis(record(CORPUS), TestConfig, service())

    contradictions = [
        item for item in report["results"]
        if item["relationship"] == "CONTRADICTION"
    ]
    assert contradictions

    leave = next(
        item for item in contradictions
        if {item["sentence_id_a"], item["sentence_id_b"]}
        == {"sent_001", "sent_003"}
    )
    assert leave["category"] == "NUMERICAL_CONFLICT"
    assert leave["document_a"]["document_name"] == "doc_001.txt"
    assert leave["document_b"]["document_name"] == "doc_002.txt"
    assert leave["evidence_a"] == CORPUS[0]["text"]
    assert leave["evidence_b"] == CORPUS[2]["text"]
    assert leave["conflicting_span_a"] in leave["evidence_a"]
    assert leave["conflicting_span_b"] in leave["evidence_b"]


def test_every_result_cites_a_real_statement():
    report = run_analysis(record(CORPUS), TestConfig, service())
    by_id = {s["sentence_id"]: s for s in CORPUS}

    for item in report["results"]:
        first = by_id[item["sentence_id_a"]]
        second = by_id[item["sentence_id_b"]]
        assert item["evidence_a"] == first["text"]
        assert item["evidence_b"] == second["text"]
        assert item["document_a"]["document_id"] == first["document_id"]
        assert item["document_b"]["page_number"] == second["page_number"]
        # No pair is ever drawn from one document.
        assert first["document_id"] != second["document_id"]


def test_usage_is_reported():
    report = run_analysis(record(CORPUS), TestConfig, service())
    usage = report["usage"]

    assert usage["embedding_requests"] == 1
    assert usage["reasoning_requests"] >= 1
    assert usage["total_tokens"] > 0
    assert usage["estimated_cost_usd"] >= 0
    assert "Not a billing figure" in usage["cost_basis"]


# ------------------------------------------------------------- warnings

def test_no_candidates_is_explained_not_silent():
    # Two statements about entirely different subjects: nothing should
    # clear the similarity gate, and the report must say why.
    unrelated = [
        sentence(1, "Visitor parking is on the second level of the garage.",
                 "doc_001"),
        sentence(2, "Performance reviews take place once per year in December.",
                 "doc_002"),
    ]
    report = run_analysis(record(unrelated), TestConfig, service())

    assert report["results"] == []
    assert report["stages"]["candidate_selection"]["selected"] == 0
    assert report["stages"]["reasoning"]["requests"] == 0
    assert any(w["kind"] == "no_candidates" for w in report["warnings"])
    message = next(
        w["message"] for w in report["warnings"] if w["kind"] == "no_candidates"
    )
    assert "SIMILARITY_THRESHOLD" in message


def test_pair_cap_is_reported():
    class TinyCap(TestConfig):
        MAX_CANDIDATE_PAIRS = 1
        SIMILARITY_THRESHOLD = 0.1

    report = run_analysis(record(CORPUS), TinyCap, service())

    assert len(report["results"]) <= 1
    assert any(w["kind"] == "pair_cap_reached" for w in report["warnings"])


def test_reasoning_failure_still_returns_a_report():
    client = StubClient()
    client.reasoning_errors = [
        type("RateLimitError", (Exception,), {})("quota")
    ] * 12

    class OnePerRequest(TestConfig):
        REASONING_BATCH_SIZE = 1
        SIMILARITY_THRESHOLD = 0.5

    report = run_analysis(record(CORPUS), OnePerRequest, service(client))

    assert report["results"] == []
    assert report["overview"]["pairs_unclassified"] > 0
    assert any(w["kind"] == "rate_limit" for w in report["warnings"])
    # Honest reporting: nothing is presented as classified.
    assert report["overview"]["contradictions_detected"] == 0


def test_embedding_failure_is_fatal_and_typed():
    client = StubClient()
    client.embedding_error = type("APIConnectionError", (Exception,), {})("down")

    with pytest.raises(OpenAIServiceError) as exc:
        run_analysis(record(CORPUS), TestConfig, service(client))
    assert exc.value.kind == "connection"


def test_missing_key_surfaces_as_an_auth_error():
    class NoKey(TestConfig):
        OPENAI_API_KEY = ""

    with pytest.raises(OpenAIServiceError) as exc:
        run_analysis(record(CORPUS), NoKey)
    assert exc.value.kind == "auth"
    assert "OPENAI_API_KEY" in exc.value.message


# -------------------------------------------------------------- caching

def test_second_run_reuses_cached_embeddings():
    first_client = StubClient()
    run_analysis(record(CORPUS), TestConfig, service(first_client))

    second_client = StubClient()
    report = run_analysis(record(CORPUS), TestConfig, service(second_client))

    assert report["stages"]["embedding"]["cache_hits"] == 5
    assert report["stages"]["embedding"]["api_calls"] == 0
    assert second_client.embedding_calls == []
    assert report["usage"]["embedding_tokens"] == 0


def test_build_service_is_used_when_none_is_injected(monkeypatch):
    calls = []

    def fake_build(config, client=None):
        calls.append(config)
        return service()

    monkeypatch.setattr(analysis_pipeline, "build_service", fake_build)
    run_analysis(record(CORPUS), TestConfig)
    assert len(calls) == 1

"""Classification: batching, validation, recovery, no fabrication."""
import pytest

from app.services.contradiction_service import (
    ContradictionService,
    build_user_prompt,
    verified_span,
)
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.services.similarity_service import pair_key
from stubs import StubClient

LEAVE_20 = "Full-time employees are entitled to 20 days of paid annual leave."
LEAVE_15 = "Full-time employees receive 15 days of paid annual leave."
SICK_A = "Employees may take up to 10 days of paid sick leave."
SICK_B = "The company provides 10 days of paid sick leave annually."
PARKING = "Visitor parking is located on the second level of the garage."


def sentence(index, text, document, page=None):
    return {
        "document_id": document,
        "document_name": f"{document}.txt",
        "page_number": page,
        "sentence_id": f"sent_{index:03d}",
        "text": text,
    }


SENTENCES = [
    sentence(1, LEAVE_20, "doc_001", page=2),
    sentence(2, SICK_A, "doc_001"),
    sentence(3, LEAVE_15, "doc_002", page=1),
    sentence(4, SICK_B, "doc_002"),
    sentence(5, PARKING, "doc_003"),
]


def pair(index_a, index_b, similarity=0.9):
    first, second = SENTENCES[index_a], SENTENCES[index_b]
    return {
        "index_a": index_a,
        "index_b": index_b,
        "sentence_id_a": first["sentence_id"],
        "sentence_id_b": second["sentence_id"],
        "pair_key": pair_key(first["sentence_id"], second["sentence_id"]),
        "similarity": similarity,
    }


LEAVE_PAIR = pair(0, 2)
SICK_PAIR = pair(1, 3)
UNRELATED_PAIR = pair(0, 4, similarity=0.66)


def build(client=None, **kwargs):
    client = client or StubClient()
    service = OpenAIService(api_key="test-key", client=client)
    return ContradictionService(service, **kwargs), client


# ----------------------------------------------------------- happy path

def test_classifies_the_three_relationship_types():
    classifier, _ = build()
    results, warnings, stats = classifier.classify(
        [LEAVE_PAIR, SICK_PAIR, UNRELATED_PAIR], SENTENCES
    )

    assert warnings == []
    assert stats["pairs_classified"] == 3
    assert stats["unclassified"] == []

    by_key = {item["pair_key"]: item for item in results}
    assert by_key[LEAVE_PAIR["pair_key"]]["relationship"] == "CONTRADICTION"
    assert by_key[SICK_PAIR["pair_key"]]["relationship"] == "SUPPORT"
    assert by_key[UNRELATED_PAIR["pair_key"]]["relationship"] == "NEUTRAL"


def test_contradictions_are_listed_first():
    classifier, _ = build()
    results, _, _ = classifier.classify(
        [UNRELATED_PAIR, SICK_PAIR, LEAVE_PAIR], SENTENCES
    )
    assert results[0]["relationship"] == "CONTRADICTION"


def test_result_carries_evidence_and_source_references():
    classifier, _ = build()
    results, _, _ = classifier.classify([LEAVE_PAIR], SENTENCES)
    item = results[0]

    # Evidence comes from our own records, not from the model.
    assert item["evidence_a"] == LEAVE_20
    assert item["evidence_b"] == LEAVE_15
    assert item["document_a"] == {
        "document_id": "doc_001",
        "document_name": "doc_001.txt",
        "page_number": 2,
    }
    assert item["document_b"]["page_number"] == 1
    assert item["similarity"] == 0.9
    assert item["confidence_basis"] == "model_reported"
    assert 0.0 <= item["confidence"] <= 1.0
    assert item["category"] == "NUMERICAL_CONFLICT"


def test_category_is_dropped_for_non_contradictions():
    classifier, _ = build()
    results, _, _ = classifier.classify([SICK_PAIR], SENTENCES)
    assert results[0]["relationship"] == "SUPPORT"
    assert results[0]["category"] is None


def test_requests_are_batched():
    pairs = [LEAVE_PAIR, SICK_PAIR, UNRELATED_PAIR]
    classifier, client = build(batch_size=2)
    _, _, stats = classifier.classify(pairs, SENTENCES)

    assert stats["requests"] == 2
    assert [len(call) for call in client.reasoning_calls] == [2, 1]


def test_no_pairs_means_no_requests():
    classifier, client = build()
    results, warnings, stats = classifier.classify([], SENTENCES)

    assert (results, warnings) == ([], [])
    assert stats["requests"] == 0
    assert client.reasoning_calls == []


# ------------------------------------------------------- span verification

def test_span_must_appear_verbatim():
    assert verified_span("20 days", LEAVE_20) == "20 days"
    assert verified_span("twenty days", LEAVE_20) is None
    assert verified_span("", LEAVE_20) is None
    assert verified_span(None, LEAVE_20) is None
    assert verified_span(12, LEAVE_20) is None


def test_span_case_difference_maps_back_to_the_source_slice():
    assert verified_span("FULL-TIME", LEAVE_20) == "Full-time"


def test_invented_span_is_discarded_but_the_result_is_kept():
    client = StubClient()
    client.result_overrides[LEAVE_PAIR["pair_key"]] = {
        "relationship": "CONTRADICTION",
        "category": "NUMERICAL_CONFLICT",
        "explanation": "20 versus 15 days.",
        "conflicting_span_a": "thirty working days",   # not in the text
        "conflicting_span_b": "15 days",
        "confidence": 0.9,
    }
    classifier, _ = build(client)
    results, _, _ = classifier.classify([LEAVE_PAIR], SENTENCES)

    assert results[0]["conflicting_span_a"] is None
    assert results[0]["conflicting_span_b"] == "15 days"
    assert results[0]["relationship"] == "CONTRADICTION"


# ------------------------------------------------------------- validation

def test_unknown_relationship_is_rejected():
    client = StubClient()
    client.result_overrides[LEAVE_PAIR["pair_key"]] = {
        "relationship": "MAYBE_CONFLICT",
        "category": "NUMERICAL_CONFLICT",
        "explanation": "x",
        "conflicting_span_a": "", "conflicting_span_b": "", "confidence": 0.9,
    }
    classifier, _ = build(client)
    results, warnings, stats = classifier.classify([LEAVE_PAIR], SENTENCES)

    assert results == []
    assert stats["unclassified"] == [LEAVE_PAIR["pair_key"]]
    assert any(w["kind"] == "invalid_relationship" for w in warnings)


def test_missing_explanation_is_rejected():
    client = StubClient()
    client.result_overrides[LEAVE_PAIR["pair_key"]] = {
        "relationship": "CONTRADICTION", "category": "NUMERICAL_CONFLICT",
        "explanation": "   ",
        "conflicting_span_a": "", "conflicting_span_b": "", "confidence": 0.9,
    }
    classifier, _ = build(client)
    results, warnings, _ = classifier.classify([LEAVE_PAIR], SENTENCES)

    assert results == []
    assert any(w["kind"] == "missing_explanation" for w in warnings)


def test_unknown_category_falls_back_to_other():
    client = StubClient()
    client.result_overrides[LEAVE_PAIR["pair_key"]] = {
        "relationship": "CONTRADICTION", "category": "VIBES_CONFLICT",
        "explanation": "20 versus 15 days.",
        "conflicting_span_a": "", "conflicting_span_b": "", "confidence": 0.9,
    }
    classifier, _ = build(client)
    results, _, _ = classifier.classify([LEAVE_PAIR], SENTENCES)
    assert results[0]["category"] == "OTHER"


@pytest.mark.parametrize("value,expected", [
    (1.7, 1.0), (-0.5, 0.0), ("0.42", 0.42), ("nonsense", None), (None, None),
])
def test_confidence_is_clamped_and_sanitised(value, expected):
    client = StubClient()
    client.result_overrides[LEAVE_PAIR["pair_key"]] = {
        "relationship": "NEUTRAL", "category": "NOT_APPLICABLE",
        "explanation": "ok",
        "conflicting_span_a": "", "conflicting_span_b": "",
        "confidence": value,
    }
    classifier, _ = build(client)
    results, _, _ = classifier.classify([LEAVE_PAIR], SENTENCES)
    assert results[0]["confidence"] == expected


def test_results_for_pairs_we_never_asked_about_are_discarded():
    client = StubClient()
    client.inject_pairs = ["sent_900|sent_901"]
    classifier, _ = build(client)
    results, warnings, _ = classifier.classify([LEAVE_PAIR], SENTENCES)

    assert [item["pair_key"] for item in results] == [LEAVE_PAIR["pair_key"]]
    assert any(w["kind"] == "unexpected_pair" for w in warnings)


def test_reply_without_a_results_array_is_retried():
    client = StubClient()
    client.reasoning_behaviours = ["no_results_key"]   # first request only
    classifier, _ = build(client)
    results, warnings, stats = classifier.classify([LEAVE_PAIR], SENTENCES)

    assert any(w["kind"] == "invalid_response" for w in warnings)
    assert stats["retries"] == 1
    # The retry succeeded, so the pair is classified rather than lost.
    assert [item["pair_key"] for item in results] == [LEAVE_PAIR["pair_key"]]
    assert stats["unclassified"] == []


def test_persistently_malformed_reply_ends_as_unclassified():
    client = StubClient()
    client.reasoning_behaviours = ["no_results_key"] * 4
    classifier, _ = build(client)
    results, warnings, stats = classifier.classify([LEAVE_PAIR], SENTENCES)

    assert results == []
    assert stats["unclassified"] == [LEAVE_PAIR["pair_key"]]
    assert any(w["kind"] == "invalid_response" for w in warnings)
    assert any(w["kind"] == "missing_results" for w in warnings)


# --------------------------------------------------------------- recovery

def test_omitted_pair_is_asked_again_individually():
    client = StubClient()
    client.omit_pairs = {SICK_PAIR["pair_key"]}

    classifier, _ = build(client, batch_size=2)
    results, warnings, stats = classifier.classify(
        [LEAVE_PAIR, SICK_PAIR], SENTENCES
    )

    assert stats["retries"] == 1
    assert any(w["kind"] == "missing_results" for w in warnings)
    # Still omitted on the retry, so it is reported as unclassified
    # rather than invented.
    assert stats["unclassified"] == [SICK_PAIR["pair_key"]]
    assert [item["pair_key"] for item in results] == [LEAVE_PAIR["pair_key"]]


def test_omitted_pair_recovered_on_retry():
    class Recovering(StubClient):
        def next_reasoning_behaviour(self):
            # Drop the pair on the first request only.
            self.omit_pairs = set() if self.reasoning_calls[:-1] else {
                SICK_PAIR["pair_key"]
            }
            return "ok"

    classifier, _ = build(Recovering(), batch_size=2)
    results, _, stats = classifier.classify([LEAVE_PAIR, SICK_PAIR], SENTENCES)

    assert stats["retries"] == 1
    assert stats["unclassified"] == []
    assert len(results) == 2


def test_truncated_reply_splits_the_batch():
    client = StubClient()
    client.reasoning_behaviours = ["truncated"]   # first request only

    classifier, _ = build(client, batch_size=2)
    results, warnings, stats = classifier.classify(
        [LEAVE_PAIR, SICK_PAIR], SENTENCES
    )

    assert stats["batch_splits"] == 1
    assert any(w["kind"] == "truncated" for w in warnings)
    assert len(results) == 2        # both recovered in smaller batches


def test_rate_limit_on_one_batch_keeps_the_other_results():
    client = StubClient()
    rate_limit = type("RateLimitError", (Exception,), {})("quota exceeded")
    # Fails the first single-pair batch, succeeds for the second.
    client.reasoning_errors = [rate_limit]

    classifier, _ = build(client, batch_size=1)
    results, warnings, stats = classifier.classify(
        [LEAVE_PAIR, SICK_PAIR], SENTENCES
    )

    assert len(results) == 1
    assert stats["unclassified"] == [LEAVE_PAIR["pair_key"]]
    assert any(w["kind"] == "rate_limit" for w in warnings)


def test_auth_failure_aborts_immediately():
    client = StubClient()
    auth = type("AuthenticationError", (Exception,), {})("bad key")
    client.reasoning_errors = [auth]

    classifier, _ = build(client, batch_size=1)
    with pytest.raises(OpenAIServiceError) as exc:
        classifier.classify([LEAVE_PAIR, SICK_PAIR], SENTENCES)

    assert exc.value.kind == "auth"
    # Aborted on the first request rather than burning the second.
    assert len(client.reasoning_calls) == 1


def test_request_budget_caps_runaway_splitting():
    client = StubClient()
    client.reasoning_behaviours = ["truncated"] * 20

    classifier, _ = build(client, batch_size=2, max_requests=3)
    _, warnings, stats = classifier.classify(
        [LEAVE_PAIR, SICK_PAIR, UNRELATED_PAIR], SENTENCES
    )

    assert stats["requests"] <= 3
    assert any(w["kind"] == "request_budget_exhausted" for w in warnings)
    assert stats["unclassified"]


# ----------------------------------------------------------------- prompt

def test_user_prompt_contains_pair_keys_and_both_statements():
    by_index = dict(enumerate(SENTENCES))
    prompt = build_user_prompt([LEAVE_PAIR], by_index)

    assert f"PAIR {LEAVE_PAIR['pair_key']}" in prompt
    assert LEAVE_20 in prompt
    assert LEAVE_15 in prompt
    assert "doc_001.txt" in prompt and "doc_002.txt" in prompt

"""Grounded document chat: retrieval, citation integrity, limits."""
import pytest

from app.services import chat_service
from app.services.chat_service import (
    FALLBACK_SUGGESTIONS,
    MAX_QUESTION_CHARS,
    ChatRejected,
    ask,
    build_context,
    build_digest,
    build_user_prompt,
    clean_question,
    related_findings,
    retrieve,
    suggest_questions,
)
from app.services.embedding_service import cache_clear
from app.services.openai_service import OpenAIService, OpenAIServiceError
from config import TestConfig
from stubs import StubClient


@pytest.fixture(autouse=True)
def clean_embedding_cache():
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


SENTENCES = [
    sentence(1, "Full-time employees are entitled to 20 days of paid annual leave.", "doc_001", 2),
    sentence(2, "Employees may take up to 10 days of paid sick leave per year.", "doc_001"),
    sentence(3, "Full-time employees receive 15 days of paid annual leave.", "doc_002", 1),
    sentence(4, "Visitor parking is located on the second level of the garage.", "doc_003"),
]

REPORT = {
    "overview": {
        "contradictions_detected": 1,
        "supporting_relationships": 0,
        "neutral_relationships": 0,
    },
    "results": [{
        "pair_key": "sent_001|sent_003",
        "sentence_id_a": "sent_001",
        "sentence_id_b": "sent_003",
        "relationship": "CONTRADICTION",
        "category": "NUMERICAL_CONFLICT",
        "explanation": "One grants 20 days, the other 15 days.",
        "evidence_a": SENTENCES[0]["text"],
        "evidence_b": SENTENCES[2]["text"],
    }],
}


def record(sentences=None, report=REPORT):
    return {
        "analysis_id": "abc123",
        "summary": {"documents_processed": 3, "total_sentences": 4},
        "documents": [
            {"document_name": "doc_001.txt", "status": "success"},
            {"document_name": "doc_002.txt", "status": "success"},
        ],
        "sentences": sentences if sentences is not None else list(SENTENCES),
        "analysis": report,
    }


def service(client=None):
    return OpenAIService(api_key="stub", client=client or StubClient())


# ------------------------------------------------------- question hygiene

def test_whitespace_is_collapsed():
    assert clean_question("  what   is   leave?  ") == "what is leave?"


@pytest.mark.parametrize("value", ["", "   ", None, 42, {"a": 1}, []])
def test_unusable_questions_are_rejected(value):
    with pytest.raises(ChatRejected):
        clean_question(value)


def test_overlong_questions_are_rejected():
    with pytest.raises(ChatRejected, match="under"):
        clean_question("x" * (MAX_QUESTION_CHARS + 1))


def test_a_question_at_the_limit_is_accepted():
    assert len(clean_question("x" * MAX_QUESTION_CHARS)) == MAX_QUESTION_CHARS


# -------------------------------------------------------------- retrieval

def test_retrieval_ranks_relevant_statements_first():
    retrieved, _stats = retrieve(
        "How many days of annual leave?", SENTENCES, service(), TestConfig
    )

    ids = [item["sentence_id"] for item in retrieved]
    # The two annual-leave statements should outrank parking.
    assert set(ids[:2]) == {"sent_001", "sent_003"}
    assert ids[-1] == "sent_004"


def test_retrieval_reports_similarity_for_each_statement():
    retrieved, _stats = retrieve(
        "annual leave", SENTENCES, service(), TestConfig
    )
    assert all("similarity" in item for item in retrieved)
    scores = [item["similarity"] for item in retrieved]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_respects_top_k():
    retrieved, _stats = retrieve(
        "leave", SENTENCES, service(), TestConfig, top_k=2
    )
    assert len(retrieved) == 2


def test_statement_vectors_come_from_the_cache_on_a_second_question():
    client = StubClient()
    svc = service(client)

    retrieve("annual leave", SENTENCES, svc, TestConfig)
    first_calls = len(client.embedding_calls)

    _retrieved, stats = retrieve("sick leave", SENTENCES, svc, TestConfig)

    # Only the new question is embedded; the statements are cached.
    assert stats["cache_hits"] == len(SENTENCES)
    assert stats["api_calls"] == 0
    assert len(client.embedding_calls) == first_calls + 1


def test_retrieval_with_no_statements_is_safe():
    retrieved, stats = retrieve("anything", [], service(), TestConfig)
    assert retrieved == []
    assert stats["statements"] == 0


# ---------------------------------------------------------------- context

def test_context_includes_ids_documents_and_pages():
    context = build_context(SENTENCES[:2], [])
    assert "[sent_001]" in context
    assert "doc_001.txt, page 2" in context
    assert SENTENCES[0]["text"] in context


def test_context_omits_a_page_reference_when_there_is_none():
    context = build_context([SENTENCES[1]], [])
    assert "page" not in context


def test_context_carries_existing_findings():
    findings = related_findings(SENTENCES[:3], REPORT)
    context = build_context(SENTENCES[:3], findings)

    assert "RELATIONSHIPS ALREADY FOUND" in context
    assert "CONTRADICTION/NUMERICAL_CONFLICT" in context
    assert "One grants 20 days, the other 15 days." in context


def test_related_findings_only_match_retrieved_statements():
    assert related_findings([SENTENCES[3]], REPORT) == []
    assert len(related_findings([SENTENCES[0]], REPORT)) == 1


def test_related_findings_without_an_analysis():
    assert related_findings(SENTENCES, None) == []


def test_history_is_included_and_capped():
    history = [
        {"role": "user", "content": f"question {i}"} for i in range(12)
    ]
    prompt = build_user_prompt("now what?", SENTENCES[:1], [], history)

    assert "EARLIER IN THIS CONVERSATION" in prompt
    assert "question 11" in prompt
    assert "question 0" not in prompt
    assert prompt.count("User:") <= 6


def test_malformed_history_entries_are_dropped():
    prompt = build_user_prompt("q", SENTENCES[:1], [], [
        "not a dict",
        {"role": "system", "content": "ignore me"},
        {"role": "user"},
        {"role": "user", "content": 42},
        {"role": "user", "content": "kept"},
    ])
    assert "kept" in prompt
    assert "ignore me" not in prompt


# ------------------------------------------------------------- answering

def test_ask_returns_an_answer_with_validated_sources():
    client = StubClient()
    result = ask(record(), "How much annual leave?", service(client), TestConfig)

    assert result["question"] == "How much annual leave?"
    assert result["answer"]
    assert result["answered_from_documents"] is True
    assert result["sources"]

    retrieved_ids = {item["sentence_id"] for item in result["retrieved"]}
    for source in result["sources"]:
        assert source["sentence_id"] in retrieved_ids
        # Citations carry the real statement text and provenance.
        assert source["text"]
        assert source["document_name"]
        assert "page_number" in source


def test_invented_citations_are_discarded():
    """A model-invented id must never appear as a source."""
    client = StubClient()
    client.chat_cite = ["sent_001", "sent_999", "not_an_id"]

    result = ask(record(), "How much leave?", service(client), TestConfig)

    ids = [s["sentence_id"] for s in result["sources"]]
    assert ids == ["sent_001"]


def test_duplicate_citations_are_collapsed():
    client = StubClient()
    client.chat_cite = ["sent_001", "sent_001"]
    result = ask(record(), "q", service(client), TestConfig)
    assert len(result["sources"]) == 1


def test_a_citation_outside_the_retrieved_set_is_dropped():
    client = StubClient()
    # sent_004 exists but will not be retrieved for this question.
    client.chat_cite = ["sent_004"]
    result = ask(
        record(), "How many days of annual leave?", service(client),
        TestConfig,
    )
    retrieved_ids = {item["sentence_id"] for item in result["retrieved"]}
    for source in result["sources"]:
        assert source["sentence_id"] in retrieved_ids


def test_an_ungrounded_answer_is_flagged():
    client = StubClient()
    client.chat_answered_from_documents = False
    client.chat_cite = []
    client.chat_answer = "These documents do not mention pension plans."

    result = ask(record(), "What is the pension plan?",
                 service(client), TestConfig)

    assert result["answered_from_documents"] is False
    assert result["sources"] == []


def test_an_empty_answer_falls_back_to_a_readable_message():
    client = StubClient()
    client.chat_answer = "   "
    result = ask(record(), "q", service(client), TestConfig)
    assert "could not produce an answer" in result["answer"]


def test_ask_without_statements_is_rejected():
    with pytest.raises(ChatRejected, match="no extracted statements"):
        ask(record(sentences=[]), "q", service(), TestConfig)


def test_ask_works_before_an_analysis_has_run():
    client = StubClient()
    result = ask(
        record(report=None), "How much annual leave?",
        service(client), TestConfig,
    )
    assert result["answer"]
    assert result["related_findings"] == []


def test_api_failure_propagates_for_the_route_to_classify():
    client = StubClient()
    client.chat_error = type("RateLimitError", (Exception,), {})("slow down")

    with pytest.raises(OpenAIServiceError) as exc:
        ask(record(), "q", service(client), TestConfig)
    assert exc.value.kind == "rate_limit"


def test_question_is_cleaned_before_use():
    client = StubClient()
    result = ask(record(), "  how    much   leave?  ",
                 service(client), TestConfig)
    assert result["question"] == "how much leave?"


# ----------------------------------------------------------- suggestions

def test_suggestions_are_generated_from_the_analysis():
    client = StubClient()
    questions, source = suggest_questions(record(), service(client), TestConfig)

    assert source == "generated"
    assert len(questions) == 4
    assert all(isinstance(q, str) and q for q in questions)
    # The digest the model saw described this analysis.
    digest = client.suggestion_calls[0]
    assert "doc_001.txt" in digest
    assert "TOP DISAGREEMENTS" in digest


def test_suggestions_fall_back_when_the_call_fails():
    client = StubClient()
    client.suggestion_error = type("APITimeoutError", (Exception,), {})("slow")

    questions, source = suggest_questions(record(), service(client), TestConfig)

    assert source == "fallback"
    assert questions == list(FALLBACK_SUGGESTIONS)


def test_suggestions_fall_back_for_an_empty_record():
    questions, source = suggest_questions(
        {"documents": [], "sentences": []}, service(), TestConfig
    )
    assert source == "fallback"
    assert questions == list(FALLBACK_SUGGESTIONS)


def test_digest_describes_documents_findings_and_statements():
    digest = build_digest(record())
    assert "DOCUMENTS:" in digest
    assert "FINDINGS: 1 contradiction(s)" in digest
    assert "SAMPLE STATEMENTS" in digest


def test_digest_is_bounded():
    many = [sentence(i, f"Statement number {i} about leave.", "doc_001")
            for i in range(100)]
    digest = build_digest(record(sentences=many), max_statements=5)
    assert digest.count("- Statement number") == 5


def test_duplicate_suggestions_are_removed():
    client = StubClient()
    import json as _json
    from types import SimpleNamespace

    def one_repeat(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=_json.dumps(
                    {"questions": ["Same?", "Same?", "Other?"]}
                )),
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )

    client.chat.completions.create = one_repeat
    questions, source = suggest_questions(record(), service(client), TestConfig)

    assert source == "generated"
    assert questions == ["Same?", "Other?"]


def test_chat_service_never_imports_the_test_stub():
    source = open(chat_service.__file__, encoding="utf-8").read()
    assert "stub" not in source.lower().replace("stubborn", "")

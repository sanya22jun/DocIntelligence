"""POST /api/chat and GET /api/chat/suggestions/<id>."""
import pytest

from app.services import chat_service
from app.services.chat_service import MAX_QUESTION_CHARS, FALLBACK_SUGGESTIONS
from app.services.embedding_service import cache_clear
from app.services.openai_service import OpenAIService
from conftest import upload_file
from stubs import StubClient

DOC_A = (
    b"Full-time employees are entitled to 20 days of paid annual leave.\n"
    b"Employees may take up to 10 days of paid sick leave each year.\n"
)
DOC_B = (
    b"Full-time employees receive 15 days of paid annual leave.\n"
    b"Visitor parking is located on the second level of the garage.\n"
)


@pytest.fixture(autouse=True)
def clean_embedding_cache():
    cache_clear()
    yield
    cache_clear()


@pytest.fixture
def stub(app, monkeypatch):
    """Point both the pipeline and the chat at a stub client."""
    client = StubClient()
    app.config["OPENAI_API_KEY"] = "test-key"

    def fake_build(config, client_override=None):
        return OpenAIService(api_key="test-key", client=client)

    # The chat route builds its own service, so patch the symbol each
    # module actually calls.
    monkeypatch.setattr("app.routes.build_service", fake_build)
    monkeypatch.setattr(
        "app.services.analysis_pipeline.build_service", fake_build
    )
    return client


def upload(client):
    response = client.post(
        "/api/upload",
        data={"documents": [
            upload_file(DOC_A, "policy_2023.txt"),
            upload_file(DOC_B, "handbook_2024.txt"),
        ]},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    return response.get_json()["analysis_id"]


def ask(client, analysis_id, question, **extra):
    body = {"analysis_id": analysis_id, "question": question}
    body.update(extra)
    return client.post("/api/chat", json=body)


# ------------------------------------------------------------ validation

def test_analysis_id_is_required(client, stub):
    response = client.post("/api/chat", json={"question": "hi"})
    assert response.status_code == 400
    assert "analysis_id" in response.get_json()["error"]


@pytest.mark.parametrize("bad", [{"a": 1}, ["x"], 7, True])
def test_non_string_analysis_id_is_a_400(client, stub, bad):
    response = client.post(
        "/api/chat", json={"analysis_id": bad, "question": "hi"}
    )
    assert response.status_code == 400


def test_unknown_analysis_is_a_404(client, stub):
    response = ask(client, "ghost", "hi")
    assert response.status_code == 404
    assert response.get_json()["error"]


@pytest.mark.parametrize("question", ["", "   ", None, 42, {"a": 1}])
def test_an_unusable_question_is_a_400(client, stub, question):
    analysis_id = upload(client)
    response = ask(client, analysis_id, question)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_an_overlong_question_is_refused(client, stub):
    analysis_id = upload(client)
    response = ask(client, analysis_id, "x" * (MAX_QUESTION_CHARS + 50))

    assert response.status_code == 400
    assert str(MAX_QUESTION_CHARS) in response.get_json()["error"]
    # Refused before any API call, so nothing was spent.
    assert stub.chat_calls == []


def test_invalid_json_body_is_a_400(client, stub):
    response = client.post(
        "/api/chat", data="{not json", content_type="application/json"
    )
    assert response.status_code == 400


def test_missing_key_is_a_503(client):
    # No stub fixture, so the app has no key configured.
    analysis_id = upload(client)
    response = ask(client, analysis_id, "How much leave?")

    assert response.status_code == 503
    body = response.get_json()
    assert body["kind"] == "auth"
    assert "OPENAI_API_KEY" in body["error"]
    assert "sk-" not in body["error"]


# ------------------------------------------------------------ happy path

def test_a_question_is_answered_with_cited_sources(client, stub):
    analysis_id = upload(client)
    response = ask(client, analysis_id, "How many days of annual leave?")

    assert response.status_code == 200
    body = response.get_json()

    assert body["analysis_id"] == analysis_id
    assert body["question"] == "How many days of annual leave?"
    assert body["answer"]
    assert body["answered_from_documents"] is True
    assert body["sources"]

    for source in body["sources"]:
        assert source["sentence_id"]
        assert source["document_name"].endswith(".txt")
        assert source["text"]
        assert "page_number" in source


def test_sources_are_always_statements_that_were_retrieved(client, stub):
    analysis_id = upload(client)
    body = ask(client, analysis_id, "annual leave").get_json()

    retrieved = {item["sentence_id"] for item in body["retrieved"]}
    cited = {source["sentence_id"] for source in body["sources"]}
    assert cited <= retrieved


def test_an_invented_citation_never_reaches_the_client(client, stub):
    analysis_id = upload(client)
    stub.chat_cite = ["sent_001", "sent_404", "fabricated"]

    body = ask(client, analysis_id, "How much leave?").get_json()
    ids = [source["sentence_id"] for source in body["sources"]]

    assert "sent_404" not in ids
    assert "fabricated" not in ids


def test_usage_is_reported_per_question(client, stub):
    analysis_id = upload(client)
    body = ask(client, analysis_id, "How much leave?").get_json()

    usage = body["usage"]
    assert usage["reasoning_requests"] == 1
    assert usage["total_tokens"] > 0
    assert "Not a billing figure" in usage["cost_basis"]


def test_chat_works_before_any_analysis_has_run(client, stub):
    analysis_id = upload(client)
    body = ask(client, analysis_id, "How much annual leave?").get_json()

    assert body["answer"]
    assert body["related_findings"] == []


def test_findings_are_offered_as_context_after_an_analysis(client, stub):
    import time

    analysis_id = upload(client)
    client.post("/api/analyze", json={"analysis_id": analysis_id})
    deadline = time.time() + 15
    while time.time() < deadline:
        state = client.get(f"/api/results/{analysis_id}").get_json()["state"]
        if state in ("done", "failed"):
            break
        time.sleep(0.02)

    body = ask(client, analysis_id, "Why do the documents disagree?").get_json()

    assert body["related_findings"]
    # The prompt carried the finding so the answer can explain it.
    assert "RELATIONSHIPS ALREADY FOUND" in stub.chat_calls[-1]


def test_history_is_accepted_for_follow_ups(client, stub):
    analysis_id = upload(client)
    response = ask(
        client, analysis_id, "Why?",
        history=[
            {"role": "user", "content": "How much annual leave?"},
            {"role": "assistant", "content": "The documents disagree."},
        ],
    )

    assert response.status_code == 200
    assert "EARLIER IN THIS CONVERSATION" in stub.chat_calls[-1]


def test_malformed_history_does_not_break_the_request(client, stub):
    analysis_id = upload(client)
    response = ask(
        client, analysis_id, "Why?",
        history=["nonsense", {"role": "root", "content": "x"}, 42],
    )
    assert response.status_code == 200


# --------------------------------------------------------------- failures

def test_an_api_failure_is_a_502(client, stub):
    analysis_id = upload(client)
    stub.chat_error = type("APIConnectionError", (Exception,), {})("down")

    response = ask(client, analysis_id, "How much leave?")
    assert response.status_code == 502
    assert response.get_json()["kind"] == "connection"


def test_a_rate_limit_is_a_502_with_a_readable_reason(client, stub):
    analysis_id = upload(client)
    stub.chat_error = type("RateLimitError", (Exception,), {})("quota")

    response = ask(client, analysis_id, "How much leave?")
    assert response.status_code == 502
    assert "rate limit" in response.get_json()["error"].lower()


def test_an_unexpected_crash_is_a_500_without_a_traceback(client, stub,
                                                          monkeypatch):
    analysis_id = upload(client)

    def boom(*_args, **_kwargs):
        raise RuntimeError("internal detail that must not leak")

    monkeypatch.setattr(chat_service, "ask", boom)
    response = ask(client, analysis_id, "How much leave?")
    body = response.get_json()

    assert response.status_code == 500
    assert "internal detail" not in body["error"]
    assert "Traceback" not in body["error"]


# ------------------------------------------------------------ suggestions

def test_suggestions_are_returned_and_cached(client, stub):
    analysis_id = upload(client)

    first = client.get(f"/api/chat/suggestions/{analysis_id}")
    assert first.status_code == 200
    body = first.get_json()
    assert body["source"] == "generated"
    assert len(body["questions"]) == 4
    assert body["cached"] is False

    calls = len(stub.suggestion_calls)
    second = client.get(f"/api/chat/suggestions/{analysis_id}").get_json()

    assert second["cached"] is True
    assert second["questions"] == body["questions"]
    # Cached, so no second request was paid for.
    assert len(stub.suggestion_calls) == calls


def test_suggestions_fall_back_without_a_key(client):
    analysis_id = upload(client)
    body = client.get(f"/api/chat/suggestions/{analysis_id}").get_json()

    assert body["source"] == "fallback"
    assert body["questions"] == list(FALLBACK_SUGGESTIONS)


def test_suggestions_fall_back_when_the_api_fails(client, stub):
    analysis_id = upload(client)
    stub.suggestion_error = type("APITimeoutError", (Exception,), {})("slow")

    body = client.get(f"/api/chat/suggestions/{analysis_id}").get_json()
    assert body["source"] == "fallback"
    # A fallback is not cached, so a later attempt can still generate.
    assert body["cached"] is False


def test_suggestions_for_an_unknown_analysis_is_a_404(client, stub):
    assert client.get("/api/chat/suggestions/ghost").status_code == 404

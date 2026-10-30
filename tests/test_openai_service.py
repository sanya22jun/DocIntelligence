"""Client wrapper: error translation, usage accounting, key handling."""
import pytest

from app.services.openai_service import (
    KIND_AUTH,
    KIND_BAD_REQUEST,
    KIND_CONNECTION,
    KIND_INVALID_RESPONSE,
    KIND_RATE_LIMIT,
    KIND_TIMEOUT,
    KIND_TRUNCATED,
    KIND_UNKNOWN,
    OpenAIService,
    OpenAIServiceError,
    UsageRecord,
    build_service,
    _translate,
)
from config import TestConfig
from stubs import StubClient


def service(client=None, **kwargs):
    return OpenAIService(
        api_key="test-key", client=client or StubClient(), **kwargs
    )


# ------------------------------------------------------------------ keys

def test_missing_key_raises_auth_error_on_first_use():
    bare = OpenAIService(api_key="")
    with pytest.raises(OpenAIServiceError) as exc:
        bare.client
    assert exc.value.kind == KIND_AUTH
    assert "OPENAI_API_KEY" in exc.value.message


def test_building_the_service_never_requires_a_key():
    # Importing and constructing must work with no key configured, so
    # the app starts and the offline tests run.
    built = build_service(TestConfig)
    assert built.embedding_model == "text-embedding-3-small"
    assert built.reasoning_model == "gpt-4.1-mini"


def test_error_details_never_echo_a_key_fragment():
    error = _translate(RuntimeError("bad key sk-abcdef123 rejected"))
    assert "sk-" not in error.message


# ---------------------------------------------------------- translation

@pytest.mark.parametrize("name,expected", [
    ("AuthenticationError", KIND_AUTH),
    ("PermissionDeniedError", KIND_AUTH),
    ("RateLimitError", KIND_RATE_LIMIT),
    ("APITimeoutError", KIND_TIMEOUT),
    ("APIConnectionError", KIND_CONNECTION),
    ("BadRequestError", KIND_BAD_REQUEST),
    ("NotFoundError", KIND_BAD_REQUEST),
    ("LengthFinishReasonError", KIND_TRUNCATED),
    ("InternalServerError", "server_error"),
    ("SomethingBrandNew", KIND_UNKNOWN),
])
def test_sdk_errors_are_classified(name, expected):
    exc = type(name, (Exception,), {})("boom")
    assert _translate(exc).kind == expected


def test_retryable_and_splittable_flags():
    assert OpenAIServiceError(KIND_RATE_LIMIT, "x").retryable
    assert OpenAIServiceError(KIND_TIMEOUT, "x").retryable
    assert not OpenAIServiceError(KIND_AUTH, "x").retryable
    assert OpenAIServiceError(KIND_TRUNCATED, "x").splittable
    assert not OpenAIServiceError(KIND_RATE_LIMIT, "x").splittable


# ------------------------------------------------------------ embeddings

def test_embed_returns_vectors_in_input_order():
    client = StubClient()
    vectors = service(client).embed(["annual leave", "parking garage"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 8
    assert client.embedding_calls == [["annual leave", "parking garage"]]


def test_embed_rejects_a_short_response():
    client = StubClient()
    client.drop_one_vector = True
    with pytest.raises(OpenAIServiceError) as exc:
        service(client).embed(["a", "b"])
    assert exc.value.kind == KIND_INVALID_RESPONSE


def test_embed_translates_sdk_failures():
    client = StubClient()
    client.embedding_error = type("RateLimitError", (Exception,), {})("slow down")
    with pytest.raises(OpenAIServiceError) as exc:
        service(client).embed(["a"])
    assert exc.value.kind == KIND_RATE_LIMIT


def test_embed_of_nothing_makes_no_call():
    client = StubClient()
    assert service(client).embed([]) == []
    assert client.embedding_calls == []


# ------------------------------------------------------------- reasoning

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["results"], "properties": {"results": {"type": "array"}}}


def call(client):
    return service(client).complete_json("sys", "PAIR a|b", SCHEMA)


def test_complete_json_parses_a_good_reply():
    client = StubClient()
    client.result_overrides["a|b"] = {
        "relationship": "NEUTRAL", "category": "NOT_APPLICABLE",
        "explanation": "ok", "conflicting_span_a": "",
        "conflicting_span_b": "", "confidence": 0.5,
    }
    payload = call(client)
    assert payload["results"][0]["pair_key"] == "a|b"


@pytest.mark.parametrize("behaviour,kind", [
    ("truncated", KIND_TRUNCATED),
    ("bad_json", KIND_INVALID_RESPONSE),
    ("not_object", KIND_INVALID_RESPONSE),
    ("empty", KIND_INVALID_RESPONSE),
])
def test_complete_json_rejects_bad_replies(behaviour, kind):
    client = StubClient()
    client.reasoning_behaviours = [behaviour]
    with pytest.raises(OpenAIServiceError) as exc:
        call(client)
    assert exc.value.kind == kind


# ----------------------------------------------------------------- usage

def test_usage_is_measured_and_costed():
    client = StubClient()
    svc = service(client, prices={
        "embedding": 0.02, "reasoning_input": 0.40, "reasoning_output": 1.60,
    })
    svc.embed(["annual leave policy"])
    svc.complete_json("sys", "PAIR a|b", SCHEMA)

    usage = svc.usage.to_dict()
    assert usage["embedding_requests"] == 1
    assert usage["reasoning_requests"] == 1
    assert usage["reasoning_input_tokens"] == 120
    assert usage["reasoning_output_tokens"] == 90
    assert usage["total_tokens"] == usage["embedding_tokens"] + 210
    assert usage["estimated_cost_usd"] > 0
    assert "Not a billing figure" in usage["cost_basis"]


def test_cost_estimate_arithmetic():
    usage = UsageRecord({
        "embedding": 0.02, "reasoning_input": 0.40, "reasoning_output": 1.60,
    })
    usage.add_embedding(1_000_000)
    usage.add_reasoning(1_000_000, 1_000_000)
    # 0.02 + 0.40 + 1.60
    assert usage.estimated_cost_usd() == pytest.approx(2.02)


def test_failed_requests_are_counted():
    client = StubClient()
    client.embedding_error = type("APITimeoutError", (Exception,), {})("slow")
    svc = service(client)
    with pytest.raises(OpenAIServiceError):
        svc.embed(["a"])
    assert svc.usage.to_dict()["failed_requests"] == 1

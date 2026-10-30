"""POST /api/analyze - the asynchronous analysis workflow."""
import time

import pytest

from app.services import analysis_pipeline
from app.services.embedding_service import cache_clear
from app.services.openai_service import OpenAIService
from conftest import upload_file
from stubs import StubClient

LEAVE_2023 = (
    b"Full-time employees are entitled to 20 days of paid annual leave.\n"
    b"Employees may take up to 10 days of paid sick leave each year.\n"
)
LEAVE_2024 = (
    b"Full-time employees receive 15 days of paid annual leave.\n"
    b"The company provides 10 days of paid sick leave annually.\n"
)
PARKING = (
    b"Visitor parking is located on the second level of the north garage.\n"
    b"The office cafeteria operates between 08:00 and 18:00 on weekdays.\n"
)


@pytest.fixture(autouse=True)
def clean_cache():
    cache_clear()
    yield
    cache_clear()


@pytest.fixture
def stub(app, monkeypatch):
    """Route the pipeline at a stub client instead of the real API."""
    client = StubClient()
    # The route refuses to start without a configured key; this fixture
    # stands in for a properly configured application.
    app.config["OPENAI_API_KEY"] = "test-key"

    def fake_build(config, client_override=None):
        return OpenAIService(api_key="test-key", client=client)

    monkeypatch.setattr(analysis_pipeline, "build_service", fake_build)
    return client


# ---------------------------------------------------------------- helpers

def upload(client, files=None):
    files = files or [
        upload_file(LEAVE_2023, "policy_2023.txt"),
        upload_file(LEAVE_2024, "handbook_2024.txt"),
    ]
    response = client.post(
        "/api/upload",
        data={"documents": files},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    return response.get_json()["analysis_id"]


def start(client, analysis_id=None, **body):
    if analysis_id is not None:
        body["analysis_id"] = analysis_id
    return client.post("/api/analyze", json=body)


def settle(client, analysis_id, timeout=15.0):
    """Poll /api/results until the worker thread finishes or fails."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/results/{analysis_id}").get_json()
        if body["state"] in ("done", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"analysis {analysis_id} did not finish in time")


def analyze(client, analysis_id, **body):
    """Start and wait; returns the settled /api/results payload."""
    response = start(client, analysis_id, **body)
    assert response.status_code in (200, 202), response.get_json()
    return settle(client, analysis_id)


# ------------------------------------------------------------ validation

def test_analysis_id_is_required(client):
    response = start(client)
    assert response.status_code == 400
    assert "analysis_id" in response.get_json()["error"]


def test_unknown_analysis_id_returns_404(client):
    response = start(client, "nosuchthing")
    assert response.status_code == 404
    assert response.get_json()["error"]


def test_single_usable_document_is_refused_immediately(client, stub):
    analysis_id = upload(client, [
        upload_file(LEAVE_2023, "good.txt"),
        upload_file(b"HEADINGS ONLY\nSECTION A\n", "headings.txt"),
    ])
    response = start(client, analysis_id)

    assert response.status_code == 422
    assert "two different documents" in response.get_json()["error"]
    # Refused before the worker started, so nothing was spent.
    assert stub.embedding_calls == []


def test_missing_key_is_refused_before_any_work(client):
    # No stub fixture, so the app still has an empty key.
    analysis_id = upload(client)
    response = start(client, analysis_id)
    body = response.get_json()

    assert response.status_code == 503
    assert body["kind"] == "auth"
    assert "OPENAI_API_KEY" in body["error"]
    assert "sk-" not in body["error"]


# ------------------------------------------------------------ happy path

def test_analyze_starts_a_run_and_reports_the_stages(client, stub):
    analysis_id = upload(client)
    response = start(client, analysis_id)
    body = response.get_json()

    assert response.status_code == 202
    assert body["state"] == "running"
    assert body["started"] is True
    assert body["stages"] == [
        "embedding", "candidate_selection", "reasoning", "report",
    ]

    settled = settle(client, analysis_id)
    assert settled["state"] == "done"


def test_completed_run_produces_a_report(client, stub):
    analysis_id = upload(client)
    settled = analyze(client, analysis_id)

    report = settled["analysis"]
    assert report["overview"]["documents_analyzed"] == 2
    assert report["overview"]["contradictions_detected"] >= 1
    assert report["usage"]["reasoning_requests"] >= 1
    assert report["models"]["reasoning"] == "gpt-4.1-mini"


def test_results_carry_document_and_page_references(client, stub):
    analysis_id = upload(client)
    report = analyze(client, analysis_id)["analysis"]

    assert report["results"]
    for item in report["results"]:
        for side in ("document_a", "document_b"):
            assert item[side]["document_name"].endswith(".txt")
            assert "document_id" in item[side]
            assert "page_number" in item[side]
        assert (
            item["document_a"]["document_id"]
            != item["document_b"]["document_id"]
        )


def test_report_is_attached_to_the_stored_analysis(client, stub):
    analysis_id = upload(client)
    analyze(client, analysis_id)

    stored = client.get(f"/api/analysis/{analysis_id}").get_json()
    assert stored["analysis"]["overview"]["contradictions_detected"] >= 1
    # Embeddings must not be serialised into the stored payload.
    assert "embeddings" not in stored
    assert "vectors" not in stored


def test_three_documents_are_compared_across_documents(client, stub):
    analysis_id = upload(client, [
        upload_file(LEAVE_2023, "policy_2023.txt"),
        upload_file(LEAVE_2024, "handbook_2024.txt"),
        upload_file(PARKING, "facilities.txt"),
    ])
    report = analyze(client, analysis_id)["analysis"]

    assert report["overview"]["documents_analyzed"] == 3
    names = set()
    for item in report["results"]:
        names.add(item["document_a"]["document_name"])
        names.add(item["document_b"]["document_name"])
    assert len(names) >= 2


# ---------------------------------------------------------------- caching

def test_repeat_request_is_served_from_the_store(client, stub):
    analysis_id = upload(client)
    analyze(client, analysis_id)
    calls_after_first = len(stub.reasoning_calls)

    response = start(client, analysis_id)
    body = response.get_json()

    assert response.status_code == 200
    assert body["cached"] is True
    assert body["started"] is False
    # No further spend on an identical request.
    assert len(stub.reasoning_calls) == calls_after_first


def test_force_reruns_the_analysis(client, stub):
    analysis_id = upload(client)
    analyze(client, analysis_id)
    calls_after_first = len(stub.reasoning_calls)

    response = start(client, analysis_id, force=True)
    assert response.status_code == 202
    assert response.get_json()["started"] is True

    settle(client, analysis_id)
    assert len(stub.reasoning_calls) > calls_after_first


def test_duplicate_submission_does_not_start_a_second_run(client, stub, app):
    """A second POST while one is running must not spend twice."""
    from app import store

    analysis_id = upload(client)
    # Simulate the first run still being in flight.
    store.set_progress(analysis_id, store.STATE_RUNNING, stage="reasoning")

    response = start(client, analysis_id)
    body = response.get_json()

    assert response.status_code == 202
    assert body["started"] is False
    assert body["state"] == "running"
    assert stub.reasoning_calls == []


# --------------------------------------------------------------- failures

def test_api_outage_surfaces_as_a_failed_state(client, stub):
    analysis_id = upload(client)
    stub.embedding_error = type("APIConnectionError", (Exception,), {})("down")

    settled = analyze(client, analysis_id)

    assert settled["state"] == "failed"
    assert "Could not reach the OpenAI API" in settled["error"]
    assert "analysis" not in settled


def test_worker_crash_surfaces_without_a_traceback(client, stub, monkeypatch):
    analysis_id = upload(client)

    def boom(*_args, **_kwargs):
        raise RuntimeError("internal detail that must not be exposed")

    monkeypatch.setattr(analysis_pipeline, "run_analysis", boom)
    settled = analyze(client, analysis_id)

    assert settled["state"] == "failed"
    assert "internal detail" not in settled["error"]
    assert "Traceback" not in settled["error"]


def test_health_reports_key_presence_without_the_key(client):
    body = client.get("/api/health").get_json()
    assert body["openai_configured"] is False
    assert body["embedding_model"] == "text-embedding-3-small"
    assert body["reasoning_model"] == "gpt-4.1-mini"
    assert not any("sk-" in str(value) for value in body.values())


# ------------------------------------------------------- malformed input

@pytest.mark.parametrize("body", [
    {"analysis_id": {"nested": 1}},   # object instead of a string
    {"analysis_id": ["a", "b"]},      # array instead of a string
    {"analysis_id": 12345},           # number instead of a string
    {"analysis_id": True},
])
def test_non_string_analysis_id_is_a_400_not_a_crash(client, body):
    """The id becomes a dict key, so a non-string must be refused."""
    response = client.post("/api/analyze", json=body)
    assert response.status_code == 400
    assert "analysis_id" in response.get_json()["error"]


def test_invalid_json_body_is_a_400(client):
    response = client.post(
        "/api/analyze", data="{not json",
        content_type="application/json",
    )
    assert response.status_code == 400


def test_non_object_json_body_is_a_400(client):
    response = client.post("/api/analyze", json=["not", "an", "object"])
    assert response.status_code == 400


def test_an_overlong_id_is_simply_not_found(client):
    response = client.post("/api/analyze", json={"analysis_id": "x" * 10000})
    assert response.status_code == 404

"""GET /api/results and GET /api/export."""
import csv
import io
import json
import time

import pytest

from app.services import analysis_pipeline
from app.services.embedding_service import cache_clear
from app.services.export_service import CSV_COLUMNS, ExportError, build_export
from app.services.openai_service import OpenAIService
from conftest import upload_file
from stubs import StubClient

DOC_A = (
    b"Full-time employees are entitled to 20 days of paid annual leave.\n"
    b"Employees may take up to 10 days of paid sick leave each year.\n"
)
DOC_B = (
    b"Full-time employees receive 15 days of paid annual leave.\n"
    b"The company provides 10 days of paid sick leave annually.\n"
)


@pytest.fixture(autouse=True)
def clean_cache():
    cache_clear()
    yield
    cache_clear()


@pytest.fixture
def stub(app, monkeypatch):
    client = StubClient()
    app.config["OPENAI_API_KEY"] = "test-key"
    monkeypatch.setattr(
        analysis_pipeline, "build_service",
        lambda config, client_override=None: OpenAIService(
            api_key="test-key", client=client
        ),
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


def analyzed(client, analysis_id, timeout=15.0):
    client.post("/api/analyze", json={"analysis_id": analysis_id})
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/results/{analysis_id}").get_json()
        if body["state"] in ("done", "failed"):
            assert body["state"] == "done", body.get("error")
            return body
        time.sleep(0.02)
    raise AssertionError("analysis did not finish in time")


# ----------------------------------------------------------- /api/results

def test_results_before_any_analysis_reports_idle(client, stub):
    analysis_id = upload(client)
    body = client.get(f"/api/results/{analysis_id}").get_json()

    assert body["state"] == "idle"
    assert body["analysis_id"] == analysis_id
    assert "analysis" not in body
    # Document context is available even before analysis.
    assert body["summary"]["documents_processed"] == 2
    assert len(body["documents"]) == 2


def test_results_lists_the_stage_order(client, stub):
    analysis_id = upload(client)
    body = client.get(f"/api/results/{analysis_id}").get_json()

    assert body["stages"] == [
        "embedding", "candidate_selection", "reasoning", "report",
    ]


def test_results_returns_the_report_when_done(client, stub):
    analysis_id = upload(client)
    body = analyzed(client, analysis_id)

    assert body["state"] == "done"
    assert body["analysis"]["results"]
    assert body["analysis"]["overview"]["contradictions_detected"] >= 1


def test_results_for_an_unknown_id_is_404(client):
    response = client.get("/api/results/ghost")
    assert response.status_code == 404
    assert response.get_json()["error"]


# ------------------------------------------------------------ /api/export

def test_json_export_downloads_the_full_report(client, stub):
    analysis_id = upload(client)
    analyzed(client, analysis_id)

    response = client.get(f"/api/export/{analysis_id}?format=json")

    assert response.status_code == 200
    assert response.mimetype == "application/json"
    disposition = response.headers["Content-Disposition"]
    assert disposition.startswith("attachment;")
    assert ".json" in disposition

    payload = json.loads(response.data.decode("utf-8"))
    assert payload["analysis_id"] == analysis_id
    assert payload["analysis"]["results"]
    assert payload["document_summary"]["documents_processed"] == 2
    assert len(payload["documents"]) == 2
    assert payload["exported_at"]
    # The caveats travel with the data.
    assert "not statistically calibrated" in payload["notes"]["confidence"]
    assert "not evidence" in payload["notes"]["similarity"]
    assert "not a billing figure" in payload["notes"]["cost"].lower()


def test_csv_export_has_one_row_per_result(client, stub):
    analysis_id = upload(client)
    report = analyzed(client, analysis_id)["analysis"]

    response = client.get(f"/api/export/{analysis_id}?format=csv")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert ".csv" in response.headers["Content-Disposition"]

    text = response.data.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    assert list(rows[0].keys()) == list(CSV_COLUMNS)
    assert len(rows) == len(report["results"])

    first = report["results"][0]
    assert rows[0]["pair_key"] == first["pair_key"]
    assert rows[0]["relationship"] == first["relationship"]
    assert rows[0]["statement_a"] == first["evidence_a"]
    assert rows[0]["statement_b"] == first["evidence_b"]
    assert rows[0]["explanation"] == first["explanation"]
    assert rows[0]["document_a"] == first["document_a"]["document_name"]


def test_csv_leaves_the_page_cell_empty_for_txt_documents(client, stub):
    analysis_id = upload(client)
    analyzed(client, analysis_id)

    text = client.get(
        f"/api/export/{analysis_id}?format=csv"
    ).data.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    # TXT files have no pagination, so an invented page number would be a
    # false citation; the cell is blank instead.
    assert all(row["page_a"] == "" for row in rows)


def test_csv_starts_with_a_bom_for_excel(client, stub):
    analysis_id = upload(client)
    analyzed(client, analysis_id)

    data = client.get(f"/api/export/{analysis_id}?format=csv").data
    assert data.startswith(b"\xef\xbb\xbf")


def test_export_defaults_to_json(client, stub):
    analysis_id = upload(client)
    analyzed(client, analysis_id)

    response = client.get(f"/api/export/{analysis_id}")
    assert response.status_code == 200
    assert response.mimetype == "application/json"


def test_export_before_analysis_explains_itself(client, stub):
    analysis_id = upload(client)
    response = client.get(f"/api/export/{analysis_id}?format=json")

    assert response.status_code == 400
    assert "no results to export yet" in response.get_json()["error"]


def test_unsupported_export_format_is_rejected(client, stub):
    analysis_id = upload(client)
    analyzed(client, analysis_id)

    response = client.get(f"/api/export/{analysis_id}?format=pdf")
    assert response.status_code == 400
    assert "Unsupported export format" in response.get_json()["error"]


def test_export_for_an_unknown_id_is_404(client):
    response = client.get("/api/export/ghost?format=json")
    assert response.status_code == 404


# ------------------------------------------------- export builder (unit)

def test_builder_rejects_a_record_with_no_analysis():
    with pytest.raises(ExportError, match="no results to export"):
        build_export({"analysis_id": "abc"}, "json")


def test_builder_filename_is_safe_and_stamped():
    record = {
        "analysis_id": "../../etc/passwd",
        "analysis": {"results": []},
    }
    _payload, _mimetype, filename = build_export(record, "csv")

    assert ".." not in filename
    assert "/" not in filename
    assert filename.startswith("crossdoc-analysis-")
    assert filename.endswith(".csv")


def test_builder_handles_a_report_with_no_results():
    record = {"analysis_id": "abc", "analysis": {"results": []}}
    payload, _mimetype, _filename = build_export(record, "csv")

    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    assert rows == []

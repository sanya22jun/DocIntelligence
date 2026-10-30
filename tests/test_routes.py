"""End-to-end HTTP tests for the Phase 1 workflow."""
from conftest import make_pdf, sample_bytes, upload_file

TWO_SENTENCES = (
    b"Employees are entitled to 20 days of annual leave each year.\n"
    b"Unused annual leave may be carried over up to five days.\n"
)
OTHER_SENTENCES = (
    b"Employees are entitled to 15 days of annual leave each year.\n"
    b"Unused annual leave is forfeited at the end of the year.\n"
)


def post(client, files):
    return client.post(
        "/api/upload",
        data={"documents": files},
        content_type="multipart/form-data",
    )


# ------------------------------------------------------------------- the page

def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"CrossDoc Intelligence" in response.data
    assert b"AI-powered semantic document consistency analysis." in response.data


def test_health_reports_the_segmenter(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert body["segmenter"]


# --------------------------------------------------------------- upload rules

def test_upload_requires_at_least_two_documents(client):
    response = post(client, [upload_file(TWO_SENTENCES, "a.txt")])
    assert response.status_code == 400
    assert "At least 2 documents" in response.get_json()["error"]


def test_upload_with_no_documents_is_rejected(client):
    response = post(client, [])
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_upload_above_the_file_limit_is_rejected(client, app):
    limit = app.config["MAX_FILES_PER_ANALYSIS"]
    files = [
        upload_file(TWO_SENTENCES, f"doc{i}.txt") for i in range(limit + 1)
    ]
    response = post(client, files)
    assert response.status_code == 400
    assert "At most" in response.get_json()["error"]


# ------------------------------------------------------------- happy path

def test_upload_processes_txt_and_pdf_together(client):
    pdf = make_pdf([
        "Employees are entitled to 20 days of annual leave each year.",
        "Managers must approve leave requests within three working days.",
    ])
    response = post(client, [
        upload_file(TWO_SENTENCES, "handbook.txt"),
        upload_file(pdf, "policy.pdf"),
    ])

    assert response.status_code == 200
    body = response.get_json()

    assert body["analysis_id"]
    assert body["summary"]["documents_uploaded"] == 2
    assert body["summary"]["documents_processed"] == 2
    assert body["summary"]["documents_failed"] == 0
    assert body["summary"]["total_sentences"] >= 4
    assert body["errors"] == []

    # Source references survive the pipeline.
    txt = [s for s in body["sentences"] if s["document_name"] == "handbook.txt"]
    pdf_sentences = [
        s for s in body["sentences"] if s["document_name"] == "policy.pdf"
    ]
    assert txt and pdf_sentences
    assert all(s["page_number"] is None for s in txt)
    assert sorted({s["page_number"] for s in pdf_sentences}) == [1, 2]

    ids = [s["sentence_id"] for s in body["sentences"]]
    assert len(set(ids)) == len(ids)
    assert {s["document_id"] for s in body["sentences"]} == {"doc_001", "doc_002"}


def test_upload_accepts_the_sample_documents(client):
    names = [
        "hr_leave_policy_2023.txt",
        "hr_handbook_2024.txt",
        "finance_travel_policy.txt",
    ]
    response = post(client, [
        upload_file(sample_bytes(name), name) for name in names
    ])

    assert response.status_code == 200
    body = response.get_json()
    assert body["summary"]["documents_processed"] == 3
    assert body["summary"]["total_sentences"] >= 30


# -------------------------------------------------------- partial failures

def test_invalid_file_is_reported_without_losing_the_others(client):
    response = post(client, [
        upload_file(TWO_SENTENCES, "good.txt"),
        upload_file(OTHER_SENTENCES, "also_good.txt"),
        upload_file(b"binary-ish", "notes.docx"),
    ])

    assert response.status_code == 200
    body = response.get_json()

    assert body["summary"]["documents_uploaded"] == 3
    assert body["summary"]["documents_processed"] == 2
    assert body["summary"]["documents_failed"] == 1

    failed = [d for d in body["documents"] if d["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["document_name"] == "notes.docx"
    assert "not supported" in failed[0]["error"]
    assert body["errors"][0]["document_name"] == "notes.docx"


def test_all_documents_failing_returns_422(client):
    response = post(client, [
        upload_file(b"x", "a.docx"),
        upload_file(b"this is not a pdf", "b.pdf"),
    ])

    assert response.status_code == 422
    body = response.get_json()
    assert "None of the uploaded documents" in body["error"]
    assert len(body["errors"]) == 2


def test_document_with_only_headings_is_reported(client):
    response = post(client, [
        upload_file(TWO_SENTENCES, "good.txt"),
        upload_file(b"POLICY\nSECTION A\n3.\n", "headings.txt"),
    ])

    body = response.get_json()
    failed = [d for d in body["documents"] if d["status"] == "failed"]
    assert len(failed) == 1
    assert "no complete statements" in failed[0]["error"]


# ------------------------------------------------------------- retrieval

def test_stored_analysis_can_be_retrieved(client):
    created = post(client, [
        upload_file(TWO_SENTENCES, "a.txt"),
        upload_file(OTHER_SENTENCES, "b.txt"),
    ]).get_json()

    response = client.get(f"/api/analysis/{created['analysis_id']}")
    assert response.status_code == 200

    stored = response.get_json()
    assert stored["analysis_id"] == created["analysis_id"]
    assert stored["summary"] == created["summary"]
    assert len(stored["sentences"]) == created["summary"]["total_sentences"]


def test_unknown_analysis_returns_404(client):
    response = client.get("/api/analysis/doesnotexist")
    assert response.status_code == 404
    assert response.get_json()["error"]


def test_oversized_request_returns_json_413(app):
    # Werkzeug aborts the request itself once MAX_CONTENT_LENGTH is passed;
    # this checks the handler answers API routes with JSON, not HTML.
    app.config["MAX_CONTENT_LENGTH"] = 2048
    client = app.test_client()

    payload = b"Employees receive 20 days of annual leave each year.\n" * 200
    response = post(client, [
        upload_file(payload, "a.txt"),
        upload_file(payload, "b.txt"),
    ])

    assert response.status_code == 413
    assert "too large" in response.get_json()["error"]

"""Validation and extraction tests."""
import pytest

from app.services.document_processor import (
    DocumentValidationError,
    extract_document,
    validate_upload,
)
from conftest import make_pdf

ALLOWED = {"pdf", "txt"}
MAX_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------- validation

def test_accepts_supported_extensions():
    validate_upload("policy.pdf", 1024, ALLOWED, MAX_BYTES)
    validate_upload("NOTES.TXT", 1024, ALLOWED, MAX_BYTES)


def test_rejects_unsupported_extension():
    with pytest.raises(DocumentValidationError) as exc:
        validate_upload("report.docx", 1024, ALLOWED, MAX_BYTES)
    assert ".docx" in str(exc.value) or "docx" in str(exc.value)


def test_rejects_missing_extension():
    with pytest.raises(DocumentValidationError, match="no file extension"):
        validate_upload("README", 512, ALLOWED, MAX_BYTES)


def test_rejects_empty_file():
    with pytest.raises(DocumentValidationError, match="empty"):
        validate_upload("blank.txt", 0, ALLOWED, MAX_BYTES)


def test_rejects_oversized_file():
    with pytest.raises(DocumentValidationError, match="exceeds"):
        validate_upload("huge.pdf", MAX_BYTES + 1, ALLOWED, MAX_BYTES)


# ---------------------------------------------------------------- extraction

def test_extracts_txt_without_page_numbers():
    data = b"Employees receive 20 days of annual leave."
    result = extract_document(data, "leave.txt", "doc_001")

    assert result.ok
    assert result.file_type == "txt"
    assert result.page_count == 1
    # TXT files are unpaginated, so no page reference is invented.
    assert result.pages[0].page_number is None
    assert "20 days" in result.pages[0].text


def test_extracts_pdf_with_one_entry_per_page():
    data = make_pdf([
        "Page one states 20 days of annual leave.",
        "Page two states 15 days of annual leave.",
    ])
    result = extract_document(data, "policy.pdf", "doc_002")

    assert result.ok
    assert result.page_count == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert "20 days" in result.pages[0].text
    assert "15 days" in result.pages[1].text


def test_decodes_non_utf8_text():
    data = "Café allowance is 1200 rupees.".encode("cp1252")
    result = extract_document(data, "meals.txt", "doc_003")

    assert result.ok
    assert "1200 rupees" in result.pages[0].text


def test_empty_pdf_is_reported_not_dropped():
    data = make_pdf([""])
    result = extract_document(data, "blank.pdf", "doc_004")

    assert not result.ok
    assert result.status == "failed"
    assert "No extractable text" in result.error


def test_corrupt_pdf_is_reported_not_dropped():
    result = extract_document(b"this is not a pdf at all", "broken.pdf", "doc_005")

    assert not result.ok
    assert result.error
    assert "Could not read the file" in result.error


def test_unsupported_type_reaching_extractor_is_reported():
    result = extract_document(b"data", "archive.zip", "doc_006")

    assert not result.ok
    assert "Unsupported file type" in result.error


def test_document_name_is_sanitised():
    result = extract_document(b"Some sentence that is long enough.", "../../etc/passwd.txt", "doc_007")

    assert ".." not in result.document_name
    assert "/" not in result.document_name

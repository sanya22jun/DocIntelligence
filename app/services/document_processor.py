"""Upload validation and text extraction for PDF and TXT documents.

Files are read straight from the request into memory, so no temporary
files are created on disk and there is nothing to clean up afterwards.
"""
import os

import fitz  # PyMuPDF
from werkzeug.utils import secure_filename

from .models import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    DocumentExtraction,
    PageText,
)

#: Encodings tried in order when decoding a TXT upload.
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class DocumentValidationError(Exception):
    """The upload was rejected before any extraction was attempted."""


def allowed_extension(filename, allowed):
    return extension_of(filename) in allowed


def extension_of(filename):
    return os.path.splitext(filename or "")[1].lower().lstrip(".")


def safe_display_name(filename, fallback="document"):
    """A filename safe to echo back to the browser and store in results."""
    cleaned = secure_filename(filename or "")
    return cleaned or fallback


def validate_upload(filename, size_bytes, allowed_extensions, max_bytes):
    """Raise :class:`DocumentValidationError` if this file is unusable.

    Checks the extension and the size only; content is validated later
    by the extractor, which can report a far more specific reason.
    """
    if not filename:
        raise DocumentValidationError("The file has no name.")

    ext = extension_of(filename)
    if not ext:
        raise DocumentValidationError(
            f"'{filename}' has no file extension. "
            f"Allowed types: {_pretty(allowed_extensions)}."
        )
    if ext not in allowed_extensions:
        raise DocumentValidationError(
            f"'{filename}' is a .{ext} file, which is not supported. "
            f"Allowed types: {_pretty(allowed_extensions)}."
        )
    if size_bytes == 0:
        raise DocumentValidationError(f"'{filename}' is empty (0 bytes).")
    if size_bytes > max_bytes:
        raise DocumentValidationError(
            f"'{filename}' is {_mb(size_bytes)} MB, which exceeds the "
            f"{_mb(max_bytes)} MB limit per file."
        )


def extract_document(data, filename, document_id):
    """Extract per-page text from one in-memory file.

    Always returns a :class:`DocumentExtraction`; extraction problems
    are reported through ``status``/``error`` rather than as exceptions,
    so one bad file cannot abort a multi-document analysis.
    """
    display_name = safe_display_name(filename)
    ext = extension_of(filename)

    try:
        if ext == "pdf":
            pages = _extract_pdf(data)
        elif ext == "txt":
            pages = _extract_txt(data)
        else:
            return _failed(
                document_id, display_name, ext,
                f"Unsupported file type '.{ext}'.",
            )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user below
        return _failed(
            document_id, display_name, ext,
            f"Could not read the file: {_reason(exc)}",
        )

    if not any(page.text.strip() for page in pages):
        return _failed(
            document_id, display_name, ext,
            "No extractable text was found. The document may be empty, "
            "or it may be a scanned/image-only PDF that needs OCR.",
        )

    return DocumentExtraction(
        document_id=document_id,
        document_name=display_name,
        file_type=ext,
        status=STATUS_SUCCESS,
        pages=pages,
    )


# ---------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------

def _extract_pdf(data):
    """Return one :class:`PageText` per PDF page, numbered from 1."""
    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError(
                "the PDF is password-protected and cannot be opened"
            )
        for index, page in enumerate(doc, start=1):
            pages.append(PageText(page_number=index, text=page.get_text("text")))
    return pages


def _extract_txt(data):
    """Return a single page for a plain-text file.

    TXT files have no pagination, so ``page_number`` stays None rather
    than inventing a page reference that the report could not justify.
    """
    for encoding in TEXT_ENCODINGS:
        try:
            return [PageText(page_number=None, text=data.decode(encoding))]
        except UnicodeDecodeError:
            continue
    raise ValueError(
        "the text could not be decoded with any supported encoding "
        f"({', '.join(TEXT_ENCODINGS)})"
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _failed(document_id, display_name, ext, message):
    return DocumentExtraction(
        document_id=document_id,
        document_name=display_name,
        file_type=ext,
        status=STATUS_FAILED,
        pages=[],
        error=message,
    )


def _reason(exc):
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _pretty(extensions):
    return ", ".join("." + e for e in sorted(extensions))


def _mb(value):
    return round(value / (1024 * 1024), 2)

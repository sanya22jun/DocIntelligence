"""Phase 1 pipeline: validate -> extract -> clean -> segment.

This module owns the orchestration so that the Flask layer stays thin
and the NLP phases can reuse the same entry point.
"""
from .document_processor import (
    DocumentValidationError,
    extension_of,
    extract_document,
    safe_display_name,
    validate_upload,
)
from .models import STATUS_FAILED, DocumentExtraction
from .text_preprocessor import segment_document


class UploadRejected(Exception):
    """The request as a whole is unusable (wrong number of files, etc.)."""


def check_batch(files, config):
    """Validate the request-level rules before reading any file."""
    minimum = getattr(config, "MIN_FILES_PER_ANALYSIS", 2)
    maximum = getattr(config, "MAX_FILES_PER_ANALYSIS", 10)

    if not files:
        raise UploadRejected("No documents were uploaded.")
    if len(files) < minimum:
        raise UploadRejected(
            f"At least {minimum} documents are required to compare them "
            f"against each other; {len(files)} was provided."
            if len(files) == 1 else
            f"At least {minimum} documents are required to compare them "
            f"against each other; {len(files)} were provided."
        )
    if len(files) > maximum:
        raise UploadRejected(
            f"At most {maximum} documents can be analysed at once; "
            f"{len(files)} were provided."
        )


def ingest_documents(files, config):
    """Process uploaded files into documents plus segmented sentences.

    ``files`` is a list of Werkzeug FileStorage objects. Every file
    produces one entry in ``documents``, successful or not.

    Returns a dict ready to be serialised as the API payload.
    """
    check_batch(files, config)

    allowed = getattr(config, "ALLOWED_EXTENSIONS", {"pdf", "txt"})
    max_bytes = getattr(config, "MAX_FILE_SIZE_MB", 10) * 1024 * 1024

    documents = []
    sentences = []
    next_index = 1

    for position, storage in enumerate(files, start=1):
        document_id = f"doc_{position:03d}"
        filename = storage.filename or ""

        data = _read(storage)
        try:
            validate_upload(filename, len(data), allowed, max_bytes)
        except DocumentValidationError as exc:
            documents.append(
                DocumentExtraction(
                    document_id=document_id,
                    document_name=safe_display_name(filename, filename or "unnamed"),
                    file_type=extension_of(filename),
                    status=STATUS_FAILED,
                    error=str(exc),
                )
            )
            continue

        extraction = extract_document(data, filename, document_id)
        if extraction.ok:
            document_sentences, next_index = segment_document(
                extraction, next_index, config
            )
            if not document_sentences:
                extraction.status = STATUS_FAILED
                extraction.error = (
                    "Text was extracted but no complete statements could be "
                    "identified. The document may contain only headings, "
                    "tables or fragments."
                )
            else:
                sentences.extend(document_sentences)

        documents.append(extraction)

    return _payload(documents, sentences)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _payload(documents, sentences):
    processed = [d for d in documents if d.ok]
    failed = [d for d in documents if not d.ok]

    return {
        "summary": {
            "documents_uploaded": len(documents),
            "documents_processed": len(processed),
            "documents_failed": len(failed),
            "total_pages": sum(d.page_count for d in processed),
            "total_characters": sum(d.char_count for d in processed),
            "total_sentences": len(sentences),
        },
        "documents": [d.to_dict() for d in documents],
        "sentences": [s.to_dict() for s in sentences],
        "errors": [
            {"document_name": d.document_name, "error": d.error}
            for d in failed
        ],
    }


def _read(storage):
    """Read a FileStorage fully into memory, leaving the stream rewound."""
    try:
        storage.stream.seek(0)
    except Exception:  # noqa: BLE001 - non-seekable streams are fine
        pass
    data = storage.read()
    try:
        storage.stream.seek(0)
    except Exception:  # noqa: BLE001
        pass
    return data or b""

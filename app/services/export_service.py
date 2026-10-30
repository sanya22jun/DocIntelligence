"""Export a completed analysis as JSON or CSV.

Exports are built from the stored report only. Nothing is recomputed or
re-worded here, so a downloaded file says exactly what the dashboard
says.
"""
import csv
import io
import json
import re
from datetime import datetime, timezone

FORMAT_JSON = "json"
FORMAT_CSV = "csv"
SUPPORTED_FORMATS = (FORMAT_JSON, FORMAT_CSV)

CSV_COLUMNS = (
    "pair_key",
    "relationship",
    "category",
    "confidence",
    "confidence_basis",
    "similarity",
    "document_a",
    "page_a",
    "statement_a",
    "document_b",
    "page_b",
    "statement_b",
    "explanation",
    "conflicting_span_a",
    "conflicting_span_b",
)


class ExportError(Exception):
    """The analysis cannot be exported yet, or the format is unknown."""


def build_export(record, fmt):
    """Return ``(payload_bytes, mimetype, filename)`` for a download."""
    fmt = (fmt or FORMAT_JSON).lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ExportError(
            f"Unsupported export format '{fmt}'. Use "
            f"{' or '.join(SUPPORTED_FORMATS)}."
        )

    report = record.get("analysis")
    if not report:
        raise ExportError(
            "This analysis has no results to export yet. Run the analysis "
            "first."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    base = f"crossdoc-analysis-{_slug(record.get('analysis_id', 'report'))}-{stamp}"

    if fmt == FORMAT_CSV:
        return (
            _csv_bytes(report),
            "text/csv; charset=utf-8",
            f"{base}.csv",
        )
    return (
        _json_bytes(record, report),
        "application/json; charset=utf-8",
        f"{base}.json",
    )


# ---------------------------------------------------------------------

def _json_bytes(record, report):
    """The full report plus the document list it was derived from."""
    payload = {
        "analysis_id": record.get("analysis_id"),
        "created_at": record.get("created_at"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "documents": record.get("documents") or [],
        "document_summary": record.get("summary") or {},
        "analysis": report,
        "notes": {
            "confidence": (
                "Confidence values are reported by the reasoning model and "
                "are not statistically calibrated probabilities."
            ),
            "similarity": (
                "Cosine similarity was used only to shortlist candidate "
                "pairs for analysis; it is not evidence of a contradiction."
            ),
            "cost": (
                "Token counts are reported by the API. The cost figure is an "
                "estimate from configured list prices, not a billing figure."
            ),
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _csv_bytes(report):
    """One row per classified pair.

    Written with a UTF-8 BOM so Excel opens accented text correctly.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore",
        lineterminator="\r\n",
    )
    writer.writeheader()

    for item in report.get("results") or []:
        first = item.get("document_a") or {}
        second = item.get("document_b") or {}
        writer.writerow({
            "pair_key": item.get("pair_key", ""),
            "relationship": item.get("relationship", ""),
            "category": item.get("category") or "",
            "confidence": _number(item.get("confidence")),
            "confidence_basis": item.get("confidence_basis", ""),
            "similarity": _number(item.get("similarity")),
            "document_a": first.get("document_name", ""),
            "page_a": _page(first.get("page_number")),
            "statement_a": item.get("evidence_a", ""),
            "document_b": second.get("document_name", ""),
            "page_b": _page(second.get("page_number")),
            "statement_b": item.get("evidence_b", ""),
            "explanation": item.get("explanation", ""),
            "conflicting_span_a": item.get("conflicting_span_a") or "",
            "conflicting_span_b": item.get("conflicting_span_b") or "",
        })

    return buffer.getvalue().encode("utf-8-sig")


def _number(value):
    return "" if value is None else value


def _page(value):
    # TXT files have no pages; an empty cell is honest, "1" would not be.
    return "" if value is None else value


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value))[:40] or "report"

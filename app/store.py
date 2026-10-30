"""In-memory analysis store.

A prototype does not need a database: results live for the lifetime of
the process, keyed by an opaque analysis id, so the dashboard can fetch
them again and the NLP phases can pick up the extracted statements
without asking the user to upload the files a second time.

The store is bounded, so a long-running process cannot grow without
limit, and guarded by a lock because Flask serves requests on threads.
"""
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

_LOCK = threading.Lock()
_ANALYSES = OrderedDict()


def new_id():
    return uuid.uuid4().hex[:12]


def save(payload, analysis_id=None, max_entries=20):
    """Store a payload and return its analysis id.

    The oldest entries are evicted once ``max_entries`` is exceeded.
    """
    analysis_id = analysis_id or new_id()
    record = dict(payload)
    record["analysis_id"] = analysis_id
    record.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    with _LOCK:
        _ANALYSES[analysis_id] = record
        _ANALYSES.move_to_end(analysis_id)
        while len(_ANALYSES) > max(1, max_entries):
            _ANALYSES.popitem(last=False)

    return analysis_id


def get(analysis_id):
    """Return a stored payload, or None if it is unknown or evicted."""
    with _LOCK:
        return _ANALYSES.get(analysis_id)


def update(analysis_id, **fields):
    """Merge fields into a stored payload. Returns the record or None."""
    with _LOCK:
        record = _ANALYSES.get(analysis_id)
        if record is None:
            return None
        record.update(fields)
        return record


# --- Analysis progress ------------------------------------------------
# The analysis runs on a worker thread so the browser can show real
# stage progress instead of an indeterminate spinner. These helpers are
# the only shared state between that thread and the request handlers.

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"


def set_progress(analysis_id, state, stage=None, message=None, error=None):
    """Record what the analysis worker is doing."""
    with _LOCK:
        record = _ANALYSES.get(analysis_id)
        if record is None:
            return None
        progress = {"state": state}
        if stage is not None:
            progress["stage"] = stage
        if message is not None:
            progress["message"] = message
        if error is not None:
            progress["error"] = error
        record["progress"] = progress
        return progress


def get_progress(analysis_id):
    with _LOCK:
        record = _ANALYSES.get(analysis_id)
        if record is None:
            return None
        return record.get("progress") or {"state": STATE_IDLE}


def is_running(analysis_id):
    progress = get_progress(analysis_id)
    return bool(progress and progress.get("state") == STATE_RUNNING)


def clear():
    """Drop every stored analysis (used by the test suite)."""
    with _LOCK:
        _ANALYSES.clear()


def count():
    with _LOCK:
        return len(_ANALYSES)

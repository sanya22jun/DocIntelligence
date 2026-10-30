"""HTTP layer. Keeps orchestration in the services package."""
import threading

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
)

from . import store
from .services import analysis_pipeline, chat_service
from .services.analysis_pipeline import STAGES, AnalysisRejected
from .services.chat_service import MAX_QUESTION_CHARS, ChatRejected
from .services.contradiction_service import CATEGORIES, RELATIONSHIPS
from .services.export_service import ExportError, build_export
from .services.ingestion import UploadRejected, ingest_documents
from .services.openai_service import (
    KIND_AUTH,
    OpenAIServiceError,
    build_service,
)

bp = Blueprint("main", __name__)

#: Sentences returned inline with an upload response. The full set stays
#: in the store so a large corpus does not bloat every response.
SENTENCE_PREVIEW_LIMIT = 300


@bp.get("/")
def index():
    config = current_app.config
    return render_template(
        "index.html",
        max_file_size_mb=config["MAX_FILE_SIZE_MB"],
        min_files=config["MIN_FILES_PER_ANALYSIS"],
        max_files=config["MAX_FILES_PER_ANALYSIS"],
        allowed_extensions=sorted(config["ALLOWED_EXTENSIONS"]),
        relationships=list(RELATIONSHIPS),
        categories=list(CATEGORIES),
        stages=list(STAGES),
        max_question_chars=MAX_QUESTION_CHARS,
    )


@bp.post("/api/upload")
def upload():
    """Validate, extract and segment the uploaded documents."""
    files = [
        f for f in request.files.getlist("documents")
        if f and f.filename
    ]

    try:
        payload = ingest_documents(files, _config())
    except UploadRejected as exc:
        return jsonify(error=str(exc)), 400
    except Exception:  # noqa: BLE001 - never leak a traceback to the client
        current_app.logger.exception("Document ingestion failed")
        return jsonify(
            error="The documents could not be processed due to a server error."
        ), 500

    if payload["summary"]["documents_processed"] == 0:
        return jsonify(
            error="None of the uploaded documents could be processed.",
            documents=payload["documents"],
            errors=payload["errors"],
        ), 422

    analysis_id = store.save(
        payload, max_entries=current_app.config["MAX_STORED_ANALYSES"]
    )

    response = {
        "analysis_id": analysis_id,
        "summary": payload["summary"],
        "documents": payload["documents"],
        "errors": payload["errors"],
        "sentences": payload["sentences"][:SENTENCE_PREVIEW_LIMIT],
        "sentences_truncated": (
            len(payload["sentences"]) > SENTENCE_PREVIEW_LIMIT
        ),
    }
    return jsonify(response), 200


@bp.post("/api/analyze")
def analyze():
    """Start the semantic analysis for a previously uploaded set.

    The work runs on a background thread so the browser can poll
    ``/api/results/<id>`` and show which stage is actually running.
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
    analysis_id = body.get("analysis_id") or request.form.get("analysis_id")

    # The id becomes a dict key, so it has to be a string: a JSON client
    # sending a list or an object must get a 400, not a server error.
    if not analysis_id or not isinstance(analysis_id, str):
        return jsonify(error="An 'analysis_id' is required."), 400

    record = store.get(analysis_id)
    if record is None:
        return jsonify(error=_GONE), 404

    if store.is_running(analysis_id):
        # A duplicate submission must not start a second paid run.
        return jsonify(
            analysis_id=analysis_id, state=store.STATE_RUNNING, started=False
        ), 202

    # Re-running costs real money, so an existing report is reused unless
    # the caller explicitly asks for a fresh one.
    if record.get("analysis") and not body.get("force"):
        return jsonify(
            analysis_id=analysis_id,
            state=store.STATE_DONE,
            started=False,
            cached=True,
            analysis=record["analysis"],
        ), 200

    # Cheap checks run here so an impossible analysis fails immediately
    # rather than in the background.
    try:
        analysis_pipeline.precheck(record, current_app.config)
    except AnalysisRejected as exc:
        return jsonify(error=str(exc)), 422

    if not current_app.config.get("OPENAI_API_KEY"):
        return jsonify(error=_NO_KEY, kind=KIND_AUTH), 503

    _start_worker(analysis_id, record)
    return jsonify(
        analysis_id=analysis_id,
        state=store.STATE_RUNNING,
        started=True,
        stages=list(STAGES),
    ), 202


@bp.get("/api/results/<analysis_id>")
def results(analysis_id):
    """Progress while the analysis runs, the report once it is done."""
    record = store.get(analysis_id)
    if record is None:
        return jsonify(error=_GONE), 404

    progress = store.get_progress(analysis_id)
    body = {
        "analysis_id": analysis_id,
        "state": progress.get("state", store.STATE_IDLE),
        "stage": progress.get("stage"),
        "message": progress.get("message"),
        "stages": list(STAGES),
        "summary": record.get("summary"),
        "documents": record.get("documents"),
    }

    if body["state"] == store.STATE_FAILED:
        body["error"] = progress.get("error")
        return jsonify(body), 200

    if record.get("analysis"):
        body["analysis"] = record["analysis"]
        if body["state"] == store.STATE_IDLE:
            body["state"] = store.STATE_DONE

    return jsonify(body), 200


@bp.post("/api/chat")
def chat():
    """Answer a question from the statements in one analysis."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}

    analysis_id = body.get("analysis_id")
    if not analysis_id or not isinstance(analysis_id, str):
        return jsonify(error="An 'analysis_id' is required."), 400

    record = store.get(analysis_id)
    if record is None:
        return jsonify(error=_GONE), 404

    if not current_app.config.get("OPENAI_API_KEY"):
        return jsonify(error=_NO_KEY, kind=KIND_AUTH), 503

    service = build_service(current_app.config)
    try:
        answer = chat_service.ask(
            record,
            body.get("question"),
            service,
            config=current_app.config,
            history=body.get("history"),
        )
    except ChatRejected as exc:
        return jsonify(error=str(exc)), 400
    except OpenAIServiceError as exc:
        status = 503 if exc.kind == KIND_AUTH else 502
        current_app.logger.warning("Chat failed: %s", exc.kind)
        return jsonify(error=exc.message, kind=exc.kind), status
    except Exception:  # noqa: BLE001 - never leak a traceback
        current_app.logger.exception("Chat request crashed")
        return jsonify(
            error="The question could not be answered due to a server error."
        ), 500

    answer["analysis_id"] = analysis_id
    answer["usage"] = service.usage.to_dict()
    return jsonify(answer), 200


@bp.get("/api/chat/suggestions/<analysis_id>")
def chat_suggestions(analysis_id):
    """Starter questions for this analysis.

    Generated once and cached on the record, since they only depend on
    the documents and the findings.
    """
    record = store.get(analysis_id)
    if record is None:
        return jsonify(error=_GONE), 404

    cached = record.get("chat_suggestions")
    if cached:
        return jsonify(
            analysis_id=analysis_id, cached=True, **cached
        ), 200

    if not current_app.config.get("OPENAI_API_KEY"):
        # Without a key the panel still needs something to show.
        return jsonify(
            analysis_id=analysis_id,
            cached=False,
            questions=list(chat_service.FALLBACK_SUGGESTIONS),
            source="fallback",
        ), 200

    service = build_service(current_app.config)
    questions, source = chat_service.suggest_questions(
        record, service, config=current_app.config
    )
    payload = {"questions": questions, "source": source}
    if source == "generated":
        store.update(analysis_id, chat_suggestions=payload)

    return jsonify(analysis_id=analysis_id, cached=False, **payload), 200


@bp.get("/api/export/<analysis_id>")
def export(analysis_id):
    """Download the analysis as JSON or CSV."""
    record = store.get(analysis_id)
    if record is None:
        return jsonify(error=_GONE), 404

    try:
        payload, mimetype, filename = build_export(
            record, request.args.get("format", "json")
        )
    except ExportError as exc:
        return jsonify(error=str(exc)), 400

    return Response(
        payload,
        mimetype=mimetype,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(payload)),
            "Cache-Control": "no-store",
        },
    )


@bp.get("/api/analysis/<analysis_id>")
def get_analysis(analysis_id):
    """Return a previously stored analysis in full."""
    record = store.get(analysis_id)
    if record is None:
        return jsonify(error=_GONE), 404
    return jsonify(record), 200


@bp.get("/api/health")
def health():
    from .services.text_preprocessor import get_segmenter

    model = current_app.config["SPACY_MODEL"]
    return jsonify(
        status="ok",
        segmenter=model if get_segmenter(model) else "regex-fallback",
        stored_analyses=store.count(),
        # Whether a key is present, never the key itself.
        openai_configured=bool(current_app.config.get("OPENAI_API_KEY")),
        embedding_model=current_app.config["OPENAI_EMBEDDING_MODEL"],
        reasoning_model=current_app.config["OPENAI_REASONING_MODEL"],
    ), 200


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------

_GONE = (
    "That analysis is no longer available. Please upload the documents again."
)
_NO_KEY = (
    "OPENAI_API_KEY is not configured. Add it to your .env file and restart "
    "the application."
)


def _start_worker(analysis_id, record):
    """Run the analysis on a daemon thread, recording its progress."""
    app = current_app._get_current_object()
    store.set_progress(
        analysis_id, store.STATE_RUNNING,
        stage=STAGES[0], message="Starting analysis",
    )

    def on_stage(stage, message):
        store.set_progress(
            analysis_id, store.STATE_RUNNING, stage=stage, message=message
        )

    def worker():
        with app.app_context():
            try:
                report = analysis_pipeline.run_analysis(
                    record, app.config, on_stage=on_stage
                )
            except AnalysisRejected as exc:
                store.set_progress(
                    analysis_id, store.STATE_FAILED, error=str(exc)
                )
            except OpenAIServiceError as exc:
                app.logger.warning("Analysis failed: %s", exc.kind)
                store.set_progress(
                    analysis_id, store.STATE_FAILED, error=exc.message
                )
            except Exception:  # noqa: BLE001 - keep the detail in the log
                app.logger.exception("Analysis pipeline crashed")
                store.set_progress(
                    analysis_id, store.STATE_FAILED,
                    error=(
                        "The analysis could not be completed due to a server "
                        "error."
                    ),
                )
            else:
                store.update(analysis_id, analysis=report)
                store.set_progress(
                    analysis_id, store.STATE_DONE,
                    message="Analysis complete",
                )

    threading.Thread(
        target=worker, name=f"analysis-{analysis_id}", daemon=True
    ).start()


def _config():
    """Expose app config as an attribute object for the service layer."""
    return _ConfigView(current_app.config)


class _ConfigView:
    """Read-only attribute access over Flask's config mapping."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __getattr__(self, name):
        try:
            return self._mapping[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

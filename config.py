"""Application configuration.

All tunable values are read from the environment so that nothing
secret or deployment-specific is hard-coded in the source tree.
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Loads E:\NLP\.env when present. Never commit that file.
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Base configuration shared by every environment."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-secret")
    DEBUG = _env_bool("FLASK_DEBUG", False)

    # ---- Upload rules ------------------------------------------------
    ALLOWED_EXTENSIONS = {"pdf", "txt"}
    MAX_FILE_SIZE_MB = _env_int("MAX_FILE_SIZE_MB", 10)
    MIN_FILES_PER_ANALYSIS = _env_int("MIN_FILES_PER_ANALYSIS", 2)
    MAX_FILES_PER_ANALYSIS = _env_int("MAX_FILES_PER_ANALYSIS", 10)

    # Hard ceiling Werkzeug enforces on the whole multipart request.
    # Slack of 2 MB covers multipart overhead and form fields.
    MAX_CONTENT_LENGTH = (
        MAX_FILE_SIZE_MB * MAX_FILES_PER_ANALYSIS + 2
    ) * 1024 * 1024

    # ---- Sentence segmentation --------------------------------------
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
    # A statement shorter than this is almost always a heading, a table
    # cell or an extraction artifact rather than a claim we can compare.
    MIN_SENTENCE_CHARS = _env_int("MIN_SENTENCE_CHARS", 25)
    MIN_SENTENCE_WORDS = _env_int("MIN_SENTENCE_WORDS", 4)
    # Pages are segmented in chunks so very large TXT files stay under
    # spaCy's max_length guard.
    SEGMENTATION_CHUNK_CHARS = _env_int("SEGMENTATION_CHUNK_CHARS", 90_000)

    # ---- In-memory analysis store -----------------------------------
    MAX_STORED_ANALYSES = _env_int("MAX_STORED_ANALYSES", 20)

    # ---- OpenAI ------------------------------------------------------
    # The key is read from the environment only; it is never written to
    # a template, logged, or sent to the frontend.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    OPENAI_REASONING_MODEL = os.getenv(
        "OPENAI_REASONING_MODEL", "gpt-4.1-mini"
    )
    OPENAI_TIMEOUT_SECONDS = _env_float("OPENAI_TIMEOUT_SECONDS", 30.0)
    OPENAI_MAX_RETRIES = _env_int("OPENAI_MAX_RETRIES", 2)

    # ---- Candidate selection ----------------------------------------
    # Cosine similarity only shortlists pairs for the reasoning model.
    # It is never treated as evidence of a contradiction.
    SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.65)
    TOP_K_CANDIDATES = _env_int("TOP_K_CANDIDATES", 5)
    MAX_CANDIDATE_PAIRS = _env_int("MAX_CANDIDATE_PAIRS", 100)
    EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 64)
    MAX_SENTENCES_PER_ANALYSIS = _env_int("MAX_SENTENCES_PER_ANALYSIS", 400)

    # ---- Reasoning ---------------------------------------------------
    REASONING_BATCH_SIZE = _env_int("REASONING_BATCH_SIZE", 8)
    MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 1400)

    # ---- Document chat -----------------------------------------------
    # Statements retrieved as context for one question. More context
    # means a better answer and a larger prompt.
    CHAT_TOP_K = _env_int("CHAT_TOP_K", 8)
    CHAT_MAX_OUTPUT_TOKENS = _env_int("CHAT_MAX_OUTPUT_TOKENS", 700)
    CHAT_SUGGESTION_TOKENS = _env_int("CHAT_SUGGESTION_TOKENS", 200)

    # ---- Cost estimation (USD per 1M tokens) -------------------------
    # Published list prices, overridable so a price change needs no code
    # edit. Used for estimates only; measured token counts come from the
    # API response itself.
    PRICE_EMBEDDING_PER_MTOK = _env_float("PRICE_EMBEDDING_PER_MTOK", 0.02)
    PRICE_REASONING_INPUT_PER_MTOK = _env_float(
        "PRICE_REASONING_INPUT_PER_MTOK", 0.40
    )
    PRICE_REASONING_OUTPUT_PER_MTOK = _env_float(
        "PRICE_REASONING_OUTPUT_PER_MTOK", 1.60
    )

    @property
    def max_file_size_bytes(self):
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


class TestConfig(Config):
    TESTING = True
    SECRET_KEY = "testing"
    # The test suite must never read, depend on, or spend against a real
    # key, whatever happens to be in the developer's .env file. Every
    # test injects a stub client instead.
    OPENAI_API_KEY = ""

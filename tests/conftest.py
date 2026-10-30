"""Shared pytest fixtures and helpers."""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app  # noqa: E402
from app import store  # noqa: E402
from config import TestConfig  # noqa: E402

SAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sample_documents")
)


@pytest.fixture
def app():
    application = create_app(TestConfig)
    yield application
    store.clear()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def config():
    return TestConfig


def make_pdf(pages):
    """Build an in-memory PDF, one page per string in ``pages``."""
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 750), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def upload_file(data, filename):
    """Build a (stream, filename) tuple for Flask's test client."""
    return io.BytesIO(data), filename


def sample_bytes(name):
    with open(os.path.join(SAMPLE_DIR, name), "rb") as handle:
        return handle.read()

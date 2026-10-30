"""Typed structures passed between pipeline stages.

Field order inside ``Sentence`` matches the contract agreed for the
NLP pipeline, so ``to_dict()`` serialises in that order.
"""
from dataclasses import dataclass, field
from typing import List, Optional

#: ``status`` values used by :class:`DocumentExtraction`.
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


@dataclass
class Sentence:
    """A single comparable statement taken from one document."""

    document_id: str
    document_name: str
    page_number: Optional[int]
    sentence_id: str
    text: str
    word_count: int = 0
    char_count: int = 0

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "sentence_id": self.sentence_id,
            "text": self.text,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }


@dataclass
class PageText:
    """Raw text of one page. ``page_number`` is None for TXT files."""

    page_number: Optional[int]
    text: str


@dataclass
class DocumentExtraction:
    """Outcome of reading one uploaded file.

    A document that could not be read is still returned, with
    ``status == STATUS_FAILED`` and a human-readable ``error``, so that
    nothing is silently dropped from the analysis.
    """

    document_id: str
    document_name: str
    file_type: str
    status: str
    pages: List[PageText] = field(default_factory=list)
    error: Optional[str] = None
    sentence_count: int = 0

    @property
    def page_count(self):
        return len(self.pages)

    @property
    def char_count(self):
        return sum(len(page.text) for page in self.pages)

    @property
    def ok(self):
        return self.status == STATUS_SUCCESS

    def to_dict(self):
        """Summary for API responses (page text itself is excluded)."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "file_type": self.file_type,
            "status": self.status,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "sentence_count": self.sentence_count,
            "error": self.error,
        }

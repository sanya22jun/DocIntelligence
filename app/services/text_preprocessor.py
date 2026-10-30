"""Text cleaning and sentence segmentation.

Cleaning is deliberately conservative: it removes extraction artifacts
(hyphenation across line breaks, running page numbers, stray whitespace)
but never lower-cases, stems, strips stop words or rewrites numbers,
dates, negations or punctuation. Those carry the very meaning the
contradiction detector has to compare.
"""
import re
import unicodedata

from .models import Sentence

# --- Artifact patterns -------------------------------------------------
# A line that is nothing but a page marker, e.g. "3", "- 4 -", "Page 2 of 9".
_PAGE_MARKER = re.compile(
    r"^\s*[-–—(\[]?\s*(?:page|pg\.?)?\s*\d{1,4}"
    r"\s*(?:of\s*\d{1,4})?\s*[-–—)\]]?\s*$",
    re.IGNORECASE,
)
# A line made only of separators / dot leaders, e.g. "......", "_____".
_SEPARATOR_LINE = re.compile(
    r"^[\s.\-_=~|*•·▪◦‣–—]+$"
)
# Soft hyphen and zero-width characters left behind by PDF extraction.
_INVISIBLE = re.compile(r"[­​‌‍﻿]")
# A word split across a line break: "annu-\nal" -> "annual".
_LINE_BREAK_HYPHEN = re.compile(r"([a-z])-\n([a-z])")
# Leading bullet glyphs or list markers at the start of a line.
_BULLET_PREFIX = re.compile(
    r"^\s*(?:[•·▪◦‣*]|[-–—](?=\s))\s*"
)
# Spaces of every width that PDFs emit, collapsed to one plain space.
_HORIZONTAL_WS = re.compile(
    r"[ \t  -   　]+"
)
_BLANK_RUN = re.compile(r"\n{3,}")
_SINGLE_BREAK = re.compile(r"(?<!\n)\n(?!\n)")

# Abbreviations that must not end a sentence in the regex fallback.
_ABBREVIATIONS = (
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "no", "fig", "inc", "ltd", "co", "dept", "approx",
)


def clean_text(raw):
    """Normalise one page of extracted text without altering meaning."""
    if not raw:
        return ""

    text = unicodedata.normalize("NFKC", raw)
    text = _INVISIBLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _LINE_BREAK_HYPHEN.sub(r"\1\2", text)

    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and (
            _PAGE_MARKER.match(stripped) or _SEPARATOR_LINE.match(stripped)
        ):
            continue  # running header/footer or table rule, not content
        line = _BULLET_PREFIX.sub("", line)
        if _is_heading_line(line.strip()):
            # Headings have no terminal punctuation, so without a block
            # break of their own they would be glued onto the sentence
            # that follows them.
            kept.extend(["", line.strip(), ""])
        else:
            kept.append(line)
    text = "\n".join(kept)

    # Paragraph breaks are meaningful; line wraps inside a paragraph are
    # an artifact of the page layout, so they become plain spaces.
    text = _BLANK_RUN.sub("\n\n", text)
    text = _SINGLE_BREAK.sub(" ", text)
    text = _HORIZONTAL_WS.sub(" ", text)
    text = "\n".join(part.strip() for part in text.split("\n"))
    return text.strip()


# ---------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------

_SEGMENTER = None
_SEGMENTER_NAME = None


def get_segmenter(model_name="en_core_web_sm"):
    """Load the spaCy pipeline once per process.

    Returns None when the model is unavailable, in which case callers
    fall back to :func:`_regex_sentences`.
    """
    global _SEGMENTER, _SEGMENTER_NAME
    if _SEGMENTER_NAME == model_name:
        return _SEGMENTER

    try:
        import spacy

        # The dependency parser supplies sentence boundaries; NER and the
        # lemmatizer are pure overhead for this task.
        _SEGMENTER = spacy.load(
            model_name, exclude=["ner", "lemmatizer", "textcat"]
        )
    except Exception:  # noqa: BLE001 - optional model, fall back to regex
        _SEGMENTER = None
    _SEGMENTER_NAME = model_name
    return _SEGMENTER


def split_sentences(text, model_name="en_core_web_sm", chunk_chars=90_000):
    """Split cleaned text into sentence strings.

    Each paragraph is segmented on its own. Headings carry no terminal
    punctuation, so segmenting a whole page at once would glue a heading
    onto the first sentence beneath it; per-paragraph segmentation keeps
    the heading separate, where the length filter can discard it.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    nlp = get_segmenter(model_name)
    if nlp is None:
        return [s for p in paragraphs for s in _regex_sentences(p)]

    limit = max(1000, min(chunk_chars, nlp.max_length - 1000))
    blocks = [chunk for p in paragraphs for chunk in _chunk(p, limit)]

    sentences = []
    for doc in nlp.pipe(blocks):
        sentences.extend(sent.text.strip() for sent in doc.sents)
    return [s for s in sentences if s]


def is_meaningful_sentence(text, min_chars=25, min_words=4):
    """Filter out headings, table fragments and extraction noise."""
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        return False
    if len(stripped.split()) < min_words:
        return False
    if not any(ch.isalpha() for ch in stripped):
        return False
    if _is_heading_line(stripped, max_chars=None):
        return False  # a title or section heading, not a claim
    # Dot leaders, numeric table rows and similar debris are mostly
    # non-alphanumeric once whitespace is ignored.
    dense = [ch for ch in stripped if not ch.isspace()]
    if dense and sum(ch.isalnum() for ch in dense) / len(dense) < 0.6:
        return False
    return True


def segment_document(extraction, start_index=1, config=None):
    """Turn a :class:`DocumentExtraction` into a list of Sentences.

    ``start_index`` continues sentence numbering across documents so
    every sentence id is unique within one analysis run.

    Returns ``(sentences, next_index)``.
    """
    model_name = _cfg(config, "SPACY_MODEL", "en_core_web_sm")
    min_chars = _cfg(config, "MIN_SENTENCE_CHARS", 25)
    min_words = _cfg(config, "MIN_SENTENCE_WORDS", 4)
    chunk_chars = _cfg(config, "SEGMENTATION_CHUNK_CHARS", 90_000)

    sentences = []
    index = start_index

    for page in extraction.pages:
        cleaned = clean_text(page.text)
        for raw in split_sentences(cleaned, model_name, chunk_chars):
            if not is_meaningful_sentence(raw, min_chars, min_words):
                continue
            sentences.append(
                Sentence(
                    document_id=extraction.document_id,
                    document_name=extraction.document_name,
                    page_number=page.page_number,
                    sentence_id=f"sent_{index:03d}",
                    text=raw,
                    word_count=len(raw.split()),
                    char_count=len(raw),
                )
            )
            index += 1

    extraction.sentence_count = len(sentences)
    return sentences, index


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _is_heading_line(line, max_chars=60):
    """True for a line that reads as a title or section heading.

    Deliberately narrow: the line must be entirely upper-case and must
    not end like a sentence. Mixed-case wrapped lines are left alone, so
    an ordinary sentence is never mistaken for a heading.
    """
    if not line:
        return False
    if max_chars is not None and len(line) > max_chars:
        return False
    if line.endswith((".", ",", "!", "?")):
        return False
    letters = [ch for ch in line if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def _chunk(text, limit):
    """Yield blocks of at most ``limit`` chars, split on blank lines."""
    if len(text) <= limit:
        yield text
        return

    buffer = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= limit:
            buffer = candidate
            continue
        if buffer:
            yield buffer
            buffer = ""
        # A single paragraph longer than the limit is cut on whitespace.
        while len(paragraph) > limit:
            cut = paragraph.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            yield paragraph[:cut]
            paragraph = paragraph[cut:].lstrip()
        buffer = paragraph
    if buffer:
        yield buffer


def _regex_sentences(text):
    """Fallback splitter used only when spaCy is unavailable."""
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])", text)
    merged = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _ends_with_abbreviation(text):
    tail = text.rstrip().rstrip(".").split()
    if not tail:
        return False
    return tail[-1].lower().strip("(\"'") in _ABBREVIATIONS


def _cfg(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

"""Cleaning and sentence-segmentation tests."""
from app.services.models import DocumentExtraction, PageText
from app.services.text_preprocessor import (
    clean_text,
    is_meaningful_sentence,
    segment_document,
    split_sentences,
)


# ------------------------------------------------------------------ cleaning

def test_collapses_whitespace_and_blank_runs():
    cleaned = clean_text("Leave   is    granted.\n\n\n\nSick leave is separate.")
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_rejoins_words_hyphenated_across_a_line_break():
    assert "annual" in clean_text("Employees get 20 days of annu-\nal leave.")


def test_drops_running_page_markers():
    cleaned = clean_text("First line of text.\nPage 2 of 9\nSecond line of text.")
    assert "Page 2 of 9" not in cleaned
    assert "First line" in cleaned and "Second line" in cleaned


def test_drops_separator_only_lines():
    cleaned = clean_text("Contents\n..........\n-------\nAnnual leave rules apply.")
    assert "....." not in cleaned
    assert "Annual leave rules apply." in cleaned


def test_joins_wrapped_lines_but_keeps_paragraph_breaks():
    cleaned = clean_text("Unused leave does not\ncarry over.\n\nNew paragraph here.")
    assert "does not carry over." in cleaned
    assert "\n\n" in cleaned


def test_preserves_numbers_dates_negations_and_punctuation():
    raw = ("Employees shall not receive more than 20.5 days of leave "
           "before 01/04/2024; see clause 3(b).")
    cleaned = clean_text(raw)

    for token in ["not", "20.5", "01/04/2024", ";", "3(b)"]:
        assert token in cleaned
    # No case folding: meaning-bearing capitalisation survives.
    assert "Employees" in cleaned


def test_strips_bullet_glyphs_only_at_line_start():
    cleaned = clean_text("• Managers must approve leave within 3 days.")
    assert cleaned.startswith("Managers")


# -------------------------------------------------------------- segmentation

def test_splits_multiple_sentences():
    text = ("Employees receive 20 days of annual leave. "
            "Requests need 14 days of notice. "
            "Unused leave expires at year end.")
    assert len(split_sentences(text)) == 3


def test_does_not_split_on_decimal_numbers():
    sentences = split_sentences("The allowance is 1200.50 rupees per day of travel.")
    assert len(sentences) == 1


def test_headings_are_filtered_out():
    assert not is_meaningful_sentence("ANNUAL LEAVE POLICY")
    assert not is_meaningful_sentence("3.")
    assert not is_meaningful_sentence(".... 12")
    # A long all-caps title is still a title, not a comparable claim.
    assert not is_meaningful_sentence("ACME CORPORATION - EMPLOYEE LEAVE POLICY")
    assert is_meaningful_sentence("Employees receive 20 days of annual leave.")


def test_heading_is_not_glued_onto_the_following_sentence():
    raw = ("ACME CORPORATION - LEAVE POLICY\n"
           "Employees are entitled to 20 days of paid annual leave.\n")
    sentences = split_sentences(clean_text(raw))

    claims = [s for s in sentences if is_meaningful_sentence(s)]
    assert claims == ["Employees are entitled to 20 days of paid annual leave."]


def test_mixed_case_wrapped_line_is_not_treated_as_a_heading():
    # Guards the heading rule against splitting an ordinary wrapped
    # sentence that happens to lack terminal punctuation on line one.
    raw = ("Employees are entitled to 20 days of paid\n"
           "annual leave per calendar year.\n")
    sentences = split_sentences(clean_text(raw))

    assert sentences == [
        "Employees are entitled to 20 days of paid annual leave per calendar year."
    ]


def test_segment_document_builds_the_expected_record():
    extraction = DocumentExtraction(
        document_id="doc_001",
        document_name="leave_policy.pdf",
        file_type="pdf",
        status="success",
        pages=[
            PageText(1, "Employees are entitled to 20 days of annual leave."),
            PageText(2, "Unused annual leave is forfeited at the end of the year."),
        ],
    )

    sentences, next_index = segment_document(extraction, start_index=1)

    assert len(sentences) == 2
    assert next_index == 3
    assert extraction.sentence_count == 2

    first = sentences[0].to_dict()
    assert first["document_id"] == "doc_001"
    assert first["document_name"] == "leave_policy.pdf"
    assert first["page_number"] == 1
    assert first["sentence_id"] == "sent_001"
    assert "20 days" in first["text"]

    # Page references follow the page the sentence came from.
    assert sentences[1].page_number == 2
    assert sentences[1].sentence_id == "sent_002"


def test_sentence_ids_continue_across_documents():
    def build(doc_id, name, text):
        return DocumentExtraction(
            document_id=doc_id, document_name=name, file_type="txt",
            status="success", pages=[PageText(None, text)],
        )

    first, index = segment_document(
        build("doc_001", "a.txt", "Employees receive 20 days of annual leave."),
        start_index=1,
    )
    second, index = segment_document(
        build("doc_002", "b.txt", "Employees receive 15 days of annual leave."),
        start_index=index,
    )

    ids = [s.sentence_id for s in first + second]
    assert ids == ["sent_001", "sent_002"]
    assert len(set(ids)) == len(ids)
    assert second[0].page_number is None


def test_regex_fallback_matches_on_simple_text():
    from app.services import text_preprocessor as tp

    text = ("Employees receive 20 days of leave. Dr. Rao approves requests. "
            "Unused leave expires.")
    assert len(tp._regex_sentences(text)) == 3

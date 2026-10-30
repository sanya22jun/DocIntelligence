"""Frontend guards.

The dashboard builds HTML by string concatenation, so the escaping
helpers are the boundary that keeps document text and model output from
becoming markup. Their unit tests run under Node; this module runs them
as part of the Python suite and lints the call sites.
"""
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JS_DIR = os.path.join(ROOT, "app", "static", "js")
JS_TEST = os.path.join(ROOT, "tests", "js", "text-utils.test.js")
APP_JS = os.path.join(JS_DIR, "app.js")

NODE = shutil.which("node")

#: Fields holding text that came from an uploaded document or from the
#: model. Every one of them must be escaped before reaching innerHTML.
UNTRUSTED_FIELDS = (
    "evidence_a", "evidence_b", "explanation", "document_name",
    "conflicting_span_a", "conflicting_span_b", "message", "text",
    # Chat: the answer and its citations are model output over document
    # text, so they are untrusted on both counts.
    "answer", "sentence_id", "questions",
)

#: A property read such as ``item.explanation``. The trailing lookahead
#: keeps ``.textContent`` and the CSS class ``conflict-explanation`` from
#: matching.
FIELD_ACCESS = re.compile(
    r"\.\s*(?:" + "|".join(UNTRUSTED_FIELDS) + r")(?![A-Za-z0-9_])"
)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_text_utils_unit_tests_pass():
    result = subprocess.run(
        [NODE, JS_TEST], capture_output=True, text=True, timeout=120,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checks passed" in result.stdout


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_dashboard_scripts_parse():
    for name in ("text-utils.js", "app.js"):
        result = subprocess.run(
            [NODE, "--check", os.path.join(JS_DIR, name)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"


def lint_unescaped(source):
    """Return markup-building lines that interpolate untrusted text."""
    offenders = []
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        if not FIELD_ACCESS.search(line):
            continue
        # Already escaped at this call site.
        if "escapeHtml(" in line or "highlight(" in line:
            continue
        # Only markup-building lines are a risk. A line that merely joins
        # untrusted text into a plain string is fine, because the
        # consumer (showAlert) escapes every item it is given.
        if "<" in line or ">" in line:
            offenders.append(f"{number}: {stripped}")
    return offenders


def test_the_lint_itself_detects_an_unescaped_interpolation():
    """Guards against the lint silently matching nothing."""
    offenders = lint_unescaped(
        'row.innerHTML = "<td>" + item.explanation + "</td>";'
    )
    assert len(offenders) == 1

    # ...and does not fire on the safe shapes it must tolerate.
    assert lint_unescaped('return f.document_name + ": " + f.error;') == []
    assert lint_unescaped('el.progressMessage.textContent = message || "";') == []
    assert lint_unescaped('"<footer class=\\"conflict-explanation\\">" +') == []
    assert lint_unescaped('"<td>" + escapeHtml(item.explanation) + "</td>"') == []


def test_untrusted_fields_are_always_escaped_at_the_call_site():
    offenders = lint_unescaped(open(APP_JS, encoding="utf-8").read())
    assert not offenders, (
        "Untrusted text reaches HTML without escaping:\n" + "\n".join(offenders)
    )


def test_statement_text_is_escaped_in_the_statements_table():
    source = open(APP_JS, encoding="utf-8").read()
    row = re.search(
        r"sentencesBody\.innerHTML = visible\.map\((.*?)\}\)\.join",
        source, re.S,
    )
    assert row, "could not locate the statements table renderer"
    assert "escapeHtml(item.text)" in row.group(1)


def test_chat_answers_and_citations_are_escaped():
    source = open(APP_JS, encoding="utf-8").read()
    assert "escapeHtml(body.answer)" in source
    assert "escapeHtml(source.text)" in source
    assert "escapeHtml(source.document_name)" in source
    assert "escapeHtml(question)" in source


def test_highlight_is_used_for_statement_evidence():
    source = open(APP_JS, encoding="utf-8").read()
    assert "highlight(item.evidence_a, item.conflicting_span_a)" in source
    assert "highlight(item.evidence_b, item.conflicting_span_b)" in source


def test_shared_helpers_are_not_redefined_in_the_dashboard_script():
    """One escaping implementation means one place to get it right."""
    source = open(APP_JS, encoding="utf-8").read()
    for name in ("escapeHtml", "highlight", "titleCase"):
        assert f"function {name}(" not in source

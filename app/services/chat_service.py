"""Grounded question answering over the uploaded documents.

The chat is deliberately *not* a general assistant. It answers only from
statements retrieved out of the user's own documents, cites them, and
says plainly when the documents do not contain the answer. The same
no-fabrication rule used for the conflict report applies here: a cited
statement id that was not actually retrieved is discarded rather than
shown, so every citation in the UI points at real text.
"""
import numpy as np

from .embedding_service import EmbeddingService
from .similarity_service import cosine_matrix

#: Statements handed to the model as context for one question.
DEFAULT_TOP_K = 8
#: Findings about those statements, included so "why is this a conflict?"
#: can be answered from the analysis that was already paid for.
MAX_RELATED_FINDINGS = 5
#: Prior turns kept for follow-up questions ("why?", "what about X?").
MAX_HISTORY_MESSAGES = 6

MAX_QUESTION_CHARS = 500
MAX_ANSWER_CHARS = 1500

SYSTEM_PROMPT = """\
You answer questions about a specific set of documents. You are given \
numbered statements extracted from those documents, and sometimes the \
relationships a previous analysis found between them.

Rules:
- Answer ONLY from the statements provided. They are the entire source \
of truth available to you.
- Cite the id of every statement you used, in cited_sentence_ids.
- If the statements do not contain the answer, say so plainly in one \
sentence and set answered_from_documents to false. Do not guess, and do \
not fall back on general knowledge.
- When the statements disagree with each other, say so explicitly and \
give both values together with the document each came from. A \
disagreement is the most important thing to surface.
- Never invent numbers, dates, document names or page numbers. Every \
figure in your answer must appear in a statement you were given.
- Be concise: two to four sentences unless the question needs a list.
- Write for the reader, not about the statements: say "the 2023 policy \
grants 20 days", not "statement sent_016 says".\
"""

SUGGESTION_PROMPT = """\
You write starter questions for a document question-answering panel.

Given a digest of the documents a user just analysed, write four short \
questions they would plausibly want answered, in this order:
1. one about the most significant disagreement between the documents;
2. one about what the documents agree on;
3. one asking for a specific value or rule that the documents cover;
4. one asking for a summary of a topic the documents actually discuss.

Each question must be answerable from these documents, under 12 words, \
phrased as a user would type it. Do not invent topics the digest does \
not mention. Return exactly four.\
"""

#: Used when suggestions cannot be generated. Deliberately generic, and
#: only ever shown as prompts the user may click, never as findings.
FALLBACK_SUGGESTIONS = (
    "What are the main contradictions between these documents?",
    "What do these documents agree on?",
    "Which document should I follow?",
    "Summarise the key rules in these documents.",
)


class ChatRejected(Exception):
    """The question itself is unusable."""


def answer_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "cited_sentence_ids", "answered_from_documents"],
        "properties": {
            "answer": {"type": "string"},
            "cited_sentence_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "answered_from_documents": {"type": "boolean"},
        },
    }


def suggestion_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["questions"],
        "properties": {
            "questions": {"type": "array", "items": {"type": "string"}},
        },
    }


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

def retrieve(question, sentences, service, config=None, top_k=DEFAULT_TOP_K):
    """Return the statements most related to the question.

    Statement vectors are requested through the embedding cache, so for
    an analysis that has already run this costs one small embedding call
    for the question itself and nothing for the documents.
    """
    if not sentences:
        return [], {"statements": 0, "api_calls": 0, "cache_hits": 0}

    embedder = EmbeddingService(
        service, batch_size=_cfg(config, "EMBEDDING_BATCH_SIZE", 64)
    )
    matrix, stats = embedder.embed_sentences(sentences)

    question_vector = np.asarray(
        service.embed([question])[0], dtype=np.float32
    ).reshape(1, -1)

    combined = np.vstack([question_vector, matrix])
    scores = cosine_matrix(combined)[0][1:]

    ranked = np.argsort(-scores)[:max(1, top_k)]
    retrieved = []
    for index in ranked:
        sentence = dict(sentences[int(index)])
        sentence["similarity"] = round(float(scores[int(index)]), 4)
        retrieved.append(sentence)

    return retrieved, stats


def related_findings(retrieved, report, limit=MAX_RELATED_FINDINGS):
    """Findings from the stored analysis that involve a retrieved statement."""
    if not report:
        return []

    ids = {item["sentence_id"] for item in retrieved}
    findings = []
    for result in report.get("results") or []:
        if result["sentence_id_a"] in ids or result["sentence_id_b"] in ids:
            findings.append(result)
        if len(findings) >= limit:
            break
    return findings


# ---------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------

def build_context(retrieved, findings):
    """Render the retrieved statements and findings as the user message."""
    lines = ["STATEMENTS"]
    for item in retrieved:
        location = item["document_name"]
        if item.get("page_number") is not None:
            location += f", page {item['page_number']}"
        lines.append(f"[{item['sentence_id']}] ({location}) {item['text']}")

    if findings:
        lines.append("")
        lines.append("RELATIONSHIPS ALREADY FOUND BETWEEN THESE STATEMENTS")
        for finding in findings:
            category = f"/{finding['category']}" if finding.get("category") else ""
            lines.append(
                f"- {finding['sentence_id_a']} vs {finding['sentence_id_b']}: "
                f"{finding['relationship']}{category} - {finding['explanation']}"
            )
    return "\n".join(lines)


def build_user_prompt(question, retrieved, findings, history=None):
    sections = []
    trimmed = _trim_history(history)
    if trimmed:
        sections.append("EARLIER IN THIS CONVERSATION")
        for message in trimmed:
            role = "User" if message["role"] == "user" else "You"
            sections.append(f"{role}: {message['content']}")
        sections.append("")

    sections.append(build_context(retrieved, findings))
    sections.append("")
    sections.append(f"QUESTION: {question}")
    return "\n".join(sections)


# ---------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------

def ask(record, question, service, config=None, history=None):
    """Answer one question about a stored analysis.

    Returns a dict with the answer, validated citations, and what was
    retrieved. Raises :class:`ChatRejected` for an unusable question and
    ``OpenAIServiceError`` for an API failure, which the route maps to a
    status code.
    """
    question = clean_question(question)
    sentences = record.get("sentences") or []
    if not sentences:
        raise ChatRejected(
            "This analysis has no extracted statements to answer from."
        )

    retrieved, retrieval_stats = retrieve(
        question, sentences, service, config,
        top_k=_cfg(config, "CHAT_TOP_K", DEFAULT_TOP_K),
    )
    findings = related_findings(retrieved, record.get("analysis"))

    payload = service.complete_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(question, retrieved, findings, history),
        schema=answer_schema(),
        schema_name="grounded_answer",
        max_output_tokens=_cfg(config, "CHAT_MAX_OUTPUT_TOKENS", 700),
    )

    answer = str(payload.get("answer") or "").strip()
    if not answer:
        answer = (
            "I could not produce an answer from these documents. Try "
            "rephrasing the question."
        )

    return {
        "question": question,
        "answer": answer[:MAX_ANSWER_CHARS],
        "answered_from_documents": bool(
            payload.get("answered_from_documents")
        ),
        "sources": _validated_sources(
            payload.get("cited_sentence_ids"), retrieved
        ),
        "retrieved": [
            {
                "sentence_id": item["sentence_id"],
                "document_name": item["document_name"],
                "page_number": item.get("page_number"),
                "similarity": item["similarity"],
            }
            for item in retrieved
        ],
        "related_findings": [
            {
                "pair_key": finding["pair_key"],
                "relationship": finding["relationship"],
                "category": finding.get("category"),
            }
            for finding in findings
        ],
        "retrieval": retrieval_stats,
    }


def clean_question(question):
    if not isinstance(question, str):
        raise ChatRejected("A question is required.")
    cleaned = " ".join(question.split())
    if not cleaned:
        raise ChatRejected("A question is required.")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ChatRejected(
            f"That question is {len(cleaned)} characters long; please keep "
            f"it under {MAX_QUESTION_CHARS}."
        )
    return cleaned


def _validated_sources(cited, retrieved):
    """Keep only citations pointing at statements actually retrieved.

    A model-invented id would otherwise appear in the UI as a real
    source, which is exactly the kind of false citation this project
    refuses to produce.
    """
    by_id = {item["sentence_id"]: item for item in retrieved}
    sources = []
    seen = set()

    for sentence_id in cited or []:
        if not isinstance(sentence_id, str):
            continue
        item = by_id.get(sentence_id.strip())
        if item is None or item["sentence_id"] in seen:
            continue
        seen.add(item["sentence_id"])
        sources.append({
            "sentence_id": item["sentence_id"],
            "document_id": item["document_id"],
            "document_name": item["document_name"],
            "page_number": item.get("page_number"),
            "text": item["text"],
        })
    return sources


def _trim_history(history):
    """Keep the last few well-formed turns, newest last."""
    if not isinstance(history, list):
        return []

    cleaned = []
    for message in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = " ".join(content.split())[:MAX_QUESTION_CHARS]
        if content:
            cleaned.append({"role": role, "content": content})
    return cleaned


# ---------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------

def build_digest(record, max_statements=10):
    """A compact description of the analysis, for suggestion generation."""
    lines = []
    documents = [
        d["document_name"] for d in record.get("documents") or []
        if d.get("status") == "success"
    ]
    if documents:
        lines.append("DOCUMENTS: " + ", ".join(documents))

    report = record.get("analysis") or {}
    overview = report.get("overview") or {}
    if overview:
        lines.append(
            "FINDINGS: {} contradiction(s), {} agreement(s), {} neutral".format(
                overview.get("contradictions_detected", 0),
                overview.get("supporting_relationships", 0),
                overview.get("neutral_relationships", 0),
            )
        )

    conflicts = [
        r for r in report.get("results") or []
        if r["relationship"] == "CONTRADICTION"
    ][:3]
    if conflicts:
        lines.append("TOP DISAGREEMENTS")
        for conflict in conflicts:
            lines.append(f"- {conflict['explanation']}")

    statements = record.get("sentences") or []
    if statements:
        lines.append("SAMPLE STATEMENTS")
        for item in statements[:max_statements]:
            lines.append(f"- {item['text']}")

    return "\n".join(lines)


def suggest_questions(record, service, config=None):
    """Four starter questions grounded in this analysis.

    Falls back to generic starters if the call fails, so the panel always
    has something to show. Suggestions are prompts, never findings.
    """
    digest = build_digest(record)
    if not digest.strip():
        return list(FALLBACK_SUGGESTIONS), "fallback"

    try:
        payload = service.complete_json(
            system_prompt=SUGGESTION_PROMPT,
            user_prompt=digest,
            schema=suggestion_schema(),
            schema_name="suggested_questions",
            max_output_tokens=_cfg(config, "CHAT_SUGGESTION_TOKENS", 200),
        )
    except Exception:  # noqa: BLE001 - suggestions must never block the UI
        return list(FALLBACK_SUGGESTIONS), "fallback"

    questions = []
    for item in payload.get("questions") or []:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.split())
        if cleaned and cleaned not in questions:
            questions.append(cleaned[:MAX_QUESTION_CHARS])

    if not questions:
        return list(FALLBACK_SUGGESTIONS), "fallback"
    return questions[:4], "generated"


def _cfg(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

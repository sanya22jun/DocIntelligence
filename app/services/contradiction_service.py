"""GPT-based relationship classification for shortlisted pairs.

Everything the model returns is treated as a claim to be checked, not as
truth: pair ids are matched against what was asked, enums are validated,
confidence is clamped, and highlight spans are accepted only when they
appear verbatim in the statement they belong to. Evidence text and all
source references come from our own stored statements, never from the
model, so a conflict report cannot cite something that was invented.
"""
import math

from .openai_service import KIND_AUTH, OpenAIServiceError

RELATIONSHIP_CONTRADICTION = "CONTRADICTION"
RELATIONSHIP_SUPPORT = "SUPPORT"
RELATIONSHIP_NEUTRAL = "NEUTRAL"

RELATIONSHIPS = (
    RELATIONSHIP_CONTRADICTION,
    RELATIONSHIP_SUPPORT,
    RELATIONSHIP_NEUTRAL,
)

CATEGORIES = (
    "NUMERICAL_CONFLICT",
    "DATE_CONFLICT",
    "FACTUAL_CONFLICT",
    "POLICY_CONFLICT",
    "LOGICAL_CONFLICT",
    "OTHER",
)
#: Emitted by the model for SUPPORT/NEUTRAL rows, dropped afterwards.
CATEGORY_NA = "NOT_APPLICABLE"

MAX_EXPLANATION_CHARS = 600

SYSTEM_PROMPT = """\
You are a document-consistency analyst. Each pair below contains two \
statements taken from two DIFFERENT documents. Decide how they relate.

Choose exactly one relationship per pair:
- CONTRADICTION: they describe the same subject under the same scope and \
cannot both hold at the same time.
- SUPPORT: they agree with, restate or reinforce each other, including \
paraphrases and the same value expressed differently.
- NEUTRAL: unrelated, about different subjects or scopes, or simply \
compatible details.

Do NOT return CONTRADICTION merely because:
- the wording differs, or one statement is more specific than the other;
- numbers differ while the SCOPE differs (for example "managers receive 20 \
days" versus "interns receive 15 days" concern different populations and \
are compatible);
- one states a general rule and the other a stated exception to it;
- they concern different entities, departments, locations or document \
types that can coexist;
- one simply adds information the other leaves out.

Return CONTRADICTION when the same subject and scope are given \
incompatible values, dates, quantities, obligations or permissions, \
including when one forbids what the other permits or requires.

If the two documents carry different effective dates or versions, still \
report a genuine incompatibility as a CONTRADICTION, and name both \
effective dates or versions in the explanation so the reader can judge \
which document governs.

category applies only to CONTRADICTION. Use NUMERICAL_CONFLICT, \
DATE_CONFLICT, FACTUAL_CONFLICT, POLICY_CONFLICT, LOGICAL_CONFLICT or \
OTHER. For SUPPORT and NEUTRAL use NOT_APPLICABLE.

explanation: one or two sentences, at most about 40 words, naming the \
specific values that clash. Describe only what the two statements say. \
Never invent values, sources, dates or page numbers that are not present \
in them.

conflicting_span_a and conflicting_span_b: a short phrase copied VERBATIM \
from statement A and statement B respectively, usually the clashing \
value. Never paraphrase a span. Use an empty string when no single phrase \
captures the issue.

confidence: your own confidence in the classification, between 0 and 1.

Return one result object for every pair you were given, echoing its \
pair_key exactly. Do not merge, skip or invent pairs.\
"""


def result_schema():
    """Strict JSON schema for the reasoning response."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "pair_key",
                        "relationship",
                        "category",
                        "explanation",
                        "conflicting_span_a",
                        "conflicting_span_b",
                        "confidence",
                    ],
                    "properties": {
                        "pair_key": {"type": "string"},
                        "relationship": {
                            "type": "string",
                            "enum": list(RELATIONSHIPS),
                        },
                        "category": {
                            "type": "string",
                            "enum": list(CATEGORIES) + [CATEGORY_NA],
                        },
                        "explanation": {"type": "string"},
                        "conflicting_span_a": {"type": "string"},
                        "conflicting_span_b": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            }
        },
    }


class ContradictionService:
    def __init__(self, service, batch_size=8, max_output_tokens=1400,
                 max_requests=None):
        self.service = service
        self.batch_size = max(1, int(batch_size))
        self.max_output_tokens = max_output_tokens
        #: Hard ceiling on reasoning requests, so that batch splitting
        #: after a truncation cannot multiply the cost of a run.
        self.max_requests = max_requests

    # -----------------------------------------------------------------

    def classify(self, pairs, sentences):
        """Classify shortlisted pairs.

        Returns ``(results, warnings, stats)``. Pairs that could not be
        classified appear in ``stats['unclassified']`` and in warnings;
        they are never reported as a relationship we did not get.
        """
        stats = {
            "pairs_submitted": len(pairs),
            "pairs_classified": 0,
            "unclassified": [],
            "requests": 0,
            "retries": 0,
            "batch_splits": 0,
            "request_budget": 0,
        }
        warnings = []
        if not pairs:
            return [], warnings, stats

        by_index = {index: sentence for index, sentence in enumerate(sentences)}
        pair_by_key = {pair["pair_key"]: pair for pair in pairs}

        budget = self.max_requests
        if budget is None:
            # Enough for one pass plus a round of splits/retries.
            batches = math.ceil(len(pairs) / self.batch_size)
            budget = batches * 2 + 2
        stats["request_budget"] = budget

        queue = [list(batch) for batch in _batched(pairs, self.batch_size)]
        answered = {}
        retried_keys = set()

        while queue:
            batch = queue.pop(0)

            if stats["requests"] >= budget:
                warnings.append({
                    "stage": "reasoning",
                    "kind": "request_budget_exhausted",
                    "message": (
                        f"Stopped after {stats['requests']} reasoning "
                        f"requests to stay within the configured budget; "
                        f"{sum(len(b) for b in queue) + len(batch)} pairs "
                        f"were left unclassified."
                    ),
                })
                stats["unclassified"].extend(
                    p["pair_key"] for p in batch + [q for b in queue for q in b]
                )
                break

            try:
                payload = self._request(batch, by_index)
                stats["requests"] += 1
            except OpenAIServiceError as exc:
                stats["requests"] += 1
                if exc.kind == KIND_AUTH:
                    raise  # no point continuing without a usable key

                if (exc.splittable or exc.retryable) and len(batch) > 1:
                    middle = len(batch) // 2
                    queue.insert(0, batch[middle:])
                    queue.insert(0, batch[:middle])
                    stats["batch_splits"] += 1
                    warnings.append({
                        "stage": "reasoning",
                        "kind": exc.kind,
                        "message": (
                            f"{exc.message} Retrying as two smaller batches."
                        ),
                    })
                    continue

                warnings.append({
                    "stage": "reasoning",
                    "kind": exc.kind,
                    "message": (
                        f"{exc.message} "
                        f"{len(batch)} pair(s) could not be analysed."
                    ),
                })
                stats["unclassified"].extend(p["pair_key"] for p in batch)
                continue

            requested = {pair["pair_key"] for pair in batch}
            accepted, rejected = self._validate(
                payload, requested, pair_by_key, by_index
            )
            answered.update(accepted)
            warnings.extend(rejected)

            missing = sorted(requested - set(accepted))
            if missing:
                fresh = [key for key in missing if key not in retried_keys]
                retried_keys.update(fresh)
                if fresh:
                    stats["retries"] += 1
                    queue.insert(0, [pair_by_key[key] for key in fresh])
                    warnings.append({
                        "stage": "reasoning",
                        "kind": "missing_results",
                        "message": (
                            f"The model omitted {len(fresh)} pair(s) from a "
                            f"batch; asking again individually."
                        ),
                    })
                else:
                    stats["unclassified"].extend(
                        key for key in missing if key not in answered
                    )
                    warnings.append({
                        "stage": "reasoning",
                        "kind": "missing_results",
                        "message": (
                            f"{len(missing)} pair(s) were still missing after "
                            f"a retry and are reported as unclassified."
                        ),
                    })

        stats["unclassified"] = sorted(
            {key for key in stats["unclassified"] if key not in answered}
        )
        stats["pairs_classified"] = len(answered)

        # Contradictions first, then by model-reported confidence.
        results = sorted(
            answered.values(),
            key=lambda item: (
                item["relationship"] != RELATIONSHIP_CONTRADICTION,
                -(item["confidence"] or 0),
                item["pair_key"],
            ),
        )
        return results, warnings, stats

    # -----------------------------------------------------------------

    def _request(self, batch, by_index):
        return self.service.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(batch, by_index),
            schema=result_schema(),
            schema_name="relationship_results",
            max_output_tokens=self.max_output_tokens,
        )

    def _validate(self, payload, requested, pair_by_key, by_index):
        """Keep only well-formed results for pairs we actually asked about."""
        accepted = {}
        problems = []

        items = payload.get("results")
        if not isinstance(items, list):
            problems.append({
                "stage": "reasoning",
                "kind": "invalid_response",
                "message": "The model's reply had no 'results' array.",
            })
            return accepted, problems

        for item in items:
            if not isinstance(item, dict):
                continue

            key = item.get("pair_key")
            if key not in requested:
                # Either a hallucinated pair or one from another batch.
                problems.append({
                    "stage": "reasoning",
                    "kind": "unexpected_pair",
                    "message": (
                        "Discarded a result for a pair that was not part of "
                        "this request."
                    ),
                })
                continue
            if key in accepted:
                continue  # duplicate answer for the same pair

            relationship = item.get("relationship")
            if relationship not in RELATIONSHIPS:
                problems.append({
                    "stage": "reasoning",
                    "kind": "invalid_relationship",
                    "message": (
                        f"Discarded {key}: unrecognised relationship "
                        f"'{relationship}'."
                    ),
                })
                continue

            explanation = str(item.get("explanation") or "").strip()
            if not explanation:
                problems.append({
                    "stage": "reasoning",
                    "kind": "missing_explanation",
                    "message": f"Discarded {key}: no explanation was given.",
                })
                continue

            pair = pair_by_key[key]
            first = by_index[pair["index_a"]]
            second = by_index[pair["index_b"]]

            accepted[key] = {
                "pair_key": key,
                "sentence_id_a": pair["sentence_id_a"],
                "sentence_id_b": pair["sentence_id_b"],
                "relationship": relationship,
                "category": _category_for(relationship, item.get("category")),
                "explanation": explanation[:MAX_EXPLANATION_CHARS],
                # Evidence and provenance come from our own records.
                "evidence_a": first["text"],
                "evidence_b": second["text"],
                "document_a": _source(first),
                "document_b": _source(second),
                "conflicting_span_a": verified_span(
                    item.get("conflicting_span_a"), first["text"]
                ),
                "conflicting_span_b": verified_span(
                    item.get("conflicting_span_b"), second["text"]
                ),
                "similarity": pair["similarity"],
                "confidence": _confidence(item.get("confidence")),
                "confidence_basis": "model_reported",
            }

        return accepted, problems


# ---------------------------------------------------------------------
# Prompt + validation helpers
# ---------------------------------------------------------------------

def build_user_prompt(batch, by_index):
    """Render a batch of pairs as the user message."""
    lines = []
    for pair in batch:
        first = by_index[pair["index_a"]]
        second = by_index[pair["index_b"]]
        lines.append(f"PAIR {pair['pair_key']}")
        lines.append(f"A ({first['document_name']}): {first['text']}")
        lines.append(f"B ({second['document_name']}): {second['text']}")
        lines.append("")
    return "\n".join(lines).strip()


def verified_span(span, text):
    """Return the span only if it occurs verbatim in ``text``.

    Highlighting is only honest if the highlighted phrase is really in
    the statement, so a paraphrased or invented span is dropped rather
    than shown. A case-only difference is tolerated by returning the
    slice as it actually appears in the source.
    """
    if not span or not isinstance(span, str):
        return None

    candidate = span.strip()
    if not candidate:
        return None
    if candidate in text:
        return candidate

    position = text.lower().find(candidate.lower())
    if position == -1:
        return None
    return text[position:position + len(candidate)]


def _category_for(relationship, category):
    """Categories describe conflicts; they are meaningless otherwise."""
    if relationship != RELATIONSHIP_CONTRADICTION:
        return None
    return category if category in CATEGORIES else "OTHER"


def _confidence(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(min(1.0, max(0.0, number)), 4)


def _source(sentence):
    return {
        "document_id": sentence["document_id"],
        "document_name": sentence["document_name"],
        "page_number": sentence["page_number"],
    }


def _batched(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]

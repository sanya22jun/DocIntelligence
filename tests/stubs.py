"""Offline stand-ins for the OpenAI SDK client.

These exist so the whole engine can be exercised with no network access
and no API spend. They are test scaffolding only: nothing in ``app``
imports them, so stub output can never reach the application UI and be
mistaken for real model output.
"""
import json
import re
from types import SimpleNamespace

VECTOR_DIM = 8

#: Topic keywords -> the dimension they load. Sentences about the same
#: topic therefore land close together under cosine similarity, which is
#: all the candidate-selection stage needs in order to be testable.
TOPICS = (
    ("leave", ("annual leave", "days of leave", "carried over", "carry over")),
    ("sick", ("sick leave", "medical certificate", "doctor's note", "illness")),
    ("remote", ("remote", "probation", "work from home")),
    ("review", ("performance review", "appraisal")),
    ("expense", ("expense", "reimburse", "claim", "receipt")),
    ("travel", ("flight", "hotel", "travel", "economy")),
    ("meal", ("meal", "allowance", "cafeteria")),
    ("misc", ("parking", "garage", "visitor")),
)


def fake_vector(text):
    """A deterministic, topic-aware unit-ish vector for a sentence."""
    lowered = text.lower()
    vector = [0.05] * VECTOR_DIM
    for index, (_name, keywords) in enumerate(TOPICS):
        for keyword in keywords:
            if keyword in lowered:
                vector[index % VECTOR_DIM] += 1.0
    # A little text-dependent jitter so distinct sentences are not
    # perfectly identical vectors.
    vector[-1] += (len(lowered) % 7) / 100.0
    return vector


class StubEmbeddings:
    def __init__(self, parent):
        self._parent = parent

    def create(self, model, input):
        self._parent.embedding_calls.append(list(input))
        if self._parent.embedding_error is not None:
            raise self._parent.embedding_error

        data = [
            SimpleNamespace(index=i, embedding=fake_vector(text))
            for i, text in enumerate(input)
        ]
        if self._parent.drop_one_vector and data:
            data = data[:-1]

        tokens = sum(max(1, len(t.split())) for t in input)
        return SimpleNamespace(
            data=data, usage=SimpleNamespace(total_tokens=tokens)
        )


class StubCompletions:
    def __init__(self, parent):
        self._parent = parent

    def create(self, model, messages, **kwargs):
        parent = self._parent
        user_prompt = messages[-1]["content"]

        # Chat and suggestion requests have their own shapes; the
        # relationship classifier is the default.
        if "QUESTION:" in user_prompt:
            return parent.build_chat_reply(user_prompt)
        if "DOCUMENTS:" in user_prompt or "SAMPLE STATEMENTS" in user_prompt:
            return parent.build_suggestions_reply(user_prompt)

        pair_keys = re.findall(r"^PAIR (\S+)$", user_prompt, re.MULTILINE)
        parent.reasoning_calls.append(pair_keys)

        error = parent.next_reasoning_error()
        if error is not None:
            raise error

        behaviour = parent.next_reasoning_behaviour()
        if behaviour == "truncated":
            return _completion("", finish_reason="length")
        if behaviour == "bad_json":
            return _completion("{not json at all")
        if behaviour == "not_object":
            return _completion("[1, 2, 3]")
        if behaviour == "no_results_key":
            return _completion(json.dumps({"items": []}))
        if behaviour == "empty":
            return _completion(None)

        results = []
        for key in pair_keys:
            if key in parent.omit_pairs:
                continue
            results.append(parent.build_result(key, user_prompt))
        for extra in parent.inject_pairs:
            results.append(parent.build_result(extra, user_prompt))

        return _completion(json.dumps({"results": results}))


class StubClient:
    """Configurable stub with the same call surface the service uses."""

    def __init__(self):
        self.embeddings = StubEmbeddings(self)
        self.chat = SimpleNamespace(completions=StubCompletions(self))

        self.embedding_calls = []
        self.reasoning_calls = []

        self.embedding_error = None
        self.drop_one_vector = False

        #: Queues consumed one entry per reasoning request.
        self.reasoning_errors = []
        self.reasoning_behaviours = []

        self.omit_pairs = set()
        self.inject_pairs = []
        self.result_overrides = {}

        # -- chat --
        self.chat_calls = []
        self.suggestion_calls = []
        self.chat_error = None
        self.suggestion_error = None
        #: Statement ids the stubbed answer will claim to have used. None
        #: means "cite everything that was retrieved".
        self.chat_cite = None
        self.chat_answered_from_documents = True
        self.chat_answer = None

    # -- queues ---------------------------------------------------------

    def next_reasoning_error(self):
        return self.reasoning_errors.pop(0) if self.reasoning_errors else None

    def next_reasoning_behaviour(self):
        return (
            self.reasoning_behaviours.pop(0)
            if self.reasoning_behaviours else "ok"
        )

    # -- result construction -------------------------------------------

    def build_result(self, pair_key, user_prompt):
        """Classify by a crude numeric rule, or use an override."""
        if pair_key in self.result_overrides:
            override = dict(self.result_overrides[pair_key])
            override.setdefault("pair_key", pair_key)
            return override

        block = _pair_block(user_prompt, pair_key)
        numbers_a = _numbers(block.get("a", ""))
        numbers_b = _numbers(block.get("b", ""))

        if numbers_a and numbers_b and numbers_a != numbers_b:
            span_a = numbers_a[0]
            span_b = numbers_b[0]
            return {
                "pair_key": pair_key,
                "relationship": "CONTRADICTION",
                "category": "NUMERICAL_CONFLICT",
                "explanation": (
                    f"One statement specifies {span_a} while the other "
                    f"specifies {span_b}."
                ),
                "conflicting_span_a": span_a,
                "conflicting_span_b": span_b,
                "confidence": 0.9,
            }
        if numbers_a and numbers_a == numbers_b:
            return {
                "pair_key": pair_key,
                "relationship": "SUPPORT",
                "category": "NOT_APPLICABLE",
                "explanation": "Both statements give the same value.",
                "conflicting_span_a": "",
                "conflicting_span_b": "",
                "confidence": 0.8,
            }
        return {
            "pair_key": pair_key,
            "relationship": "NEUTRAL",
            "category": "NOT_APPLICABLE",
            "explanation": "The statements concern different subjects.",
            "conflicting_span_a": "",
            "conflicting_span_b": "",
            "confidence": 0.6,
        }


    # -- chat ----------------------------------------------------------

    def build_chat_reply(self, user_prompt):
        self.chat_calls.append(user_prompt)
        if self.chat_error is not None:
            raise self.chat_error

        available = re.findall(r"^\[(\S+)\] ", user_prompt, re.MULTILINE)
        cited = self.chat_cite if self.chat_cite is not None else available
        question = ""
        match = re.search(r"^QUESTION: (.*)$", user_prompt, re.MULTILINE)
        if match:
            question = match.group(1)

        answer = self.chat_answer or (
            f"Stub answer about: {question}" if question else "Stub answer."
        )
        return _completion(json.dumps({
            "answer": answer,
            "cited_sentence_ids": list(cited),
            "answered_from_documents": self.chat_answered_from_documents,
        }))

    def build_suggestions_reply(self, user_prompt):
        self.suggestion_calls.append(user_prompt)
        if self.suggestion_error is not None:
            raise self.suggestion_error
        return _completion(json.dumps({"questions": [
            "What is the annual leave entitlement?",
            "What do the documents agree on?",
            "Which document forfeits unused leave?",
            "Summarise the remote work rules.",
        ]}))


# ---------------------------------------------------------------------

def _completion(content, finish_reason="stop", prompt_tokens=120,
                completion_tokens=90):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content),
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def _pair_block(user_prompt, pair_key):
    """Pull the A/B statement lines for one pair out of the prompt."""
    pattern = (
        rf"^PAIR {re.escape(pair_key)}$\n^A \([^)]*\): (?P<a>.*)$\n"
        rf"^B \([^)]*\): (?P<b>.*)$"
    )
    match = re.search(pattern, user_prompt, re.MULTILINE)
    return match.groupdict() if match else {}


def _numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", text)

"""The only module that talks to OpenAI.

It owns the client, the API key, timeouts, retries, the translation of
SDK exceptions into one predictable error type, and token accounting.
Nothing above this layer imports the ``openai`` package, and the key is
never logged, echoed in an error message, or sent to the frontend.
"""
import json
import logging

log = logging.getLogger(__name__)

# --- Error kinds -------------------------------------------------------
KIND_AUTH = "auth"
KIND_RATE_LIMIT = "rate_limit"
KIND_TIMEOUT = "timeout"
KIND_CONNECTION = "connection"
KIND_BAD_REQUEST = "bad_request"
KIND_TRUNCATED = "truncated"
KIND_INVALID_RESPONSE = "invalid_response"
KIND_SERVER = "server_error"
KIND_UNKNOWN = "unknown"

#: Kinds where retrying the same call (or a smaller one) may succeed.
RETRYABLE_KINDS = frozenset(
    {KIND_RATE_LIMIT, KIND_TIMEOUT, KIND_CONNECTION, KIND_SERVER}
)
#: Kinds where splitting the batch is the right response.
SPLITTABLE_KINDS = frozenset({KIND_TRUNCATED, KIND_BAD_REQUEST})


class OpenAIServiceError(Exception):
    """A failed OpenAI call, classified for the caller to act on."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.message = message

    @property
    def retryable(self):
        return self.kind in RETRYABLE_KINDS

    @property
    def splittable(self):
        return self.kind in SPLITTABLE_KINDS

    def to_dict(self):
        return {"kind": self.kind, "message": self.message}


class UsageRecord:
    """Token counters measured from API responses, plus a cost estimate.

    The token counts are reported by the API. The cost is an estimate
    derived from configured list prices, not a billing figure.
    """

    def __init__(self, prices=None):
        self.embedding_tokens = 0
        self.reasoning_input_tokens = 0
        self.reasoning_output_tokens = 0
        self.embedding_requests = 0
        self.reasoning_requests = 0
        self.failed_requests = 0
        self._prices = prices or {}

    def add_embedding(self, tokens):
        self.embedding_tokens += int(tokens or 0)
        self.embedding_requests += 1

    def add_reasoning(self, input_tokens, output_tokens):
        self.reasoning_input_tokens += int(input_tokens or 0)
        self.reasoning_output_tokens += int(output_tokens or 0)
        self.reasoning_requests += 1

    def add_failure(self):
        self.failed_requests += 1

    @property
    def total_tokens(self):
        return (
            self.embedding_tokens
            + self.reasoning_input_tokens
            + self.reasoning_output_tokens
        )

    def estimated_cost_usd(self):
        per_m = 1_000_000.0
        cost = (
            self.embedding_tokens
            * self._prices.get("embedding", 0.0) / per_m
            + self.reasoning_input_tokens
            * self._prices.get("reasoning_input", 0.0) / per_m
            + self.reasoning_output_tokens
            * self._prices.get("reasoning_output", 0.0) / per_m
        )
        return round(cost, 6)

    def to_dict(self):
        return {
            "embedding_tokens": self.embedding_tokens,
            "reasoning_input_tokens": self.reasoning_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
            "embedding_requests": self.embedding_requests,
            "reasoning_requests": self.reasoning_requests,
            "failed_requests": self.failed_requests,
            "estimated_cost_usd": self.estimated_cost_usd(),
            "cost_basis": (
                "Estimated from configured list prices and API-reported "
                "token counts. Not a billing figure."
            ),
        }


class OpenAIService:
    """Thin, testable wrapper over the two calls this project makes."""

    def __init__(
        self,
        api_key=None,
        embedding_model="text-embedding-3-small",
        reasoning_model="gpt-4.1-mini",
        timeout=30.0,
        max_retries=2,
        client=None,
        prices=None,
    ):
        self.embedding_model = embedding_model
        self.reasoning_model = reasoning_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = UsageRecord(prices)

        self._api_key = api_key or ""
        self._client = client  # tests inject a stub here

    # -- client ---------------------------------------------------------

    @property
    def client(self):
        """Build the SDK client on first use.

        Deferred so that importing the app, or running any offline test,
        never requires a key.
        """
        if self._client is None:
            if not self._api_key:
                raise OpenAIServiceError(
                    KIND_AUTH,
                    "OPENAI_API_KEY is not configured. Add it to your .env "
                    "file and restart the application.",
                )
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    # -- embeddings -----------------------------------------------------

    def embed(self, texts):
        """Embed a batch of strings, returning vectors in input order."""
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=list(texts)
            )
        except OpenAIServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            self.usage.add_failure()
            raise _translate(exc) from exc

        items = list(response.data or [])
        if len(items) != len(texts):
            self.usage.add_failure()
            raise OpenAIServiceError(
                KIND_INVALID_RESPONSE,
                f"The embedding API returned {len(items)} vectors for "
                f"{len(texts)} inputs.",
            )

        # The API supplies an index per item; sorting by it guarantees the
        # vectors line up with the sentences they belong to.
        items.sort(key=lambda item: getattr(item, "index", 0))
        self.usage.add_embedding(_usage_field(response, "total_tokens"))
        return [list(item.embedding) for item in items]

    # -- reasoning ------------------------------------------------------

    def complete_json(self, system_prompt, user_prompt, schema,
                      schema_name="result", max_output_tokens=1400):
        """Run one reasoning request constrained to ``schema``.

        Returns the parsed object. Raises :class:`OpenAIServiceError`
        with ``KIND_TRUNCATED`` if the model ran out of output budget, so
        the caller can split the batch instead of guessing.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                temperature=0,
                max_tokens=max_output_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except OpenAIServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            self.usage.add_failure()
            raise _translate(exc) from exc

        choices = list(response.choices or [])
        if not choices:
            self.usage.add_failure()
            raise OpenAIServiceError(
                KIND_INVALID_RESPONSE, "The model returned no choices."
            )

        choice = choices[0]
        self.usage.add_reasoning(
            _usage_field(response, "prompt_tokens"),
            _usage_field(response, "completion_tokens"),
        )

        if getattr(choice, "finish_reason", None) == "length":
            raise OpenAIServiceError(
                KIND_TRUNCATED,
                "The model's reply hit the output token limit and was cut "
                "off before the JSON was complete.",
            )

        content = getattr(choice.message, "content", None)
        if not content:
            raise OpenAIServiceError(
                KIND_INVALID_RESPONSE, "The model returned an empty reply."
            )

        try:
            parsed = json.loads(content)
        except (ValueError, TypeError) as exc:
            raise OpenAIServiceError(
                KIND_INVALID_RESPONSE,
                f"The model's reply was not valid JSON: {exc}",
            ) from exc

        if not isinstance(parsed, dict):
            raise OpenAIServiceError(
                KIND_INVALID_RESPONSE,
                "The model's reply was not a JSON object.",
            )
        return parsed


# ---------------------------------------------------------------------
# Factory + helpers
# ---------------------------------------------------------------------

def build_service(config, client=None):
    """Construct a service from app config (attribute or mapping)."""
    get = config.get if isinstance(config, dict) else (
        lambda key, default=None: getattr(config, key, default)
    )
    return OpenAIService(
        api_key=get("OPENAI_API_KEY", ""),
        embedding_model=get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        reasoning_model=get("OPENAI_REASONING_MODEL", "gpt-4.1-mini"),
        timeout=get("OPENAI_TIMEOUT_SECONDS", 30.0),
        max_retries=get("OPENAI_MAX_RETRIES", 2),
        client=client,
        prices={
            "embedding": get("PRICE_EMBEDDING_PER_MTOK", 0.02),
            "reasoning_input": get("PRICE_REASONING_INPUT_PER_MTOK", 0.40),
            "reasoning_output": get("PRICE_REASONING_OUTPUT_PER_MTOK", 1.60),
        },
    )


def _usage_field(response, name):
    usage = getattr(response, "usage", None)
    return getattr(usage, name, 0) if usage is not None else 0


def _translate(exc):
    """Map an SDK exception onto an :class:`OpenAIServiceError` kind.

    Matching is done on class names so that this module does not have to
    import the SDK just to classify an error, and so an SDK upgrade that
    adds an error type degrades to KIND_UNKNOWN rather than crashing.
    """
    name = type(exc).__name__
    detail = _safe_detail(exc)

    mapping = {
        "AuthenticationError": (
            KIND_AUTH,
            "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env "
            "file.",
        ),
        "PermissionDeniedError": (
            KIND_AUTH,
            "The API key does not have access to the requested model.",
        ),
        "RateLimitError": (
            KIND_RATE_LIMIT,
            "OpenAI rate limit or quota reached. Analysis of this batch was "
            f"skipped. ({detail})",
        ),
        "APITimeoutError": (
            KIND_TIMEOUT,
            f"The OpenAI request timed out. ({detail})",
        ),
        "APIConnectionError": (
            KIND_CONNECTION,
            f"Could not reach the OpenAI API. ({detail})",
        ),
        "BadRequestError": (
            KIND_BAD_REQUEST,
            f"OpenAI rejected the request. ({detail})",
        ),
        "NotFoundError": (
            KIND_BAD_REQUEST,
            f"The requested model was not found. ({detail})",
        ),
        "LengthFinishReasonError": (
            KIND_TRUNCATED,
            "The model's reply hit the output token limit.",
        ),
        "InternalServerError": (
            KIND_SERVER,
            f"OpenAI reported a server error. ({detail})",
        ),
    }

    if name in mapping:
        kind, message = mapping[name]
        return OpenAIServiceError(kind, message)

    log.warning("Unclassified OpenAI error %s", name)
    return OpenAIServiceError(
        KIND_UNKNOWN, f"Unexpected error calling OpenAI ({name}): {detail}"
    )


def _safe_detail(exc):
    """A short message for the user, with no credential material in it."""
    text = str(exc).strip().replace("\n", " ")
    if len(text) > 160:
        text = text[:157] + "..."
    # Defensive: never let an echoed key fragment reach a response.
    return "details withheld" if "sk-" in text else (text or "no details")

"""Embedding generation with batching, de-duplication and caching.

Three mechanisms keep API usage down:

* identical sentence text is embedded once per run, however many
  documents it appears in;
* a process-wide cache means re-analysing the same corpus costs nothing
  on the embedding side;
* requests are batched, so N sentences cost ceil(N / batch_size) calls.
"""
import hashlib
import threading
from collections import OrderedDict

import numpy as np

#: Bounded process-wide cache: (model, text) -> vector.
_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_ENTRIES = 5000


def cache_key(model, text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"


def cache_clear():
    with _CACHE_LOCK:
        _CACHE.clear()


def cache_size():
    with _CACHE_LOCK:
        return len(_CACHE)


def _cache_get(key):
    with _CACHE_LOCK:
        vector = _CACHE.get(key)
        if vector is not None:
            _CACHE.move_to_end(key)
        return vector


def _cache_put(key, vector):
    with _CACHE_LOCK:
        _CACHE[key] = vector
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX_ENTRIES:
            _CACHE.popitem(last=False)


class EmbeddingService:
    def __init__(self, service, batch_size=64, use_cache=True):
        self.service = service
        self.batch_size = max(1, int(batch_size))
        self.use_cache = use_cache

    def embed_sentences(self, sentences):
        """Embed sentence dicts, returning a matrix aligned to their order.

        Returns ``(matrix, stats)`` where ``matrix`` has one row per
        sentence in the order given, so ``matrix[i]`` always belongs to
        ``sentences[i]``.
        """
        texts = [s["text"] for s in sentences]
        stats = {
            "sentences": len(texts),
            "unique_texts": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "embedded": 0,
        }
        if not texts:
            return np.zeros((0, 0), dtype=np.float32), stats

        model = self.service.embedding_model

        # One slot per distinct text; duplicates share it.
        order = []           # distinct texts, in first-seen order
        slot_of_text = {}
        for text in texts:
            if text not in slot_of_text:
                slot_of_text[text] = len(order)
                order.append(text)
        stats["unique_texts"] = len(order)

        vectors = [None] * len(order)

        pending = []
        if self.use_cache:
            for slot, text in enumerate(order):
                cached = _cache_get(cache_key(model, text))
                if cached is None:
                    pending.append(slot)
                else:
                    vectors[slot] = cached
                    stats["cache_hits"] += 1
        else:
            pending = list(range(len(order)))

        for chunk in _batched(pending, self.batch_size):
            batch_texts = [order[slot] for slot in chunk]
            embedded = self.service.embed(batch_texts)
            stats["api_calls"] += 1
            stats["embedded"] += len(embedded)

            for slot, vector in zip(chunk, embedded):
                vectors[slot] = vector
                if self.use_cache:
                    _cache_put(cache_key(model, order[slot]), vector)

        missing = [slot for slot, vector in enumerate(vectors) if vector is None]
        if missing:
            raise ValueError(
                f"{len(missing)} sentences were left without an embedding."
            )

        matrix = np.asarray(
            [vectors[slot_of_text[text]] for text in texts], dtype=np.float32
        )
        return matrix, stats


def _batched(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]

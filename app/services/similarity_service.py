"""Cosine-similarity candidate selection.

Similarity decides which statement pairs are *worth asking about*. It is
never treated as evidence of a contradiction: two statements can be
near-identical in wording and perfectly compatible, or share little
vocabulary and still conflict. Only the reasoning model classifies.
"""
import numpy as np


def cosine_matrix(matrix):
    """Pairwise cosine similarity for L2-normalised row vectors.

    Rows with zero magnitude are left as zeros rather than producing
    NaNs, so a degenerate embedding cannot poison the whole matrix.
    """
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalised = matrix / safe
    return np.clip(normalised @ normalised.T, -1.0, 1.0)


def build_candidate_pairs(sentences, matrix, threshold=0.65, top_k=5,
                          max_pairs=100):
    """Shortlist cross-document statement pairs for classification.

    Returns ``(pairs, stats)``. Each pair is a dict with the indices,
    sentence ids, a canonical ``pair_key`` and the similarity score.

    Selection rules:
      * only pairs from *different* documents (self-comparison and
        within-document pairs are excluded);
      * similarity must reach ``threshold``;
      * at most ``top_k`` neighbours per statement, which avoids an
        all-against-all explosion on large corpora;
      * identical text is skipped - the same sentence in two documents
        cannot contradict itself, so it is not worth a request;
      * the final list is capped at ``max_pairs``, keeping the strongest
        candidates.
    """
    count = len(sentences)
    stats = {
        "statements": count,
        "cross_document_pairs": 0,
        "above_threshold": 0,
        "skipped_identical": 0,
        "selected": 0,
        "dropped_by_cap": 0,
        "threshold": threshold,
        "top_k": top_k,
        "max_pairs": max_pairs,
        "highest_similarity": None,
    }

    if count < 2 or matrix.size == 0:
        return [], stats

    documents = [s["document_id"] for s in sentences]
    if len(set(documents)) < 2:
        # Nothing to compare across: a single document cannot be checked
        # against another one.
        return [], stats

    similarity = cosine_matrix(matrix)

    same_document = np.equal.outer(np.array(documents), np.array(documents))
    comparable = ~same_document
    stats["cross_document_pairs"] = int(np.triu(comparable, k=1).sum())

    scored = np.where(comparable, similarity, -np.inf)
    if stats["cross_document_pairs"]:
        best = scored.max()
        stats["highest_similarity"] = (
            round(float(best), 4) if np.isfinite(best) else None
        )

    # Top-K neighbours per statement, then merge into canonical pairs.
    neighbours = min(top_k, count - 1)
    candidates = {}
    for index in range(count):
        row = scored[index]
        if neighbours < 1:
            continue
        ranked = np.argpartition(-row, neighbours - 1)[:neighbours]
        for other in ranked:
            score = float(row[other])
            if not np.isfinite(score) or score < threshold:
                continue
            low, high = (index, int(other)) if index < other else (int(other), index)
            key = (low, high)
            if key not in candidates or score > candidates[key]:
                candidates[key] = score

    stats["above_threshold"] = len(candidates)

    pairs = []
    for (low, high), score in candidates.items():
        first, second = sentences[low], sentences[high]
        if first["text"] == second["text"]:
            stats["skipped_identical"] += 1
            continue
        pairs.append({
            "index_a": low,
            "index_b": high,
            "sentence_id_a": first["sentence_id"],
            "sentence_id_b": second["sentence_id"],
            "pair_key": pair_key(first["sentence_id"], second["sentence_id"]),
            "similarity": round(score, 4),
        })

    # Strongest candidates first, with a stable tie-break so a run is
    # reproducible and the cap always keeps the same pairs.
    pairs.sort(key=lambda item: (-item["similarity"], item["pair_key"]))

    if len(pairs) > max_pairs:
        stats["dropped_by_cap"] = len(pairs) - max_pairs
        pairs = pairs[:max_pairs]

    stats["selected"] = len(pairs)
    return pairs, stats


def pair_key(sentence_id_a, sentence_id_b):
    """A canonical, order-independent identifier for a pair."""
    first, second = sorted((sentence_id_a, sentence_id_b))
    return f"{first}|{second}"

"""End-to-end analysis: statements in, conflict report out.

Stages: embed -> shortlist candidates -> classify -> assemble report.
Each stage records what it did, so the report explains how many pairs
were considered and why any were left out.
"""
import logging
from datetime import datetime, timezone

from .contradiction_service import (
    RELATIONSHIP_CONTRADICTION,
    RELATIONSHIP_NEUTRAL,
    RELATIONSHIP_SUPPORT,
    ContradictionService,
)
from .embedding_service import EmbeddingService
from .openai_service import OpenAIServiceError, build_service
from .similarity_service import build_candidate_pairs

log = logging.getLogger(__name__)

STAGE_EMBEDDING = "embedding"
STAGE_CANDIDATES = "candidate_selection"
STAGE_REASONING = "reasoning"
STAGE_REPORT = "report"

#: Stage order, shared with the dashboard progress display.
STAGES = (STAGE_EMBEDDING, STAGE_CANDIDATES, STAGE_REASONING, STAGE_REPORT)


class AnalysisRejected(Exception):
    """The analysis cannot be attempted (too large, nothing to compare)."""


def precheck(record, config):
    """Validate what can be checked without spending anything.

    Called synchronously before the worker thread starts, so an
    impossible analysis is refused immediately instead of failing in the
    background.
    """
    _guard(record.get("sentences") or [], config)


def run_analysis(record, config, service=None, on_stage=None):
    """Analyse a stored Phase 1 record and return the report.

    ``record`` is the payload produced by the ingestion pipeline.
    ``service`` is injectable so tests run without network access.
    ``on_stage(stage, message)`` is called as each stage begins, which
    is how the dashboard shows real progress rather than a guess.
    """
    sentences = record.get("sentences") or []
    _guard(sentences, config)

    service = service or build_service(config)
    warnings = []
    announce = on_stage or (lambda *_args, **_kwargs: None)

    # ---- 1. Embeddings ------------------------------------------------
    announce(STAGE_EMBEDDING, f"Generating embeddings for {len(sentences)} statements")
    embedder = EmbeddingService(
        service, batch_size=_cfg(config, "EMBEDDING_BATCH_SIZE", 64)
    )
    try:
        matrix, embedding_stats = embedder.embed_sentences(sentences)
    except OpenAIServiceError as exc:
        # Without embeddings there are no candidates, so this is fatal.
        log.warning("Embedding stage failed: %s", exc.kind)
        raise

    # ---- 2. Candidate selection --------------------------------------
    announce(STAGE_CANDIDATES, "Comparing statements across documents")
    pairs, similarity_stats = build_candidate_pairs(
        sentences,
        matrix,
        threshold=_cfg(config, "SIMILARITY_THRESHOLD", 0.65),
        top_k=_cfg(config, "TOP_K_CANDIDATES", 5),
        max_pairs=_cfg(config, "MAX_CANDIDATE_PAIRS", 100),
    )

    if similarity_stats["dropped_by_cap"]:
        warnings.append({
            "stage": STAGE_CANDIDATES,
            "kind": "pair_cap_reached",
            "message": (
                f"{similarity_stats['dropped_by_cap']} candidate pairs above "
                f"the similarity threshold were not analysed because the cap "
                f"of {similarity_stats['max_pairs']} pairs per analysis was "
                f"reached. The strongest candidates were kept."
            ),
        })

    if not pairs:
        warnings.append({
            "stage": STAGE_CANDIDATES,
            "kind": "no_candidates",
            "message": _no_candidate_reason(similarity_stats),
        })

    # ---- 3. Classification -------------------------------------------
    announce(
        STAGE_REASONING,
        f"Analysing {len(pairs)} candidate pair(s) for contradictions",
    )
    classifier = ContradictionService(
        service,
        batch_size=_cfg(config, "REASONING_BATCH_SIZE", 8),
        max_output_tokens=_cfg(config, "MAX_OUTPUT_TOKENS", 1400),
    )
    results, reasoning_warnings, reasoning_stats = classifier.classify(
        pairs, sentences
    )
    warnings.extend(reasoning_warnings)

    # ---- 4. Report ----------------------------------------------------
    announce(STAGE_REPORT, "Building the conflict report")
    return {
        "analysed_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "embedding": service.embedding_model,
            "reasoning": service.reasoning_model,
        },
        "overview": _overview(record, similarity_stats, reasoning_stats, results),
        "results": results,
        "warnings": warnings,
        "stages": {
            STAGE_EMBEDDING: embedding_stats,
            STAGE_CANDIDATES: similarity_stats,
            STAGE_REASONING: reasoning_stats,
        },
        "usage": service.usage.to_dict(),
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _guard(sentences, config):
    if len(sentences) < 2:
        raise AnalysisRejected(
            "At least two statements are needed to compare anything."
        )

    documents = {s["document_id"] for s in sentences}
    if len(documents) < 2:
        raise AnalysisRejected(
            "Statements from at least two different documents are needed; "
            "this analysis only has usable text from one document."
        )

    cap = _cfg(config, "MAX_SENTENCES_PER_ANALYSIS", 400)
    if len(sentences) > cap:
        raise AnalysisRejected(
            f"This analysis contains {len(sentences)} statements, above the "
            f"limit of {cap} per run. Upload fewer or shorter documents."
        )


def _overview(record, similarity_stats, reasoning_stats, results):
    counts = {
        RELATIONSHIP_CONTRADICTION: 0,
        RELATIONSHIP_SUPPORT: 0,
        RELATIONSHIP_NEUTRAL: 0,
    }
    for item in results:
        counts[item["relationship"]] = counts.get(item["relationship"], 0) + 1

    summary = record.get("summary") or {}
    return {
        "documents_analyzed": summary.get("documents_processed", 0),
        "statements_extracted": summary.get("total_sentences", 0),
        "candidate_pairs_analyzed": reasoning_stats.get("pairs_classified", 0),
        "candidate_pairs_shortlisted": similarity_stats.get("selected", 0),
        "contradictions_detected": counts[RELATIONSHIP_CONTRADICTION],
        "supporting_relationships": counts[RELATIONSHIP_SUPPORT],
        "neutral_relationships": counts[RELATIONSHIP_NEUTRAL],
        "pairs_unclassified": len(reasoning_stats.get("unclassified") or []),
    }


def _no_candidate_reason(stats):
    if stats["cross_document_pairs"] == 0:
        return (
            "No statement pairs from different documents were available to "
            "compare."
        )
    highest = stats.get("highest_similarity")
    detail = (
        f" The closest cross-document pair scored {highest}."
        if highest is not None else ""
    )
    return (
        f"No cross-document pair reached the similarity threshold of "
        f"{stats['threshold']}, so nothing was sent for classification."
        f"{detail} Lowering SIMILARITY_THRESHOLD will widen the search."
    )


def _cfg(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

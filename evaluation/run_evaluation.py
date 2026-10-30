"""Evaluate relationship classification against the labelled dataset.

Two modes, and the distinction is reported in every output file:

  --mode stub   (default) runs the labelled pairs through the pipeline
                with the test stub standing in for OpenAI. This checks
                that the harness, batching, validation and metrics work.
                **It measures the harness, not the model.** The stub
                classifies by comparing digits, so its scores say
                nothing about classification quality.

  --mode live   calls the real OpenAI API with the key from .env and
                produces a genuine evaluation. Costs money.

Usage:
    python -m evaluation.run_evaluation --mode stub
    python -m evaluation.run_evaluation --mode live
    python -m evaluation.run_evaluation --mode live --limit 10
    python -m evaluation.run_evaluation --mode live --retrieval
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.contradiction_service import (  # noqa: E402
    CATEGORIES,
    ContradictionService,
)
from app.services.openai_service import (  # noqa: E402
    OpenAIService,
    OpenAIServiceError,
    build_service,
)
from config import Config  # noqa: E402
from evaluation import metrics  # noqa: E402
from evaluation.dataset import LABELS, PAIRS, validate  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

STUB_WARNING = (
    "STUB MODE - these figures measure the evaluation harness, not the "
    "reasoning model. The stub classifies by comparing digits. Do not "
    "report these as model accuracy."
)


# ---------------------------------------------------------------------
# Turning the dataset into pipeline inputs
# ---------------------------------------------------------------------

def build_inputs(rows):
    """Convert labelled rows into (pairs, sentences) for the classifier.

    Each row contributes two statements, deliberately placed in two
    different documents, and one candidate pair joining them. No
    similarity gate is applied: the dataset supplies the pairs, so this
    isolates the classifier from candidate selection.
    """
    sentences = []
    pairs = []

    for index, row in enumerate(rows):
        position_a = len(sentences)
        sentences.append({
            "document_id": f"{row['pair_id']}_a",
            "document_name": row["document_a"],
            "page_number": None,
            "sentence_id": f"{row['pair_id']}_a",
            "text": row["text_a"],
        })
        position_b = len(sentences)
        sentences.append({
            "document_id": f"{row['pair_id']}_b",
            "document_name": row["document_b"],
            "page_number": None,
            "sentence_id": f"{row['pair_id']}_b",
            "text": row["text_b"],
        })
        pairs.append({
            "index_a": position_a,
            "index_b": position_b,
            "sentence_id_a": sentences[position_a]["sentence_id"],
            "sentence_id_b": sentences[position_b]["sentence_id"],
            # The dataset id doubles as the pair key so results map back.
            "pair_key": row["pair_id"],
            "similarity": None,
        })

    return pairs, sentences


def make_service(mode):
    if mode == "live":
        if not Config.OPENAI_API_KEY:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Put it in .env to run a live "
                "evaluation, or use --mode stub."
            )
        return build_service(Config)

    # Stub mode: the harness exercises itself with no network access.
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "tests")
    )
    from stubs import StubClient  # noqa: E402

    return OpenAIService(api_key="stub", client=StubClient())


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_relationships(rows, service, batch_size, max_output_tokens):
    pairs, sentences = build_inputs(rows)
    classifier = ContradictionService(
        service, batch_size=batch_size, max_output_tokens=max_output_tokens
    )
    results, warnings, stats = classifier.classify(pairs, sentences)
    predictions = {item["pair_key"]: item for item in results}

    scored = []
    rows_out = []
    for row in rows:
        predicted = predictions.get(row["pair_id"])
        relationship = predicted["relationship"] if predicted else None
        scored.append((row["gold_relationship"], relationship))

        rows_out.append({
            "pair_id": row["pair_id"],
            "gold_relationship": row["gold_relationship"],
            "predicted_relationship": relationship,
            "correct": relationship == row["gold_relationship"],
            "gold_category": row["gold_category"],
            "predicted_category": predicted["category"] if predicted else None,
            "confidence": predicted["confidence"] if predicted else None,
            "explanation": predicted["explanation"] if predicted else None,
            "is_hard_negative": row["note"].startswith("HARD NEGATIVE"),
            "text_a": row["text_a"],
            "text_b": row["text_b"],
            "note": row["note"],
        })

    report = metrics.evaluate(scored, LABELS)
    report["contradiction_detection"] = metrics.binary_detection(
        scored, "CONTRADICTION"
    )
    report["category_accuracy"] = _category_accuracy(rows_out)
    report["hard_negatives"] = _slice_accuracy(
        [r for r in rows_out if r["is_hard_negative"]]
    )
    report["by_gold_label"] = {
        label: _slice_accuracy(
            [r for r in rows_out if r["gold_relationship"] == label]
        )
        for label in LABELS
    }

    return report, rows_out, warnings, stats


def _category_accuracy(rows_out):
    """Category agreement, counted only where both agree it is a conflict.

    Category boundaries are fuzzy, so this is indicative. Reported
    separately from the relationship metrics for that reason.
    """
    relevant = [
        r for r in rows_out
        if r["gold_relationship"] == "CONTRADICTION"
        and r["predicted_relationship"] == "CONTRADICTION"
    ]
    matched = [r for r in relevant if r["predicted_category"] == r["gold_category"]]
    confusions = {}
    for row in relevant:
        if row["predicted_category"] != row["gold_category"]:
            key = f"{row['gold_category']} -> {row['predicted_category']}"
            confusions[key] = confusions.get(key, 0) + 1

    return {
        "evaluated_on": len(relevant),
        "matched": len(matched),
        "accuracy": round(len(matched) / len(relevant), 4) if relevant else 0.0,
        "confusions": confusions,
        "note": (
            "Counted only on pairs both the gold set and the model call a "
            "contradiction. Category boundaries overlap (a changed deadline "
            "is arguably both NUMERICAL and DATE), so read this as "
            "indicative."
        ),
    }


def _slice_accuracy(rows_out):
    if not rows_out:
        return {"count": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(1 for r in rows_out if r["correct"])
    return {
        "count": len(rows_out),
        "correct": correct,
        "accuracy": round(correct / len(rows_out), 4),
    }


# ---------------------------------------------------------------------
# Retrieval (candidate selection) evaluation
# ---------------------------------------------------------------------

def evaluate_retrieval(service, config):
    """Does candidate selection actually shortlist the planted pairs?

    Classification quality is irrelevant if the pair never reaches the
    model, so this measures the recall of the similarity gate over the
    sample documents, using real embeddings.
    """
    from evaluation.planted import PLANTED, locate
    from app.services.embedding_service import EmbeddingService
    from app.services.ingestion import ingest_documents
    from app.services.similarity_service import build_candidate_pairs

    sample_dir = os.path.join(
        os.path.dirname(__file__), "..", "sample_documents"
    )
    names = sorted(
        n for n in os.listdir(sample_dir) if n.lower().endswith(".txt")
    )

    class _File:
        """Minimal stand-in for a Werkzeug FileStorage."""

        def __init__(self, path, name):
            self.filename = name
            self._data = open(path, "rb").read()
            import io
            self.stream = io.BytesIO(self._data)

        def read(self):
            return self._data

    payload = ingest_documents(
        [_File(os.path.join(sample_dir, n), n) for n in names], config
    )
    sentences = payload["sentences"]

    matrix, _ = EmbeddingService(
        service, batch_size=getattr(config, "EMBEDDING_BATCH_SIZE", 64)
    ).embed_sentences(sentences)

    pairs, stats = build_candidate_pairs(
        sentences, matrix,
        threshold=getattr(config, "SIMILARITY_THRESHOLD", 0.65),
        top_k=getattr(config, "TOP_K_CANDIDATES", 5),
        max_pairs=getattr(config, "MAX_CANDIDATE_PAIRS", 100),
    )
    shortlisted = {
        frozenset((p["sentence_id_a"], p["sentence_id_b"])): p["similarity"]
        for p in pairs
    }

    found = []
    for planted in PLANTED:
        ids = locate(sentences, planted)
        row = {
            "label": planted["label"],
            "expected_relationship": planted["expected_relationship"],
            "snippet_a": planted["snippet_a"],
            "snippet_b": planted["snippet_b"],
            "located": bool(ids),
        }
        if ids:
            key = frozenset(ids)
            row["sentence_ids"] = sorted(ids)
            row["shortlisted"] = key in shortlisted
            row["similarity"] = shortlisted.get(key)
        else:
            row["shortlisted"] = False
            row["similarity"] = None
        found.append(row)

    locatable = [r for r in found if r["located"]]
    retrieved = [r for r in locatable if r["shortlisted"]]

    return {
        "documents": names,
        "statements": len(sentences),
        "candidate_selection": stats,
        "planted_pairs": found,
        "planted_total": len(PLANTED),
        "planted_located": len(locatable),
        "planted_shortlisted": len(retrieved),
        "retrieval_recall": (
            round(len(retrieved) / len(locatable), 4) if locatable else 0.0
        ),
        "note": (
            "Recall of the similarity gate: of the deliberately planted "
            "relationships, how many were shortlisted for the model. A "
            "planted pair that was not located in the segmented text is "
            "excluded from the ratio and reported under planted_located."
        ),
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def render_text_report(payload):
    report = payload.get("relationship_evaluation")
    lines = []
    lines.append("=" * 70)
    lines.append(f"CrossDoc Intelligence - evaluation ({payload['mode']} mode)")
    lines.append(f"run at {payload['run_at']}")
    if payload["mode"] == "stub":
        lines.append("")
        lines.append(STUB_WARNING)
    lines.append("=" * 70)

    if report:
        counts = report["counts"]
        lines.append("")
        lines.append(f"Pairs: {counts['total']}  "
                     f"classified: {counts['classified']}  "
                     f"unclassified: {counts['unclassified']}")
        lines.append(f"Accuracy (classified pairs): {report['accuracy']}")
        lines.append("Accuracy (unclassified counted as errors): "
                     f"{report['accuracy_including_unclassified']}")
        lines.append("")
        lines.append("Confusion matrix")
        lines.append(metrics.format_confusion_matrix(
            report["confusion_matrix"], report["labels"],
            report["unclassified_by_gold"],
        ))
        lines.append("")
        lines.append(f"{'label':<16}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>6}")
        for label in report["labels"]:
            row = report["per_class"][label]
            lines.append(
                f"{label:<16}{row['precision']:>8}{row['recall']:>8}"
                f"{row['f1']:>8}{row['support']:>6}"
            )
        lines.append(f"{'macro avg':<16}{report['macro_avg']['precision']:>8}"
                     f"{report['macro_avg']['recall']:>8}"
                     f"{report['macro_avg']['f1']:>8}")
        lines.append(f"{'weighted avg':<16}"
                     f"{report['weighted_avg']['precision']:>8}"
                     f"{report['weighted_avg']['recall']:>8}"
                     f"{report['weighted_avg']['f1']:>8}")

        detection = report["contradiction_detection"]
        lines.append("")
        lines.append("Contradiction detection (one-vs-rest)")
        lines.append(f"  precision {detection['precision']}  "
                     f"recall {detection['recall']}  f1 {detection['f1']}")
        lines.append(f"  tp {detection['true_positive']}  "
                     f"fp {detection['false_positive']}  "
                     f"fn {detection['false_negative']}  "
                     f"tn {detection['true_negative']}")

        hard = report["hard_negatives"]
        lines.append("")
        lines.append(f"Hard negatives: {hard['correct']}/{hard['count']} "
                     f"correct (accuracy {hard['accuracy']})")

        category = report["category_accuracy"]
        lines.append(f"Category agreement: {category['matched']}/"
                     f"{category['evaluated_on']} "
                     f"(accuracy {category['accuracy']})")
        if category["confusions"]:
            for key, count in sorted(category["confusions"].items()):
                lines.append(f"    {key}: {count}")

    retrieval = payload.get("retrieval_evaluation")
    if retrieval:
        lines.append("")
        lines.append("-" * 70)
        lines.append("Candidate selection (similarity gate)")
        lines.append(f"  statements: {retrieval['statements']}  "
                     f"shortlisted pairs: "
                     f"{retrieval['candidate_selection']['selected']}")
        lines.append(f"  planted pairs located: {retrieval['planted_located']}"
                     f"/{retrieval['planted_total']}")
        lines.append(f"  planted pairs shortlisted: "
                     f"{retrieval['planted_shortlisted']}"
                     f"/{retrieval['planted_located']}  "
                     f"(recall {retrieval['retrieval_recall']})")
        for row in retrieval["planted_pairs"]:
            mark = "ok  " if row["shortlisted"] else "MISS"
            similarity = row["similarity"]
            lines.append(
                f"    {mark} {row['label']:<34} "
                f"sim={similarity if similarity is not None else '-'}"
            )

    usage = payload.get("usage")
    if usage:
        lines.append("")
        lines.append("-" * 70)
        lines.append("API usage (reported by the API)")
        lines.append(f"  embedding tokens     {usage['embedding_tokens']}")
        lines.append(f"  reasoning in tokens  {usage['reasoning_input_tokens']}")
        lines.append(f"  reasoning out tokens {usage['reasoning_output_tokens']}")
        lines.append(f"  total tokens         {usage['total_tokens']}")
        lines.append(f"  requests             "
                     f"{usage['embedding_requests']} embedding, "
                     f"{usage['reasoning_requests']} reasoning, "
                     f"{usage['failed_requests']} failed")
        lines.append(f"  estimated cost (USD) {usage['estimated_cost_usd']}")
        lines.append(f"  {usage['cost_basis']}")

    warnings = payload.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append(f"Warnings ({len(warnings)})")
        for warning in warnings:
            lines.append(f"  [{warning['kind']}] {warning['message']}")

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stub", "live"), default="stub")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N labelled pairs")
    parser.add_argument("--retrieval", action="store_true",
                        help="also measure the similarity gate on the "
                             "sample documents")
    parser.add_argument("--skip-relationships", action="store_true",
                        help="run only the retrieval evaluation")
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args(argv)

    validate()
    rows = PAIRS[:args.limit] if args.limit else list(PAIRS)
    service = make_service(args.mode)

    payload = {
        "mode": args.mode,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "embedding": service.embedding_model,
            "reasoning": service.reasoning_model,
        },
        "dataset": {
            "pairs_available": len(PAIRS),
            "pairs_evaluated": 0 if args.skip_relationships else len(rows),
            "source": "evaluation/dataset.py (hand-labelled)",
        },
        "warnings": [],
    }
    if args.mode == "stub":
        payload["caveat"] = STUB_WARNING

    if not args.skip_relationships:
        try:
            report, rows_out, warnings, stats = evaluate_relationships(
                rows, service,
                batch_size=Config.REASONING_BATCH_SIZE,
                max_output_tokens=Config.MAX_OUTPUT_TOKENS,
            )
        except OpenAIServiceError as exc:
            raise SystemExit(f"Evaluation aborted: [{exc.kind}] {exc.message}")

        payload["relationship_evaluation"] = report
        payload["predictions"] = rows_out
        payload["reasoning_stats"] = stats
        payload["warnings"].extend(warnings)

    if args.retrieval:
        try:
            payload["retrieval_evaluation"] = evaluate_retrieval(
                service, Config
            )
        except OpenAIServiceError as exc:
            raise SystemExit(
                f"Retrieval evaluation aborted: [{exc.kind}] {exc.message}"
            )

    payload["usage"] = service.usage.to_dict()

    text = render_text_report(payload)
    print(text)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = os.path.join(args.out, f"evaluation-{args.mode}-{stamp}")
    with open(f"{base}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(f"{base}.txt", "w", encoding="utf-8") as handle:
        handle.write(text + "\n")

    print(f"\nwritten: {base}.json")
    print(f"written: {base}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

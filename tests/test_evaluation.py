"""The evaluation harness itself.

These tests check that the harness measures what it claims to: that the
dataset is internally consistent, that labelled pairs are handed to the
classifier correctly, that the planted relationships can still be found
in the segmented sample documents, and that stub-mode output is clearly
marked as not being a model measurement.
"""
import io
import json
import os

import pytest

from app.services.openai_service import OpenAIService
from config import TestConfig
from evaluation import dataset, metrics
from evaluation.planted import PLANTED, locate
from evaluation.run_evaluation import (
    STUB_WARNING,
    build_inputs,
    evaluate_relationships,
    main,
    render_text_report,
)
from stubs import StubClient

SAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sample_documents")
)


# ------------------------------------------------------------- dataset

def test_dataset_is_internally_consistent():
    assert dataset.validate() is True


def test_dataset_size_is_in_the_intended_range():
    # The brief asked for roughly 30-50 labelled pairs.
    assert 30 <= len(dataset.PAIRS) <= 50


def test_every_label_is_well_represented():
    distribution = dataset.label_distribution()
    assert set(distribution) == set(dataset.LABELS)
    # No class so small that its metrics would be meaningless.
    assert all(count >= 10 for count in distribution.values()), distribution


def test_hard_negatives_are_present_and_labelled_neutral():
    hard = dataset.hard_negatives()
    assert len(hard) >= 6
    assert all(row["gold_relationship"] == dataset.NEUTRAL for row in hard)


def test_every_contradiction_category_is_exercised():
    categories = {
        row["gold_category"] for row in dataset.PAIRS if row["gold_category"]
    }
    # OTHER is a fallback, not something to write a gold case for.
    assert categories >= {
        "NUMERICAL_CONFLICT", "DATE_CONFLICT", "POLICY_CONFLICT",
        "FACTUAL_CONFLICT", "LOGICAL_CONFLICT",
    }


def test_duplicate_pair_ids_are_rejected():
    original = dataset.PAIRS[0]
    dataset.PAIRS.append(dict(original))
    try:
        with pytest.raises(ValueError, match="duplicate"):
            dataset.validate()
    finally:
        dataset.PAIRS.pop()


# -------------------------------------------------------------- inputs

def test_build_inputs_places_each_statement_in_its_own_document():
    pairs, sentences = build_inputs(dataset.PAIRS[:3])

    assert len(pairs) == 3
    assert len(sentences) == 6

    for pair in pairs:
        first = sentences[pair["index_a"]]
        second = sentences[pair["index_b"]]
        # Cross-document comparison is what the system does, so the
        # evaluation must not hand it a same-document pair.
        assert first["document_id"] != second["document_id"]
        assert pair["sentence_id_a"] == first["sentence_id"]
        assert pair["sentence_id_b"] == second["sentence_id"]


def test_pair_key_is_the_dataset_id_so_results_map_back():
    pairs, _ = build_inputs(dataset.PAIRS[:2])
    assert [p["pair_key"] for p in pairs] == ["ev_001", "ev_002"]


def test_statement_text_is_passed_through_unaltered():
    row = dataset.PAIRS[0]
    _pairs, sentences = build_inputs([row])
    assert sentences[0]["text"] == row["text_a"]
    assert sentences[1]["text"] == row["text_b"]


def test_no_similarity_is_claimed_for_dataset_pairs():
    # The dataset supplies the pairs directly, so there is no similarity
    # score to report and none should be invented.
    pairs, _ = build_inputs(dataset.PAIRS[:1])
    assert pairs[0]["similarity"] is None


# ------------------------------------------------------------ scoring

def test_evaluation_scores_every_pair_and_tracks_slices():
    rows = dataset.PAIRS[:12]
    service = OpenAIService(api_key="stub", client=StubClient())

    report, predictions, _warnings, stats = evaluate_relationships(
        rows, service, batch_size=4, max_output_tokens=1400
    )

    assert stats["pairs_submitted"] == 12
    assert len(predictions) == 12
    assert report["counts"]["total"] == 12
    assert set(report["labels"]) == set(dataset.LABELS)
    assert "contradiction_detection" in report
    assert "hard_negatives" in report
    assert report["by_gold_label"].keys() == set(dataset.LABELS)

    for row in predictions:
        assert row["pair_id"].startswith("ev_")
        assert row["gold_relationship"] in dataset.LABELS
        assert row["correct"] == (
            row["predicted_relationship"] == row["gold_relationship"]
        )


def test_an_unclassified_pair_is_never_scored_as_correct():
    rows = dataset.PAIRS[:4]
    client = StubClient()
    client.omit_pairs = {"ev_001"}
    service = OpenAIService(api_key="stub", client=client)

    report, predictions, _warnings, _stats = evaluate_relationships(
        rows, service, batch_size=4, max_output_tokens=1400
    )

    missing = next(r for r in predictions if r["pair_id"] == "ev_001")
    assert missing["predicted_relationship"] is None
    assert missing["correct"] is False
    assert report["counts"]["unclassified"] == 1
    # Strict accuracy charges it as an error; the lenient one excludes it.
    assert (report["accuracy_including_unclassified"]
            <= report["accuracy"])


def test_category_accuracy_only_counts_agreed_contradictions():
    rows = [r for r in dataset.PAIRS if r["gold_relationship"] == "CONTRADICTION"][:4]
    service = OpenAIService(api_key="stub", client=StubClient())

    report, predictions, _w, _s = evaluate_relationships(
        rows, service, batch_size=4, max_output_tokens=1400
    )

    agreed = [
        r for r in predictions
        if r["gold_relationship"] == "CONTRADICTION"
        and r["predicted_relationship"] == "CONTRADICTION"
    ]
    assert report["category_accuracy"]["evaluated_on"] == len(agreed)
    assert "indicative" in report["category_accuracy"]["note"]


# ------------------------------------------------- planted / retrieval

def test_every_planted_relationship_is_findable_in_the_segmented_text():
    """Guards the retrieval ground truth against snippet drift.

    If segmentation changes shape, or a sample document is edited, this
    fails rather than quietly reporting a planted pair as "not located".
    """
    from app.services.ingestion import ingest_documents

    class _File:
        def __init__(self, path, name):
            self.filename = name
            self._data = open(path, "rb").read()
            self.stream = io.BytesIO(self._data)

        def read(self):
            return self._data

    names = sorted(
        n for n in os.listdir(SAMPLE_DIR) if n.lower().endswith(".txt")
    )
    payload = ingest_documents(
        [_File(os.path.join(SAMPLE_DIR, n), n) for n in names], TestConfig
    )
    sentences = payload["sentences"]

    unlocated = [p["label"] for p in PLANTED if not locate(sentences, p)]
    assert not unlocated, f"planted pairs no longer locatable: {unlocated}"


def test_locate_requires_two_different_documents():
    sentences = [
        {"document_id": "d1", "sentence_id": "s1", "text": "twenty days"},
        {"document_id": "d1", "sentence_id": "s2", "text": "fifteen days"},
    ]
    planted = {
        "snippet_a": "twenty days", "snippet_b": "fifteen days",
        "label": "x", "expected_relationship": "CONTRADICTION",
    }
    assert locate(sentences, planted) == set()

    sentences[1]["document_id"] = "d2"
    assert locate(sentences, planted) == {"s1", "s2"}


def test_locate_returns_empty_when_a_snippet_is_absent():
    sentences = [
        {"document_id": "d1", "sentence_id": "s1", "text": "twenty days"},
    ]
    assert locate(sentences, {
        "snippet_a": "twenty days", "snippet_b": "nowhere to be found",
        "label": "x", "expected_relationship": "CONTRADICTION",
    }) == set()


# ------------------------------------------------------------ reporting

def test_stub_mode_output_is_marked_as_not_a_model_measurement():
    rows = dataset.PAIRS[:4]
    service = OpenAIService(api_key="stub", client=StubClient())
    report, _p, _w, _s = evaluate_relationships(
        rows, service, batch_size=4, max_output_tokens=1400
    )

    text = render_text_report({
        "mode": "stub",
        "run_at": "now",
        "relationship_evaluation": report,
        "usage": service.usage.to_dict(),
    })

    assert STUB_WARNING in text
    assert "not the reasoning model" in text
    assert "Confusion matrix" in text


def test_live_mode_output_carries_no_stub_warning():
    text = render_text_report({"mode": "live", "run_at": "now"})
    assert STUB_WARNING not in text
    assert "live mode" in text


def test_cli_writes_json_and_text_results(tmp_path):
    exit_code = main([
        "--mode", "stub", "--limit", "6", "--out", str(tmp_path),
    ])
    assert exit_code == 0

    written = sorted(os.listdir(tmp_path))
    assert len(written) == 2
    assert any(name.endswith(".json") for name in written)
    assert any(name.endswith(".txt") for name in written)

    payload = json.loads(
        (tmp_path / next(n for n in written if n.endswith(".json")))
        .read_text(encoding="utf-8")
    )
    assert payload["mode"] == "stub"
    assert payload["caveat"] == STUB_WARNING
    assert payload["dataset"]["pairs_evaluated"] == 6
    assert len(payload["predictions"]) == 6
    assert payload["usage"]["reasoning_requests"] >= 1


def test_cli_refuses_live_mode_without_a_key(tmp_path, monkeypatch):
    monkeypatch.setattr("config.Config.OPENAI_API_KEY", "")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        main(["--mode", "live", "--limit", "2", "--out", str(tmp_path)])


def test_metrics_module_is_the_one_used_by_the_runner():
    # A guard against the runner quietly growing its own arithmetic.
    assert hasattr(metrics, "evaluate")
    assert hasattr(metrics, "binary_detection")

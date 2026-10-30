"""Metric arithmetic, checked against hand-computed values."""
import pytest

from evaluation.metrics import (
    binary_detection,
    confusion_matrix,
    evaluate,
    format_confusion_matrix,
    per_class,
)

LABELS = ("CONTRADICTION", "SUPPORT", "NEUTRAL")
C, S, N = LABELS


# ------------------------------------------------------- confusion matrix

def test_perfect_predictions_fill_the_diagonal():
    pairs = [(C, C), (S, S), (N, N), (C, C)]
    table = confusion_matrix(pairs, LABELS)

    assert table["matrix"][C][C] == 2
    assert table["matrix"][S][S] == 1
    assert table["matrix"][C][S] == 0
    assert sum(table["unclassified"].values()) == 0


def test_unclassified_predictions_are_tallied_apart():
    table = confusion_matrix([(C, None), (C, C), (S, None)], LABELS)

    assert table["unclassified"] == {C: 1, S: 1, N: 0}
    # The missing prediction is not charged to any predicted column.
    assert sum(table["matrix"][C].values()) == 1


def test_unknown_labels_are_rejected():
    with pytest.raises(ValueError, match="gold"):
        confusion_matrix([("MAYBE", C)], LABELS)
    with pytest.raises(ValueError, match="predicted"):
        confusion_matrix([(C, "MAYBE")], LABELS)


# ------------------------------------------------------------- per class

def test_per_class_arithmetic_is_hand_checked():
    # 3 contradictions: 2 found, 1 called neutral.
    # 1 support pair wrongly called a contradiction.
    pairs = [(C, C), (C, C), (C, N), (S, C), (S, S), (N, N)]
    report = per_class(confusion_matrix(pairs, LABELS)["matrix"], LABELS)

    contradiction = report[C]
    assert contradiction["true_positive"] == 2
    assert contradiction["false_positive"] == 1   # the SUPPORT pair
    assert contradiction["false_negative"] == 1   # called NEUTRAL
    assert contradiction["support"] == 3
    assert contradiction["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert contradiction["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert contradiction["f1"] == pytest.approx(2 / 3, abs=1e-4)


def test_a_label_never_predicted_scores_zero_not_an_error():
    pairs = [(C, N), (S, N), (N, N)]
    report = per_class(confusion_matrix(pairs, LABELS)["matrix"], LABELS)

    assert report[C]["precision"] == 0.0
    assert report[C]["recall"] == 0.0
    assert report[C]["f1"] == 0.0
    assert report[N]["recall"] == 1.0


# -------------------------------------------------------------- evaluate

def test_evaluate_reports_both_accuracy_readings():
    pairs = [(C, C), (C, C), (C, N), (S, S), (N, N), (N, None)]
    result = evaluate(pairs, LABELS)

    # 5 classified, 4 correct.
    assert result["counts"] == {
        "total": 6, "classified": 5, "correct": 4,
        "incorrect": 1, "unclassified": 1,
    }
    assert result["accuracy"] == pytest.approx(0.8)
    # The unclassified pair counts as an error in the stricter reading.
    assert result["accuracy_including_unclassified"] == pytest.approx(4 / 6,
                                                                      abs=1e-4)
    assert result["unclassified_by_gold"][N] == 1


def test_macro_average_is_the_unweighted_mean():
    pairs = [(C, C), (S, S), (N, N)]
    result = evaluate(pairs, LABELS)

    assert result["macro_avg"]["f1"] == 1.0
    assert result["weighted_avg"]["f1"] == 1.0
    assert result["accuracy"] == 1.0


def test_macro_and_weighted_differ_on_imbalanced_data():
    # NEUTRAL dominates and is perfect; CONTRADICTION is always missed.
    pairs = [(C, N)] + [(N, N)] * 9
    result = evaluate(pairs, LABELS)

    assert result["per_class"][C]["recall"] == 0.0
    assert result["weighted_avg"]["recall"] > result["macro_avg"]["recall"]


def test_empty_input_does_not_divide_by_zero():
    result = evaluate([], LABELS)
    assert result["accuracy"] == 0.0
    assert result["macro_avg"]["f1"] == 0.0
    assert result["counts"]["total"] == 0


# ------------------------------------------------------ binary detection

def test_binary_detection_treats_one_label_as_positive():
    pairs = [(C, C), (C, N), (S, C), (N, N), (S, S)]
    result = binary_detection(pairs, C)

    assert result["true_positive"] == 1
    assert result["false_negative"] == 1     # contradiction called neutral
    assert result["false_positive"] == 1     # support called contradiction
    assert result["true_negative"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_unclassified_positive_counts_as_a_miss():
    result = binary_detection([(C, None), (N, None)], C)

    assert result["false_negative"] == 1
    assert result["false_negative_due_to_unclassified"] == 1
    # An unclassified negative was never going to be flagged anyway.
    assert result["true_negative"] == 1
    assert result["recall"] == 0.0


# ---------------------------------------------------------------- output

def test_matrix_renders_as_a_readable_table():
    pairs = [(C, C), (S, N), (N, None)]
    result = evaluate(pairs, LABELS)
    text = format_confusion_matrix(
        result["confusion_matrix"], LABELS, result["unclassified_by_gold"]
    )

    assert "CONTRADICTION" in text
    assert "rows = gold label" in text
    assert "none" in text          # the unclassified column
    assert len(text.splitlines()) == len(LABELS) + 2

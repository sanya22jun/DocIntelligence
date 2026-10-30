"""Classification metrics, implemented directly.

Written out rather than pulled from scikit-learn so the project has no
extra dependency and so the arithmetic behind the reported numbers is
visible and unit tested.

Predictions may legitimately be missing: a pair the model failed to
classify is counted in ``unclassified`` and excluded from the per-class
metrics, never silently scored as a wrong answer or a right one.
"""

def confusion_matrix(pairs, labels):
    """Build a gold x predicted count table.

    ``pairs`` is a sequence of ``(gold, predicted)``. A predicted value
    of None is tallied separately under ``unclassified``.
    """
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    unclassified = {gold: 0 for gold in labels}

    for gold, predicted in pairs:
        if gold not in matrix:
            raise ValueError(f"unknown gold label: {gold!r}")
        if predicted is None:
            unclassified[gold] += 1
            continue
        if predicted not in labels:
            raise ValueError(f"unknown predicted label: {predicted!r}")
        matrix[gold][predicted] += 1

    return {"matrix": matrix, "unclassified": unclassified}


def _divide(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


def per_class(matrix, labels):
    """Precision, recall, F1 and support for each label."""
    report = {}
    for label in labels:
        true_positive = matrix[label][label]
        false_negative = sum(
            matrix[label][other] for other in labels if other != label
        )
        false_positive = sum(
            matrix[other][label] for other in labels if other != label
        )

        precision = _divide(true_positive, true_positive + false_positive)
        recall = _divide(true_positive, true_positive + false_negative)
        f1 = _divide(2 * precision * recall, precision + recall)

        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": true_positive + false_negative,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    return report


def evaluate(pairs, labels):
    """Full metric set for a list of ``(gold, predicted)`` pairs.

    ``accuracy`` is computed over the pairs that were actually
    classified; ``accuracy_including_unclassified`` charges every
    unclassified pair as an error, which is the stricter reading.
    """
    table = confusion_matrix(pairs, labels)
    matrix = table["matrix"]
    classes = per_class(matrix, labels)

    scored = sum(
        matrix[gold][pred] for gold in labels for pred in labels
    )
    correct = sum(matrix[label][label] for label in labels)
    unclassified_total = sum(table["unclassified"].values())

    macro = {
        key: round(
            _divide(sum(classes[label][key] for label in labels), len(labels)),
            4,
        )
        for key in ("precision", "recall", "f1")
    }

    total_support = sum(classes[label]["support"] for label in labels)
    weighted = {
        key: round(
            _divide(
                sum(
                    classes[label][key] * classes[label]["support"]
                    for label in labels
                ),
                total_support,
            ),
            4,
        )
        for key in ("precision", "recall", "f1")
    }

    return {
        "labels": list(labels),
        "confusion_matrix": matrix,
        "unclassified_by_gold": table["unclassified"],
        "per_class": classes,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "accuracy": round(_divide(correct, scored), 4),
        "accuracy_including_unclassified": round(
            _divide(correct, scored + unclassified_total), 4
        ),
        "counts": {
            "total": scored + unclassified_total,
            "classified": scored,
            "correct": correct,
            "incorrect": scored - correct,
            "unclassified": unclassified_total,
        },
    }


def binary_detection(pairs, positive):
    """Treat one label as the positive class, everything else negative.

    For contradiction detection this is the number that matters: how
    often a flagged conflict is real, and how many real conflicts were
    found. Unclassified pairs count as missed positives.
    """
    true_positive = false_positive = false_negative = true_negative = 0
    missed_unclassified = 0

    for gold, predicted in pairs:
        gold_positive = gold == positive
        if predicted is None:
            if gold_positive:
                false_negative += 1
                missed_unclassified += 1
            else:
                true_negative += 1
            continue

        predicted_positive = predicted == positive
        if gold_positive and predicted_positive:
            true_positive += 1
        elif gold_positive:
            false_negative += 1
        elif predicted_positive:
            false_positive += 1
        else:
            true_negative += 1

    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, true_positive + false_negative)

    return {
        "positive_label": positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_negative_due_to_unclassified": missed_unclassified,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_divide(2 * precision * recall, precision + recall), 4),
    }


def format_confusion_matrix(matrix, labels, unclassified=None):
    """Render the matrix as a fixed-width text table."""
    short = {label: label[:4] for label in labels}
    width = max(len(label) for label in labels) + 2

    header = " " * width + "".join(f"{short[label]:>7}" for label in labels)
    if unclassified:
        header += f"{'none':>7}"
    lines = [header]

    for gold in labels:
        row = f"{gold:<{width}}"
        row += "".join(f"{matrix[gold][pred]:>7}" for pred in labels)
        if unclassified:
            row += f"{unclassified.get(gold, 0):>7}"
        lines.append(row)

    lines.append("(rows = gold label, columns = predicted)")
    return "\n".join(lines)

"""Ground truth for the three sample documents.

The sample documents were written with known relationships in them.
These entries identify each one by a short snippet of its statements, so
the retrieval evaluation can check whether the similarity gate actually
shortlisted the pair for the model. Snippets are matched against the
segmented statements, which keeps this file independent of how
segmentation happens to split the text.
"""

PLANTED = [
    {
        "label": "annual leave: 20 vs 15 days",
        "snippet_a": "entitled to 20 days of paid annual leave",
        "snippet_b": "receive 15 days of paid annual leave",
        "expected_relationship": "CONTRADICTION",
    },
    {
        "label": "leave notice: 14 vs 30 days",
        "snippet_a": "at least 14 days before the intended start date",
        "snippet_b": "at least 30 days in advance of the start date",
        "expected_relationship": "CONTRADICTION",
    },
    {
        "label": "remote work: 2 vs 3 days per week",
        "snippet_a": "work remotely for up to 2 days per week with manager approval",
        "snippet_b": "maximum of 3 days per week, subject to manager approval",
        "expected_relationship": "CONTRADICTION",
    },
    {
        "label": "leave year: 1 April vs 1 January",
        "snippet_a": "leave year begins on 1 April",
        "snippet_b": "leave year begins on 1 January",
        "expected_relationship": "CONTRADICTION",
    },
    {
        "label": "carry-over: 5 days vs forfeited",
        "snippet_a": "may be carried over to the following year",
        "snippet_b": "cannot be carried over and is forfeited",
        "expected_relationship": "CONTRADICTION",
    },
    {
        "label": "reviews: twice vs once per year",
        "snippet_a": "conducted twice per year, in June and December",
        "snippet_b": "take place once per year, in December",
        "expected_relationship": "CONTRADICTION",
    },
    {
        "label": "sick leave: 10 days in both",
        "snippet_a": "up to 10 days of paid sick leave per calendar year",
        "snippet_b": "provides 10 days of paid sick leave annually",
        "expected_relationship": "SUPPORT",
    },
    {
        "label": "medical note after 2 days in both",
        "snippet_a": "medical certificate is required for any sick leave absence",
        "snippet_b": "doctor's note is required when an employee is absent",
        "expected_relationship": "SUPPORT",
    },
    {
        "label": "probation excluded from remote work",
        "snippet_a": "not available to employees serving their probation period",
        "snippet_b": "on probation are not eligible for remote work",
        "expected_relationship": "SUPPORT",
    },
    {
        "label": "manager response within 3 working days",
        "snippet_a": "approve or reject a leave request within 3 working days",
        "snippet_b": "respond to a submitted leave request within 3 working days",
        "expected_relationship": "SUPPORT",
    },
    {
        "label": "remote work: 2 days (HR vs Finance)",
        "snippet_a": "work remotely for up to 2 days per week with manager approval",
        "snippet_b": "permitted to work remotely for up to 2 days per week",
        "expected_relationship": "SUPPORT",
    },
]


def locate(sentences, planted):
    """Find the sentence ids whose text contains each snippet.

    Returns a set of two ids, or an empty set if either snippet could not
    be matched to exactly one statement in a distinct document. Being
    strict here keeps a mislocated pair from being silently scored.
    """
    matches_a = [
        s for s in sentences if planted["snippet_a"].lower() in s["text"].lower()
    ]
    matches_b = [
        s for s in sentences if planted["snippet_b"].lower() in s["text"].lower()
    ]
    if not matches_a or not matches_b:
        return set()

    for first in matches_a:
        for second in matches_b:
            if first["document_id"] != second["document_id"]:
                return {first["sentence_id"], second["sentence_id"]}
    return set()

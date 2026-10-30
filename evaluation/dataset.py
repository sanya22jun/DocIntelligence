"""Manually labelled evaluation set for relationship classification.

Each entry is a statement pair with a gold label assigned by hand, and a
note recording *why* that label is correct. Pairs were written to probe
the behaviour that matters:

* real conflicts of every category the system reports;
* agreements that are worded differently, or state the same quantity in
  different units, so that paraphrase is not mistaken for conflict;
* **hard negatives** - pairs that look like conflicts because their
  numbers differ, but whose scope, subject or population differs too, so
  the correct answer is NEUTRAL. These are the cases a naive
  number-comparison approach gets wrong, and they are deliberately
  over-represented.

This is a small, hand-built set. It measures whether the classifier
behaves sensibly on representative cases; it is not a benchmark and the
figures from it should not be generalised.
"""

CONTRADICTION = "CONTRADICTION"
SUPPORT = "SUPPORT"
NEUTRAL = "NEUTRAL"

HR_2023 = "hr_leave_policy_2023.txt"
HR_2024 = "hr_handbook_2024.txt"
FINANCE = "finance_travel_policy.txt"

#: ``gold_category`` is only meaningful for CONTRADICTION rows. Category
#: boundaries are genuinely fuzzy (a changed deadline is arguably both
#: NUMERICAL and DATE), so category accuracy is reported separately and
#: should be read as indicative only.
PAIRS = [
    # ---------------- contradictions: numerical ----------------
    {
        "pair_id": "ev_001",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
        "text_b": "Full-time employees receive 15 days of paid annual leave each calendar year.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "NUMERICAL_CONFLICT",
        "note": "Same subject, same scope, different entitlement.",
    },
    {
        "pair_id": "ev_002",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Annual leave requests must be submitted at least 14 days before the intended start date.",
        "text_b": "All leave requests must be submitted at least 30 days in advance of the start date.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "NUMERICAL_CONFLICT",
        "note": "Incompatible notice periods for the same action.",
    },
    {
        "pair_id": "ev_003",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Employees are permitted to work remotely for up to 2 days per week with manager approval.",
        "text_b": "Remote work is permitted for a maximum of 3 days per week, subject to manager approval.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "NUMERICAL_CONFLICT",
        "note": "Two different stated maxima for the same allowance.",
    },
    {
        "pair_id": "ev_004",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "The probation period for new employees is three months.",
        "text_b": "All new employees serve a probation period of six months.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "NUMERICAL_CONFLICT",
        "note": "Same population, different duration.",
    },
    {
        "pair_id": "ev_005",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Hotel accommodation is reimbursed up to 4000 rupees per night in metropolitan cities.",
        "text_b": "Hotel stays in metropolitan cities are reimbursed up to 6000 rupees per night.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "NUMERICAL_CONFLICT",
        "note": "Same scope (metro cities), different cap.",
    },
    {
        "pair_id": "ev_006",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "The notice period for resignation is one month.",
        "text_b": "Employees resigning must serve a notice period of 60 days.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "NUMERICAL_CONFLICT",
        "note": "Unit mismatch hides the conflict: one month is not 60 days.",
    },

    # ---------------- contradictions: dates ----------------
    {
        "pair_id": "ev_007",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "The leave year begins on 1 April and ends on 31 March.",
        "text_b": "The leave year begins on 1 January and ends on 31 December.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "DATE_CONFLICT",
        "note": "The same period is given two different start dates.",
    },
    {
        "pair_id": "ev_008",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "The vendor agreement remains valid until 31 December 2025.",
        "text_b": "The vendor agreement expires on 30 June 2026.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "DATE_CONFLICT",
        "note": "Two end dates for one agreement.",
    },
    {
        "pair_id": "ev_009",
        "document_a": HR_2023, "document_b": FINANCE,
        "text_a": "This policy takes effect on 01 April 2023.",
        "text_b": "This policy takes effect on 15 February 2024.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "DATE_CONFLICT",
        "note": "Borderline by design: different documents may each have "
                "their own effective date, but as written both claim to be "
                "'this policy'.",
    },

    # ---------------- contradictions: policy / permission ----------------
    {
        "pair_id": "ev_010",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Unused annual leave may be carried over to the following year, up to a maximum of 5 days.",
        "text_b": "Unused annual leave cannot be carried over and is forfeited at the end of the year.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "POLICY_CONFLICT",
        "note": "One permits what the other forbids.",
    },
    {
        "pair_id": "ev_011",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Alcoholic beverages are not reimbursable under this policy.",
        "text_b": "Alcoholic drinks may be claimed up to 500 rupees per evening meal.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "POLICY_CONFLICT",
        "note": "A flat prohibition against a stated allowance.",
    },
    {
        "pair_id": "ev_012",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Remote work does not require prior approval from a manager.",
        "text_b": "Remote work is permitted only with prior written approval from a manager.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "POLICY_CONFLICT",
        "note": "Directly opposed approval requirements.",
    },
    {
        "pair_id": "ev_013",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "International travel requires approval from the Finance Director.",
        "text_b": "International travel is approved by the employee's line manager.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "POLICY_CONFLICT",
        "note": "Different approving authority for the same action.",
    },

    # ---------------- contradictions: factual / logical ----------------
    {
        "pair_id": "ev_014",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Formal performance reviews are conducted twice per year, in June and December.",
        "text_b": "Performance reviews take place once per year, in December.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "FACTUAL_CONFLICT",
        "note": "Incompatible review frequency.",
    },
    {
        "pair_id": "ev_015",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "The office cafeteria operates between 08:00 and 18:00 on weekdays.",
        "text_b": "The office cafeteria is open from 08:00 to 20:00 on weekdays.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "FACTUAL_CONFLICT",
        "note": "Same facility, same days, different closing time.",
    },
    {
        "pair_id": "ev_016",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "All employees are eligible for the annual performance bonus.",
        "text_b": "No employee is eligible for the annual performance bonus.",
        "gold_relationship": CONTRADICTION,
        "gold_category": "LOGICAL_CONFLICT",
        "note": "Universal affirmative against universal negative.",
    },

    # ---------------- support: agreement and paraphrase ----------------
    {
        "pair_id": "ev_017",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Employees may take up to 10 days of paid sick leave per calendar year.",
        "text_b": "The company provides 10 days of paid sick leave annually to every full-time employee.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "Same entitlement, different wording.",
    },
    {
        "pair_id": "ev_018",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "A medical certificate is required for any sick leave absence longer than 2 consecutive days.",
        "text_b": "A doctor's note is required when an employee is absent due to illness for more than 2 consecutive days.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "'Medical certificate' and 'doctor's note' are the same requirement.",
    },
    {
        "pair_id": "ev_019",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Remote work is not available to employees serving their probation period.",
        "text_b": "Employees on probation are not eligible for remote work arrangements.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "Identical restriction, inverted phrasing.",
    },
    {
        "pair_id": "ev_020",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
        "text_b": "Full-time staff receive four weeks of paid annual leave each year.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "Equivalent quantities in different units: 20 working days "
                "is four weeks. Differing numbers must not be read as a "
                "conflict here.",
    },
    {
        "pair_id": "ev_021",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Expense claims must be submitted within 30 days of the date the expense was incurred.",
        "text_b": "Employees must file expense claims inside one month of incurring the cost.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "30 days and one month are the same window.",
    },
    {
        "pair_id": "ev_022",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Domestic flights must be booked in economy class for all employees below director level.",
        "text_b": "Non-director staff travel in economy class on domestic flights.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "Straight paraphrase.",
    },
    {
        "pair_id": "ev_023",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "The daily meal allowance for domestic travel is 1200 rupees per day.",
        "text_b": "Domestic travel carries a meal allowance of Rs 1,200 each day.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "Same value, different number formatting.",
    },
    {
        "pair_id": "ev_024",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "All expense claims above 5000 rupees require a receipt and prior written approval.",
        "text_b": "Any claim over Rs 5000 needs a receipt and written approval beforehand.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "Same threshold and same requirement.",
    },
    {
        "pair_id": "ev_025",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "International travel requires approval from the Finance Director at least 21 days in advance.",
        "text_b": "Book international trips at least three weeks ahead with Finance Director sign-off.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "21 days is three weeks; the authority matches too.",
    },
    {
        "pair_id": "ev_026",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Line managers are required to approve or reject a leave request within 3 working days.",
        "text_b": "Managers must respond to a submitted leave request within 3 working days.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "Same obligation and deadline.",
    },
    {
        "pair_id": "ev_027",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Sick leave does not accumulate and cannot be carried over into the next year.",
        "text_b": "Unused sick leave is forfeited at the end of each year and does not roll over.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "Agreement expressed with different negations.",
    },
    {
        "pair_id": "ev_028",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Alcoholic beverages are not reimbursable under this policy.",
        "text_b": "The company does not reimburse alcoholic drinks.",
        "gold_relationship": SUPPORT, "gold_category": None,
        "note": "The same prohibition stated twice.",
    },

    # ---------------- neutral: hard negatives ----------------
    {
        "pair_id": "ev_029",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Managers are entitled to 25 days of paid annual leave per year.",
        "text_b": "Interns are entitled to 15 days of paid annual leave per year.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "HARD NEGATIVE. Numbers differ but the populations differ, "
                "so both can hold at once.",
    },
    {
        "pair_id": "ev_030",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Employees are entitled to 20 days of paid annual leave per year.",
        "text_b": "Employees are entitled to 10 days of paid sick leave per year.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "HARD NEGATIVE. Near-identical wording, different kinds of "
                "leave.",
    },
    {
        "pair_id": "ev_031",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Hotel accommodation is reimbursed up to 4000 rupees per night in metropolitan cities.",
        "text_b": "Hotel accommodation is reimbursed up to 2500 rupees per night in non-metropolitan cities.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "HARD NEGATIVE. Different caps for explicitly different "
                "location tiers.",
    },
    {
        "pair_id": "ev_032",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Sick leave cannot be carried over into the next year.",
        "text_b": "Annual leave may be carried over up to a maximum of 5 days.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "HARD NEGATIVE. Opposite carry-over rules, but for two "
                "different leave types.",
    },
    {
        "pair_id": "ev_033",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Expense claims must be submitted within 30 days of the date the expense was incurred.",
        "text_b": "Claims delayed by certified medical leave may be accepted after 30 days with director approval.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "HARD NEGATIVE. The second is a stated exception to the "
                "first, not a contradiction of it.",
    },
    {
        "pair_id": "ev_034",
        "document_a": HR_2023, "document_b": FINANCE,
        "text_a": "The leave year begins on 1 April.",
        "text_b": "The financial year begins on 1 April.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "HARD NEGATIVE. Same date, different subjects; agreement "
                "about nothing in common.",
    },
    {
        "pair_id": "ev_035",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Visitor parking is located on the second level of the north garage.",
        "text_b": "Staff parking is located on the third level of the north garage.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "HARD NEGATIVE. Different populations, compatible facts.",
    },
    {
        "pair_id": "ev_036",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Employees working remotely may claim a one-time equipment allowance of 8000 rupees.",
        "text_b": "Remote work is permitted for a maximum of 3 days per week.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "Related topic, unrelated provisions.",
    },

    # ---------------- neutral: unrelated ----------------
    {
        "pair_id": "ev_037",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "The office cafeteria operates between 08:00 and 18:00 on weekdays.",
        "text_b": "Domestic flights must be booked in economy class for staff below director level.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "Entirely unrelated subjects.",
    },
    {
        "pair_id": "ev_038",
        "document_a": FINANCE, "document_b": HR_2023,
        "text_a": "Visitor parking is located on the second level of the north garage.",
        "text_b": "The daily meal allowance for domestic travel is 1200 rupees per day.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "Unrelated.",
    },
    {
        "pair_id": "ev_039",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Formal performance reviews are conducted in June and December.",
        "text_b": "Salary revisions take effect from 1 April each year.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "Different processes on different calendars.",
    },
    {
        "pair_id": "ev_040",
        "document_a": HR_2023, "document_b": HR_2024,
        "text_a": "Employees may take up to 10 days of paid sick leave per calendar year.",
        "text_b": "Employees must inform their manager before 10:00 on the first day of absence.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "Same topic, complementary rather than overlapping.",
    },
    {
        "pair_id": "ev_041",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "Claims submitted after 60 days will not be reimbursed under any circumstances.",
        "text_b": "Reimbursements are paid with the following month's salary.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "Submission deadline versus payment timing.",
    },
    {
        "pair_id": "ev_042",
        "document_a": FINANCE, "document_b": HR_2024,
        "text_a": "The office cafeteria operates between 08:00 and 18:00 on weekdays.",
        "text_b": "The staff gym is open from 06:00 to 22:00 every day.",
        "gold_relationship": NEUTRAL, "gold_category": None,
        "note": "HARD NEGATIVE of a milder kind: both state opening hours "
                "for different facilities.",
    },
]

LABELS = (CONTRADICTION, SUPPORT, NEUTRAL)


def label_distribution():
    counts = {label: 0 for label in LABELS}
    for item in PAIRS:
        counts[item["gold_relationship"]] += 1
    return counts


def hard_negatives():
    """Pairs explicitly written to trip a naive number comparison."""
    return [p for p in PAIRS if p["note"].startswith("HARD NEGATIVE")]


def validate():
    """Check the dataset's own integrity. Raises on a problem."""
    seen = set()
    for item in PAIRS:
        pair_id = item["pair_id"]
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen.add(pair_id)

        if item["gold_relationship"] not in LABELS:
            raise ValueError(f"{pair_id}: bad gold label")
        if item["gold_relationship"] == CONTRADICTION:
            if not item["gold_category"]:
                raise ValueError(f"{pair_id}: contradiction needs a category")
        elif item["gold_category"] is not None:
            raise ValueError(
                f"{pair_id}: only contradictions may carry a category"
            )
        for field in ("text_a", "text_b", "document_a", "document_b", "note"):
            if not str(item.get(field, "")).strip():
                raise ValueError(f"{pair_id}: empty {field}")
        if item["text_a"] == item["text_b"]:
            raise ValueError(f"{pair_id}: both statements are identical")
    return True

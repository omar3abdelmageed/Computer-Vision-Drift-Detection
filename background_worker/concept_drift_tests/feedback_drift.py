from __future__ import annotations

from collections import Counter
from typing import Sequence


def review_summary(feedback_types: Sequence[str]) -> dict:
    counts = Counter(feedback_types)
    total = len(feedback_types)
    approved = counts.get("approve", 0)
    return {
        "total_reviews": total,
        "accuracy_percent": (approved / total * 100.0) if total else None,
        "counts": {
            "good_response": approved,
            "false_positive": counts.get("false_positive", 0),
            "missed_object": counts.get("missed_object", 0),
            "wrong_class": counts.get("wrong_class", 0),
            "corrected_box": counts.get("corrected_box", 0),
            "corrected_label": counts.get("corrected_label", 0),
            "reject": counts.get("reject", 0),
        },
    }

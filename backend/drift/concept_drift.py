from __future__ import annotations

from collections import Counter
from typing import Sequence

from backend.common import DriftResult, DriftStatus, DriftType


def classification_concept_drift(y_true: Sequence[str], y_pred: Sequence[str], feedback_types: Sequence[str] = ()) -> list[DriftResult]:
    if not y_true and not feedback_types:
        return []
    results: list[DriftResult] = []
    if y_true:
        total = len(y_true)
        correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
        accuracy = correct / total if total else 0.0
        results.append(DriftResult(DriftType.CONCEPT, "rolling_accuracy", accuracy, 0.8, DriftStatus.CRITICAL if accuracy < 0.65 else DriftStatus.WARNING if accuracy < 0.8 else DriftStatus.OK))
    if feedback_types:
        counts = Counter(feedback_types)
        rejected = counts.get("reject", 0) + counts.get("corrected_label", 0)
        rate = rejected / len(feedback_types)
        results.append(DriftResult(DriftType.CONCEPT, "human_rejection_rate", rate, 0.2, DriftStatus.CRITICAL if rate >= 0.4 else DriftStatus.WARNING if rate >= 0.2 else DriftStatus.OK, {"counts": dict(counts)}))
    return results


def detection_feedback_concept_drift(feedback_types: Sequence[str]) -> list[DriftResult]:
    if not feedback_types:
        return []
    counts = Counter(feedback_types)
    total = len(feedback_types)
    metrics = {
        "human_rejection_rate": counts.get("reject", 0) / total,
        "false_positive_flag_rate": counts.get("false_positive", 0) / total,
        "missed_object_flag_rate": counts.get("missed_object", 0) / total,
        "wrong_class_flag_rate": counts.get("wrong_class", 0) / total,
    }
    return [
        DriftResult(DriftType.CONCEPT, name, value, 0.2, DriftStatus.CRITICAL if value >= 0.4 else DriftStatus.WARNING if value >= 0.2 else DriftStatus.OK, {"counts": dict(counts)})
        for name, value in metrics.items()
    ]

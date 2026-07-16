from __future__ import annotations

from collections import Counter
from typing import Iterable


def class_distribution(labels: Iterable[str]) -> dict[str, int]:
    return dict(Counter(str(label) for label in labels if label is not None and str(label) != ""))


def class_percentages(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {label: 0.0 for label in counts}
    return {label: count / total * 100.0 for label, count in counts.items()}


def positive_int_counts(counts: dict[str, int]) -> dict[str, int]:
    clean = {}
    for label, count in counts.items():
        if not isinstance(count, (int, float)):
            continue
        value = int(count)
        if value > 0:
            clean[str(label)] = value
    return clean


def chi_square_prediction_drift(training_counts: dict[str, int], production_counts: dict[str, int]) -> dict:
    clean_training = positive_int_counts(training_counts)
    clean_production = positive_int_counts(production_counts)
    training_total = sum(clean_training.values())
    production_total = sum(clean_production.values())
    unseen_classes = sorted(label for label in clean_production if label not in clean_training)
    if training_total <= 0:
        return {"statistic": None, "p_value": None, "expected_counts": {}, "observed_counts": clean_production, "unseen_classes": unseen_classes, "reason": "Training class distribution is empty. Rebuild the baseline after validating train labels/classes."}
    if production_total <= 0:
        return {"statistic": None, "p_value": None, "expected_counts": {}, "observed_counts": {}, "unseen_classes": [], "reason": "No production predictions are available for the current window."}

    labels = sorted(clean_training)
    expected_counts = {label: clean_training[label] / training_total * production_total for label in labels}
    observed_counts = {label: clean_production.get(label, 0) for label in labels}
    if unseen_classes:
        statistic = sum(((observed_counts[label] - expected_counts[label]) ** 2) / expected_counts[label] for label in labels if expected_counts[label] > 0)
        return {"statistic": float(statistic), "p_value": None, "expected_counts": expected_counts, "observed_counts": observed_counts, "unseen_classes": unseen_classes, "reason": "Production predictions include classes that are absent from the training distribution."}

    from scipy.stats import chisquare

    result = chisquare(f_obs=[observed_counts[label] for label in labels], f_exp=[expected_counts[label] for label in labels])
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "expected_counts": expected_counts, "observed_counts": observed_counts, "unseen_classes": [], "reason": None}

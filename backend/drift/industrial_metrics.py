from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Iterable, Sequence


PROPERTY_KEYS = ("brightness", "contrast", "colour_distribution", "noise")

PROPERTY_LABELS = {
    "brightness": "Brightness",
    "contrast": "Contrast",
    "colour_distribution": "Colour distribution",
    "noise": "Noise",
}

PROPERTY_HELP = {
    "total_data_drift": "Total data drift uses RBF-kernel Maximum Mean Discrepancy (MMD) comparing fixed training-set feature vectors with the current production window.",
    "brightness": "Brightness is the mean grayscale pixel intensity. The reference line is the fixed training-set average.",
    "contrast": "Contrast is the standard deviation of grayscale pixel intensity. The reference line is the fixed training-set average.",
    "colour_distribution": "Colour distribution uses an RGB colorfulness score based on red-green and yellow-blue channel differences. The reference line is the fixed training-set average.",
    "noise": "Noise is the mean absolute difference between the grayscale image and a lightly blurred grayscale image. The reference line is the fixed training-set average.",
    "prediction_class_distribution": "Prediction drift compares fixed training-set class percentages with current production predicted-class counts using a Chi-Square goodness-of-fit test.",
    "human_review_summary": "Concept drift uses human review outcomes. Accuracy is approved predictions divided by total reviewed predictions.",
}


def colorfulness_from_rgb_channel_stats(rgb_means: Sequence[float], rgb_stds: Sequence[float] = ()) -> float:
    if len(rgb_means) < 3:
        return 0.0
    red, green, blue = [float(value) for value in rgb_means[:3]]
    rg_mean = red - green
    yb_mean = 0.5 * (red + green) - blue
    rg_std = float(rgb_stds[0]) if len(rgb_stds) > 0 else 0.0
    yb_std = 0.5 * (float(rgb_stds[0]) + float(rgb_stds[1])) if len(rgb_stds) > 1 else 0.0
    return sqrt(rg_std**2 + yb_std**2) + 0.3 * sqrt(rg_mean**2 + yb_mean**2)


def feature_vector(row: dict) -> list[float]:
    return [
        float(row.get("brightness_mean") or 0.0),
        float(row.get("contrast") or 0.0),
        property_value("colour_distribution", row),
        property_value("noise", row),
    ]


def property_value(property_key: str, row: dict) -> float:
    if property_key == "brightness":
        return float(row.get("brightness_mean") or 0.0)
    if property_key == "contrast":
        return float(row.get("contrast") or 0.0)
    if property_key == "colour_distribution":
        if isinstance(row.get("colorfulness"), (int, float)):
            return float(row["colorfulness"])
        return colorfulness_from_rgb_channel_stats(row.get("rgb_channel_means") or [], row.get("rgb_channel_stds") or [])
    if property_key == "noise":
        if isinstance(row.get("noise_level"), (int, float)):
            return float(row["noise_level"])
        return float(row.get("brightness_std") or 0.0)
    return 0.0


def mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def property_averages(feature_rows: Sequence[dict]) -> dict[str, float | None]:
    return {key: mean([property_value(key, row) for row in feature_rows]) for key in PROPERTY_KEYS}


def feature_vectors(feature_rows: Sequence[dict]) -> list[list[float]]:
    return [feature_vector(row) for row in feature_rows]


def rbf_mmd(reference_vectors: Sequence[Sequence[float]], current_vectors: Sequence[Sequence[float]]) -> float:
    if not reference_vectors or not current_vectors:
        return 0.0
    import numpy as np

    ref = np.asarray(reference_vectors, dtype=float)
    cur = np.asarray(current_vectors, dtype=float)
    combined = np.vstack([ref, cur])
    variance = float(np.var(combined))
    gamma = 1.0 / (combined.shape[1] * variance + 1e-8)
    k_xx = rbf_kernel_mean(ref, ref, gamma)
    k_yy = rbf_kernel_mean(cur, cur, gamma)
    k_xy = rbf_kernel_mean(ref, cur, gamma)
    return float(max(0.0, k_xx + k_yy - 2 * k_xy))


def rbf_kernel_mean(left, right, gamma: float) -> float:
    import numpy as np

    diff = left[:, None, :] - right[None, :, :]
    squared_distances = np.sum(diff * diff, axis=2)
    return float(np.exp(-gamma * squared_distances).mean())


def class_distribution(labels: Iterable[str]) -> dict[str, int]:
    return dict(Counter(str(label) for label in labels if label is not None and str(label) != ""))


def class_percentages(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {label: 0.0 for label in counts}
    return {label: count / total * 100.0 for label, count in counts.items()}


def total_variation_distance(reference_counts: dict[str, int], current_counts: dict[str, int]) -> float:
    reference = class_percentages(reference_counts)
    current = class_percentages(current_counts)
    labels = set(reference) | set(current)
    return 0.5 * sum(abs(reference.get(label, 0.0) - current.get(label, 0.0)) for label in labels)


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
        return {
            "statistic": None,
            "p_value": None,
            "expected_counts": {},
            "observed_counts": clean_production,
            "unseen_classes": unseen_classes,
            "reason": "Training class distribution is empty. Rebuild the baseline after validating train labels/classes.",
        }
    if production_total <= 0:
        return {
            "statistic": None,
            "p_value": None,
            "expected_counts": {},
            "observed_counts": {},
            "unseen_classes": [],
            "reason": "No production predictions are available for the current window.",
        }

    labels = sorted(clean_training)
    expected_counts = {
        label: clean_training[label] / training_total * production_total
        for label in labels
    }
    observed_counts = {label: clean_production.get(label, 0) for label in labels}
    if unseen_classes:
        statistic = sum(
            ((observed_counts[label] - expected_counts[label]) ** 2) / expected_counts[label]
            for label in labels
            if expected_counts[label] > 0
        )
        return {
            "statistic": float(statistic),
            "p_value": None,
            "expected_counts": expected_counts,
            "observed_counts": observed_counts,
            "unseen_classes": unseen_classes,
            "reason": "Production predictions include classes that are absent from the training distribution.",
        }

    from scipy.stats import chisquare

    observed = [observed_counts[label] for label in labels]
    expected = [expected_counts[label] for label in labels]
    result = chisquare(f_obs=observed, f_exp=expected)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "unseen_classes": [],
        "reason": None,
    }


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

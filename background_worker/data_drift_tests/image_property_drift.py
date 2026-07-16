from __future__ import annotations

from math import sqrt
from typing import Sequence


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

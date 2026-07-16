from __future__ import annotations

def confidence_gauge_color(value: float, medium_threshold: float, good_threshold: float) -> str:
    if value < medium_threshold:
        return "#ef4444"
    if value < good_threshold:
        return "#f59e0b"
    return "#22c55e"


def confidence_gauge_status(value: float, medium_threshold: float, good_threshold: float) -> str:
    if value < medium_threshold:
        return "Low confidence"
    if value < good_threshold:
        return "Medium confidence"
    return "Good confidence"

from __future__ import annotations

from backend.common import FeedbackType


def build_feedback_payload(feedback_type: FeedbackType | str, corrected_payload: dict | None = None, comment: str | None = None) -> dict:
    return {
        "feedback_type": FeedbackType(feedback_type).value,
        "corrected_payload": corrected_payload or {},
        "comment": comment,
    }

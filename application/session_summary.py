from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def build_session_summary(images, predictions, feedback, session: dict, source: dict | None, ended_at: str) -> dict[str, Any]:
    image_rows = images.where(session_id=session["id"])
    complete_image_rows = [row for row in image_rows if (row.get("inference_status") or "complete") == "complete"]
    prediction_rows = predictions_for_images(predictions, complete_image_rows)
    latest_feedback_rows = latest_feedback_for_predictions(feedback, prediction_rows)
    confidences = [row.get("confidence") for row in prediction_rows if row.get("confidence") is not None]
    started_at = session.get("started_at")
    return {
        "source_uri": source.get("source_uri") if source else None,
        "processed_image_count": len(image_rows),
        "completed_image_count": len(complete_image_rows),
        "failed_image_count": sum(1 for row in image_rows if row.get("inference_status") == "failed"),
        "pending_image_count": sum(1 for row in image_rows if row.get("inference_status") == "pending"),
        "prediction_count": len(prediction_rows),
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "reviewed_prediction_count": len(latest_feedback_rows),
        "approved_prediction_count": sum(1 for row in latest_feedback_rows if row.get("feedback_type") == "approve"),
        "rejected_prediction_count": sum(1 for row in latest_feedback_rows if row.get("feedback_type") == "reject"),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds(started_at, ended_at),
    }


def predictions_for_images(predictions, image_rows: list[dict]) -> list[dict]:
    rows = []
    for image in image_rows:
        rows.extend(predictions.where(image_id=image["id"]))
    return rows


def prediction_rows_for_images(predictions, image_rows: list[dict], image_ids: list[str] | None = None) -> list[dict]:
    ids = image_ids if image_ids is not None else [image["id"] for image in image_rows if image.get("id")]
    if not ids:
        return []
    if hasattr(predictions, "where_in"):
        return predictions.where_in("image_id", ids)
    return predictions_for_images(predictions, image_rows) if hasattr(predictions, "where") else []


def latest_feedback_for_predictions(feedback, prediction_rows: list[dict]) -> list[dict]:
    rows = []
    for prediction in prediction_rows:
        prediction_feedback = feedback.where_ordered(prediction_id=prediction["id"], limit=1)
        if prediction_feedback:
            rows.append(prediction_feedback[0])
    return rows


def duration_seconds(started_at: str | None, ended_at: str | None) -> int | None:
    start = parse_datetime(started_at)
    end = parse_datetime(ended_at)
    if not start or not end:
        return None
    return max(0, int((end - start).total_seconds()))


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def predictions_by_image_id(predictions, image_rows: list[dict]) -> dict[str, list[dict]]:
    image_ids = [image["id"] for image in image_rows if image.get("id")]
    if not image_ids:
        return {}
    rows = prediction_rows_for_images(predictions, image_rows, image_ids)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        image_id = row.get("image_id")
        if image_id:
            grouped[image_id].append(row)
    for image_id, grouped_rows in grouped.items():
        grouped[image_id] = sorted(grouped_rows, key=lambda row: (str(row.get("created_at") or ""), str(row.get("id") or "")))
    return grouped

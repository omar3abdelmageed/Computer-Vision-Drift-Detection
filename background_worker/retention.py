from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from database.repositories import Repository


DEFAULT_COMPLETED_SESSION_RETENTION_DAYS = 30


def cleanup_completed_session_data(client, retention_days: int = DEFAULT_COMPLETED_SESSION_RETENTION_DAYS) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(retention_days)))
    sessions = Repository(client, "monitoring_sessions")
    images = Repository(client, "images")
    predictions = Repository(client, "predictions")
    feedback = Repository(client, "feedback")
    drift = Repository(client, "drift_results")
    alerts = Repository(client, "alerts")

    cleaned_sessions = 0
    deleted_images = 0
    deleted_predictions = 0
    deleted_feedback = 0
    deleted_drift = 0
    deleted_alerts = 0

    for session in sessions.where(status="completed"):
        if session.get("archived_at"):
            continue
        ended_at = parse_datetime(session.get("ended_at"))
        if ended_at is None or ended_at > cutoff:
            continue

        image_rows = images.where(session_id=session["id"])
        for image in image_rows:
            for row in feedback.delete_where(image_id=image["id"]):
                deleted_feedback += 1
            for row in predictions.delete_where(image_id=image["id"]):
                deleted_predictions += 1
            if images.delete(image["id"]):
                deleted_images += 1

        for row in drift.delete_where(session_id=session["id"]):
            deleted_drift += 1
        for alert in alerts.where(session_id=session["id"]):
            if alert.get("is_resolved"):
                if alerts.delete(alert["id"]):
                    deleted_alerts += 1

        sessions.update(session["id"], {"archived_at": datetime.now(timezone.utc).isoformat()})
        cleaned_sessions += 1

    return {
        "cleaned_sessions": cleaned_sessions,
        "deleted_images": deleted_images,
        "deleted_predictions": deleted_predictions,
        "deleted_feedback": deleted_feedback,
        "deleted_drift": deleted_drift,
        "deleted_alerts": deleted_alerts,
    }


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

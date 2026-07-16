from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from database.repositories import Repository
from database.local import create_local_database
from database.tables import BASELINE_PROFILES, MODEL_ARTIFACTS, MODELS, MONITORING_SESSIONS, PRODUCTION_SOURCES

from background_worker.session_processor import process_source_once
from background_worker.retention import cleanup_completed_session_data
from background_worker.supervisor import worker_owner_id as local_worker_owner_id
from background_worker.supervisor import write_heartbeat


LOGGER = logging.getLogger(__name__)


LEASE_SECONDS = 90
REQUIRED_BASELINE_PROFILE_TYPES = {"feature_rows", "class_distribution"}


def running_sessions(sessions: Repository, owner_id: str | None = None) -> list[dict[str, Any]]:
    rows = sessions.where(status="running_live_monitoring")
    if owner_id is None:
        return rows
    return sorted(
        [row for row in rows if worker_can_process_session(row, owner_id)],
        key=lambda row: (row.get("worker_owner_id") != owner_id, str(row.get("created_at") or "")),
    )


def worker_can_process_session(session: dict[str, Any], owner_id: str) -> bool:
    session_owner = session.get("worker_owner_id")
    return not session_owner or session_owner == owner_id or lease_expired(session.get("worker_lease_expires_at"))


def lease_expired(value: str | None) -> bool:
    if not value:
        return True
    parsed = parse_datetime(value)
    return parsed is None or parsed <= datetime.now(timezone.utc)


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


def process_running_sessions(client, worker_owner_id: str | None = None, lease_seconds: int = LEASE_SECONDS, run_cleanup: bool = False) -> dict[str, Any]:
    owner_id = worker_owner_id or local_worker_owner_id()
    models = Repository(client, MODELS)
    artifacts = Repository(client, MODEL_ARTIFACTS)
    sources = Repository(client, PRODUCTION_SOURCES)
    sessions = Repository(client, MONITORING_SESSIONS)
    baselines = Repository(client, BASELINE_PROFILES)

    processed_sessions = 0
    processed_images = 0
    failed_sessions = 0
    details: list[dict[str, Any]] = []

    for session in running_sessions(sessions, owner_id):
        processed_sessions += 1
        session_detail: dict[str, Any] = {
            "session_id": session.get("id"),
            "processed": 0,
            "status": "started",
        }
        try:
            leased_session = acquire_session_lease(sessions, session, owner_id, lease_seconds)
            if not leased_session:
                session_detail.update({"status": "skipped", "reason": "lease_not_acquired"})
                details.append(session_detail)
                continue
            session = leased_session
            model = models.get(session["model_id"]) if session.get("model_id") else None
            artifact = artifacts.get(session["artifact_id"]) if session.get("artifact_id") else None
            source = sources.get(session["source_id"]) if session.get("source_id") else None
            if not model or not artifact or not source:
                reason = "Missing model, artifact, or source for running session."
                sessions.update(session["id"], {"worker_status": "failed", "last_worker_error": reason})
                session_detail.update({"status": "failed", "reason": reason})
                details.append(session_detail)
                failed_sessions += 1
                continue

            baseline_filters = {"model_id": model["id"], "artifact_id": artifact.get("id")}
            dataset_id = session.get("dataset_id")
            if dataset_id:
                baseline_filters["dataset_id"] = dataset_id
            baseline_profiles = baselines.where_ordered(**baseline_filters)
            missing_profile_types = missing_baseline_profile_types(baseline_profiles)
            if missing_profile_types:
                reason = f"Missing baseline profiles for running session: {', '.join(sorted(missing_profile_types))}."
                sessions.update(session["id"], {"worker_status": "failed", "last_worker_error": reason})
                session_detail.update({"status": "failed", "reason": reason})
                details.append(session_detail)
                failed_sessions += 1
                continue

            previous_worker_status = session.get("worker_status")
            refresh_session_lease(sessions, session["id"], owner_id, lease_seconds)
            summary = process_source_once(
                client,
                model,
                artifact,
                session_source_snapshot(source, session),
                session,
                baseline_profiles,
                owner_id=owner_id,
                lease_seconds=lease_seconds,
            )
            processed = int(summary.get("processed") or 0)
            processed_images += processed
            session_detail.update(processing_detail_counts(summary))
            reason = summary.get("reason")
            if reason == "session_not_running":
                session_detail.update({"processed": processed, "status": "stopped"})
                details.append(session_detail)
                continue
            if reason:
                error_message = summary.get("error") or str(reason)
                sessions.update(
                    session["id"],
                    {
                        "worker_status": "failed",
                        "last_worker_error": error_message,
                    },
                )
                session_detail.update({"processed": processed, "status": "failed", "reason": reason, "error": error_message})
                details.append(session_detail)
                failed_sessions += 1
                continue
            latest_session = sessions.get(session["id"]) or {}
            if latest_session.get("status") != "running_live_monitoring":
                session_detail.update({"processed": processed, "status": "stopped"})
                details.append(session_detail)
                continue
            update_payload = {
                "worker_status": "caught_up",
                "last_worker_error": None,
                "worker_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "worker_lease_expires_at": lease_expiry(lease_seconds).isoformat(),
            }
            if processed > 0:
                update_payload["last_processed_at"] = datetime.now(timezone.utc).isoformat()
            if summary.get("last_processed_image_id"):
                update_payload["last_processed_image_id"] = summary["last_processed_image_id"]
            if processed > 0 or previous_worker_status != "caught_up":
                sessions.update(session["id"], update_payload)
            session_detail.update({"processed": processed, "status": "caught_up"})
            details.append(session_detail)
        except Exception as exc:
            LOGGER.exception("Worker failed for session %s", session.get("id"))
            sessions.update(session["id"], {"worker_status": "failed", "last_worker_error": str(exc)})
            session_detail.update({"status": "failed", "reason": str(exc)})
            details.append(session_detail)
            failed_sessions += 1

    cleanup_summary = cleanup_completed_session_data(client) if run_cleanup else {}
    return {
        "sessions": processed_sessions,
        "processed_images": processed_images,
        "failed_sessions": failed_sessions,
        "cleanup": cleanup_summary,
        "details": details,
    }


def processing_detail_counts(summary: dict[str, Any]) -> dict[str, int]:
    keys = ("ready_files", "source_files", "skipped_unready", "skipped_by_stride", "skipped_duplicates", "insert_errors", "inference_errors")
    return {key: int(summary.get(key) or 0) for key in keys if key in summary}


def acquire_session_lease(sessions: Repository, session: dict[str, Any], owner_id: str, lease_seconds: int) -> dict[str, Any] | None:
    current = sessions.get(session["id"]) or session
    if current.get("status") != "running_live_monitoring" or not worker_can_process_session(current, owner_id):
        return None
    now = datetime.now(timezone.utc)
    payload = {
        "worker_status": "running",
        "worker_owner_id": owner_id,
        "worker_pid": os.getpid(),
        "worker_heartbeat_at": now.isoformat(),
        "worker_lease_expires_at": lease_expiry(lease_seconds).isoformat(),
        "processing_started_at": current.get("processing_started_at") or now.isoformat(),
        "processing_attempt_count": int(current.get("processing_attempt_count") or 0) + 1,
        "last_worker_error": None,
    }
    return sessions.update(current["id"], payload)


def refresh_session_lease(sessions: Repository, session_id: str, owner_id: str, lease_seconds: int) -> dict[str, Any] | None:
    return sessions.update(
        session_id,
        {
            "worker_owner_id": owner_id,
            "worker_pid": os.getpid(),
            "worker_heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "worker_lease_expires_at": lease_expiry(lease_seconds).isoformat(),
        },
    )


def lease_expiry(lease_seconds: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=max(10, int(lease_seconds)))


def missing_baseline_profile_types(baseline_profiles: list[dict[str, Any]]) -> set[str]:
    profile_types = {row.get("profile_type") for row in baseline_profiles}
    return REQUIRED_BASELINE_PROFILE_TYPES - profile_types


def session_source_snapshot(source: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(source)
    if session.get("source_uri"):
        snapshot["source_uri"] = session["source_uri"]
    if session.get("source_type"):
        snapshot["source_type"] = session["source_type"]
    if session.get("polling_interval_seconds"):
        snapshot["polling_interval_seconds"] = session["polling_interval_seconds"]
    return snapshot


def run_forever(client, interval_seconds: int, owner_id: str | None = None) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    worker_id = owner_id or local_worker_owner_id()
    while True:
        write_heartbeat(os.getpid(), max(1, interval_seconds), started_at, worker_id)
        summary = process_running_sessions(client, worker_owner_id=worker_id, run_cleanup=True)
        LOGGER.info("worker cycle complete: %s", summary)
        time.sleep(interval_seconds)


def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    parser = argparse.ArgumentParser(description="Process running live monitoring sessions.")
    parser.add_argument("--once", action="store_true", help="Process running sessions once and exit.")
    parser.add_argument("--interval-seconds", type=int, default=30, help="Worker polling interval.")
    parser.add_argument("--worker-owner-id", default=None, help="Stable local worker owner id.")
    args = parser.parse_args()

    client = create_local_database()
    if args.once:
        print(process_running_sessions(client, worker_owner_id=args.worker_owner_id or local_worker_owner_id(), run_cleanup=True))
        return
    interval_seconds = max(1, args.interval_seconds)
    started_at = datetime.now(timezone.utc).isoformat()
    worker_id = args.worker_owner_id or local_worker_owner_id()
    while True:
        write_heartbeat(os.getpid(), interval_seconds, started_at, worker_id)
        summary = process_running_sessions(client, worker_owner_id=worker_id, run_cleanup=True)
        LOGGER.info("worker cycle complete: %s", summary)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()

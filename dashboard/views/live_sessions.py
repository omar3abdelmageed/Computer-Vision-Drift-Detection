from __future__ import annotations

from datetime import timedelta
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import altair as alt

from background_worker.session_processor import (
    DEFAULT_THRESHOLD_CONFIG,
    GRAPH_WINDOW_DEFAULT,
    GRAPH_WINDOW_MAX,
    GRAPH_WINDOW_MIN,
    normalized_threshold_config,
    threshold_config_from_slider_values,
)
from background_worker.session_processor import record_feedback_concept_drift, source_image_stride
from background_worker.supervisor import ensure_worker_running, heartbeat_is_stale, read_heartbeat, stop_worker, worker_owner_id
from dashboard.components.image_viewer import draw_detection_overlays
from dashboard.components.status_cards import status_card
from dashboard.presenters import confidence_gauge_color, confidence_gauge_status
from database.repositories import Repository
from application.session_summary import (
    build_session_summary,
    duration_seconds,
    latest_feedback_for_predictions,
    parse_datetime,
    prediction_rows_for_images,
    predictions_by_image_id,
    predictions_for_images,
)
from application.detection_monitoring import (
    DETECTION_CLUSTER_IOU_THRESHOLD,
    DetectionCluster,
    cluster_detection_predictions,
    cluster_member_label,
    summarize_detection_overlaps,
)


CONCEPT_DRIFT_ACCURACY_WARNING_PERCENT = 80.0
DRIFT_WINDOW_SIZE = 25
TOTAL_DATA_DRIFT_WARNING_THRESHOLD = 0.1
TOTAL_DATA_DRIFT_CRITICAL_THRESHOLD = 0.25
PROPERTY_DRIFT_WARNING_THRESHOLD = 20.0
PROPERTY_DRIFT_CRITICAL_THRESHOLD = 50.0
ACTIVE_SESSION_STATUSES = {"running_live_monitoring", "paused"}
IDLE_WORKER_RETRY_SECONDS = 10
LOW_CONFIDENCE_THRESHOLD = 0.5


def render_live_sessions_tab(client) -> None:
    st.header("Live Sessions")
    models = Repository(client, "models")
    datasets = Repository(client, "datasets")
    artifacts = Repository(client, "model_artifacts")
    sources = Repository(client, "production_sources")
    sessions = Repository(client, "monitoring_sessions")
    images = Repository(client, "images")
    predictions = Repository(client, "predictions")
    drift = Repository(client, "drift_results")
    alerts = Repository(client, "alerts")
    feedback = Repository(client, "feedback")
    baselines = Repository(client, "baseline_profiles")

    selected_model = select_registered_model(models, datasets, artifacts, baselines)
    if not selected_model:
        return

    model_id = selected_model["id"]
    source = latest_source(sources, model_id)
    latest_session = latest_monitoring_session(sessions, model_id)
    active_session = active_monitoring_session(sessions)
    selected_active_session = active_session if active_session and active_session.get("model_id") == model_id else None
    session_for_controls = selected_active_session or latest_session

    source_tab, session_tab, predictions_tab, drift_tab = st.tabs(["Source", "Session", "Predictions", "Drift"])
    with source_tab:
        protected_source_id = active_session.get("source_id") if active_session else None
        source = render_source_tab(sources, model_id, source, protected_source_id=protected_source_id)
    with session_tab:
        active_source = source_for_session(sources, source, session_for_controls)
        session_for_controls = render_session_tab(
            client,
            sessions,
            images,
            predictions,
            feedback,
            selected_model,
            active_source,
            session_for_controls,
            active_session=active_session,
        )
    active_source = source_for_session(sources, source, session_for_controls)
    maybe_auto_poll(client, selected_model, active_source, session_for_controls)
    with predictions_tab:
        render_predictions_tab(client, selected_model, images, predictions, feedback, session_for_controls)
    with drift_tab:
        render_drift_tab(drift, alerts, session_for_controls)


def select_registered_model(models, datasets, artifacts, baselines) -> dict | None:
    st.subheader("Registered Model")
    rows = models.list()
    if not rows:
        st.info("No registered models yet.")
        return None

    options = {f"{row.get('name')} ({row.get('baseline_status')})": row for row in rows}
    selected = dict(options[st.selectbox("Registered model", list(options.keys()))])
    dataset = datasets.latest_where(model_id=selected["id"]) or {}
    artifact = artifacts.latest_where(model_id=selected["id"]) or {}
    baseline_profiles = current_baseline_profiles(baselines, selected["id"], dataset, artifact)
    has_current_baseline = len(baseline_profiles) > 0
    eligible = (
        dataset.get("validation_status") == "valid"
        and artifact.get("compatibility_status") != "invalid"
        and selected.get("baseline_status") == "completed"
        and has_current_baseline
    )

    selected["_latest_dataset"] = dataset
    selected["_latest_artifact"] = artifact
    selected["_baseline_profiles"] = baseline_profiles
    selected["_live_eligible"] = eligible

    status_card("Eligibility", "valid" if eligible else "warning", "Ready" if eligible else "Complete validation, compatibility, and baseline first.")
    baseline_status = "current" if has_current_baseline else "missing for current dataset/artifact"
    st.caption(
        f"Dataset: {dataset.get('validation_status', 'missing')} | "
        f"Artifact: {artifact.get('compatibility_status', 'missing')} | "
        f"Baseline: {selected.get('baseline_status')} ({baseline_status})"
    )
    return selected


def current_baseline_profiles(baselines, model_id: str, dataset: dict, artifact: dict) -> list[dict]:
    dataset_id = dataset.get("id")
    artifact_id = artifact.get("id")
    if not dataset_id or not artifact_id:
        return []
    return baselines.where_ordered(model_id=model_id, dataset_id=dataset_id, artifact_id=artifact_id)


def latest_source(sources, model_id: str) -> dict | None:
    return sources.latest_where(model_id=model_id)


def latest_monitoring_session(sessions, model_id: str) -> dict | None:
    return sessions.latest_where(model_id=model_id)


def active_monitoring_session(sessions, model_id: str | None = None) -> dict | None:
    active_rows = []
    for status in ACTIVE_SESSION_STATUSES:
        filters = {"status": status}
        if model_id is not None:
            filters["model_id"] = model_id
        active_rows.extend(sessions.where_ordered(**filters))
    if not active_rows:
        return None
    return sorted(active_rows, key=lambda row: (str(row.get("created_at") or ""), str(row.get("id") or "")), reverse=True)[0]


def source_for_session(sources, fallback_source: dict | None, session: dict | None) -> dict | None:
    if not session or session.get("status") not in ACTIVE_SESSION_STATUSES:
        return fallback_source
    source_id = session.get("source_id")
    return sources.get(source_id) if source_id else fallback_source


def render_source_tab(sources, model_id: str, source: dict | None, protected_source_id: str | None = None) -> dict | None:
    st.subheader("Image Source")
    source_locked = protected_source_id is not None
    if source:
        source = {**source, "source_uri": normalize_source_uri(source.get("source_uri"))}
    with st.form("source_config"):
        source_uri = st.text_input("Path to pull images from", value=source.get("source_uri", "") if source else "")
        interval = st.number_input(
            "Polling interval seconds",
            min_value=1,
            value=int(source.get("polling_interval_seconds") or 30) if source else 30,
        )
        image_stride = st.number_input(
            "Use every Nth image",
            min_value=1,
            value=source_image_stride(source),
            help="1 uses every image. 4 uses only every fourth ready image.",
        )
        submitted = st.form_submit_button("Save source", icon=":material/save:", disabled=source_locked)
    if source_locked:
        st.caption("Stop the running session before configuring a new source.")

    if submitted:
        payload = {
            "model_id": model_id,
            "source_type": "watched_folder",
            "source_uri": normalize_source_uri(source_uri),
            "mode": "live_monitoring",
            "polling_interval_seconds": int(interval),
            "is_active": True,
            "config": {
                **(source.get("config") if source and isinstance(source.get("config"), dict) else {}),
                "image_stride": int(image_stride),
            },
        }
        if source and source.get("id") != protected_source_id:
            source = sources.update(source["id"], payload) or source
        else:
            source = sources.insert(payload)
        st.success("Source saved.")

    if source:
        source_path = Path(source.get("source_uri") or "")
        stride = source_image_stride(source)
        sampling_detail = "all images" if stride == 1 else f"every {stride}th image"
        detail = f"{source.get('source_uri') or 'No path'} | every {source.get('polling_interval_seconds') or 30}s | {sampling_detail}"
        status_card("Source", "valid" if source_path.exists() else "warning", detail)
    else:
        st.info("Save a source path before starting a session.")
    return source


def render_session_tab(
    client,
    sessions,
    images,
    predictions,
    feedback,
    model: dict,
    source: dict | None,
    latest_session: dict | None,
    active_session: dict | None = None,
) -> dict | None:
    st.subheader("Session Controls")
    model_id = model["id"]
    artifact = model.get("_latest_artifact") or {}
    dataset = model.get("_latest_dataset") or {}
    default_name = f"{model.get('name') or 'Session'} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    session_name = st.text_input("Session name", value=latest_session.get("name") or default_name if latest_session else default_name)
    threshold_config = render_threshold_controls(sessions, latest_session, model_id)
    if latest_session and latest_session.get("status") == "paused":
        latest_session = {**latest_session, "threshold_config": threshold_config}

    cols = st.columns(4)
    start_blockers = session_start_blockers(source, model, active_session)
    start_disabled = bool(start_blockers)
    if cols[0].button("Start", icon=":material/play_arrow:", disabled=start_disabled, width="stretch"):
        stop_result = stop_worker(client=client, reason="session_start_reset")
        if not stop_result.ok:
            st.error(f"Background worker could not be reset before start: {stop_result.error}")
            return latest_session
        latest_session = sessions.insert(
            {
                "model_id": model_id,
                "dataset_id": dataset.get("id"),
                "artifact_id": artifact.get("id"),
                "source_id": source.get("id") if source else None,
                "source_type": source.get("source_type") if source else None,
                "source_uri": source.get("source_uri") if source else None,
                "polling_interval_seconds": int(source.get("polling_interval_seconds") or 30) if source else 30,
                "name": session_name,
                "status": "running_live_monitoring",
                "worker_status": "idle",
                "worker_owner_id": worker_owner_id(),
                "last_worker_error": None,
                "threshold_config": threshold_config,
                "summary": {},
            }
        )
        if latest_session:
            st.success("Session started.")
            worker_result = ensure_worker_running(int(source.get("polling_interval_seconds") or 30) if source else 30, client=client)
            if worker_result.ok:
                st.caption("Background worker is running.")
                st.rerun()
            else:
                error = worker_result.error or "unknown error"
                sessions.update(latest_session["id"], {"worker_status": "failed", "last_worker_error": error})
                st.error(f"Session started, but the background worker could not be started: {error}")
        else:
            st.error("Session could not be started.")

    if latest_session:
        if cols[1].button("Pause", icon=":material/pause:", disabled=latest_session.get("status") != "running_live_monitoring", width="stretch"):
            updated = sessions.update(
                latest_session["id"],
                {
                    "status": "paused",
                    "worker_status": "idle",
                    "worker_pid": None,
                    "worker_heartbeat_at": None,
                    "worker_lease_expires_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if updated:
                stop_worker(client=client, reason="session_paused")
                st.success("Session paused.")
                st.rerun()
            else:
                st.error("Session could not be paused.")
        if cols[2].button("Resume", icon=":material/restart_alt:", disabled=latest_session.get("status") != "paused", width="stretch"):
            updated = sessions.update(
                latest_session["id"],
                {
                    "status": "running_live_monitoring",
                    "worker_status": "idle",
                    "worker_owner_id": worker_owner_id(),
                    "worker_pid": None,
                    "worker_heartbeat_at": None,
                    "worker_lease_expires_at": datetime.now(timezone.utc).isoformat(),
                    "last_worker_error": None,
                    "diagnostic_status": None,
                    "diagnostic_error": None,
                },
            )
            if updated:
                stop_worker(client=client, reason="session_resume_reset")
                ensure_worker_running(int(source.get("polling_interval_seconds") or 30) if source else 30, client=client)
                st.success("Session resumed.")
                st.rerun()
            else:
                st.error("Session could not be resumed.")
        if cols[3].button("Stop", icon=":material/stop:", disabled=latest_session.get("status") not in ACTIVE_SESSION_STATUSES, width="stretch"):
            ended_at = datetime.now(timezone.utc).isoformat()
            updated = sessions.update(
                latest_session["id"],
                {
                    "status": "completed",
                    "ended_at": ended_at,
                    "completed_reason": "user_stopped",
                    "worker_status": "idle",
                    "worker_pid": None,
                    "worker_heartbeat_at": None,
                    "worker_lease_expires_at": ended_at,
                },
            )
            if updated:
                clear_threshold_widget_state(latest_session["id"])
                clear_threshold_widget_state(model_id)
                stop_worker(client=client, reason="session_stopped")
                refreshed_session = sessions.get(latest_session["id"]) or updated
                summary = build_session_summary(images, predictions, feedback, refreshed_session, source, ended_at)
                sessions.update(latest_session["id"], {"summary": summary})
                st.success("Session completed.")
                st.rerun()
            else:
                st.error("Session could not be completed.")

        status_card("Session", latest_session.get("status"), latest_session.get("name") or "")
        recover_idle_worker(sessions, latest_session, source, client=client)
        render_worker_status(latest_session)
        render_session_confidence_metrics(images, predictions, latest_session)
        if st.button("Refresh session data", icon=":material/refresh:"):
            st.rerun()
        if latest_session.get("summary"):
            st.json(latest_session["summary"])
    else:
        st.info("No session has been started for this model yet.")

    if start_disabled:
        st.warning("Session cannot start yet:\n\n- " + "\n- ".join(start_blockers))
    return latest_session




def threshold_widget_key_suffix(session: dict | None, model_id: str) -> str:
    if session and session.get("status") in ACTIVE_SESSION_STATUSES and session.get("id"):
        return str(session["id"])
    return model_id


def threshold_widget_keys(key_suffix: str) -> tuple[str, str, str, str, str, str]:
    return (
        f"threshold_prediction_{key_suffix}",
        f"threshold_data_{key_suffix}",
        f"threshold_concept_{key_suffix}",
        f"threshold_confidence_medium_{key_suffix}",
        f"threshold_confidence_good_{key_suffix}",
        f"graph_window_{key_suffix}",
    )


def clear_threshold_widget_state(key_suffix: str) -> None:
    for key in threshold_widget_keys(key_suffix):
        st.session_state.pop(key, None)


def session_control_config(session: dict | None) -> dict:
    session_status = (session or {}).get("status")
    config_source = (session or {}).get("threshold_config") if session_status in ACTIVE_SESSION_STATUSES else DEFAULT_THRESHOLD_CONFIG
    return normalized_threshold_config(config_source)


def render_threshold_controls(sessions, session: dict | None, model_id: str) -> dict:
    st.markdown("#### Drift Thresholds")
    session_status = (session or {}).get("status")
    current = session_control_config(session)
    locked = session_status == "running_live_monitoring"
    key_suffix = threshold_widget_key_suffix(session, model_id)

    cols = st.columns(3)

    threshold_help_button(
        cols[0],
        "Prediction Drift",
        "Choose the p-value where class-distribution changes should become a warning. Lower values are stricter statistical evidence; the critical cutoff is derived from this value and frozen with the session.",
        f"prediction_{key_suffix}",
    )
    prediction_warning = cols[0].slider(
        "p-value warning",
        min_value=0.001,
        max_value=0.5,
        value=float(current["prediction"]["p_value_warning"]),
        step=0.001,
        format="%.3f",
        disabled=locked,
        key=f"threshold_prediction_{key_suffix}",
    )

    threshold_help_button(
        cols[1],
        "Data Drift",
        "Choose the MMD score where feature-distribution changes should become a warning. Use a lower value for sensitive monitoring and a higher value to tolerate expected image-source variation.",
        f"data_{key_suffix}",
    )
    data_warning = cols[1].slider(
        "MMD warning",
        min_value=0.001,
        max_value=1.0,
        value=float(current["data"]["mmd_warning"]),
        step=0.001,
        format="%.3f",
        disabled=locked,
        key=f"threshold_data_{key_suffix}",
    )

    threshold_help_button(
        cols[2],
        "Concept Drift",
        "Choose the review accuracy below which human feedback should flag concept drift. Higher values are stricter and expect more approved predictions.",
        f"concept_{key_suffix}",
    )
    concept_warning = cols[2].slider(
        "Accuracy warning %",
        min_value=1.0,
        max_value=100.0,
        value=float(current["concept"]["accuracy_warning_percent"]),
        step=1.0,
        format="%.0f%%",
        disabled=locked,
        key=f"threshold_concept_{key_suffix}",
    )

    st.markdown("#### Confidence Display Thresholds")

    confidence_cols = st.columns(2)
    current_medium_percent = float(current["confidence"]["medium_threshold"]) * 100
    current_good_percent = float(current["confidence"]["good_threshold"]) * 100

    threshold_help_button(
        confidence_cols[0],
        "Medium Confidence",
        "Choose the confidence value from which predictions should no longer be shown as low confidence. This should depend on the model, number of classes and calibration.",
        f"confidence_medium_{key_suffix}",
    )
    confidence_medium_percent = confidence_cols[0].slider(
        "Medium from",
        min_value=0.0,
        max_value=100.0,
        value=current_medium_percent,
        step=1.0,
        format="%.0f%%",
        disabled=locked,
        key=f"threshold_confidence_medium_{key_suffix}",
    )

    threshold_help_button(
        confidence_cols[1],
        "Good Confidence",
        "Choose the confidence value from which predictions should be shown as high confidence. This threshold should be at least as high as the medium threshold.",
        f"confidence_good_{key_suffix}",
    )
    confidence_good_percent = confidence_cols[1].slider(
        "Good from",
        min_value=confidence_medium_percent,
        max_value=100.0,
        value=max(current_good_percent, confidence_medium_percent),
        step=1.0,
        format="%.0f%%",
        disabled=locked,
        key=f"threshold_confidence_good_{key_suffix}",
    )

    st.markdown("#### Graph Display")
    graph_window = st.slider(
        "Graph window",
        min_value=GRAPH_WINDOW_MIN,
        max_value=GRAPH_WINDOW_MAX,
        value=int(current["display"]["graph_window"]),
        step=1,
        disabled=locked,
        key=f"graph_window_{key_suffix}",
        help="Sets how many recent samples every line graph displays. Line graph x-axes are numbered from 1 to this value.",
    )

    config = threshold_config_from_slider_values(
        prediction_warning,
        data_warning,
        concept_warning,
        confidence_medium_percent / 100,
        confidence_good_percent / 100,
        graph_window,
    )

    st.caption(
        "Thresholds are snapshotted on the session and used for deterministic drift decisions and confidence visualization. "
        f"Critical cutoffs: p < {config['prediction']['p_value_critical']:.3f}, "
        f"MMD >= {config['data']['mmd_critical']:.3f}, "
        f"accuracy < {config['concept']['accuracy_critical_percent']:.0f}%. "
        f"Confidence bands: low < {config['confidence']['medium_threshold']:.0%}, "
        f"medium < {config['confidence']['good_threshold']:.0%}, "
        f"good >= {config['confidence']['good_threshold']:.0%}. "
        f"Line graph window: {config['display']['graph_window']} samples."
    )

    if locked:
        st.caption("Pause the running session before changing session configuration.")
    elif session and session_status == "paused":
        if st.button("Save session configuration", icon=":material/save:"):
            updated = sessions.update(session["id"], {"threshold_config": config})
            if updated:
                session["threshold_config"] = config
                st.success("Session configuration saved.")
                st.rerun()
            else:
                st.error("Session configuration could not be saved.")

    return config


def threshold_help_button(container, title: str, help_text: str, key: str) -> None:
    label_col, help_col = container.columns([0.82, 0.18], vertical_alignment="center")
    label_col.markdown(f"**{title}**")
    help_col.button("?", key=f"threshold_help_{key}", help=help_text, disabled=True, width="stretch")


def source_path_is_valid(source: dict | None) -> bool:
    if not source:
        return False
    source_uri = normalize_source_uri(source.get("source_uri"))
    return bool(source_uri and Path(source_uri).expanduser().exists() and Path(source_uri).expanduser().is_dir())


def normalize_source_uri(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def session_start_blockers(
    source: dict | None,
    model: dict,
    active_session: dict | None,
) -> list[str]:
    blockers: list[str] = []
    dataset = model.get("_latest_dataset") or {}
    artifact = model.get("_latest_artifact") or {}
    if dataset.get("validation_status") != "valid":
        blockers.append("Validate the current dataset successfully.")
    if artifact.get("compatibility_status") != "valid":
        blockers.append("Complete a valid model/dataset compatibility check.")
    if model.get("baseline_status") != "completed" or not model.get("_baseline_profiles"):
        blockers.append("Build a baseline for the current dataset and model artifact.")
    if not artifact.get("local_path"):
        blockers.append("Register a local model artifact.")
    if source is None:
        blockers.append("Save a production image source folder.")
    elif not source_path_is_valid(source):
        shown_path = normalize_source_uri(source.get("source_uri")) or "(empty)"
        blockers.append(f"Choose an existing production image folder; the saved path is not valid: {shown_path}")
    if active_session and active_session.get("status") in ACTIVE_SESSION_STATUSES:
        blockers.append("Stop the existing running or paused session first.")
    return blockers


def recover_idle_worker(sessions, session: dict, source: dict | None, client=None) -> None:
    if session.get("status") != "running_live_monitoring" or session.get("worker_status") not in {"idle", "stale"}:
        return
    retry_key = f"worker_idle_retry_at_{session['id']}"
    now = datetime.now(timezone.utc)
    previous_retry = st.session_state.get(retry_key)
    if isinstance(previous_retry, datetime) and (now - previous_retry).total_seconds() < IDLE_WORKER_RETRY_SECONDS:
        return
    st.session_state[retry_key] = now
    worker_result = ensure_worker_running(int(source.get("polling_interval_seconds") or 30) if source else 30, client=client)
    if worker_result.ok:
        st.caption("Background worker is running.")
        return
    error = worker_result.error or "unknown error"
    sessions.update(session["id"], {"worker_status": "failed", "last_worker_error": error})
    session["worker_status"] = "failed"
    session["last_worker_error"] = error
    st.error(f"Background worker could not be started: {error}")


def maybe_auto_poll(client, model: dict, source: dict | None, session: dict | None) -> None:
    if not source or not session or session.get("status") != "running_live_monitoring":
        return
    interval = max(2, int(source.get("polling_interval_seconds") or 30))
    if not hasattr(st, "fragment"):
        return

    @st.fragment(run_every=interval)
    def auto_refresh_fragment() -> None:
        token_key = f"session_change_token_{session['id']}"
        token = session_change_token(client, session["id"])
        previous_token = st.session_state.get(token_key)
        st.session_state[token_key] = token
        if previous_token is not None and token != previous_token and rerun_cooldown_elapsed(session["id"], interval):
            st.rerun()

    auto_refresh_fragment()


def rerun_cooldown_elapsed(session_id: str, interval_seconds: int) -> bool:
    key = f"session_rerun_at_{session_id}"
    now = datetime.now(timezone.utc)
    previous = st.session_state.get(key)
    if previous is not None:
        try:
            if (now - previous).total_seconds() < max(1, int(interval_seconds)):
                return False
        except TypeError:
            pass
    st.session_state[key] = now
    return True


def render_worker_status(session: dict) -> None:
    worker_status = session.get("worker_status") or "not started"
    last_processed = format_timestamp(session.get("last_processed_at"))
    processing_started = format_timestamp(session.get("processing_started_at"))
    worker_error = session.get("last_worker_error")
    heartbeat = read_heartbeat()
    heartbeat_age = heartbeat_age_label(heartbeat)
    stale = heartbeat_is_stale(heartbeat, int(session.get("polling_interval_seconds") or 30))

    cols = st.columns(4)
    cols[0].metric("Worker", str(worker_status).replace("_", " ").title())
    cols[1].metric("Processing started", processing_started)
    cols[2].metric("Last processed", last_processed)
    cols[3].metric("Heartbeat", "Stale" if stale else heartbeat_age)
    st.caption(f"Worker owner: {session.get('worker_owner_id') or (heartbeat or {}).get('owner_id') or 'N/A'} | PID: {session.get('worker_pid') or (heartbeat or {}).get('pid') or 'N/A'}")
    if worker_error:
        st.error(f"Worker error: {worker_error}")


def heartbeat_age_label(heartbeat: dict | None) -> str:
    if not heartbeat or not heartbeat.get("last_seen_at"):
        return "N/A"
    parsed = parse_datetime(heartbeat.get("last_seen_at"))
    if not parsed:
        return "N/A"
    seconds = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    return f"{seconds}s ago"


def session_change_token(client, session_id: str) -> tuple:
    sessions = Repository(client, "monitoring_sessions")
    images = Repository(client, "images")
    predictions = Repository(client, "predictions")
    drift = Repository(client, "drift_results")
    alerts = Repository(client, "alerts")

    session = sessions.get(session_id) or {}
    image_rows = images.where(session_id=session_id) if hasattr(images, "where") else images.where_ordered(session_id=session_id)
    image_ids = [row["id"] for row in image_rows if row.get("id")]
    prediction_rows = prediction_rows_for_images(predictions, image_rows, image_ids)
    latest_prediction = latest_row_from_rows(prediction_rows)
    latest_drift = latest_row_token(drift, session_id)
    latest_alert = latest_row_token(alerts, session_id)
    return (
        session.get("status"),
        session.get("last_worker_error"),
        session.get("diagnostic_status"),
        session.get("diagnostic_error"),
        session.get("last_processed_image_id"),
        sum(1 for row in image_rows if (row.get("inference_status") or "complete") == "complete"),
        sum(1 for row in image_rows if row.get("inference_status") == "failed"),
        sum(1 for row in image_rows if row.get("inference_status") == "pending"),
        latest_prediction,
        len(prediction_rows),
        latest_drift,
        latest_alert,
    )


def latest_row_token(repository: Repository, session_id: str) -> tuple | None:
    rows = repository.where_ordered(session_id=session_id, limit=1)
    if not rows:
        return None
    row = rows[0]
    return row.get("id"), row.get("window_end") or row.get("created_at")


def latest_row_from_rows(rows: list[dict]) -> tuple | None:
    if not rows:
        return None
    row = sorted(rows, key=lambda item: (str(item.get("created_at") or ""), str(item.get("id") or "")), reverse=True)[0]
    return row.get("id"), row.get("created_at")


def render_session_confidence_metrics(images, predictions, session: dict | None) -> None:
    stats = session_prediction_stats(images, predictions, session)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Avg. confidence", format_confidence(stats["average_confidence"]))
    metric_cols[1].metric(
        "Low confidence",
        stats["low_confidence_count"],
        f"<{stats['confidence_medium_threshold']:.0%}",
    )
    metric_cols[2].metric("Predictions", stats["prediction_count"])
    metric_cols[3].metric("Processed images", stats["completed_image_count"])

    if stats["detection_location_count"]:
        st.caption(
            f"Detection overlap · {stats['ambiguous_location_count']} ambiguous locations · "
            f"{stats['same_class_duplicate_count']} same-class duplicate locations across "
            f"{stats['overlap_affected_image_count']} affected images · "
            f"IoU ≥ {DETECTION_CLUSTER_IOU_THRESHOLD:.0%}"
        )

    st.markdown("#### Average confidence")
    _, chart_col, _ = st.columns([1, 4, 1])
    with chart_col:
        render_confidence_gauge(
            stats["average_confidence"],
            stats["confidence_medium_threshold"],
            stats["confidence_good_threshold"],
        )

    st.caption(
        "Average confidence is calculated over all stored predictions in this session. "
        "It is a model uncertainty indicator, not a direct accuracy value. "
        f"Configured bands: low < {stats['confidence_medium_threshold']:.0%}, "
        f"medium < {stats['confidence_good_threshold']:.0%}, "
        f"good >= {stats['confidence_good_threshold']:.0%}."
    )

def render_confidence_gauge(value: float | None, medium_threshold: float = 0.5, good_threshold: float = 0.75) -> None:
    if value is None:
        st.info("No confidence values available yet.")
        return

    clamped = max(0.0, min(1.0, float(value)))
    medium_threshold = max(0.0, min(1.0, float(medium_threshold)))
    good_threshold = max(medium_threshold, min(1.0, float(good_threshold)))

    percent_value = clamped * 100
    gauge_color = confidence_gauge_color(clamped, medium_threshold, good_threshold)
    gauge_status = confidence_gauge_status(clamped, medium_threshold, good_threshold)
    medium_percent = medium_threshold * 100
    good_percent = good_threshold * 100
    bands = pd.DataFrame(
        [
            {"start": 0.0, "end": medium_percent, "band": "Low"},
            {"start": medium_percent, "end": good_percent, "band": "Medium"},
            {"start": good_percent, "end": 100.0, "band": "Good"},
        ]
    )
    marker = pd.DataFrame([{"confidence": percent_value, "label": f"{percent_value:.1f}%"}])
    axis_values = sorted({0.0, medium_percent, good_percent, 100.0})
    label_align = "left" if percent_value <= 5 else "right" if percent_value >= 95 else "center"
    label_dx = 5 if percent_value <= 5 else -5 if percent_value >= 95 else 0

    band_chart = alt.Chart(bands).mark_bar(size=24).encode(
        x=alt.X(
            "start:Q",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(title=None, values=axis_values, labelExpr="datum.value + '%'"),
        ),
        x2="end:Q",
        color=alt.Color(
            "band:N",
            scale=alt.Scale(domain=["Low", "Medium", "Good"], range=["#f87171", "#fbbf24", "#4ade80"]),
            legend=None,
        ),
        tooltip=[alt.Tooltip("band:N", title="Band"), alt.Tooltip("start:Q", title="From", format=".0f"), alt.Tooltip("end:Q", title="To", format=".0f")],
    )
    marker_chart = alt.Chart(marker).mark_rule(color=gauge_color, strokeWidth=3).encode(
        x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 100]))
    )
    point_chart = alt.Chart(marker).mark_point(color=gauge_color, filled=True, size=70).encode(
        x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 100]))
    )
    label_chart = alt.Chart(marker).mark_text(
        dy=-23,
        dx=label_dx,
        align=label_align,
        fontSize=14,
        fontWeight="bold",
    ).encode(
        x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 100])),
        text="label:N",
    )
    chart = (band_chart + marker_chart + point_chart + label_chart).properties(
        height=75,
        padding={"top": 28, "right": 6, "bottom": 0, "left": 6},
    ).configure_view(stroke=None)

    st.altair_chart(chart, width="stretch")
    st.caption(f"{gauge_status} · {percent_value:.1f}%")


def session_prediction_stats(images, predictions, session: dict | None) -> dict[str, Any]:
    confidence_medium, confidence_good = confidence_thresholds_from_session(session)

    if not session or not session.get("id"):
        return {
            "completed_image_count": 0,
            "prediction_count": 0,
            "average_confidence": None,
            "low_confidence_count": 0,
            "detection_location_count": 0,
            "ambiguous_location_count": 0,
            "same_class_duplicate_count": 0,
            "overlap_affected_image_count": 0,
            "confidence_medium_threshold": confidence_medium,
            "confidence_good_threshold": confidence_good,
        }

    image_rows = images.where(session_id=session["id"])
    complete_image_rows = [
        row for row in image_rows
        if (row.get("inference_status") or "complete") == "complete"
    ]

    prediction_rows = prediction_rows_for_images(predictions, complete_image_rows)
    rows_by_image: dict[str, list[dict]] = {}
    for row in prediction_rows:
        image_id = str(row.get("image_id") or "")
        rows_by_image.setdefault(image_id, []).append(row)
    overlap_summary = summarize_detection_overlaps(rows_by_image.values())

    confidences = [
        float(row["confidence"])
        for row in prediction_rows
        if isinstance(row.get("confidence"), (int, float))
    ]

    return {
        "completed_image_count": len(complete_image_rows),
        "prediction_count": len(prediction_rows),
        "average_confidence": sum(confidences) / len(confidences) if confidences else None,
        "low_confidence_count": sum(
            1 for confidence in confidences
            if confidence < confidence_medium
        ),
        "detection_location_count": overlap_summary["location_count"],
        "ambiguous_location_count": overlap_summary["ambiguous_location_count"],
        "same_class_duplicate_count": overlap_summary["same_class_duplicate_count"],
        "overlap_affected_image_count": overlap_summary["affected_image_count"],
        "confidence_medium_threshold": confidence_medium,
        "confidence_good_threshold": confidence_good,
    }

def confidence_thresholds_from_session(session: dict | None) -> tuple[float, float]:
    config = normalized_threshold_config((session or {}).get("threshold_config"))
    confidence = config["confidence"]
    return confidence["medium_threshold"], confidence["good_threshold"]


def format_confidence(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.1f}%"


def render_predictions_tab(client, model: dict, images, predictions, feedback, session: dict | None) -> None:
    st.subheader("Latest Session Predictions")
    if not session:
        st.info("Start a session before reviewing predictions.")
        return

    st.caption(f"{session.get('name') or 'Untitled session'} | {session.get('status')}")
    image_rows = images.where_ordered(session_id=session["id"], limit=50)
    if not image_rows:
        st.info("No images have been processed for the latest session.")
        return

    render_session_confidence_metrics(images, predictions, session)
    st.divider()
    st.markdown(
        """
        <style>
        [class*="st-key-review_table_"] {
            border-color: #111111 !important;
            border-radius: 12px !important;
            margin-top: 2.6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    prediction_rows_by_image = predictions_by_image_id(predictions, image_rows)
    for image_index, image in enumerate(image_rows):
        prediction_rows = prediction_rows_by_image.get(image["id"], [])
        render_image_review_group(client, model, session, image, prediction_rows, feedback)
        if image_index < len(image_rows) - 1:
            st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)


def render_image_review_group(client, model: dict, session: dict, image: dict, prediction_rows: list[dict], feedback) -> None:
    preview_col, detail_col = st.columns([0.42, 0.58], gap="medium", vertical_alignment="top")
    preview_col.markdown(f"**{image.get('filename') or 'Image'}**")
    local_path = image.get("local_path")
    detection_rows = [row for row in prediction_rows if row.get("task_type") == "object_detection"]
    detection_clusters = cluster_detection_predictions(detection_rows)
    if local_path and Path(local_path).exists():
        if detection_rows:
            preview_col.image(draw_detection_overlays(local_path, detection_clusters), width="stretch")
        else:
            preview_col.image(local_path, width="stretch")
    else:
        preview_col.caption("Image unavailable")

    inference_status = image.get("inference_status") or "complete"
    if inference_status == "pending":
        detail_col.dataframe([{"Status": "Prediction pending", "Detail": "Waiting for inference to complete."}], width="stretch")
        st.info("Prediction pending")
        return
    if inference_status == "failed":
        detail_col.dataframe([{"Status": "Inference failed", "Detail": image.get("inference_error") or "No error details available."}], width="stretch")
        st.warning("Inference failed")
        return
    if not prediction_rows:
        message = "No detections" if model.get("selected_task_type") == "object_detection" else "No prediction"
        detail_col.dataframe([{"Status": message, "Detail": "No feedback controls available."}], width="stretch")
        st.info(message)
        return

    render_feedback_table(client, model, session, image, prediction_rows, feedback, detail_col, detection_clusters)


def render_feedback_table(
    client,
    model: dict,
    session: dict,
    image: dict,
    prediction_rows: list[dict],
    feedback,
    container,
    detection_clusters: list[DetectionCluster],
) -> None:
    with container:
        table_key = "review_table_" + "".join(character if character.isalnum() else "_" for character in str(image["id"]))
        with st.container(key=table_key, border=True):
            if detection_clusters:
                ambiguous_count = sum(cluster.is_ambiguous for cluster in detection_clusters)
                st.caption(
                    f"{len(detection_clusters)} locations · {ambiguous_count} ambiguous · "
                    f"{sum(len(cluster.members) for cluster in detection_clusters)} raw predictions"
                )
            header_cols = st.columns([0.05, 0.17, 0.12, 0.24, 0.12, 0.2, 0.1])
            for column, label in zip(header_cols, ("#", "Class", "Confidence", "Box", "Latest", "Decision", "")):
                column.caption(label)
            if detection_clusters:
                for cluster in detection_clusters:
                    for member_index, prediction in enumerate(cluster.members):
                        render_prediction_feedback_form(
                            client,
                            model,
                            session,
                            image,
                            prediction,
                            feedback,
                            cluster_member_label(cluster, member_index),
                        )
            else:
                for index, prediction in enumerate(prediction_rows, start=1):
                    render_prediction_feedback_form(client, model, session, image, prediction, feedback, str(index))


def normalized_box_label(prediction: dict) -> str:
    values = [prediction.get(key) for key in ("x_center", "y_center", "width", "height")]
    if any(not isinstance(value, (int, float)) for value in values):
        return "N/A"
    return ", ".join(f"{float(value):.3f}" for value in values)


def render_prediction_feedback_form(client, model: dict, session: dict, image: dict, prediction: dict, feedback, index: str) -> None:
    existing_feedback = feedback.where_ordered(prediction_id=prediction["id"], limit=1)
    latest_feedback = existing_feedback[0].get("feedback_type") if existing_feedback else None
    task_type = prediction.get("task_type") or model.get("selected_task_type")
    feedback_options = ["approve", "reject"] if task_type == "classification" else ["approve", "reject", "false_positive", "missed_object"]
    predicted_class = prediction.get("predicted_class_name") or "No prediction"
    confidence = prediction.get("confidence")
    confidence_label = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "N/A"
    with st.form(f"review_{prediction['id']}", border=False):
        row_cols = st.columns([0.05, 0.17, 0.12, 0.24, 0.12, 0.2, 0.1], vertical_alignment="center")
        row_cols[0].caption(str(index))
        row_cols[1].caption(predicted_class)
        row_cols[2].caption(confidence_label)
        row_cols[3].caption(normalized_box_label(prediction))
        row_cols[4].caption(latest_feedback or "None")
        feedback_type = row_cols[5].selectbox("Decision", feedback_options, key=f"feedback_type_{prediction['id']}", label_visibility="collapsed")
        submitted = row_cols[6].form_submit_button("Save", icon=":material/save:", width="stretch")
    if submitted:
        feedback_payload = {
            "image_id": image["id"],
            "prediction_id": prediction["id"],
            "feedback_type": feedback_type,
            "corrected_payload": {},
            "comment": None,
        }
        if existing_feedback:
            feedback.update(existing_feedback[0]["id"], feedback_payload)
        else:
            feedback.insert(feedback_payload)
        record_feedback_concept_drift(client, model, session)
        st.success("Review saved.")


def render_drift_tab(drift, alerts, session: dict | None) -> None:
    st.subheader("Drift")
    if not session:
        st.info("Start a session before viewing drift.")
        return

    rows = drift.where(session_id=session["id"])
    scored_rows = [with_drift_score(row) for row in rows]
    if not scored_rows:
        st.info("Prediction drift will appear after production predictions are available.")
        return
    alert_rows = alerts.where(session_id=session["id"])
    unresolved_alerts = unresolved_drift_alerts(alert_rows)

    latest_total = latest_metric_row(scored_rows, "total_data_drift")
    latest_prediction = latest_metric_row(scored_rows, "prediction_class_distribution")
    latest_concept = latest_metric_row(scored_rows, "human_review_summary")
    sections = available_drift_sections(scored_rows)
    session_threshold_config = normalized_threshold_config(session.get("threshold_config"))
    graph_window = graph_window_from_session(session)

    render_diagnostic_state(session)
    render_drift_overview(session, scored_rows, unresolved_alerts)
    render_drift_alerts(unresolved_alerts, scored_rows)

    metric_count = 1 + int(sections["data"]) + int(sections["concept"])
    metric_cols = st.columns(metric_count)
    prediction_details = (latest_prediction or {}).get("details") or {}
    metric_cols[0].metric(
        "Prediction drift",
        drift_status_label(latest_prediction),
        f"Chi-Square {format_score(latest_prediction.get('metric_value') if latest_prediction else None)} | p {format_score(prediction_details.get('p_value'))}",
    )
    next_col = 1
    if sections["data"]:
        metric_cols[next_col].metric(
            "Total data drift",
            drift_status_label(latest_total),
            f"MMD {format_score(latest_total.get('metric_value') if latest_total else None)}",
        )
        next_col += 1
    if sections["concept"]:
        concept_accuracy = ((latest_concept or {}).get("details") or {}).get("accuracy_percent")
        metric_cols[next_col].metric("Review accuracy", concept_status_label(latest_concept), format_percent(concept_accuracy))

    render_prediction_drift_history(
        metric_rows(scored_rows, "prediction_class_distribution"),
        session_threshold_config,
        graph_window,
    )
    render_prediction_distribution_chart(latest_prediction)
    if sections["data"]:
        render_total_data_drift_chart(scored_rows, session_threshold_config, graph_window)
        render_data_property_charts(scored_rows, graph_window)
    else:
        st.info("Data drift diagnostics are calculated after critical prediction drift pauses the session.")
    if sections["concept"]:
        render_concept_drift_history(
            metric_rows(scored_rows, "human_review_summary"),
            session_threshold_config,
            graph_window,
        )
        render_concept_summary(latest_concept)
    else:
        st.info("Concept drift is calculated after prediction feedback is submitted.")


def available_drift_sections(rows: list[dict]) -> dict[str, bool]:
    drift_types = {row.get("drift_type") for row in rows}
    return {
        "prediction": "prediction" in drift_types,
        "data": "data" in drift_types,
        "concept": "concept" in drift_types,
    }


def render_diagnostic_state(session: dict) -> None:
    status = session.get("diagnostic_status")
    reason = str(session.get("diagnostic_reason") or "").replace("_", " ")
    if status == "running":
        st.warning(f"Production monitoring was automatically paused because of {reason}. Data drift diagnostics are running.")
    elif status == "completed":
        st.error(f"Production monitoring was automatically paused because of {reason}. Data drift diagnostics are available below.")
    elif status == "failed":
        error = session.get("diagnostic_error") or "Unknown diagnostic error."
        st.error(f"Production monitoring was automatically paused because of {reason}, but data drift diagnostics failed: {error}")

def render_drift_overview(session: dict, rows: list[dict], unresolved_alerts: list[dict]) -> None:
    latest_row = latest_drift_row(rows)
    latest_time = format_timestamp(latest_row.get("created_at") if latest_row else None)
    image_count = latest_row.get("num_images") if latest_row else None
    overall_status = alert_overall_status(unresolved_alerts)
    cols = st.columns(4)
    cols[0].metric("Overall state", overall_status)
    cols[1].metric("Session", session.get("status") or "N/A", session.get("name") or "Untitled")
    cols[2].metric("Window", f"Latest {DRIFT_WINDOW_SIZE} images", f"{image_count or 0} used")
    cols[3].metric("Latest check", latest_time)


def render_drift_alerts(alert_rows: list[dict], drift_rows: list[dict]) -> None:
    st.markdown("#### Active Alerts")
    if not alert_rows:
        st.success("No unresolved drift alerts for this session.")
        return
    drift_by_id = {row.get("id"): row for row in drift_rows if row.get("id")}
    for severity in ("critical", "warning", "info"):
        severity_rows = [row for row in alert_rows if row.get("severity") == severity]
        if not severity_rows:
            continue
        for alert in sorted(severity_rows, key=lambda row: str(row.get("created_at") or ""), reverse=True):
            linked_metric = alert_metric_name(alert, drift_by_id)
            message = alert.get("message") or "Review this drift alert."
            created = format_timestamp(alert.get("created_at"))
            text = f"{alert.get('title') or severity.title()}: {message} Metric: {linked_metric}. Created: {created}."
            if severity == "critical":
                st.error(text)
            elif severity == "warning":
                st.warning(text)
            else:
                st.info(text)


def unresolved_drift_alerts(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row.get("is_resolved") and row.get("drift_result_id")]


def alert_metric_name(alert: dict, drift_by_id: dict[str, dict]) -> str:
    drift_row = drift_by_id.get(alert.get("drift_result_id"))
    if drift_row and drift_row.get("metric_name"):
        return readable_metric_name(drift_row["metric_name"])
    return "linked drift result"


def alert_overall_status(alert_rows: list[dict]) -> str:
    severities = {row.get("severity") for row in alert_rows}
    if "critical" in severities:
        return "Critical"
    if "warning" in severities:
        return "Warning"
    return "OK"


def latest_drift_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: str(drift_chart_time(row) or ""))[-1]


def is_non_ok_drift(row: dict | None) -> bool:
    return bool(row and row.get("status") in {"warning", "critical"})


def is_concept_drift_high(row: dict | None) -> bool:
    if is_non_ok_drift(row):
        return True
    details = (row or {}).get("details") or {}
    accuracy = details.get("accuracy_percent")
    threshold = concept_warning_threshold_from_row(row)
    return isinstance(accuracy, (int, float)) and accuracy < threshold


def concept_warning_threshold_from_row(row: dict | None) -> float:
    details = (row or {}).get("details") or {}
    threshold_config = details.get("threshold_config") if isinstance(details.get("threshold_config"), dict) else {}
    concept = threshold_config.get("concept") if isinstance(threshold_config.get("concept"), dict) else {}
    value = concept.get("accuracy_warning_percent") or (row or {}).get("threshold") or CONCEPT_DRIFT_ACCURACY_WARNING_PERCENT
    try:
        return float(value)
    except (TypeError, ValueError):
        return CONCEPT_DRIFT_ACCURACY_WARNING_PERCENT


def with_drift_score(row: dict) -> dict:
    enriched = dict(row)
    enriched["score"] = normalized_drift_score(row)
    return enriched


def normalized_drift_score(row: dict) -> float:
    metric_value = row.get("metric_value")
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return {"ok": 0.0, "warning": 60.0, "critical": 100.0}.get(row.get("status"), 0.0)


def format_score(score: float | None) -> str:
    return "N/A" if score is None else f"{float(score):.3f}"


def format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.1f}%"


def format_timestamp(value: str | None) -> str:
    if not value:
        return "N/A"
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)


def drift_status_label(row: dict | None) -> str:
    if not row:
        return "N/A"
    return str(row.get("status") or "N/A").replace("_", " ").title()


def concept_status_label(row: dict | None) -> str:
    if not row:
        return "N/A"
    return "Warning" if is_concept_drift_high(row) else "OK"


def readable_metric_name(metric_name: str) -> str:
    return str(metric_name).replace("_", " ").title()


def graph_heading(title: str, help_text: str, key: str) -> None:
    title_col, help_col = st.columns([0.94, 0.06], vertical_alignment="center")
    title_col.markdown(f"#### {title}")
    help_col.button("?", key=f"help_{key}", help=help_text, disabled=True, width="stretch")


def latest_metric_row(rows: list[dict], metric_name: str) -> dict | None:
    matches = [row for row in rows if row.get("metric_name") == metric_name]
    if not matches:
        return None
    return sorted(matches, key=lambda row: str(drift_chart_time(row) or ""))[-1]


def metric_rows(rows: list[dict], metric_name: str) -> list[dict]:
    return sorted([row for row in rows if row.get("metric_name") == metric_name], key=lambda row: str(drift_chart_time(row) or ""))


def drift_chart_time(row: dict) -> str | None:
    return row.get("window_end") or row.get("created_at")


def graph_window_from_session(session: dict | None) -> int:
    return int(normalized_threshold_config((session or {}).get("threshold_config"))["display"]["graph_window"])


def windowed_rows(rows: list[dict], graph_window: int) -> list[dict]:
    window = max(GRAPH_WINDOW_MIN, min(int(graph_window), GRAPH_WINDOW_MAX))
    ordered = sorted(rows, key=lambda row: str(drift_chart_time(row) or ""))
    return ordered[-window:]


def graph_sample_axis(graph_window: int):
    window = max(GRAPH_WINDOW_MIN, min(int(graph_window), GRAPH_WINDOW_MAX))
    return alt.X(
        "Sample:Q",
        title="Sample",
        scale=alt.Scale(domain=[1, window], nice=False),
        axis=alt.Axis(format="d", tickCount=min(window, 10)),
    )


def render_total_data_drift_chart(
    rows: list[dict],
    threshold_config: dict | None = None,
    graph_window: int = GRAPH_WINDOW_DEFAULT,
) -> None:
    total_rows = metric_rows(rows, "total_data_drift")
    graph_heading(
        "Total Data Drift",
        "Total data drift uses RBF-kernel Maximum Mean Discrepancy (MMD) comparing fixed training-set feature vectors with the current production window.",
        "total_data_drift",
    )
    if not total_rows:
        st.info("No total data drift data available.")
        return
    warning_threshold, critical_threshold = mmd_thresholds_from_rows(total_rows, threshold_config)
    st.caption(f"Reference lines show current session thresholds: warning starts at {warning_threshold:.3f}; critical starts at {critical_threshold:.3f}. Existing drift points keep their original persisted status.")
    frame = total_data_drift_frame(total_rows, graph_window)
    threshold_frame = pd.DataFrame(
        [
            {"Threshold": "Warning", "MMD": warning_threshold},
            {"Threshold": "Critical", "MMD": critical_threshold},
        ]
    )
    line_chart = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=graph_sample_axis(graph_window),
            y=alt.Y("MMD:Q", title="MMD score"),
            tooltip=["CreatedAt:T", "WindowStart:T", "WindowEnd:T", "Images:Q", "MMD:Q"],
        )
    )
    threshold_chart = (
        alt.Chart(threshold_frame)
        .mark_rule(strokeDash=[6, 4])
        .encode(
            y="MMD:Q",
            color=alt.Color("Threshold:N", scale=alt.Scale(domain=["Warning", "Critical"], range=["#F58518", "#E45756"])),
            tooltip=["Threshold", "MMD:Q"],
        )
    )
    chart = (line_chart + threshold_chart).properties(height=300)
    st.altair_chart(chart, width="stretch")


def mmd_thresholds_from_rows(rows: list[dict], threshold_config: dict | None = None) -> tuple[float, float]:
    if threshold_config is not None:
        data = normalized_threshold_config(threshold_config)["data"]
        return data["mmd_warning"], data["mmd_critical"]

    latest = rows[-1] if rows else {}
    details = latest.get("details") or {}
    row_threshold_config = details.get("threshold_config") if isinstance(details.get("threshold_config"), dict) else {}
    data = row_threshold_config.get("data") if isinstance(row_threshold_config.get("data"), dict) else {}
    warning = latest.get("threshold") or data.get("mmd_warning") or TOTAL_DATA_DRIFT_WARNING_THRESHOLD
    critical = data.get("mmd_critical") or TOTAL_DATA_DRIFT_CRITICAL_THRESHOLD
    try:
        warning = float(warning)
    except (TypeError, ValueError):
        warning = TOTAL_DATA_DRIFT_WARNING_THRESHOLD
    try:
        critical = float(critical)
    except (TypeError, ValueError):
        critical = TOTAL_DATA_DRIFT_CRITICAL_THRESHOLD
    return warning, critical


def total_data_drift_frame(rows: list[dict], graph_window: int = GRAPH_WINDOW_DEFAULT) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sample": sample,
            "CreatedAt": pd.to_datetime(drift_chart_time(row)),
            "WindowStart": pd.to_datetime(row.get("window_start")),
            "WindowEnd": pd.to_datetime(row.get("window_end")),
            "Images": int(row.get("num_images") or 0),
            "MMD": float(row.get("metric_value") or 0.0),
        }
        for sample, row in enumerate(windowed_rows(rows, graph_window), start=1)
        if drift_chart_time(row)
    )


def render_data_property_charts(rows: list[dict], graph_window: int = GRAPH_WINDOW_DEFAULT) -> None:
    properties = [
        ("brightness", "Brightness", "Brightness is the mean grayscale pixel intensity. The reference line is the fixed training-set average."),
        ("contrast", "Contrast", "Contrast is the standard deviation of grayscale pixel intensity. The reference line is the fixed training-set average."),
        ("colour_distribution", "Colour Distribution", "Colour distribution uses an RGB colorfulness score based on red-green and yellow-blue channel differences. The reference line is the fixed training-set average."),
        ("noise", "Noise", "Noise is the mean absolute difference between the grayscale image and a lightly blurred grayscale image. The reference line is the fixed training-set average."),
    ]
    for metric_name, title, help_text in properties:
        graph_heading(title, help_text, metric_name)
        property_rows = metric_rows(rows, metric_name)
        if not property_rows:
            st.info(f"No {title.lower()} data available.")
            continue
        latest_row = property_rows[-1]
        st.caption(
            f"Latest status: {drift_status_label(latest_row)} | "
            f"Difference from training: {format_percent(latest_row.get('metric_value'))}. "
            f"Each production point is the latest monitoring window average. Warning: {PROPERTY_DRIFT_WARNING_THRESHOLD:.0f}%+, critical: {PROPERTY_DRIFT_CRITICAL_THRESHOLD:.0f}%+."
        )
        frame = property_line_frame(property_rows, graph_window)
        chart = (
            alt.Chart(frame)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=graph_sample_axis(graph_window),
                y=alt.Y("Value:Q", title=title),
                color=alt.Color("Series:N", scale=alt.Scale(domain=["Training average", "Production average"], range=["#4C78A8", "#F58518"])),
                tooltip=["CreatedAt:T", "WindowStart:T", "WindowEnd:T", "Images:Q", "Series", "Value"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch")


def property_line_frame(rows: list[dict], graph_window: int = GRAPH_WINDOW_DEFAULT) -> pd.DataFrame:
    frame_rows = []
    for sample, row in enumerate(windowed_rows(rows, graph_window), start=1):
        details = row.get("details") or {}
        created_at = drift_chart_time(row)
        if not created_at:
            continue
        common = {
            "Sample": sample,
            "CreatedAt": pd.to_datetime(created_at),
            "WindowStart": pd.to_datetime(row.get("window_start")),
            "WindowEnd": pd.to_datetime(row.get("window_end")),
            "Images": int(row.get("num_images") or 0),
        }
        frame_rows.append({**common, "Series": "Training average", "Value": details.get("training_average")})
        frame_rows.append({**common, "Series": "Production average", "Value": details.get("production_average")})
    return pd.DataFrame(frame_rows)


def render_prediction_drift_history(
    rows: list[dict],
    threshold_config: dict | None = None,
    graph_window: int = GRAPH_WINDOW_DEFAULT,
) -> None:
    graph_heading(
        "Prediction Drift History",
        "The newest configured graph window of prediction-drift p-values. Lower values indicate stronger evidence of distribution drift.",
        "prediction_history",
    )
    frame = prediction_drift_history_frame(rows, graph_window)
    if frame.empty:
        st.info("No prediction drift history available.")
        return
    thresholds = normalized_threshold_config(threshold_config)["prediction"]
    threshold_frame = pd.DataFrame(
        [
            {"Threshold": "Warning", "PValue": thresholds["p_value_warning"]},
            {"Threshold": "Critical", "PValue": thresholds["p_value_critical"]},
        ]
    )
    line_chart = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=graph_sample_axis(graph_window),
            y=alt.Y("PValue:Q", title="p-value"),
            tooltip=["CreatedAt:T", "WindowStart:T", "WindowEnd:T", "Images:Q", "ChiSquare:Q", "PValue:Q", "Status:N"],
        )
    )
    threshold_chart = (
        alt.Chart(threshold_frame)
        .mark_rule(strokeDash=[6, 4])
        .encode(
            y="PValue:Q",
            color=alt.Color("Threshold:N", scale=alt.Scale(domain=["Warning", "Critical"], range=["#F58518", "#E45756"])),
            tooltip=["Threshold", "PValue:Q"],
        )
    )
    st.altair_chart((line_chart + threshold_chart).properties(height=280), width="stretch")


def prediction_drift_history_frame(rows: list[dict], graph_window: int = GRAPH_WINDOW_DEFAULT) -> pd.DataFrame:
    frame_rows = []
    for sample, row in enumerate(windowed_rows(rows, graph_window), start=1):
        created_at = drift_chart_time(row)
        p_value = (row.get("details") or {}).get("p_value")
        if not created_at or not isinstance(p_value, (int, float)):
            continue
        frame_rows.append(
            {
                "Sample": sample,
                "CreatedAt": pd.to_datetime(created_at),
                "WindowStart": pd.to_datetime(row.get("window_start")),
                "WindowEnd": pd.to_datetime(row.get("window_end")),
                "Images": int(row.get("num_images") or 0),
                "ChiSquare": row.get("metric_value"),
                "PValue": float(p_value),
                "Status": drift_status_label(row),
            }
        )
    return pd.DataFrame(frame_rows)


def render_prediction_distribution_chart(row: dict | None) -> None:
    graph_heading(
        "Prediction Class Distribution",
        "Prediction drift compares fixed training-set class percentages with current production predicted-class counts using a Chi-Square goodness-of-fit test.",
        "prediction_distribution",
    )
    if not row:
        st.info("No prediction drift data available.")
        return
    st.caption("Production percentages are calculated from predictions in the latest monitoring window and compared with the fixed training baseline.")
    details = row.get("details") or {}
    cols = st.columns(2)
    cols[0].metric("Chi-Square statistic", format_score(row.get("metric_value")))
    cols[1].metric("p-value", format_score(details.get("p_value")))
    if details.get("reason"):
        st.warning(details["reason"])
    if details.get("unseen_classes"):
        st.error(f"Unseen production classes: {', '.join(details['unseen_classes'])}")
    frame = class_distribution_frame(details.get("training_percentages") or {}, details.get("production_percentages") or {})
    if frame.empty:
        st.info("No class distribution percentages available to chart.")
        return
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("Class:N", title="Class"),
            y=alt.Y("Percentage:Q", title="Percentage"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Training set", "Production"], range=["#4C78A8", "#F58518"])),
            xOffset="Series:N",
            tooltip=["Class", "Series", "Percentage"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def class_distribution_frame(training_percentages: dict, production_percentages: dict) -> pd.DataFrame:
    labels = sorted(set(training_percentages) | set(production_percentages))
    rows = []
    for label in labels:
        rows.append({"Class": label, "Series": "Training set", "Percentage": float(training_percentages.get(label, 0.0))})
        rows.append({"Class": label, "Series": "Production", "Percentage": float(production_percentages.get(label, 0.0))})
    return pd.DataFrame(rows)


def render_concept_drift_history(
    rows: list[dict],
    threshold_config: dict | None = None,
    graph_window: int = GRAPH_WINDOW_DEFAULT,
) -> None:
    graph_heading(
        "Concept Drift History",
        "The newest configured graph window of cumulative human-review accuracy after each saved review.",
        "concept_history",
    )
    frame = concept_drift_history_frame(rows, graph_window)
    if frame.empty:
        st.info("No concept drift history available.")
        return
    threshold = normalized_threshold_config(threshold_config)["concept"]["accuracy_warning_percent"]
    line_chart = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=graph_sample_axis(graph_window),
            y=alt.Y("Accuracy:Q", title="Review accuracy (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["CreatedAt:T", "Accuracy:Q", "Reviews:Q", "Status:N"],
        )
    )
    threshold_chart = alt.Chart(pd.DataFrame([{"Accuracy": threshold}])).mark_rule(strokeDash=[6, 4], color="#F58518").encode(y="Accuracy:Q")
    st.altair_chart((line_chart + threshold_chart).properties(height=280), width="stretch")


def concept_drift_history_frame(rows: list[dict], graph_window: int = GRAPH_WINDOW_DEFAULT) -> pd.DataFrame:
    frame_rows = []
    for sample, row in enumerate(windowed_rows(rows, graph_window), start=1):
        created_at = drift_chart_time(row)
        details = row.get("details") or {}
        accuracy = details.get("accuracy_percent")
        if not created_at or not isinstance(accuracy, (int, float)):
            continue
        frame_rows.append(
            {
                "Sample": sample,
                "CreatedAt": pd.to_datetime(created_at),
                "Accuracy": float(accuracy),
                "Reviews": int(details.get("total_reviews") or 0),
                "Status": concept_status_label(row),
            }
        )
    return pd.DataFrame(frame_rows)


def render_concept_summary(row: dict | None) -> None:
    graph_heading(
        "Concept Drift Review Summary",
        "Concept drift uses human review outcomes. Accuracy is approved predictions divided by total reviewed predictions.",
        "concept_summary",
    )
    if not row:
        st.info("No human review data available.")
        return
    details = row.get("details") or {}
    counts = details.get("counts") or {}
    cols = st.columns(2)
    cols[0].metric("Accuracy", format_percent(details.get("accuracy_percent")))
    cols[1].metric("Reviews", details.get("total_reviews", 0))
    st.dataframe(
        pd.DataFrame([{"Outcome": label.replace("_", " ").title(), "Count": count} for label, count in counts.items()]),
        width="stretch",
    )

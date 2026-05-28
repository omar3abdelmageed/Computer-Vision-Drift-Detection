from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.components.feedback_widgets import render_feedback_form
from app.components.status_cards import status_card
from backend.database.repositories import SupabaseRepository
from backend.monitoring.session_manager import process_source_once


def render_live_sessions_tab(client) -> None:
    st.header("Live Sessions")
    models = SupabaseRepository(client, "models")
    datasets = SupabaseRepository(client, "datasets")
    artifacts = SupabaseRepository(client, "model_artifacts")
    sources = SupabaseRepository(client, "production_sources")
    sessions = SupabaseRepository(client, "monitoring_sessions")
    images = SupabaseRepository(client, "images")
    predictions = SupabaseRepository(client, "predictions")
    drift = SupabaseRepository(client, "drift_results")
    alerts = SupabaseRepository(client, "alerts")
    feedback = SupabaseRepository(client, "feedback")
    baselines = SupabaseRepository(client, "baseline_profiles")

    selected_model = select_registered_model(models, datasets, artifacts, baselines)
    if not selected_model:
        return
    model_id = selected_model["id"]
    render_initial_test_evaluation(datasets, sources, sessions, model_id)
    source = render_source_configuration(sources, model_id)
    session = render_session_controls(client, sessions, artifacts, baselines, selected_model, source)
    render_prediction_feed(images, predictions, session)
    render_drift_dashboard(drift, session)
    render_alerts(alerts, session, model_id)
    render_feedback_form(feedback, images, predictions, selected_model, session)


def select_registered_model(models, datasets, artifacts, baselines) -> dict | None:
    st.subheader("1. Select Registered Model")
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
    eligible = dataset.get("validation_status") == "valid" and artifact.get("compatibility_status") != "invalid" and selected.get("baseline_status") == "completed" and has_current_baseline
    selected["_latest_dataset"] = dataset
    selected["_latest_artifact"] = artifact
    selected["_baseline_profiles"] = baseline_profiles
    selected["_live_eligible"] = eligible
    status_card("Eligibility", "valid" if eligible else "warning", "Ready" if eligible else "Complete validation, compatibility, and baseline first.")
    baseline_status = "current" if has_current_baseline else "missing for current dataset/artifact"
    st.caption(f"Dataset: {dataset.get('validation_status', 'missing')} | Artifact: {artifact.get('compatibility_status', 'missing')} | Baseline: {selected.get('baseline_status')} ({baseline_status})")
    return selected


def current_baseline_profiles(baselines, model_id: str, dataset: dict, artifact: dict) -> list[dict]:
    dataset_id = dataset.get("id")
    artifact_id = artifact.get("id")
    if not dataset_id or not artifact_id:
        return []
    return baselines.where_ordered(model_id=model_id, dataset_id=dataset_id, artifact_id=artifact_id)


def render_initial_test_evaluation(datasets, sources, sessions, model_id: str) -> None:
    st.subheader("2. Initial Test Evaluation")
    dataset = datasets.latest_where(model_id=model_id) or {}
    if not dataset.get("has_test_split"):
        st.info("No test split found. Configure a live production source to begin monitoring.")
        return
    st.write("A test split is available.")
    if st.button("Run initial test evaluation"):
        source = sources.insert({"model_id": model_id, "source_type": "test_folder", "source_uri": resolve_test_source_uri(dataset.get("dataset_root_path")), "mode": "initial_evaluation", "polling_interval_seconds": 30, "is_active": True, "config": {"test_has_labels": dataset.get("test_has_labels")}})
        sessions.insert({"model_id": model_id, "source_id": source["id"] if source else None, "source_type": "test_folder", "status": "running_test_evaluation"})
        st.success("Initial test evaluation session created.")


def resolve_test_source_uri(dataset_root_path: str | None) -> str | None:
    if not dataset_root_path:
        return None
    root = Path(dataset_root_path)
    for candidate in (root / "test", root / "images" / "test"):
        if candidate.exists():
            return str(candidate)
    return str(root)


def render_source_configuration(sources, model_id: str) -> dict | None:
    st.subheader("3. Production Source Configuration")
    with st.form("source_config"):
        source_type = st.selectbox("Source type", ["test_folder", "manual_upload", "watched_folder"])
        source_uri = st.text_input("Source URI or local path")
        label_uri = st.text_input("Optional label URI")
        interval = st.number_input("Polling interval seconds", min_value=1, value=30)
        mode = st.selectbox("Mode", ["live_monitoring", "manual_batch", "initial_evaluation"])
        config_text = st.text_area("Config JSON", value="{}")
        submitted = st.form_submit_button("Save source")
        if submitted:
            source = sources.insert({"model_id": model_id, "source_type": source_type, "source_uri": source_uri, "label_uri": label_uri or None, "mode": mode, "polling_interval_seconds": int(interval), "is_active": True, "config": {"raw": config_text}})
            st.success("Source saved.")
            return source
    rows = sources.where_ordered(model_id=model_id)
    st.dataframe(rows, use_container_width=True)
    return rows[0] if rows else None


def render_session_controls(client, sessions, artifacts, baselines, model: dict, source: dict | None) -> dict | None:
    st.subheader("4. Session Controls")
    model_id = model["id"]
    artifact = model.get("_latest_artifact") or artifacts.latest_where(model_id=model_id) or {}
    baseline_profiles = model.get("_baseline_profiles") or current_baseline_profiles(baselines, model_id, model.get("_latest_dataset") or {}, artifact)
    eligible = bool(model.get("_live_eligible"))
    cols = st.columns(5)
    if cols[0].button("Start session", disabled=source is None or not eligible):
        sessions.insert({"model_id": model_id, "artifact_id": artifact.get("id"), "source_id": source.get("id") if source else None, "source_type": source.get("source_type") if source else None, "status": "running_live_monitoring"})
        st.success("Session started.")
    rows = sessions.where_ordered(model_id=model_id)
    session = rows[0] if rows else None
    if session:
        if cols[1].button("Pause session"):
            sessions.update(session["id"], {"status": "paused"})
        if cols[2].button("Resume session"):
            sessions.update(session["id"], {"status": "running_live_monitoring"})
        if cols[3].button("Stop session"):
            sessions.update(session["id"], {"status": "completed"})
        cols[4].button("Refresh results")
        if st.button("Process source once", disabled=source is None or artifact.get("local_path") is None or not eligible):
            try:
                summary = process_source_once(client, model, artifact, source, session, baseline_profiles)
                st.success(f"Processed {summary.get('processed', 0)} new images.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
        status_card("Session", session.get("status"), session.get("source_type") or "")
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No sessions yet.")
    return session


def render_prediction_feed(images, predictions, session: dict | None) -> None:
    st.subheader("5. Live Prediction Feed")
    if not session:
        st.info("Start or select a session.")
        return
    rows = images.where(session_id=session["id"])[:50]
    st.dataframe(rows, use_container_width=True)
    if rows:
        image_id = rows[0]["id"]
        st.dataframe(predictions.where(image_id=image_id), use_container_width=True)


def render_drift_dashboard(drift, session: dict | None) -> None:
    st.subheader("6. Drift Dashboard")
    if not session:
        st.info("No session selected.")
        return
    rows = drift.where(session_id=session["id"])
    tabs = st.tabs(["Data Drift", "Prediction Drift", "Concept Drift"])
    for tab, drift_type in zip(tabs, ["data", "prediction", "concept"]):
        with tab:
            typed_rows = [row for row in rows if row.get("drift_type") == drift_type]
            if drift_type == "concept" and not typed_rows:
                st.info("Concept drift requires labels or human feedback.")
            else:
                st.dataframe(typed_rows, use_container_width=True)


def render_alerts(alerts, session: dict | None, model_id: str) -> None:
    st.subheader("7. Alerts")
    show_resolved = st.toggle("Show resolved alerts", value=False)
    rows = alerts.where(model_id=model_id)
    rows = rows if show_resolved else [row for row in rows if not row.get("is_resolved")]
    st.dataframe(rows, use_container_width=True)

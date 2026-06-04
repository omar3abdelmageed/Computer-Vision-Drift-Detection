from __future__ import annotations

from datetime import timedelta
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import pandas as pd
import streamlit as st
import altair as alt

from app.components.image_viewer import draw_detection_overlays
from app.components.status_cards import status_card
from backend.database.repositories import SupabaseRepository
from backend.monitoring.session_manager import process_source_once, record_feedback_concept_drift


CONCEPT_DRIFT_ACCURACY_WARNING_PERCENT = 80.0
DRIFT_WINDOW_SIZE = 25
TOTAL_DATA_DRIFT_WARNING_THRESHOLD = 0.1
TOTAL_DATA_DRIFT_CRITICAL_THRESHOLD = 0.25
PROPERTY_DRIFT_WARNING_THRESHOLD = 20.0
PROPERTY_DRIFT_CRITICAL_THRESHOLD = 50.0


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
    source = latest_source(sources, model_id)
    latest_session = latest_monitoring_session(sessions, model_id)

    source_tab, session_tab, predictions_tab, drift_tab = st.tabs(["Source", "Session", "Predictions", "Drift"])
    with source_tab:
        protected_source_id = latest_session.get("source_id") if latest_session and latest_session.get("status") in {"running_live_monitoring", "paused"} else None
        source = render_source_tab(sources, model_id, source, protected_source_id=protected_source_id)
    with session_tab:
        active_source = source_for_session(sources, source, latest_session)
        latest_session = render_session_tab(client, sessions, images, predictions, feedback, selected_model, active_source, latest_session)
    active_source = source_for_session(sources, source, latest_session)
    maybe_auto_poll(client, selected_model, active_source, latest_session)
    with predictions_tab:
        render_predictions_tab(client, selected_model, images, predictions, feedback, latest_session)
    with drift_tab:
        render_drift_tab(drift, alerts, latest_session)


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


def source_for_session(sources, fallback_source: dict | None, session: dict | None) -> dict | None:
    if not session or session.get("status") not in {"running_live_monitoring", "paused"}:
        return fallback_source
    source_id = session.get("source_id")
    return sources.get(source_id) if source_id else fallback_source


def render_source_tab(sources, model_id: str, source: dict | None, protected_source_id: str | None = None) -> dict | None:
    st.subheader("Image Source")
    source_locked = protected_source_id is not None
    with st.form("source_config"):
        source_uri = st.text_input("Path to pull images from", value=source.get("source_uri", "") if source else "")
        interval = st.number_input(
            "Polling interval seconds",
            min_value=1,
            value=int(source.get("polling_interval_seconds") or 30) if source else 30,
        )
        submitted = st.form_submit_button("Save source", icon=":material/save:", disabled=source_locked)
    if source_locked:
        st.caption("Stop the running session before configuring a new source.")

    if submitted:
        payload = {
            "model_id": model_id,
            "source_type": "watched_folder",
            "source_uri": source_uri,
            "mode": "live_monitoring",
            "polling_interval_seconds": int(interval),
            "is_active": True,
            "config": {},
        }
        if source and source.get("id") != protected_source_id:
            source = sources.update(source["id"], payload) or source
        else:
            source = sources.insert(payload)
        st.success("Source saved.")

    if source:
        source_path = Path(source.get("source_uri") or "")
        detail = f"{source.get('source_uri') or 'No path'} | every {source.get('polling_interval_seconds') or 30}s"
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
) -> dict | None:
    st.subheader("Session Controls")
    model_id = model["id"]
    artifact = model.get("_latest_artifact") or {}
    eligible = bool(model.get("_live_eligible"))
    default_name = f"{model.get('name') or 'Session'} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    session_name = st.text_input("Session name", value=latest_session.get("name") or default_name if latest_session else default_name)

    cols = st.columns(4)
    has_active_session = latest_session and latest_session.get("status") in {"running_live_monitoring", "paused"}
    start_disabled = source is None or not eligible or not artifact.get("local_path") or bool(has_active_session)
    if cols[0].button("Start", icon=":material/play_arrow:", disabled=start_disabled, use_container_width=True):
        latest_session = sessions.insert(
            {
                "model_id": model_id,
                "artifact_id": artifact.get("id"),
                "source_id": source.get("id") if source else None,
                "source_type": source.get("source_type") if source else None,
                "name": session_name,
                "status": "running_live_monitoring",
                "summary": {},
            }
        )
        st.success("Session started.")
        if latest_session and source:
            process_source_once_and_report(client, model, source, latest_session)
            st.rerun()

    if latest_session:
        if cols[1].button("Pause", icon=":material/pause:", disabled=latest_session.get("status") != "running_live_monitoring", use_container_width=True):
            latest_session = sessions.update(latest_session["id"], {"status": "paused"}) or latest_session
            st.success("Session paused.")
        if cols[2].button("Resume", icon=":material/restart_alt:", disabled=latest_session.get("status") != "paused", use_container_width=True):
            latest_session = sessions.update(latest_session["id"], {"status": "running_live_monitoring"}) or latest_session
            st.success("Session resumed.")
        if cols[3].button("Stop", icon=":material/stop:", disabled=latest_session.get("status") == "completed", use_container_width=True):
            ended_at = datetime.now(timezone.utc).isoformat()
            summary = build_session_summary(images, predictions, feedback, latest_session, source, ended_at)
            latest_session = sessions.update(latest_session["id"], {"status": "completed", "ended_at": ended_at, "summary": summary}) or latest_session
            st.success("Session completed.")

        status_card("Session", latest_session.get("status"), latest_session.get("name") or "")
        if latest_session.get("summary"):
            st.json(latest_session["summary"])
    else:
        st.info("No session has been started for this model yet.")

    if start_disabled:
        st.caption("Start requires an eligible model, a saved source, a local model artifact path, and no running or paused latest session.")
    return latest_session


def maybe_auto_poll(client, model: dict, source: dict | None, session: dict | None) -> None:
    if not source or not session or session.get("status") != "running_live_monitoring":
        return
    if not model.get("_live_eligible"):
        return
    if not (model.get("_latest_artifact") or {}).get("local_path"):
        return

    interval = int(source.get("polling_interval_seconds") or 30)
    if not hasattr(st, "fragment"):
        st.caption("Automatic polling requires a Streamlit version with st.fragment.")
        return

    @st.fragment(run_every=interval)
    def auto_poll_fragment() -> None:
        poll_key = f"last_poll_{session['id']}"
        now = monotonic()
        last_poll = st.session_state.get(poll_key, 0)
        if now - last_poll >= interval:
            st.session_state[poll_key] = now
            summary = process_source_once_and_report(client, model, source, session, show_empty=False)
            if summary and summary.get("processed", 0) > 0:
                st.rerun()

    auto_poll_fragment()


def process_source_once_and_report(client, model: dict, source: dict, session: dict, show_empty: bool = True) -> dict | None:
    artifact = model.get("_latest_artifact") or {}
    baseline_profiles = model.get("_baseline_profiles") or []
    if client is None:
        st.error("Processing failed: Supabase client is unavailable.")
        return None
    try:
        summary = process_source_once(client, model, artifact, source, session, baseline_profiles)
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        return None
    report_processing_summary(summary, show_empty=show_empty)
    return summary


def report_processing_summary(summary: dict, show_empty: bool = True) -> None:
    processed = summary.get("processed", 0)
    reason = summary.get("reason")
    if processed:
        st.success(f"Processed {processed} new images.")
    elif reason == "source_path_missing":
        st.warning("Session is running, but the configured source path does not exist.")
    elif show_empty:
        st.info("No new images found to process.")


def build_session_summary(images, predictions, feedback, session: dict, source: dict | None, ended_at: str) -> dict[str, Any]:
    image_rows = images.where(session_id=session["id"])
    prediction_rows = predictions_for_images(predictions, image_rows)
    latest_feedback_rows = latest_feedback_for_predictions(feedback, prediction_rows)
    confidences = [row.get("confidence") for row in prediction_rows if row.get("confidence") is not None]
    started_at = session.get("started_at")

    return {
        "source_uri": source.get("source_uri") if source else None,
        "processed_image_count": len(image_rows),
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
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


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

    for image in image_rows:
        prediction_rows = predictions.where(image_id=image["id"])
        render_image_review_group(client, model, session, image, prediction_rows, feedback)


def render_image_review_group(client, model: dict, session: dict, image: dict, prediction_rows: list[dict], feedback) -> None:
    preview_col, detail_col = st.columns([0.28, 0.72], vertical_alignment="center")
    local_path = image.get("local_path")
    detection_rows = [row for row in prediction_rows if row.get("task_type") == "object_detection"]
    if local_path and Path(local_path).exists():
        if detection_rows:
            preview_col.image(draw_detection_overlays(local_path, detection_rows), use_container_width=True)
        else:
            preview_col.image(local_path, use_container_width=True)
    else:
        preview_col.caption(image.get("filename") or "Image")

    detail_col.markdown(f"**{image.get('filename') or 'Image'}**")
    if not prediction_rows:
        detail_col.write("Predicted class: No prediction")
        detail_col.write("Confidence: N/A")
    elif detection_rows:
        detail_col.dataframe(detection_summary_rows(detection_rows), use_container_width=True)
    else:
        prediction = prediction_rows[0]
        predicted_class = prediction.get("predicted_class_name") or "No prediction"
        confidence = prediction.get("confidence")
        detail_col.write(f"Predicted class: {predicted_class}")
        detail_col.write(f"Confidence: {confidence:.3f}" if isinstance(confidence, (int, float)) else "Confidence: N/A")

    if not prediction_rows:
        st.info("No prediction")
        return

    render_feedback_table(client, model, session, image, prediction_rows, feedback)


def render_feedback_table(client, model: dict, session: dict, image: dict, prediction_rows: list[dict], feedback) -> None:
    st.markdown("##### Feedback")
    header_cols = st.columns([0.2, 0.16, 0.18, 0.24, 0.16, 0.12])
    for column, label in zip(header_cols, ("Prediction", "Latest", "Decision", "Correction", "Comment", "")):
        column.markdown(f"**{label}**")
    for index, prediction in enumerate(prediction_rows, start=1):
        render_prediction_feedback_form(client, model, session, image, prediction, feedback, index)


def detection_summary_rows(prediction_rows: list[dict]) -> list[dict]:
    rows = []
    for index, prediction in enumerate(prediction_rows, start=1):
        confidence = prediction.get("confidence")
        rows.append(
            {
                "#": index,
                "Class": prediction.get("predicted_class_name"),
                "Confidence": round(float(confidence), 3) if isinstance(confidence, (int, float)) else None,
                "Box": normalized_box_label(prediction),
            }
        )
    return rows


def normalized_box_label(prediction: dict) -> str:
    values = [prediction.get(key) for key in ("x_center", "y_center", "width", "height")]
    if any(not isinstance(value, (int, float)) for value in values):
        return "N/A"
    return ", ".join(f"{float(value):.3f}" for value in values)


def render_prediction_feedback_form(client, model: dict, session: dict, image: dict, prediction: dict, feedback, index: int) -> None:
    existing_feedback = feedback.where_ordered(prediction_id=prediction["id"], limit=1)
    latest_feedback = existing_feedback[0].get("feedback_type") if existing_feedback else None
    task_type = prediction.get("task_type") or model.get("selected_task_type")
    feedback_options = ["approve", "reject", "corrected_label"] if task_type == "classification" else ["approve", "reject", "false_positive", "missed_object", "wrong_class", "corrected_box"]
    label = prediction.get("predicted_class_name") or f"Prediction {index}"
    confidence = prediction.get("confidence")
    if isinstance(confidence, (int, float)):
        label = f"{index}. {label} ({confidence:.2f})"
    with st.form(f"review_{prediction['id']}"):
        row_cols = st.columns([0.2, 0.16, 0.18, 0.24, 0.16, 0.12], vertical_alignment="center")
        row_cols[0].caption(label)
        row_cols[1].caption(latest_feedback or "None")
        feedback_type = row_cols[2].selectbox("Decision", feedback_options, key=f"feedback_type_{prediction['id']}", label_visibility="collapsed")
        corrected_value = ""
        corrected_box = {}
        if feedback_type in {"corrected_label", "wrong_class"}:
            corrected_value = row_cols[3].text_input("Correction", key=f"corrected_class_{prediction['id']}", label_visibility="collapsed", placeholder="Correct class")
        elif feedback_type == "corrected_box":
            corrected_value = row_cols[3].text_input("Correction", value=normalized_box_label(prediction), key=f"corrected_box_{prediction['id']}", label_visibility="collapsed", placeholder="x,y,w,h")
            corrected_box = parse_corrected_box(corrected_value)
        else:
            row_cols[3].caption("N/A")
        comment = row_cols[4].text_input("Comment", key=f"comment_{prediction['id']}", label_visibility="collapsed", placeholder="Optional")
        submitted = row_cols[5].form_submit_button("Save", icon=":material/save:", use_container_width=True)
    if submitted:
        payload = {}
        if corrected_value and feedback_type in {"corrected_label", "wrong_class"}:
            payload["corrected_class"] = corrected_value
        if feedback_type == "corrected_box":
            if not corrected_box:
                st.error("Corrected box must contain four values between 0 and 1: x,y,w,h")
                return
            payload["corrected_box"] = corrected_box
        feedback.insert({"image_id": image["id"], "prediction_id": prediction["id"], "feedback_type": feedback_type, "corrected_payload": payload, "comment": comment or None})
        record_feedback_concept_drift(client, model, session)
        st.success("Review saved.")


def parse_corrected_box(value: str) -> dict:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError:
        return {}
    if len(parts) != 4 or any(part < 0 or part > 1 for part in parts):
        return {}
    return {"x_center": parts[0], "y_center": parts[1], "width": parts[2], "height": parts[3]}


def render_drift_tab(drift, alerts, session: dict | None) -> None:
    st.subheader("Drift")
    if not session:
        st.info("Start a session before viewing drift.")
        return

    rows = drift.where(session_id=session["id"])
    scored_rows = [with_drift_score(row) for row in rows]
    if not scored_rows:
        st.info("No drift data available yet.")
        return
    alert_rows = alerts.where(session_id=session["id"])
    unresolved_alerts = unresolved_drift_alerts(alert_rows)

    latest_total = latest_metric_row(scored_rows, "total_data_drift")
    latest_prediction = latest_metric_row(scored_rows, "prediction_class_distribution")
    latest_concept = latest_metric_row(scored_rows, "human_review_summary")

    render_drift_overview(session, scored_rows, unresolved_alerts)
    render_drift_alerts(unresolved_alerts, scored_rows)

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Total data drift",
        drift_status_label(latest_total),
        f"MMD {format_score(latest_total.get('metric_value') if latest_total else None)}",
    )
    prediction_details = (latest_prediction or {}).get("details") or {}
    metric_cols[1].metric(
        "Prediction drift",
        drift_status_label(latest_prediction),
        f"Chi-Square {format_score(latest_prediction.get('metric_value') if latest_prediction else None)} | p {format_score(prediction_details.get('p_value'))}",
    )
    concept_accuracy = ((latest_concept or {}).get("details") or {}).get("accuracy_percent")
    metric_cols[2].metric("Review accuracy", concept_status_label(latest_concept), format_percent(concept_accuracy))

    render_total_data_drift_chart(scored_rows)
    render_data_property_charts(scored_rows)
    render_prediction_distribution_chart(latest_prediction)
    render_concept_summary(latest_concept)


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
    return sorted(rows, key=lambda row: str(row.get("created_at") or ""))[-1]


def latest_rows_by_metric(rows: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: str(item.get("created_at") or "")):
        metric_name = row.get("metric_name")
        if metric_name:
            latest[metric_name] = row
    return latest


def is_non_ok_drift(row: dict | None) -> bool:
    return bool(row and row.get("status") in {"warning", "critical"})


def is_concept_drift_high(row: dict | None) -> bool:
    if is_non_ok_drift(row):
        return True
    accuracy = ((row or {}).get("details") or {}).get("accuracy_percent")
    return isinstance(accuracy, (int, float)) and accuracy < CONCEPT_DRIFT_ACCURACY_WARNING_PERCENT


def with_drift_score(row: dict) -> dict:
    enriched = dict(row)
    enriched["score"] = normalized_drift_score(row)
    return enriched


def normalized_drift_score(row: dict) -> float:
    metric_value = row.get("metric_value")
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return {"ok": 0.0, "warning": 60.0, "critical": 100.0}.get(row.get("status"), 0.0)


def drift_scores(rows: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rows:
        drift_type = row.get("drift_type")
        if drift_type:
            scores[drift_type] = max(scores.get(drift_type, 0.0), row.get("score", 0.0))
    return scores


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
    help_col.button("?", key=f"help_{key}", help=help_text, disabled=True, use_container_width=True)


def latest_metric_row(rows: list[dict], metric_name: str) -> dict | None:
    matches = [row for row in rows if row.get("metric_name") == metric_name]
    if not matches:
        return None
    return sorted(matches, key=lambda row: str(row.get("created_at") or ""))[-1]


def metric_rows(rows: list[dict], metric_name: str) -> list[dict]:
    return sorted([row for row in rows if row.get("metric_name") == metric_name], key=lambda row: str(row.get("created_at") or ""))


def render_total_data_drift_chart(rows: list[dict]) -> None:
    total_rows = metric_rows(rows, "total_data_drift")
    graph_heading(
        "Total Data Drift",
        "Total data drift uses RBF-kernel Maximum Mean Discrepancy (MMD) comparing fixed training-set feature vectors with the current production window.",
        "total_data_drift",
    )
    if not total_rows:
        st.info("No total data drift data available.")
        return
    st.caption("Each point is calculated from the latest production monitoring window, not the whole session. Warning starts at 0.1; critical starts at 0.25.")
    frame = pd.DataFrame(
        {"CreatedAt": pd.to_datetime(row.get("created_at")), "MMD": float(row.get("metric_value") or 0.0)}
        for row in total_rows
        if row.get("created_at")
    )
    threshold_frame = pd.DataFrame(
        [
            {"Threshold": "Warning", "MMD": TOTAL_DATA_DRIFT_WARNING_THRESHOLD},
            {"Threshold": "Critical", "MMD": TOTAL_DATA_DRIFT_CRITICAL_THRESHOLD},
        ]
    )
    line_chart = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=padded_time_axis(frame),
            y=alt.Y("MMD:Q", title="MMD score"),
            tooltip=["CreatedAt:T", "MMD:Q"],
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
    st.altair_chart(chart, use_container_width=True)


def render_data_property_charts(rows: list[dict]) -> None:
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
        frame = property_line_frame(property_rows)
        chart = (
            alt.Chart(frame)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=padded_time_axis(frame),
                y=alt.Y("Value:Q", title=title),
                color=alt.Color("Series:N", scale=alt.Scale(domain=["Training average", "Production average"], range=["#4C78A8", "#F58518"])),
                tooltip=["CreatedAt:T", "Series", "Value"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)


def property_line_frame(rows: list[dict]) -> pd.DataFrame:
    frame_rows = []
    for row in rows:
        details = row.get("details") or {}
        created_at = row.get("created_at")
        if not created_at:
            continue
        frame_rows.append({"CreatedAt": pd.to_datetime(created_at), "Series": "Training average", "Value": details.get("training_average")})
        frame_rows.append({"CreatedAt": pd.to_datetime(created_at), "Series": "Production average", "Value": details.get("production_average")})
    return pd.DataFrame(frame_rows)


def padded_time_axis(frame: pd.DataFrame):
    if frame.empty or "CreatedAt" not in frame:
        return alt.X("CreatedAt:T", title="Time")
    timestamps = pd.to_datetime(frame["CreatedAt"]).dropna()
    if timestamps.empty:
        return alt.X("CreatedAt:T", title="Time")
    start = timestamps.min().to_pydatetime()
    end = timestamps.max().to_pydatetime()
    if start == end:
        padding = timedelta(minutes=5)
    else:
        span = end - start
        padding = max(span / 12, timedelta(minutes=1))
    return alt.X(
        "CreatedAt:T",
        title="Time",
        scale=alt.Scale(domain=[start - padding, end + padding], nice=True),
    )


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
    st.altair_chart(chart, use_container_width=True)


def class_distribution_frame(training_percentages: dict, production_percentages: dict) -> pd.DataFrame:
    labels = sorted(set(training_percentages) | set(production_percentages))
    rows = []
    for label in labels:
        rows.append({"Class": label, "Series": "Training set", "Percentage": float(training_percentages.get(label, 0.0))})
        rows.append({"Class": label, "Series": "Production", "Percentage": float(production_percentages.get(label, 0.0))})
    return pd.DataFrame(rows)


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
        use_container_width=True,
    )

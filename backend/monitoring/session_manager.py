from __future__ import annotations

from pathlib import Path

from backend.alerts.alert_rules import alerts_from_drift
from backend.common import DriftResult, DriftStatus, DriftType
from backend.database.repositories import SupabaseRepository
from backend.drift.industrial_metrics import (
    PROPERTY_HELP,
    PROPERTY_KEYS,
    PROPERTY_LABELS,
    chi_square_prediction_drift,
    class_distribution,
    class_percentages,
    feature_vectors,
    property_averages,
    rbf_mmd,
    review_summary,
)
from backend.features.image_properties import extract_image_features, extract_image_metadata
from backend.ingestion.image_scanner import scan_images
from backend.model.inference import run_yolo_inference
from backend.registration.local_files import resolve_local_path
from backend.utils.serialization import to_jsonable


def process_source_once(client, model: dict, artifact: dict, source: dict, session: dict, baseline_profiles: list[dict], window_size: int = 25) -> dict:
    images_repo = SupabaseRepository(client, "images")
    predictions_repo = SupabaseRepository(client, "predictions")
    drift_repo = SupabaseRepository(client, "drift_results")
    alerts_repo = SupabaseRepository(client, "alerts")
    source_path = resolve_source_path(source.get("source_uri") or "")
    if source_path is None:
        return {"processed": 0, "reason": "source_path_missing"}
    paths = scan_images(source_path)
    processed = 0
    current_features: list[dict] = []
    current_predictions: list[str] = []
    for image_path in paths:
        metadata = extract_image_metadata(image_path)
        existing = images_repo.where(model_id=model["id"], session_id=session["id"], content_hash=metadata.content_hash)
        if existing:
            continue
        features = extract_image_features(image_path)
        image_row = images_repo.insert(
            {
                "model_id": model["id"],
                "session_id": session["id"],
                "source": normalize_image_source(source["source_type"]),
                "split": "production",
                "local_path": str(image_path),
                "filename": image_path.name,
                "content_hash": metadata.content_hash,
                "image_format": metadata.image_format,
                "color_mode": metadata.color_mode,
                "num_channels": metadata.num_channels,
                "bit_depth": metadata.bit_depth,
                "width": metadata.width,
                "height": metadata.height,
                "aspect_ratio": metadata.aspect_ratio,
                "file_size_bytes": metadata.file_size_bytes,
                "brightness_mean": features.brightness_mean,
                "brightness_std": features.brightness_std,
                "contrast": features.contrast,
                "sharpness": features.sharpness,
                "saturation_mean": features.saturation_mean,
                "saturation_std": features.saturation_std,
                "edge_density": features.edge_density,
                "feature_payload": to_jsonable(features),
            }
        )
        if image_row and artifact.get("local_path"):
            for pred in run_yolo_inference(Path(artifact["local_path"]), image_path, model["selected_task_type"]):
                pred["image_id"] = image_row["id"]
                pred["artifact_id"] = artifact.get("id")
                predictions_repo.insert(pred)
                if pred.get("predicted_class_name"):
                    current_predictions.append(pred["predicted_class_name"])
        current_features.append(to_jsonable(features))
        processed += 1
    window_features, window_predictions = current_session_window(images_repo, predictions_repo, session["id"], window_size)
    if len(window_features) >= 2 or window_predictions:
        results = calculate_drift_for_window(baseline_profiles, window_features, window_predictions)
        persist_drift_results(drift_repo, alerts_repo, model["id"], session["id"], len(window_features), results)
    return {"processed": processed}


def normalize_image_source(source_type: str) -> str:
    if source_type == "test_folder":
        return "test"
    if source_type in {"manual_upload", "watched_folder", "supabase_bucket"}:
        return source_type
    if source_type in {"rtsp", "usb_camera"}:
        return "camera"
    return "production"


def resolve_source_path(raw_path: str) -> Path | None:
    try:
        path = resolve_local_path(raw_path)
    except ValueError:
        return None
    if not path.exists() or not path.is_dir():
        return None
    return path


def current_session_window(images_repo: SupabaseRepository, predictions_repo: SupabaseRepository, session_id: str, window_size: int) -> tuple[list[dict], list[str]]:
    image_rows = images_repo.where_ordered(session_id=session_id, limit=window_size)
    features = [row.get("feature_payload") or feature_payload_from_image_row(row) for row in image_rows]
    predictions: list[str] = []
    for image in image_rows:
        for prediction in predictions_repo.where(image_id=image["id"]):
            if prediction.get("predicted_class_name"):
                predictions.append(prediction["predicted_class_name"])
    return [row for row in features if row], predictions


def feature_payload_from_image_row(row: dict) -> dict:
    keys = ("brightness_mean", "brightness_std", "contrast", "sharpness", "saturation_mean", "saturation_std", "edge_density", "noise_level", "colorfulness")
    return {key: row[key] for key in keys if row.get(key) is not None}


def calculate_drift_for_window(baseline_profiles: list[dict], current_features: list[dict], current_predictions: list[str]):
    feature_profile = next((row.get("metrics") for row in baseline_profiles if row.get("profile_type") == "feature_rows"), {}) or {}
    class_profile = next((row.get("metrics") for row in baseline_profiles if row.get("profile_type") == "class_distribution"), {}) or {}
    training_features = feature_profile.get("training_features") or feature_profile.get("features") or []
    training_averages = feature_profile.get("training_averages") or property_averages(training_features)
    training_vectors = feature_profile.get("training_vectors") or feature_vectors(training_features)
    production_averages = property_averages(current_features)
    production_vectors = feature_vectors(current_features)
    results = []
    if training_vectors and production_vectors:
        mmd_score = rbf_mmd(training_vectors, production_vectors)
        results.append(
            DriftResult(
                DriftType.DATA,
                "total_data_drift",
                mmd_score,
                0.0,
                status_from_mmd(mmd_score),
                {
                    "calculation": "rbf_mmd",
                    "help": PROPERTY_HELP["total_data_drift"],
                    "training_vector_count": len(training_vectors),
                    "production_vector_count": len(production_vectors),
                    "training_reference_fixed": True,
                },
            )
        )
    for key in PROPERTY_KEYS:
        training_average = training_averages.get(key)
        production_average = production_averages.get(key)
        if training_average is None or production_average is None:
            continue
        score = normalized_average_delta(float(training_average), float(production_average))
        results.append(
            DriftResult(
                DriftType.DATA,
                key,
                score,
                0.0,
                status_from_score(score),
                {
                    "property_key": key,
                    "property_label": PROPERTY_LABELS[key],
                    "help": PROPERTY_HELP[key],
                    "training_average": float(training_average),
                    "production_average": float(production_average),
                    "training_reference_fixed": True,
                },
            )
        )
    if current_predictions:
        training_counts = class_profile.get("training_counts") or {}
        production_counts = class_distribution(current_predictions)
        chi_square = chi_square_prediction_drift(training_counts, production_counts)
        p_value = chi_square.get("p_value")
        unseen_classes = chi_square.get("unseen_classes") or []
        statistic = chi_square.get("statistic")
        has_training_counts = sum(int(count) for count in training_counts.values() if isinstance(count, (int, float))) > 0
        results.append(
            DriftResult(
                DriftType.PREDICTION,
                "prediction_class_distribution",
                statistic,
                0.05,
                status_from_chi_square(p_value, bool(unseen_classes), has_training_counts),
                {
                    "calculation": "chi_square_goodness_of_fit",
                    "help": PROPERTY_HELP["prediction_class_distribution"],
                    "p_value": p_value,
                    "expected_counts": chi_square.get("expected_counts") or {},
                    "observed_counts": chi_square.get("observed_counts") or {},
                    "training_counts": training_counts,
                    "training_percentages": class_percentages(training_counts),
                    "production_counts": production_counts,
                    "production_percentages": class_percentages(production_counts),
                    "unseen_classes": unseen_classes,
                    "reason": chi_square.get("reason"),
                    "training_reference_fixed": True,
                },
            )
        )
    return results


def normalized_average_delta(training_average: float, production_average: float) -> float:
    denominator = max(abs(training_average), 1e-6)
    return abs(production_average - training_average) / denominator * 100.0


def status_from_score(score: float) -> DriftStatus:
    if score >= 50:
        return DriftStatus.CRITICAL
    if score >= 20:
        return DriftStatus.WARNING
    return DriftStatus.OK


def status_from_mmd(score: float) -> DriftStatus:
    if score >= 0.25:
        return DriftStatus.CRITICAL
    if score >= 0.1:
        return DriftStatus.WARNING
    return DriftStatus.OK


def status_from_chi_square(p_value: float | None, has_unseen_classes: bool, has_training_counts: bool) -> DriftStatus:
    if has_unseen_classes:
        return DriftStatus.CRITICAL
    if not has_training_counts or p_value is None:
        return DriftStatus.WARNING
    if p_value < 0.01:
        return DriftStatus.CRITICAL
    if p_value < 0.05:
        return DriftStatus.WARNING
    return DriftStatus.OK


def persist_drift_results(drift_repo: SupabaseRepository, alerts_repo: SupabaseRepository, model_id: str, session_id: str, num_images: int, results) -> None:
    for result in results:
        drift_row = drift_repo.insert(
            {
                "model_id": model_id,
                "session_id": session_id,
                "num_images": num_images,
                "drift_type": result.drift_type.value,
                "metric_name": result.metric_name,
                "metric_value": result.metric_value,
                "threshold": result.threshold,
                "status": result.status.value,
                "details": result.details,
            }
        )
        for alert in alerts_from_drift([result]):
            alerts_repo.insert(
                {
                    "model_id": model_id,
                    "session_id": session_id,
                    "severity": alert["severity"],
                    "title": alert["title"],
                    "message": alert["message"],
                    "drift_result_id": drift_row.get("id") if drift_row else None,
                }
            )


def record_feedback_concept_drift(client, model: dict, session: dict | None) -> None:
    if client is None or not session:
        return
    images_repo = SupabaseRepository(client, "images")
    feedback_repo = SupabaseRepository(client, "feedback")
    drift_repo = SupabaseRepository(client, "drift_results")
    alerts_repo = SupabaseRepository(client, "alerts")
    image_rows = images_repo.where(session_id=session["id"])
    feedback_types: list[str] = []
    for image in image_rows:
        feedback_types.extend(row["feedback_type"] for row in feedback_repo.where(image_id=image["id"]) if row.get("feedback_type"))
    if not feedback_types:
        return
    summary = review_summary(feedback_types)
    accuracy = summary.get("accuracy_percent")
    result = DriftResult(
        DriftType.CONCEPT,
        "human_review_summary",
        float(accuracy) if isinstance(accuracy, (int, float)) else 0.0,
        0.0,
        DriftStatus.OK,
        {"calculation": "review_accuracy", "help": PROPERTY_HELP["human_review_summary"], **summary},
    )
    persist_drift_results(drift_repo, alerts_repo, model["id"], session["id"], len(image_rows), [result])

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from background_worker.alerts import METRIC_LABELS, alerts_from_drift
from background_worker.concept_drift_tests.feedback_drift import review_summary
from background_worker.data_drift_tests.image_property_drift import (
    PROPERTY_HELP,
    PROPERTY_KEYS,
    PROPERTY_LABELS,
    property_averages,
)
from background_worker.data_drift_tests.mmd import feature_vectors, rbf_mmd
from background_worker.prediction_drift_tests.class_distribution_drift import chi_square_prediction_drift, class_distribution, class_percentages
from background_worker.image_features import extract_image_features, extract_image_metadata
from background_worker.inference import run_yolo_inference
from background_worker.source_scanner import scan_image_readiness
from core.local_files import resolve_local_path
from core.types import DriftResult, DriftStatus, DriftType
from database.repositories import Repository
from database.serialization import to_jsonable


CONCEPT_DRIFT_WARNING_PERCENT = 80.0
CONCEPT_DRIFT_CRITICAL_PERCENT = 60.0
PREDICTION_DRIFT_WARNING_P_VALUE = 0.05
PREDICTION_DRIFT_CRITICAL_P_VALUE = 0.01
TOTAL_DATA_DRIFT_WARNING_THRESHOLD = 0.1
TOTAL_DATA_DRIFT_CRITICAL_THRESHOLD = 0.25
PROPERTY_DRIFT_WARNING_THRESHOLD = 20.0
PROPERTY_DRIFT_CRITICAL_THRESHOLD = 50.0
CONFIDENCE_MEDIUM_THRESHOLD = 0.5
CONFIDENCE_GOOD_THRESHOLD = 0.75
GRAPH_WINDOW_MIN = 5
GRAPH_WINDOW_MAX = 50
GRAPH_WINDOW_DEFAULT = 25

DEFAULT_THRESHOLD_CONFIG = {
    "prediction": {
        "p_value_warning": PREDICTION_DRIFT_WARNING_P_VALUE,
        "p_value_critical": PREDICTION_DRIFT_CRITICAL_P_VALUE,
    },
    "data": {
        "mmd_warning": TOTAL_DATA_DRIFT_WARNING_THRESHOLD,
        "mmd_critical": TOTAL_DATA_DRIFT_CRITICAL_THRESHOLD,
    },
    "concept": {
        "accuracy_warning_percent": CONCEPT_DRIFT_WARNING_PERCENT,
        "accuracy_critical_percent": CONCEPT_DRIFT_CRITICAL_PERCENT,
    },
    "confidence": {
        "medium_threshold": CONFIDENCE_MEDIUM_THRESHOLD,
        "good_threshold": CONFIDENCE_GOOD_THRESHOLD,
    },
    "display": {
        "graph_window": GRAPH_WINDOW_DEFAULT,
    },
}


def source_image_stride(source: dict | None) -> int:
    if not isinstance(source, dict):
        return 1
    config = source.get("config") if isinstance(source.get("config"), dict) else {}
    value = source.get("image_stride") or config.get("image_stride")
    try:
        stride = int(value)
    except (TypeError, ValueError):
        stride = 1
    return max(1, stride)


def sampled_source_paths(paths: list[Path], stride: int) -> list[Path]:
    stride = max(1, int(stride or 1))
    if stride == 1:
        return paths
    return [path for index, path in enumerate(paths) if index % stride == 0]


def process_source_once(
    client,
    model: dict,
    artifact: dict,
    source: dict,
    session: dict,
    baseline_profiles: list[dict],
    window_size: int = 25,
    owner_id: str | None = None,
    lease_seconds: int = 90,
) -> dict:
    sessions_repo = Repository(client, "monitoring_sessions")
    images_repo = Repository(client, "images")
    predictions_repo = Repository(client, "predictions")
    drift_repo = Repository(client, "drift_results")
    alerts_repo = Repository(client, "alerts")
    threshold_config = normalized_threshold_config(session.get("threshold_config"))
    source_path = resolve_source_path(source.get("source_uri") or "")
    if source_path is None:
        return {"processed": 0, "reason": "source_path_missing"}
    if not session_is_running(sessions_repo, session):
        return {"processed": 0, "reason": "session_not_running"}
    readiness = scan_image_readiness(source_path)
    paths = readiness["ready"]
    paths = sampled_source_paths(paths, source_image_stride(source))
    processed = 0
    ready_files = len(readiness["ready"])
    source_files = int(readiness.get("total") or len(paths))
    skipped_unready = int(readiness.get("skipped_unready") or 0)
    skipped_by_stride = max(0, ready_files - len(paths))
    skipped_duplicates = 0
    insert_errors = 0
    inference_errors = 0
    last_processed_image_id = None
    prediction_drift_paused = False
    for image_path in paths:
        if not session_is_running(sessions_repo, session):
            return {
                "processed": processed,
                "ready_files": ready_files,
                "source_files": source_files,
                "skipped_unready": skipped_unready,
                "skipped_by_stride": skipped_by_stride,
                "skipped_duplicates": skipped_duplicates,
                "insert_errors": insert_errors,
                "inference_errors": inference_errors,
                "reason": "session_not_running",
            }
        refresh_session_lease(sessions_repo, session["id"], owner_id, lease_seconds)
        metadata = extract_image_metadata(image_path)
        existing = images_repo.where(model_id=model["id"], session_id=session["id"], content_hash=metadata.content_hash)
        if existing:
            skipped_duplicates += 1
            continue
        features = extract_image_features(image_path)
        try:
            image_row, was_duplicate = insert_image_idempotent(images_repo, model, session, source, image_path, metadata, features)
        except Exception as exc:
            insert_errors += 1
            return {
                "processed": processed,
                "ready_files": ready_files,
                "source_files": source_files,
                "skipped_unready": skipped_unready,
                "skipped_by_stride": skipped_by_stride,
                "skipped_duplicates": skipped_duplicates,
                "insert_errors": insert_errors,
                "inference_errors": inference_errors,
                "reason": "image_insert_failed",
                "error": str(exc),
            }
        if was_duplicate:
            skipped_duplicates += 1
            continue
        if not image_row:
            skipped_duplicates += 1
            continue
        prediction_count = 0
        if image_row and artifact.get("local_path"):
            try:
                if not predictions_repo.where(image_id=image_row["id"], artifact_id=artifact.get("id")):
                    for pred in run_yolo_inference(Path(artifact["local_path"]), image_path, model["selected_task_type"]):
                        pred["image_id"] = image_row["id"]
                        pred["artifact_id"] = artifact.get("id")
                        predictions_repo.insert(pred)
                        prediction_count += 1
                else:
                    prediction_count = len(predictions_repo.where(image_id=image_row["id"], artifact_id=artifact.get("id")))
            except Exception as exc:
                inference_errors += 1
                finalize_image_inference(images_repo, image_row, "failed", prediction_count=0, error=str(exc) or "Inference failed.")
                last_processed_image_id = image_row.get("id")
                processed += 1
                continue
        finalize_image_inference(images_repo, image_row, "complete", prediction_count=prediction_count)
        last_processed_image_id = image_row.get("id")
        processed += 1
        if image_row.get("id") and session_is_running(sessions_repo, session):
            prediction_drift_paused = persist_drift_for_completed_images(
                images_repo,
                predictions_repo,
                drift_repo,
                alerts_repo,
                model["id"],
                session["id"],
                baseline_profiles,
                [image_row["id"]],
                window_size,
                sessions_repo=sessions_repo,
                threshold_config=threshold_config,
            )
            if prediction_drift_paused:
                break
    return {
        "processed": processed,
        "ready_files": ready_files,
        "source_files": source_files,
        "skipped_unready": skipped_unready,
        "skipped_by_stride": skipped_by_stride,
        "skipped_duplicates": skipped_duplicates,
        "insert_errors": insert_errors,
        "inference_errors": inference_errors,
        "last_processed_image_id": last_processed_image_id,
        "prediction_drift_paused": prediction_drift_paused,
    }


def insert_image_idempotent(images_repo, model: dict, session: dict, source: dict, image_path: Path, metadata, features) -> tuple[dict | None, bool]:
    payload = {
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
        "inference_status": "pending",
        "prediction_count": 0,
        "inference_error": None,
        "inference_completed_at": None,
    }
    try:
        return images_repo.insert(payload), False
    except Exception as exc:
        if not is_duplicate_image_insert_error(exc):
            raise
        existing = images_repo.where(model_id=model["id"], session_id=session["id"], content_hash=metadata.content_hash)
        return (existing[0] if existing else None), True


def finalize_image_inference(images_repo, image_row: dict, status: str, prediction_count: int = 0, error: str | None = None) -> dict | None:
    payload = {
        "inference_status": status,
        "prediction_count": max(0, int(prediction_count or 0)),
        "inference_error": error,
        "inference_completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if hasattr(images_repo, "update") and image_row.get("id"):
        updated = images_repo.update(image_row["id"], payload)
        if updated:
            image_row.update(updated)
            return updated
    image_row.update(payload)
    return image_row


def is_duplicate_image_insert_error(exc: Exception) -> bool:
    message = str(exc).lower()
    duplicate_markers = (
        "23505",
        "duplicate key",
        "unique constraint",
        "idx_images_model_session_content_hash",
        "images_model_session_content_hash",
    )
    return any(marker in message for marker in duplicate_markers)


def session_is_running(sessions_repo, session: dict) -> bool:
    current = sessions_repo.get(session["id"]) if hasattr(sessions_repo, "get") else session
    return bool((current or session).get("status", "running_live_monitoring") == "running_live_monitoring")


def refresh_session_lease(sessions_repo, session_id: str, owner_id: str | None, lease_seconds: int) -> None:
    if not owner_id or not hasattr(sessions_repo, "update"):
        return
    sessions_repo.update(
        session_id,
        {
            "worker_owner_id": owner_id,
            "worker_heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "worker_lease_expires_at": (datetime.now(timezone.utc) + timedelta(seconds=max(10, int(lease_seconds)))).isoformat(),
        },
    )


def normalize_image_source(source_type: str) -> str:
    if source_type == "test_folder":
        return "test"
    if source_type in {"manual_upload", "watched_folder"}:
        return source_type
    return "production"


def resolve_source_path(raw_path: str) -> Path | None:
    try:
        path = resolve_local_path(raw_path)
    except ValueError:
        return None
    if not path.exists() or not path.is_dir():
        return None
    return path


def persist_drift_for_completed_images(
    images_repo: Repository,
    predictions_repo: Repository,
    drift_repo: Repository,
    alerts_repo: Repository,
    model_id: str,
    session_id: str,
    baseline_profiles: list[dict],
    completed_image_ids: list[str],
    window_size: int,
    sessions_repo: Repository | None = None,
    threshold_config: dict | None = None,
) -> bool:
    effective_thresholds = normalized_threshold_config(threshold_config)
    for image_id in completed_image_ids:
        window = current_session_window(images_repo, predictions_repo, session_id, window_size, end_image_id=image_id)
        if not window["predictions"]:
            continue
        prediction_results = calculate_prediction_drift_for_window(baseline_profiles, window["predictions"], effective_thresholds)
        persist_drift_results(
            drift_repo,
            alerts_repo,
            model_id,
            session_id,
            len(window["features"]),
            prediction_results,
            window.get("window_start"),
            window.get("window_end"),
        )
        if not any(result.status == DriftStatus.CRITICAL for result in prediction_results):
            continue
        if sessions_repo is None:
            return False
        trigger_prediction_diagnostics(
            sessions_repo,
            images_repo,
            drift_repo,
            alerts_repo,
            model_id,
            session_id,
            baseline_profiles,
            image_id,
            window_size,
            window,
            effective_thresholds,
        )
        return True
    return False


def trigger_prediction_diagnostics(
    sessions_repo: Repository,
    images_repo: Repository,
    drift_repo: Repository,
    alerts_repo: Repository,
    model_id: str,
    session_id: str,
    baseline_profiles: list[dict],
    trigger_image_id: str,
    window_size: int,
    trigger_window: dict,
    threshold_config: dict | None = None,
) -> None:
    effective_thresholds = normalized_threshold_config(threshold_config)
    now = datetime.now(timezone.utc).isoformat()
    sessions_repo.update(
        session_id,
        {
            "status": "paused",
            "worker_status": "idle",
            "worker_pid": None,
            "worker_heartbeat_at": None,
            "worker_lease_expires_at": now,
            "diagnostic_status": "running",
            "diagnostic_triggered_at": now,
            "diagnostic_window_start": trigger_window.get("window_start"),
            "diagnostic_window_end": trigger_window.get("window_end"),
            "diagnostic_reason": "critical_prediction_drift",
            "diagnostic_error": None,
        },
    )
    try:
        for window in historical_data_drift_windows(images_repo, session_id, trigger_image_id, window_size):
            data_results = calculate_data_drift_for_window(baseline_profiles, window["features"], effective_thresholds)
            persist_drift_results(
                drift_repo,
                alerts_repo,
                model_id,
                session_id,
                window["num_images"],
                data_results,
                window.get("window_start"),
                window.get("window_end"),
            )
        sessions_repo.update(session_id, {"diagnostic_status": "completed", "diagnostic_error": None})
    except Exception as exc:
        sessions_repo.update(
            session_id,
            {"diagnostic_status": "failed", "diagnostic_error": str(exc) or "Data drift calculation failed."},
        )


def historical_data_drift_windows(
    images_repo: Repository,
    session_id: str,
    trigger_image_id: str,
    window_size: int,
) -> list[dict]:
    rows = [row for row in images_repo.where(session_id=session_id) if (row.get("inference_status") or "complete") == "complete"]
    ordered = sorted(rows, key=lambda row: (str(image_row_time(row) or ""), str(row.get("id") or "")))
    trigger_index = next((index for index, row in enumerate(ordered) if row.get("id") == trigger_image_id), None)
    if trigger_index is None:
        return []

    windows = []
    for end_index in range(1, trigger_index + 1):
        start_index = max(0, end_index - max(2, int(window_size)) + 1)
        window_rows = ordered[start_index : end_index + 1]
        features = [row.get("feature_payload") or feature_payload_from_image_row(row) for row in window_rows]
        usable_features = [row for row in features if row]
        if len(usable_features) < 2:
            continue
        windows.append(
            {
                "features": usable_features,
                "num_images": len(window_rows),
                "window_start": image_row_time(window_rows[0]),
                "window_end": image_row_time(window_rows[-1]),
            }
        )
    return windows

def current_session_window(images_repo: Repository, predictions_repo: Repository, session_id: str, window_size: int, end_image_id: str | None = None) -> dict:
    image_rows = stable_latest_rows(images_repo, session_id, window_size, end_image_id=end_image_id)
    features = [row.get("feature_payload") or feature_payload_from_image_row(row) for row in image_rows]
    predictions: list[str] = []
    for image in image_rows:
        for prediction in predictions_repo.where(image_id=image["id"]):
            if prediction.get("predicted_class_name"):
                predictions.append(prediction["predicted_class_name"])
    chronological_rows = sorted(image_rows, key=lambda row: (str(image_row_time(row) or ""), str(row.get("id") or "")))
    return {
        "features": [row for row in features if row],
        "predictions": predictions,
        "window_start": image_row_time(chronological_rows[0]) if chronological_rows else None,
        "window_end": image_row_time(chronological_rows[-1]) if chronological_rows else None,
    }


def stable_latest_rows(images_repo: Repository, session_id: str, window_size: int, end_image_id: str | None = None) -> list[dict]:
    rows = [row for row in images_repo.where(session_id=session_id) if (row.get("inference_status") or "complete") == "complete"]
    ordered = sorted(rows, key=lambda row: (str(image_row_time(row) or ""), str(row.get("id") or "")))
    if end_image_id:
        end_index = next((index for index, row in enumerate(ordered) if row.get("id") == end_image_id), None)
        if end_index is None:
            return []
        ordered = ordered[: end_index + 1]
    return list(reversed(ordered[-window_size:]))


def image_row_time(row: dict) -> str | None:
    return row.get("created_at") or row.get("inference_completed_at")


def feature_payload_from_image_row(row: dict) -> dict:
    keys = ("brightness_mean", "brightness_std", "contrast", "sharpness", "saturation_mean", "saturation_std", "edge_density", "noise_level", "colorfulness")
    return {key: row[key] for key in keys if row.get(key) is not None}


def calculate_data_drift_for_window(baseline_profiles: list[dict], current_features: list[dict], threshold_config: dict | None = None) -> list[DriftResult]:
    thresholds = normalized_threshold_config(threshold_config)
    data_thresholds = thresholds["data"]
    feature_profile = next((row.get("metrics") for row in baseline_profiles if row.get("profile_type") == "feature_rows"), {}) or {}
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
                data_thresholds["mmd_warning"],
                status_from_mmd(mmd_score, data_thresholds),
                {
                    "calculation": "rbf_mmd",
                    "help": PROPERTY_HELP["total_data_drift"],
                    "threshold_config": thresholds,
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
                PROPERTY_DRIFT_WARNING_THRESHOLD,
                status_from_score(score),
                {
                    "property_key": key,
                    "property_label": PROPERTY_LABELS[key],
                    "help": PROPERTY_HELP[key],
                    "threshold_config": thresholds,
                    "training_average": float(training_average),
                    "production_average": float(production_average),
                    "training_reference_fixed": True,
                },
            )
        )
    return results


def calculate_prediction_drift_for_window(baseline_profiles: list[dict], current_predictions: list[str], threshold_config: dict | None = None) -> list[DriftResult]:
    if not current_predictions:
        return []
    thresholds = normalized_threshold_config(threshold_config)
    prediction_thresholds = thresholds["prediction"]
    class_profile = next((row.get("metrics") for row in baseline_profiles if row.get("profile_type") == "class_distribution"), {}) or {}
    training_counts = class_profile.get("training_counts") or {}
    production_counts = class_distribution(current_predictions)
    chi_square = chi_square_prediction_drift(training_counts, production_counts)
    p_value = chi_square.get("p_value")
    unseen_classes = chi_square.get("unseen_classes") or []
    statistic = chi_square.get("statistic")
    has_training_counts = sum(int(count) for count in training_counts.values() if isinstance(count, (int, float))) > 0
    return [
        DriftResult(
            DriftType.PREDICTION,
            "prediction_class_distribution",
            statistic,
            prediction_thresholds["p_value_warning"],
            status_from_chi_square(p_value, bool(unseen_classes), has_training_counts, prediction_thresholds),
            {
                "calculation": "chi_square_goodness_of_fit",
                "help": PROPERTY_HELP["prediction_class_distribution"],
                "threshold_config": thresholds,
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
    ]


def calculate_drift_for_window(baseline_profiles: list[dict], current_features: list[dict], current_predictions: list[str], threshold_config: dict | None = None):
    return [
        *calculate_data_drift_for_window(baseline_profiles, current_features, threshold_config),
        *calculate_prediction_drift_for_window(baseline_profiles, current_predictions, threshold_config),
    ]

def normalized_average_delta(training_average: float, production_average: float) -> float:
    denominator = max(abs(training_average), 1e-6)
    return abs(production_average - training_average) / denominator * 100.0


def status_from_score(score: float) -> DriftStatus:
    if score >= PROPERTY_DRIFT_CRITICAL_THRESHOLD:
        return DriftStatus.CRITICAL
    if score >= PROPERTY_DRIFT_WARNING_THRESHOLD:
        return DriftStatus.WARNING
    return DriftStatus.OK


def status_from_mmd(score: float, data_thresholds: dict | None = None) -> DriftStatus:
    thresholds = normalized_threshold_config({"data": data_thresholds or {}})["data"]
    if score >= thresholds["mmd_critical"]:
        return DriftStatus.CRITICAL
    if score >= thresholds["mmd_warning"]:
        return DriftStatus.WARNING
    return DriftStatus.OK


def status_from_chi_square(p_value: float | None, has_unseen_classes: bool, has_training_counts: bool, prediction_thresholds: dict | None = None) -> DriftStatus:
    thresholds = normalized_threshold_config({"prediction": prediction_thresholds or {}})["prediction"]
    if not has_training_counts:
        return DriftStatus.WARNING
    if has_unseen_classes:
        return DriftStatus.CRITICAL
    if p_value is None:
        return DriftStatus.WARNING
    if p_value < thresholds["p_value_critical"]:
        return DriftStatus.CRITICAL
    if p_value < thresholds["p_value_warning"]:
        return DriftStatus.WARNING
    return DriftStatus.OK


def normalized_threshold_config(config: dict | None = None) -> dict:
    incoming = config if isinstance(config, dict) else {}
    prediction = incoming.get("prediction") if isinstance(incoming.get("prediction"), dict) else {}
    data = incoming.get("data") if isinstance(incoming.get("data"), dict) else {}
    concept = incoming.get("concept") if isinstance(incoming.get("concept"), dict) else {}
    confidence = incoming.get("confidence") if isinstance(incoming.get("confidence"), dict) else {}
    display = incoming.get("display") if isinstance(incoming.get("display"), dict) else {}

    prediction_warning = bounded_float(prediction.get("p_value_warning"), PREDICTION_DRIFT_WARNING_P_VALUE, 0.001, 0.5)
    prediction_critical = bounded_float(prediction.get("p_value_critical"), prediction_warning * 0.2, 0.0001, prediction_warning)

    data_warning = bounded_float(data.get("mmd_warning"), TOTAL_DATA_DRIFT_WARNING_THRESHOLD, 0.001, 1.0)
    data_critical = bounded_float(data.get("mmd_critical"), data_warning * 2.5, data_warning, 2.5)

    concept_warning = bounded_float(concept.get("accuracy_warning_percent"), CONCEPT_DRIFT_WARNING_PERCENT, 1.0, 100.0)
    concept_critical = bounded_float(concept.get("accuracy_critical_percent"), concept_warning - 20.0, 0.0, concept_warning)

    confidence_medium = bounded_float(
        confidence.get("medium_threshold"),
        CONFIDENCE_MEDIUM_THRESHOLD,
        0.0,
        1.0,
    )
    confidence_good = bounded_float(
        confidence.get("good_threshold"),
        CONFIDENCE_GOOD_THRESHOLD,
        confidence_medium,
        1.0,
    )
    graph_window = int(bounded_float(display.get("graph_window"), GRAPH_WINDOW_DEFAULT, GRAPH_WINDOW_MIN, GRAPH_WINDOW_MAX))

    return {
        "prediction": {
            "p_value_warning": prediction_warning,
            "p_value_critical": prediction_critical,
        },
        "data": {
            "mmd_warning": data_warning,
            "mmd_critical": data_critical,
        },
        "concept": {
            "accuracy_warning_percent": concept_warning,
            "accuracy_critical_percent": concept_critical,
        },
        "confidence": {
            "medium_threshold": confidence_medium,
            "good_threshold": confidence_good,
        },
        "display": {
            "graph_window": graph_window,
        },
    }


def threshold_config_from_slider_values(
    prediction_warning: float,
    data_warning: float,
    concept_warning: float,
    confidence_medium: float | None = None,
    confidence_good: float | None = None,
    graph_window: int = GRAPH_WINDOW_DEFAULT,
) -> dict:
    confidence_medium = CONFIDENCE_MEDIUM_THRESHOLD if confidence_medium is None else confidence_medium
    confidence_good = CONFIDENCE_GOOD_THRESHOLD if confidence_good is None else confidence_good

    return normalized_threshold_config(
        {
            "prediction": {
                "p_value_warning": prediction_warning,
                "p_value_critical": prediction_warning * 0.2,
            },
            "data": {
                "mmd_warning": data_warning,
                "mmd_critical": data_warning * 2.5,
            },
            "concept": {
                "accuracy_warning_percent": concept_warning,
                "accuracy_critical_percent": concept_warning - 20.0,
            },
            "confidence": {
                "medium_threshold": confidence_medium,
                "good_threshold": confidence_good,
            },
            "display": {
                "graph_window": graph_window,
            },
        }
    )


def bounded_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return round(min(max(number, minimum), maximum), 6)


def persist_drift_results(
    drift_repo: Repository,
    alerts_repo: Repository,
    model_id: str,
    session_id: str,
    num_images: int,
    results,
    window_start: str | None = None,
    window_end: str | None = None,
) -> None:
    for result in results:
        if result.status == DriftStatus.OK:
            resolve_metric_alerts(alerts_repo, session_id, result.metric_name)
        drift_row = insert_drift_idempotent(
            drift_repo,
            {
                "model_id": model_id,
                "session_id": session_id,
                "window_start": window_start,
                "window_end": window_end,
                "num_images": num_images,
                "drift_type": result.drift_type.value,
                "metric_name": result.metric_name,
                "metric_value": result.metric_value,
                "threshold": result.threshold,
                "status": result.status.value,
                "details": result.details,
            },
        )
        for alert in alerts_from_drift([result]):
            upsert_unresolved_alert(alerts_repo, model_id, session_id, alert, drift_row)


def insert_drift_idempotent(drift_repo: Repository, payload: dict) -> dict | None:
    try:
        return drift_repo.insert(payload)
    except Exception:
        if payload.get("window_end"):
            existing = drift_repo.where(
                session_id=payload["session_id"],
                metric_name=payload["metric_name"],
                window_end=payload["window_end"],
            )
            return existing[0] if existing else None
        raise


def resolve_metric_alerts(alerts_repo: Repository, session_id: str, metric_name: str) -> None:
    label = METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())
    for alert in alerts_repo.where(session_id=session_id, is_resolved=False):
        if str(alert.get("title") or "").startswith(label):
            alerts_repo.update(alert["id"], {"is_resolved": True, "resolved_at": datetime.now(timezone.utc).isoformat()})


def upsert_unresolved_alert(alerts_repo: Repository, model_id: str, session_id: str, alert: dict, drift_row: dict | None) -> None:
    existing = alerts_repo.where(session_id=session_id, title=alert["title"], severity=alert["severity"], is_resolved=False)
    payload = {
        "message": alert["message"],
        "drift_result_id": drift_row.get("id") if drift_row else None,
        "last_seen_at": datetime.now(timezone.utc).isoformat(),
    }
    if existing:
        alerts_repo.update(existing[0]["id"], payload)
        return
    alerts_repo.insert(
        {
            "model_id": model_id,
            "session_id": session_id,
            "severity": alert["severity"],
            "title": alert["title"],
            **payload,
        }
    )


def record_feedback_concept_drift(client, model: dict, session: dict | None) -> None:
    if client is None or not session:
        return
    images_repo = Repository(client, "images")
    feedback_repo = Repository(client, "feedback")
    drift_repo = Repository(client, "drift_results")
    alerts_repo = Repository(client, "alerts")
    image_rows = images_repo.where(session_id=session["id"])
    feedback_types: list[str] = []
    for image in image_rows:
        latest_by_prediction = latest_feedback_by_prediction(feedback_repo.where(image_id=image["id"]))
        feedback_types.extend(row["feedback_type"] for row in latest_by_prediction if row.get("feedback_type"))
    if not feedback_types:
        return
    summary = review_summary(feedback_types)
    accuracy = summary.get("accuracy_percent")
    thresholds = normalized_threshold_config(session.get("threshold_config"))
    concept_thresholds = thresholds["concept"]
    status = concept_status_from_accuracy(accuracy, concept_thresholds)
    result = DriftResult(
        DriftType.CONCEPT,
        "human_review_summary",
        float(accuracy) if isinstance(accuracy, (int, float)) else 0.0,
        concept_thresholds["accuracy_warning_percent"],
        status,
        {"calculation": "review_accuracy", "help": PROPERTY_HELP["human_review_summary"], "threshold_config": thresholds, **summary},
    )
    persist_drift_results(drift_repo, alerts_repo, model["id"], session["id"], len(image_rows), [result])


def latest_feedback_by_prediction(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    unscoped: list[dict] = []
    for row in sorted(rows, key=lambda item: (str(item.get("created_at") or ""), str(item.get("id") or ""))):
        prediction_id = row.get("prediction_id")
        if prediction_id:
            latest[prediction_id] = row
        else:
            unscoped.append(row)
    return [*latest.values(), *unscoped]


def concept_status_from_accuracy(accuracy: float | None, concept_thresholds: dict | None = None) -> DriftStatus:
    thresholds = normalized_threshold_config({"concept": concept_thresholds or {}})["concept"]
    if not isinstance(accuracy, (int, float)):
        return DriftStatus.OK
    if accuracy < thresholds["accuracy_critical_percent"]:
        return DriftStatus.CRITICAL
    if accuracy < thresholds["accuracy_warning_percent"]:
        return DriftStatus.WARNING
    return DriftStatus.OK

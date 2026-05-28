from __future__ import annotations

from pathlib import Path

from backend.alerts.alert_rules import alerts_from_drift
from backend.common import DriftType
from backend.database.repositories import SupabaseRepository
from backend.drift.selector import categorical_prediction_drift, embedding_drift, numeric_feature_drift
from backend.features.image_properties import extract_image_features, extract_image_metadata
from backend.ingestion.image_scanner import scan_images
from backend.model.inference import run_yolo_inference
from backend.utils.serialization import to_jsonable


def process_source_once(client, model: dict, artifact: dict, source: dict, session: dict, baseline_profiles: list[dict], window_size: int = 25) -> dict:
    images_repo = SupabaseRepository(client, "images")
    predictions_repo = SupabaseRepository(client, "predictions")
    drift_repo = SupabaseRepository(client, "drift_results")
    alerts_repo = SupabaseRepository(client, "alerts")
    source_path = Path(source.get("source_uri") or "")
    if not source_path.exists():
        return {"processed": 0, "reason": "source_path_missing"}
    paths = scan_images(source_path)
    processed = 0
    current_features: list[dict] = []
    current_predictions: list[str] = []
    for image_path in paths:
        metadata = extract_image_metadata(image_path)
        existing = images_repo.where(model_id=model["id"], content_hash=metadata.content_hash)
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
    if processed >= max(2, window_size // 2):
        results = calculate_drift_for_window(baseline_profiles, current_features, current_predictions)
        for result in results:
            drift_row = drift_repo.insert({"model_id": model["id"], "session_id": session["id"], "num_images": processed, "drift_type": result.drift_type.value, "metric_name": result.metric_name, "metric_value": result.metric_value, "threshold": result.threshold, "status": result.status.value, "details": result.details})
            for alert in alerts_from_drift([result]):
                alerts_repo.insert({"model_id": model["id"], "session_id": session["id"], "severity": alert["severity"], "title": alert["title"], "message": alert["message"], "drift_result_id": drift_row.get("id") if drift_row else None})
    return {"processed": processed}


def normalize_image_source(source_type: str) -> str:
    if source_type == "test_folder":
        return "test"
    if source_type in {"manual_upload", "watched_folder", "supabase_bucket"}:
        return source_type
    if source_type in {"rtsp", "usb_camera"}:
        return "camera"
    return "production"


def calculate_drift_for_window(baseline_profiles: list[dict], current_features: list[dict], current_predictions: list[str]):
    image_stats = next((row.get("metrics") for row in baseline_profiles if row.get("profile_type") == "image_stats"), {})
    baseline_feature_rows = next((row.get("metrics", {}).get("features") for row in baseline_profiles if row.get("profile_type") == "feature_rows"), None)
    results = []
    if baseline_feature_rows:
        for key in ("brightness_mean", "contrast", "sharpness", "saturation_mean", "edge_density"):
            results.extend(numeric_feature_drift(key, [row[key] for row in baseline_feature_rows], [row[key] for row in current_features]))
        results.extend(embedding_drift([row["embedding"] for row in baseline_feature_rows if row.get("embedding")], [row["embedding"] for row in current_features if row.get("embedding")]))
    elif image_stats:
        for key, stats in image_stats.items():
            current_values = [row[key] for row in current_features if key in row]
            if current_values and abs(sum(current_values) / len(current_values) - stats.get("mean", 0)) > 0.1:
                from backend.common import DriftResult, DriftStatus
                results.append(DriftResult(DriftType.DATA, f"{key}_mean_delta", abs(sum(current_values) / len(current_values) - stats.get("mean", 0)), 0.1, DriftStatus.WARNING, {"baseline_mean": stats.get("mean")}))
    if current_predictions:
        baseline_predictions = next((row.get("metrics", {}).get("predicted_classes") for row in baseline_profiles if row.get("profile_type") == "predictions"), None)
        if baseline_predictions:
            results.append(categorical_prediction_drift("predicted_class_distribution", baseline_predictions, current_predictions))
    return results

from __future__ import annotations

from pathlib import Path

from background_worker.data_drift_tests.image_property_drift import property_averages
from background_worker.data_drift_tests.mmd import feature_vectors
from background_worker.prediction_drift_tests.class_distribution_drift import class_distribution
from background_worker.image_features import extract_image_features, extract_image_metadata
from background_worker.inference import run_yolo_inference
from background_worker.source_scanner import scan_images
from core.types import DatasetMetadata, SplitName
from database.serialization import to_jsonable


def build_baseline_profile(dataset: DatasetMetadata, artifact_path: Path | None = None, task_type: str | None = None) -> dict:
    images: list[Path] = []
    for split_name in (SplitName.TRAIN, SplitName.VAL):
        split = dataset.splits[split_name]
        images.extend(scan_images(split.images_path))
    image_rows = []
    feature_rows = []
    training_feature_rows = []
    prediction_rows = []
    for image_path in images:
        split = split_for_image_path(dataset, image_path)
        metadata = extract_image_metadata(image_path, split=split)
        features = extract_image_features(image_path)
        image_rows.append(to_jsonable(metadata))
        feature_row = to_jsonable(features)
        feature_rows.append(feature_row)
        if split == SplitName.TRAIN:
            training_feature_rows.append(feature_row)
        if artifact_path and task_type:
            predictions = run_yolo_inference(artifact_path, image_path, task_type)
            for prediction in predictions:
                prediction_rows.append({"image_path": str(image_path), **to_jsonable(prediction)})
    return {
        "dataset_profile": {
            "num_baseline_images": len(images),
            "splits": {split.value: to_jsonable(paths) for split, paths in dataset.splits.items()},
            "classes": to_jsonable(dataset.classes),
            "test_excluded": True,
        },
        "image_stats": summarize_features(training_feature_rows),
        "feature_rows": {
            "features": feature_rows,
            "training_features": training_feature_rows,
            "training_averages": property_averages(training_feature_rows),
            "training_vectors": feature_vectors(training_feature_rows),
        },
        "class_distribution": summarize_class_distribution(dataset),
        "predictions": summarize_predictions(prediction_rows),
        "images": image_rows,
        "features": feature_rows,
    }


def split_for_image_path(dataset: DatasetMetadata, image_path: Path) -> SplitName:
    for split_name in (SplitName.TRAIN, SplitName.VAL):
        split_path = dataset.splits[split_name].images_path
        if split_path and image_path.is_relative_to(split_path):
            return split_name
    return SplitName.VAL


def summarize_features(feature_rows: list[dict]) -> dict:
    if not feature_rows:
        return {}
    numeric_keys = ["brightness_mean", "brightness_std", "contrast", "sharpness", "saturation_mean", "saturation_std", "edge_density", "noise_level", "colorfulness"]
    summary = {}
    for key in numeric_keys:
        values = [row[key] for row in feature_rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    return summary


def summarize_class_distribution(dataset: DatasetMetadata) -> dict:
    counts = training_class_counts(dataset)
    return {
        "classes": [item.class_name for item in dataset.classes],
        "num_classes": len(dataset.classes),
        "training_counts": counts,
        "training_percentages": class_distribution_percentages(counts),
    }


def training_class_counts(dataset: DatasetMetadata) -> dict[str, int]:
    train_split = dataset.splits[SplitName.TRAIN]
    if dataset.selected_task_type.value == "classification":
        return classification_training_counts(train_split.images_path)
    return detection_training_counts(train_split.labels_path, [item.class_name for item in dataset.classes])


def classification_training_counts(images_path: Path | None) -> dict[str, int]:
    if not images_path or not images_path.exists():
        return {}
    labels = []
    for image_path in scan_images(images_path):
        try:
            labels.append(image_path.relative_to(images_path).parts[0])
        except IndexError:
            continue
    return class_distribution(labels)


def detection_training_counts(labels_path: Path | None, class_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    if not labels_path or not labels_path.exists():
        return counts
    for label_path in labels_path.rglob("*.txt"):
        if label_path.name == "classes.txt" or ".cache" in label_path.name or label_path.name.startswith("."):
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if 0 <= class_id < len(class_names):
                counts[class_names[class_id]] = counts.get(class_names[class_id], 0) + 1
    return counts


def class_distribution_percentages(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {label: 0.0 for label in counts}
    return {label: count / total * 100.0 for label, count in counts.items()}


def summarize_predictions(prediction_rows: list[dict]) -> dict:
    confidences = [row["confidence"] for row in prediction_rows if isinstance(row.get("confidence"), (int, float))]
    predicted_classes = [row["predicted_class_name"] for row in prediction_rows if row.get("predicted_class_name")]
    detection_rows = [row for row in prediction_rows if row.get("task_type") == "object_detection"]
    classification_rows = [row for row in prediction_rows if row.get("task_type") == "classification"]
    return {
        "prediction_count": len(prediction_rows),
        "predicted_classes": predicted_classes,
        "average_confidence": sum(confidences) / len(confidences) if confidences else None,
        "min_confidence": min(confidences) if confidences else None,
        "max_confidence": max(confidences) if confidences else None,
        "object_detection_count": len(detection_rows),
        "classification_count": len(classification_rows),
        "predictions": prediction_rows,
    }

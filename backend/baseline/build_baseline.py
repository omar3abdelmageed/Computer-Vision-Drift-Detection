from __future__ import annotations

from pathlib import Path

from backend.common import DatasetMetadata, SplitName
from backend.features.image_properties import extract_image_features, extract_image_metadata
from backend.ingestion.image_scanner import scan_images
from backend.utils.serialization import to_jsonable


def build_baseline_profile(dataset: DatasetMetadata) -> dict:
    images: list[Path] = []
    for split_name in (SplitName.TRAIN, SplitName.VAL):
        split = dataset.splits[split_name]
        images.extend(scan_images(split.images_path))
    image_rows = []
    feature_rows = []
    for image_path in images:
        split = SplitName.TRAIN if "/train/" in image_path.as_posix() else SplitName.VAL
        metadata = extract_image_metadata(image_path, split=split)
        features = extract_image_features(image_path)
        image_rows.append(to_jsonable(metadata))
        feature_rows.append(to_jsonable(features))
    return {
        "dataset_profile": {
            "num_baseline_images": len(images),
            "splits": {split.value: to_jsonable(paths) for split, paths in dataset.splits.items()},
            "classes": to_jsonable(dataset.classes),
            "test_excluded": True,
        },
        "image_stats": summarize_features(feature_rows),
        "feature_rows": {"features": feature_rows},
        "class_distribution": summarize_class_distribution(dataset),
        "images": image_rows,
        "features": feature_rows,
    }


def summarize_features(feature_rows: list[dict]) -> dict:
    if not feature_rows:
        return {}
    numeric_keys = ["brightness_mean", "brightness_std", "contrast", "sharpness", "saturation_mean", "saturation_std", "edge_density"]
    summary = {}
    for key in numeric_keys:
        values = [row[key] for row in feature_rows]
        summary[key] = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    return summary


def summarize_class_distribution(dataset: DatasetMetadata) -> dict:
    return {"classes": [item.class_name for item in dataset.classes], "num_classes": len(dataset.classes)}

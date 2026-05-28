from __future__ import annotations

from pathlib import Path

from backend.common import DetectedTaskType


def detect_task_type(dataset_root: Path) -> DetectedTaskType:
    has_yaml = any(dataset_root.rglob("*.yaml")) or any(dataset_root.rglob("*.yml"))
    has_yolo_labels = any(p for p in dataset_root.rglob("*.txt") if p.name != "classes.txt" and ".cache" not in p.name)
    has_class_folders = any((dataset_root / split).exists() and any(c.is_dir() for c in (dataset_root / split).iterdir()) for split in ("train", "val"))
    if has_yaml and has_yolo_labels and has_class_folders:
        return DetectedTaskType.AMBIGUOUS
    if has_yaml and has_yolo_labels:
        return DetectedTaskType.OBJECT_DETECTION
    if has_class_folders or ((dataset_root / "images" / "train").exists()):
        return DetectedTaskType.CLASSIFICATION
    return DetectedTaskType.INVALID

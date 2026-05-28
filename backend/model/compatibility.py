from __future__ import annotations

from backend.common import DatasetMetadata, TaskType, ValidationStatus
from backend.model.yolo import ModelArtifactMetadata


def check_compatibility(dataset: DatasetMetadata, artifact: ModelArtifactMetadata) -> dict:
    issues: list[dict] = []
    selected = dataset.selected_task_type
    yolo_task = artifact.model_task
    if selected == TaskType.OBJECT_DETECTION and yolo_task not in {"detect", "unknown"}:
        issues.append({"severity": "invalid", "code": "task_mismatch", "message": f"Selected detection but artifact task is {yolo_task}."})
    if selected == TaskType.CLASSIFICATION and yolo_task not in {"classify", "unknown"}:
        issues.append({"severity": "invalid", "code": "task_mismatch", "message": f"Selected classification but artifact task is {yolo_task}."})
    dataset_classes = [item.class_name for item in dataset.classes]
    if artifact.num_classes and len(dataset_classes) != artifact.num_classes:
        issues.append({"severity": "invalid", "code": "class_count_mismatch", "message": "Dataset and model class counts differ.", "dataset": len(dataset_classes), "model": artifact.num_classes})
    if artifact.class_names and dataset_classes and artifact.class_names != dataset_classes:
        issues.append({"severity": "warning", "code": "class_name_mismatch", "message": "Dataset and model class names differ.", "dataset": dataset_classes, "model": artifact.class_names})
    invalid = any(issue["severity"] == "invalid" for issue in issues)
    warning = any(issue["severity"] == "warning" for issue in issues)
    status = ValidationStatus.INVALID.value if invalid else ValidationStatus.WARNING.value if warning else ValidationStatus.VALID.value
    return {"is_compatible": not invalid, "compatibility_status": status, "compatibility_details": {"issues": issues}}

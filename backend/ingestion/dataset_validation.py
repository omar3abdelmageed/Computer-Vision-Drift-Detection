from __future__ import annotations

from pathlib import Path

from backend.common import TaskType
from backend.ingestion.classification_validator import validate_classification_dataset
from backend.ingestion.detection_validator import validate_detection_dataset


def validate_dataset(dataset_root: Path, selected_task_type: TaskType | str):
    task = TaskType(selected_task_type)
    if task == TaskType.OBJECT_DETECTION:
        return validate_detection_dataset(dataset_root)
    return validate_classification_dataset(dataset_root)

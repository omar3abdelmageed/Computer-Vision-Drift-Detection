from __future__ import annotations

from pathlib import Path

from core.types import TaskType
from model_management.classification_validator import validate_classification_dataset
from model_management.detection_validator import validate_detection_dataset


def validate_dataset(dataset_root: Path, selected_task_type: TaskType | str):
    task = TaskType(selected_task_type)
    if task == TaskType.OBJECT_DETECTION:
        return validate_detection_dataset(dataset_root)
    return validate_classification_dataset(dataset_root)

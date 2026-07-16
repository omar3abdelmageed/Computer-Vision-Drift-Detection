from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelArtifactMetadata:
    local_path: Path
    artifact_name: str
    model_task: str
    class_names: list[str]
    num_classes: int
    input_size: int | None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


def load_yolo_metadata(model_path: Path) -> ModelArtifactMetadata:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required to load YOLO model artifacts.") from exc
    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        class_names = [str(names[key]) for key in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    else:
        class_names = [str(item) for item in names]
    task = getattr(model, "task", None) or "unknown"
    overrides = getattr(model, "overrides", {}) or {}
    input_size = overrides.get("imgsz")
    if isinstance(input_size, (list, tuple)):
        input_size = input_size[0] if input_size else None
    return ModelArtifactMetadata(
        local_path=model_path,
        artifact_name=model_path.name,
        model_task=str(task),
        class_names=class_names,
        num_classes=len(class_names),
        input_size=int(input_size) if input_size else None,
        raw_metadata={"task": task, "names": names, "overrides": overrides},
    )

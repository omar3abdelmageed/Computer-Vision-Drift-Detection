from __future__ import annotations

from pathlib import Path

from backend.common import ClassMapping, DatasetLayout, DatasetMetadata, DatasetSplitPaths, DetectedTaskType, SplitName, TaskType, ValidationIssue, ValidationStatus
from backend.ingestion.image_scanner import scan_images


def validate_classification_dataset(dataset_root: Path) -> DatasetMetadata:
    layout, splits, class_names, issues = resolve_classification_layout(dataset_root)
    for required in (SplitName.TRAIN, SplitName.VAL):
        if not splits[required].has_images:
            issues.append(ValidationIssue(ValidationStatus.INVALID, f"missing_{required.value}_classification_images", f"Missing {required.value} class images.", str(dataset_root)))
    if splits[SplitName.VAL].has_images:
        validate_class_alignment(splits[SplitName.TRAIN].images_path, splits[SplitName.VAL].images_path, issues)
    if splits[SplitName.TEST].has_labels:
        validate_class_alignment(splits[SplitName.TRAIN].images_path, splits[SplitName.TEST].images_path, issues)
    status = ValidationStatus.INVALID if any(i.severity == ValidationStatus.INVALID for i in issues) else ValidationStatus.VALID
    return DatasetMetadata(
        dataset_root=dataset_root,
        selected_task_type=TaskType.CLASSIFICATION,
        detected_task_type=DetectedTaskType.CLASSIFICATION if class_names else DetectedTaskType.INVALID,
        layout=layout,
        selected_yaml_path=None,
        yaml_candidates=[],
        class_source="folder_names" if class_names else "unknown",
        classes=[ClassMapping(i, name) for i, name in enumerate(class_names)],
        splits=splits,
        has_test_split=splits[SplitName.TEST].has_images,
        test_has_labels=splits[SplitName.TEST].has_labels,
        supported_image_count=sum(split.image_count for split in splits.values()),
        unsupported_file_count=count_unsupported_files(dataset_root),
        validation_status=status,
        issues=issues,
    )


def resolve_classification_layout(root: Path) -> tuple[DatasetLayout, dict[SplitName, DatasetSplitPaths], list[str], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if (root / "train").exists() or (root / "val").exists():
        layout = DatasetLayout.CLASSIFICATION_SPLIT_FIRST
        base = root
    elif (root / "images" / "train").exists() or (root / "images" / "val").exists():
        layout = DatasetLayout.CLASSIFICATION_IMAGES_ROOT
        base = root / "images"
    else:
        layout = DatasetLayout.UNKNOWN
        base = root
        issues.append(ValidationIssue(ValidationStatus.INVALID, "unknown_classification_layout", "Could not find supported classification layout.", str(root)))
    splits: dict[SplitName, DatasetSplitPaths] = {}
    for split in SplitName:
        path = base / split.value
        images = scan_images(path)
        has_class_folders = any(child.is_dir() and not child.name.startswith(".") for child in path.iterdir()) if path.exists() else False
        splits[split] = DatasetSplitPaths(
            split=split,
            images_path=path,
            labels_path=None,
            has_images=bool(images),
            has_labels=has_class_folders,
            image_count=len(images),
            label_count=0,
        )
    train_path = splits[SplitName.TRAIN].images_path
    class_names = sorted(child.name for child in train_path.iterdir() if child.is_dir() and not child.name.startswith(".")) if train_path and train_path.exists() else []
    return layout, splits, class_names, issues


def validate_class_alignment(train_path: Path | None, other_path: Path | None, issues: list[ValidationIssue]) -> None:
    if not train_path or not other_path or not train_path.exists() or not other_path.exists():
        return
    train_classes = {p.name for p in train_path.iterdir() if p.is_dir()}
    other_classes = {p.name for p in other_path.iterdir() if p.is_dir()}
    if other_classes and train_classes != other_classes:
        issues.append(ValidationIssue(ValidationStatus.WARNING, "class_folder_mismatch", "Class folders do not match the train split.", str(other_path), {"train_classes": sorted(train_classes), "other_classes": sorted(other_classes)}))


def count_unsupported_files(root: Path) -> int:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    ignored = {".txt", ".cache", ".yaml", ".yml"}
    return sum(1 for path in root.rglob("*") if path.is_file() and not path.name.startswith(".") and path.suffix.lower() not in supported and path.suffix.lower() not in ignored)

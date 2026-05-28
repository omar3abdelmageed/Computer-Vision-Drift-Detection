from __future__ import annotations

from pathlib import Path

from backend.common import (
    ClassMapping,
    DatasetLayout,
    DatasetMetadata,
    DatasetSplitPaths,
    DetectedTaskType,
    DetectionLabel,
    SplitName,
    TaskType,
    ValidationIssue,
    ValidationStatus,
)
from backend.ingestion.image_scanner import scan_images
from backend.ingestion.yaml_discovery import discover_yamls, normalize_names


def validate_detection_dataset(dataset_root: Path) -> DatasetMetadata:
    issues: list[ValidationIssue] = []
    yaml_candidates = discover_yamls(dataset_root)
    selected_yaml = next((candidate for candidate in yaml_candidates if candidate.is_valid), None)
    if selected_yaml is None:
        issues.append(ValidationIssue(ValidationStatus.INVALID, "missing_valid_detection_yaml", "No valid detection YAML found.", str(dataset_root)))
        class_names: list[str] = []
    else:
        class_names = normalize_names(selected_yaml.parsed_content.get("names"))
        nc = selected_yaml.parsed_content.get("nc", len(class_names))
        if int(nc) != len(class_names):
            issues.append(ValidationIssue(ValidationStatus.INVALID, "nc_names_mismatch", "YAML nc does not match names length.", str(selected_yaml.path)))

    layout, splits = resolve_detection_layout(dataset_root)
    for required in (SplitName.TRAIN, SplitName.VAL):
        if not splits[required].has_images:
            issues.append(ValidationIssue(ValidationStatus.INVALID, f"missing_{required.value}_images", f"Missing {required.value} images.", str(dataset_root)))
        if not splits[required].has_labels:
            issues.append(ValidationIssue(ValidationStatus.INVALID, f"missing_{required.value}_labels", f"Missing {required.value} labels.", str(dataset_root)))

    for split_paths in splits.values():
        issues.extend(validate_detection_split(split_paths, class_names))

    status = ValidationStatus.INVALID if any(i.severity == ValidationStatus.INVALID for i in issues) else ValidationStatus.VALID
    image_count = sum(split.image_count for split in splits.values())
    return DatasetMetadata(
        dataset_root=dataset_root,
        selected_task_type=TaskType.OBJECT_DETECTION,
        detected_task_type=DetectedTaskType.OBJECT_DETECTION if class_names else DetectedTaskType.INVALID,
        layout=layout,
        selected_yaml_path=selected_yaml.path if selected_yaml else None,
        yaml_candidates=yaml_candidates,
        class_source="yaml_names" if class_names else "unknown",
        classes=[ClassMapping(i, name) for i, name in enumerate(class_names)],
        splits=splits,
        has_test_split=splits[SplitName.TEST].has_images,
        test_has_labels=splits[SplitName.TEST].has_labels,
        supported_image_count=image_count,
        unsupported_file_count=count_unsupported_files(dataset_root),
        validation_status=status,
        issues=issues,
    )


def resolve_detection_layout(root: Path) -> tuple[DatasetLayout, dict[SplitName, DatasetSplitPaths]]:
    images_root = root / "images"
    labels_root = root / "labels"
    if (images_root / "train").exists() or (images_root / "val").exists():
        layout = DatasetLayout.DETECTION_IMAGES_LABELS_ROOT
        splits = {
            split: DatasetSplitPaths(
                split=split,
                images_path=images_root / split.value,
                labels_path=labels_root / split.value,
            )
            for split in SplitName
        }
    else:
        layout = DatasetLayout.DETECTION_SPLIT_FIRST if (root / "train" / "images").exists() else DatasetLayout.UNKNOWN
        splits = {
            split: DatasetSplitPaths(
                split=split,
                images_path=root / split.value / "images",
                labels_path=root / split.value / "labels",
            )
            for split in SplitName
        }
    return layout, hydrate_split_counts(splits)


def hydrate_split_counts(splits: dict[SplitName, DatasetSplitPaths]) -> dict[SplitName, DatasetSplitPaths]:
    for split in splits.values():
        images = scan_images(split.images_path)
        labels = [p for p in (split.labels_path.rglob("*.txt") if split.labels_path and split.labels_path.exists() else []) if not p.name.startswith(".")]
        labels = [p for p in labels if p.name != "classes.txt" and ".cache" not in p.name]
        split.has_images = bool(images)
        split.has_labels = bool(labels)
        split.image_count = len(images)
        split.label_count = len(labels)
    return splits


def validate_detection_split(split_paths: DatasetSplitPaths, class_names: list[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not split_paths.images_path or not split_paths.labels_path or not split_paths.images_path.exists():
        return issues
    images = scan_images(split_paths.images_path)
    labels = {
        p.stem: p
        for p in split_paths.labels_path.rglob("*.txt")
        if p.name != "classes.txt" and ".cache" not in p.name and not p.name.startswith(".")
    } if split_paths.labels_path.exists() else {}
    image_stems = {p.stem for p in images}
    for image in images:
        if image.stem not in labels:
            issues.append(ValidationIssue(ValidationStatus.INVALID, "missing_label_for_image", "Image has no matching YOLO label file.", str(image)))
    for stem, label_path in labels.items():
        if stem not in image_stems:
            issues.append(ValidationIssue(ValidationStatus.WARNING, "label_without_image", "Label has no matching image.", str(label_path)))
        _, label_issues = parse_detection_label_file(label_path, class_names)
        issues.extend(label_issues)
    return issues


def parse_detection_label_file(label_path: Path, class_names: list[str]) -> tuple[list[DetectionLabel], list[ValidationIssue]]:
    labels: list[DetectionLabel] = []
    issues: list[ValidationIssue] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(ValidationIssue(ValidationStatus.INVALID, "invalid_yolo_row_length", "YOLO detection label rows must have exactly 5 values.", str(label_path), {"line_number": line_number}))
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            issues.append(ValidationIssue(ValidationStatus.INVALID, "invalid_yolo_value_type", "YOLO row contains invalid numeric values.", str(label_path), {"line_number": line_number}))
            continue
        if class_id < 0 or class_id >= len(class_names):
            issues.append(ValidationIssue(ValidationStatus.INVALID, "class_id_out_of_range", "Class id is outside the valid class range.", str(label_path), {"line_number": line_number, "class_id": class_id}))
            continue
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1) or width <= 0 or height <= 0:
            issues.append(ValidationIssue(ValidationStatus.INVALID, "invalid_box_coordinates", "YOLO normalized coordinates must be within [0, 1] and box size must be positive.", str(label_path), {"line_number": line_number}))
            continue
        labels.append(DetectionLabel(class_id, class_names[class_id], x_center, y_center, width, height))
    return labels, issues


def count_unsupported_files(root: Path) -> int:
    ignored_suffixes = {".yaml", ".yml", ".txt", ".cache"}
    return sum(1 for path in root.rglob("*") if path.is_file() and not path.name.startswith(".") and path.suffix.lower() not in ignored_suffixes and path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

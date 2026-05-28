from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TaskType(str, Enum):
    OBJECT_DETECTION = "object_detection"
    CLASSIFICATION = "classification"


class DetectedTaskType(str, Enum):
    OBJECT_DETECTION = "object_detection"
    CLASSIFICATION = "classification"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


class DatasetLayout(str, Enum):
    DETECTION_IMAGES_LABELS_ROOT = "detection_images_labels_root"
    DETECTION_SPLIT_FIRST = "detection_split_first"
    CLASSIFICATION_SPLIT_FIRST = "classification_split_first"
    CLASSIFICATION_IMAGES_ROOT = "classification_images_root"
    CLASSIFICATION_UNLABELED_TEST = "classification_unlabeled_test"
    UNKNOWN = "unknown"


class SplitName(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ValidationStatus(str, Enum):
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


class BaselineStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceType(str, Enum):
    TEST_FOLDER = "test_folder"
    MANUAL_UPLOAD = "manual_upload"
    WATCHED_FOLDER = "watched_folder"
    SUPABASE_BUCKET = "supabase_bucket"
    RTSP = "rtsp"
    USB_CAMERA = "usb_camera"


class SourceMode(str, Enum):
    INITIAL_EVALUATION = "initial_evaluation"
    LIVE_MONITORING = "live_monitoring"
    MANUAL_BATCH = "manual_batch"


class SessionStatus(str, Enum):
    CREATED = "created"
    WAITING_FOR_LIVE_DATA = "waiting_for_live_data"
    RUNNING_TEST_EVALUATION = "running_test_evaluation"
    RUNNING_LIVE_MONITORING = "running_live_monitoring"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class DriftType(str, Enum):
    DATA = "data"
    PREDICTION = "prediction"
    CONCEPT = "concept"


class DriftStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class FeedbackType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    CORRECTED_LABEL = "corrected_label"
    CORRECTED_BOX = "corrected_box"
    MISSED_OBJECT = "missed_object"
    FALSE_POSITIVE = "false_positive"
    WRONG_CLASS = "wrong_class"


@dataclass
class ValidationIssue:
    severity: ValidationStatus
    code: str
    message: str
    path: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class YamlCandidate:
    path: Path
    filename: str
    is_valid: bool
    parsed_content: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class DatasetSplitPaths:
    split: SplitName
    images_path: Optional[Path]
    labels_path: Optional[Path] = None
    has_images: bool = False
    has_labels: bool = False
    image_count: int = 0
    label_count: int = 0


@dataclass
class ClassMapping:
    class_id: int
    class_name: str


@dataclass
class DatasetMetadata:
    dataset_root: Path
    selected_task_type: TaskType
    detected_task_type: DetectedTaskType
    layout: DatasetLayout
    selected_yaml_path: Optional[Path]
    yaml_candidates: list[YamlCandidate]
    class_source: str
    classes: list[ClassMapping]
    splits: dict[SplitName, DatasetSplitPaths]
    has_test_split: bool
    test_has_labels: bool
    supported_image_count: int
    unsupported_file_count: int
    validation_status: ValidationStatus
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class ImageMetadata:
    path: Path
    filename: str
    split: Optional[SplitName]
    content_hash: Optional[str]
    image_format: str
    color_mode: str
    num_channels: Optional[int]
    bit_depth: Optional[int]
    width: int
    height: int
    aspect_ratio: float
    file_size_bytes: int


@dataclass
class ImageFeatures:
    brightness_mean: float
    brightness_std: float
    contrast: float
    sharpness: float
    saturation_mean: float
    saturation_std: float
    edge_density: float
    rgb_channel_means: list[float]
    rgb_channel_stds: list[float]
    embedding: Optional[list[float]] = None


@dataclass
class DetectionLabel:
    class_id: int
    class_name: Optional[str]
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ClassificationLabel:
    class_id: int
    class_name: str


@dataclass
class DriftResult:
    drift_type: DriftType
    metric_name: str
    metric_value: float
    threshold: float
    status: DriftStatus
    details: dict[str, Any] = field(default_factory=dict)

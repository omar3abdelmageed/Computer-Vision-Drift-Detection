from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image
import py7zr

from backend.common import TaskType, ValidationStatus
from backend.database.storage import StorageUploadError, upload_file
from backend.database.repositories import SupabaseRepository
from backend.baseline.persistence import save_baseline_profiles
from backend.drift.statistical_tests import population_stability_index, rbf_mmd
from backend.ingestion.classification_validator import validate_classification_dataset
from backend.ingestion.detection_validator import parse_detection_label_file, validate_detection_dataset
from backend.ingestion.extract_zip import extract_dataset_archive, extract_dataset_zip, find_dataset_root
from backend.registration.local_files import resolve_dataset_root, resolve_model_artifact


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(128, 120, 90)).save(path)


def test_find_dataset_root_single_folder(tmp_path):
    root = tmp_path / "extract"
    (root / "dataset").mkdir(parents=True)
    assert find_dataset_root(root) == root / "dataset"


def test_extract_dataset_zip_single_folder(tmp_path):
    source = tmp_path / "source" / "dataset"
    source.mkdir(parents=True)
    (source / "data.yaml").write_text("names: [a]\ntrain: images/train\nval: images/val\nnc: 1\n")
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(source / "data.yaml", "dataset/data.yaml")
    extracted = extract_dataset_zip(zip_path, tmp_path / "workspace", "model-1")
    assert extracted.name == "dataset"


def test_extract_dataset_7z_single_folder(tmp_path):
    source = tmp_path / "source" / "dataset"
    source.mkdir(parents=True)
    (source / "data.yaml").write_text("names: [a]\ntrain: images/train\nval: images/val\nnc: 1\n")
    archive_path = tmp_path / "dataset.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(source / "data.yaml", "dataset/data.yaml")
    extracted = extract_dataset_archive(archive_path, tmp_path / "workspace", "model-1")
    assert extracted.name == "dataset"


def test_parse_detection_label_file_validates_rows(tmp_path):
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1\n")
    labels, issues = parse_detection_label_file(label, ["ok"])
    assert len(labels) == 1
    assert issues[0].code == "invalid_yolo_row_length"


def test_detection_dataset_images_labels_layout(tmp_path):
    root = tmp_path / "det"
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "data.yaml").write_text("train: images/train\nval: images/val\nnc: 1\nnames: ['weld']\n")
    for split in ("train", "val"):
        make_image(root / "images" / split / f"{split}.bmp")
        (root / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    result = validate_detection_dataset(root)
    assert result.selected_task_type == TaskType.OBJECT_DETECTION
    assert result.validation_status == ValidationStatus.VALID


def test_classification_dataset_labeled_and_unlabeled_test(tmp_path):
    root = tmp_path / "cls"
    for split in ("train", "val"):
        make_image(root / split / "Good" / f"{split}.jpg")
    make_image(root / "test" / "image001.jpg")
    result = validate_classification_dataset(root)
    assert result.selected_task_type == TaskType.CLASSIFICATION
    assert result.has_test_split
    assert not result.test_has_labels
    assert result.validation_status == ValidationStatus.VALID


def test_drift_fallbacks_work():
    psi = population_stability_index([0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9])
    mmd = rbf_mmd([[0.0, 0.0], [0.1, 0.1]], [[2.0, 2.0], [2.1, 2.1]], threshold=0.01)
    assert psi.statistic >= 0
    assert mmd.drift_detected


def test_upload_file_wraps_supabase_payload_limit(tmp_path):
    class PayloadTooLarge(Exception):
        status = 413

    class FakeBucket:
        def upload(self, *args, **kwargs):
            raise PayloadTooLarge("too large")

    class FakeStorage:
        def from_(self, bucket):
            assert bucket == "datasets"
            return FakeBucket()

    class FakeClient:
        storage = FakeStorage()

    local_path = tmp_path / "dataset.zip"
    local_path.write_bytes(b"x" * 1024)

    try:
        upload_file(FakeClient(), "datasets", "model/raw_upload.zip", local_path)
    except StorageUploadError as exc:
        assert exc.status == 413
        assert "exceeds the Supabase Storage size limit" in str(exc)
    else:
        raise AssertionError("Expected StorageUploadError")


def test_resolve_dataset_root_accepts_existing_directory(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    assert resolve_dataset_root(str(dataset_root)) == dataset_root.resolve()


def test_resolve_dataset_root_rejects_file(tmp_path):
    dataset_file = tmp_path / "dataset.zip"
    dataset_file.write_text("not a directory")

    try:
        resolve_dataset_root(str(dataset_file))
    except ValueError as exc:
        assert "must be a directory" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_resolve_model_artifact_accepts_pt_file(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    assert resolve_model_artifact(str(model_path)) == model_path.resolve()


def test_resolve_model_artifact_rejects_non_pt_file(tmp_path):
    model_path = tmp_path / "best.onnx"
    model_path.write_bytes(b"model")

    try:
        resolve_model_artifact(str(model_path))
    except ValueError as exc:
        assert "must be a .pt file" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_save_baseline_profiles_updates_existing_and_inserts_missing():
    class FakeRepository:
        def __init__(self):
            self.rows = [
                {"id": "row-1", "model_id": "model-1", "dataset_id": "dataset-1", "artifact_id": "artifact-1", "profile_type": "dataset_profile", "metrics": {"old": True}},
                {"id": "row-2", "model_id": "model-1", "dataset_id": "dataset-1", "artifact_id": "artifact-1", "profile_type": "image_stats", "metrics": {"old": True}},
                {"id": "row-3", "model_id": "model-1", "dataset_id": "dataset-old", "artifact_id": "artifact-1", "profile_type": "dataset_profile", "metrics": {"old": "dataset"}},
            ]
            self.inserted = []
            self.updated = []

        def where(self, **filters):
            return [row for row in self.rows if all(row.get(key) == value for key, value in filters.items())]

        def update(self, row_id, payload):
            self.updated.append((row_id, payload))
            return {"id": row_id, **payload}

        def insert(self, payload):
            self.inserted.append(payload)
            return {"id": f"row-{len(self.rows) + len(self.inserted)}", **payload}

    repository = FakeRepository()
    save_baseline_profiles(
        repository,
        "model-1",
        "dataset-1",
        "artifact-1",
        {
            "dataset_profile": {"new": "dataset"},
            "image_stats": {"new": "stats"},
            "feature_rows": {"features": []},
            "class_distribution": {"classes": []},
        },
    )

    assert repository.updated == [
        ("row-1", {"metrics": {"new": "dataset"}, "dataset_id": "dataset-1", "artifact_id": "artifact-1"}),
        ("row-2", {"metrics": {"new": "stats"}, "dataset_id": "dataset-1", "artifact_id": "artifact-1"}),
    ]
    assert [row["profile_type"] for row in repository.inserted] == ["feature_rows", "class_distribution"]
    assert all(row["dataset_id"] == "dataset-1" and row["artifact_id"] == "artifact-1" for row in repository.inserted)


def test_latest_where_orders_and_limits_query():
    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def select(self, columns):
            assert columns == "*"
            return self

        def eq(self, key, value):
            self.rows = [row for row in self.rows if row.get(key) == value]
            return self

        def order(self, key, desc=True):
            self.rows = sorted(self.rows, key=lambda row: row.get(key), reverse=desc)
            return self

        def limit(self, count):
            self.rows = self.rows[:count]
            return self

        def execute(self):
            return type("Result", (), {"data": self.rows})()

    class FakeClient:
        def __init__(self):
            self.rows = [
                {"id": "old", "model_id": "model-1", "created_at": "2024-01-01T00:00:00Z"},
                {"id": "new", "model_id": "model-1", "created_at": "2024-02-01T00:00:00Z"},
                {"id": "other", "model_id": "model-2", "created_at": "2024-03-01T00:00:00Z"},
            ]

        def table(self, table_name):
            assert table_name == "datasets"
            return FakeQuery(list(self.rows))

    repository = SupabaseRepository(FakeClient(), "datasets")

    assert repository.latest_where(model_id="model-1")["id"] == "new"
    assert [row["id"] for row in repository.where_ordered(model_id="model-1", desc=False)] == ["old", "new"]

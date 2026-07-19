from __future__ import annotations

from pathlib import Path

from PIL import Image

from background_worker import session_processor as session_manager
from background_worker import source_scanner
from background_worker.retention import cleanup_completed_session_data
from background_worker.alerts import alerts_from_drift
from background_worker.concept_drift_tests.feedback_drift import review_summary
from background_worker.data_drift_tests.mmd import rbf_mmd
from background_worker.prediction_drift_tests.class_distribution_drift import chi_square_prediction_drift, class_distribution
from core.local_files import resolve_dataset_root, resolve_model_artifact
from core.runtime_paths import RuntimePaths
from core.trusted_models import import_trusted_model
from core.types import DriftResult, DriftStatus, DriftType, ImageFeatures, ImageMetadata, TaskType, ValidationStatus
from dashboard.components.image_viewer import yolo_box_to_pixel_rect
from dashboard.views.live_sessions import alert_overall_status, format_score, source_for_session, unresolved_drift_alerts, with_drift_score
from database.local import LocalDatabase
from database.repositories import Repository
from model_management import baseline_builder
from model_management.baseline_persistence import save_baseline_profiles
from model_management.classification_validator import validate_classification_dataset
from model_management.detection_validator import parse_detection_label_file, validate_detection_dataset


def test_trusted_model_is_copied_to_hashed_managed_directory(tmp_path):
    source = tmp_path / "source" / "model.pt"
    source.parent.mkdir()
    source.write_bytes(b"trusted model bytes")
    paths = RuntimePaths(tmp_path / "app", tmp_path / "app" / "logs", tmp_path / "app" / "runtime", tmp_path / "app" / "models")

    imported = import_trusted_model(str(source), paths=paths)

    assert imported.path.read_bytes() == source.read_bytes()
    assert imported.path.parent.name == imported.sha256
    assert imported.size_bytes == len(b"trusted model bytes")


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(128, 120, 90)).save(path)


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
    same = rbf_mmd([[0.0, 0.0], [0.1, 0.1]], [[0.0, 0.0], [0.1, 0.1]])
    shifted = rbf_mmd([[0.0, 0.0], [0.1, 0.1]], [[2.0, 2.0], [2.1, 2.1]])
    assert same <= shifted
    assert shifted > 0


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


def test_latest_where_orders_and_limits_query(tmp_path):
    database = LocalDatabase(tmp_path / "latest_where.sqlite3")
    models = Repository(database, "models")
    datasets = Repository(database, "datasets")
    model = models.insert({"id": "model-1", "name": "Model", "selected_task_type": "classification"})
    datasets.insert({"id": "old", "model_id": model["id"], "created_at": "2024-01-01T00:00:00Z"})
    datasets.insert({"id": "new", "model_id": model["id"], "created_at": "2024-02-01T00:00:00Z"})

    assert datasets.latest_where(model_id="model-1")["id"] == "new"
    assert [row["id"] for row in datasets.where_ordered(model_id="model-1", desc=False)] == ["old", "new"]


def test_source_for_session_uses_session_source_when_active():
    class FakeSources:
        def get(self, row_id):
            return {"id": row_id, "source_uri": "active"}

    fallback = {"id": "fallback", "source_uri": "fallback"}
    running = {"status": "running_live_monitoring", "source_id": "source-active"}
    paused = {"status": "paused", "source_id": "source-paused"}
    completed = {"status": "completed", "source_id": "source-old"}

    assert source_for_session(FakeSources(), fallback, running)["id"] == "source-active"
    assert source_for_session(FakeSources(), fallback, paused)["id"] == "source-paused"
    assert source_for_session(FakeSources(), fallback, completed) == fallback


def test_baseline_profile_saves_prediction_reference(monkeypatch, tmp_path):
    root = tmp_path / "cls"
    for split in ("train", "val"):
        make_image(root / split / "Good" / f"{split}.jpg")
    dataset = validate_classification_dataset(root)

    def fake_inference(model_path, image_path, task_type):
        return [{"task_type": task_type, "predicted_class_name": "Good", "confidence": 0.9}]

    monkeypatch.setattr(baseline_builder, "run_yolo_inference", fake_inference)
    profile = baseline_builder.build_baseline_profile(dataset, tmp_path / "best.pt", "classification")

    assert profile["predictions"]["prediction_count"] == 2
    assert profile["predictions"]["predicted_classes"] == ["Good", "Good"]
    assert profile["predictions"]["average_confidence"] == 0.9
    assert profile["feature_rows"]["training_averages"]["brightness"] is not None
    assert len(profile["feature_rows"]["training_vectors"]) == 1
    assert profile["class_distribution"]["training_counts"] == {"Good": 1}


def test_session_duplicate_images_are_scoped_to_session(monkeypatch, tmp_path):
    image_path = tmp_path / "image.jpg"
    make_image(image_path)
    stores: dict[str, list[dict]] = {}

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name
            stores.setdefault(table_name, [])

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            rows = self.where(**filters)
            rows = list(reversed(rows)) if desc else rows
            return rows[:limit] if limit else rows

        def insert(self, payload):
            row = {"id": f"{self.table_name}-{len(stores[self.table_name]) + 1}", **payload}
            stores[self.table_name].append(row)
            return row

        def upsert(self, payload, on_conflict=None):
            raise AssertionError("image processing should use insert-first duplicate handling")

    monkeypatch.setattr(session_manager, "Repository", FakeRepository)
    monkeypatch.setattr(session_manager, "scan_image_readiness", lambda path: {"ready": [image_path], "total": 1, "skipped_unready": 0})
    monkeypatch.setattr(
        session_manager,
        "extract_image_metadata",
        lambda path: ImageMetadata(path, path.name, None, "same-hash", "JPEG", "RGB", 3, None, 8, 8, 1.0, 10),
    )
    monkeypatch.setattr(
        session_manager,
        "extract_image_features",
        lambda path: ImageFeatures(0.5, 0.1, 0.1, 1.0, 0.2, 0.1, 0.05, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.5] * 12),
    )

    model = {"id": "model-1", "selected_task_type": "classification"}
    source = {"source_uri": str(tmp_path), "source_type": "watched_folder"}
    artifact = {}

    first = session_manager.process_source_once(object(), model, artifact, source, {"id": "session-1"}, [])
    duplicate = session_manager.process_source_once(object(), model, artifact, source, {"id": "session-1"}, [])
    next_session = session_manager.process_source_once(object(), model, artifact, source, {"id": "session-2"}, [])

    assert first["processed"] == 1
    assert duplicate["processed"] == 0
    assert next_session["processed"] == 1


def test_source_image_stride_reads_source_config():
    assert session_manager.source_image_stride(None) == 1
    assert session_manager.source_image_stride({"config": {"image_stride": 4}}) == 4
    assert session_manager.source_image_stride({"image_stride": "3"}) == 3
    assert session_manager.source_image_stride({"config": {"image_stride": 0}}) == 1
    assert session_manager.source_image_stride({"config": {"image_stride": "bad"}}) == 1


def test_process_source_once_samples_before_readiness_checks(monkeypatch, tmp_path):
    image_paths = [tmp_path / f"image-{index}.jpg" for index in range(1, 6)]
    for image_path in image_paths:
        make_image(image_path)
    stores: dict[str, list[dict]] = {"images": [], "predictions": [], "drift_results": [], "alerts": []}

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name
            stores.setdefault(table_name, [])

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            rows = self.where(**filters)
            return rows[:limit] if limit else rows

        def insert(self, payload):
            row = {"id": f"{self.table_name}-{len(stores[self.table_name]) + 1}", **payload}
            stores[self.table_name].append(row)
            return row

        def update(self, row_id, payload):
            row = next((item for item in stores[self.table_name] if item.get("id") == row_id), None)
            if row:
                row.update(payload)
            return row

    monkeypatch.setattr(session_manager, "Repository", FakeRepository)
    checked_paths = []
    monkeypatch.setattr(session_manager, "scan_images", lambda path: image_paths)

    def fake_readiness(paths):
        checked_paths.extend(paths)
        return {"ready": paths, "total": len(paths), "skipped_unready": 0}

    monkeypatch.setattr(session_manager, "scan_image_readiness", fake_readiness)
    monkeypatch.setattr(
        session_manager,
        "extract_image_metadata",
        lambda path: ImageMetadata(path, path.name, None, path.name, "JPEG", "RGB", 3, None, 8, 8, 1.0, 10),
    )
    monkeypatch.setattr(
        session_manager,
        "extract_image_features",
        lambda path: ImageFeatures(0.5, 0.1, 0.1, 1.0, 0.2, 0.1, 0.05, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.5] * 12),
    )

    summary = session_manager.process_source_once(
        object(),
        {"id": "model-1", "selected_task_type": "classification"},
        {},
        {"source_uri": str(tmp_path), "source_type": "watched_folder", "config": {"image_stride": 2}},
        {"id": "session-1"},
        [],
    )

    assert summary["processed"] == 3
    assert summary["ready_files"] == 3
    assert summary["skipped_by_stride"] == 2
    assert checked_paths == [image_paths[0], image_paths[2], image_paths[4]]
    assert [row["filename"] for row in stores["images"]] == ["image-1.jpg", "image-3.jpg", "image-5.jpg"]


def test_scan_ready_images_includes_stable_existing_files(tmp_path):
    image_path = tmp_path / "existing.jpg"
    make_image(image_path)

    ready = source_scanner.scan_ready_images(tmp_path, stability_seconds=0)

    assert ready == [image_path]


def test_scan_ready_images_skips_files_that_change_during_stability_check(monkeypatch, tmp_path):
    image_path = tmp_path / "copying.jpg"
    make_image(image_path)

    def mutate_during_sleep(seconds):
        image_path.write_bytes(image_path.read_bytes() + b"still-copying")

    monkeypatch.setattr(source_scanner.time, "sleep", mutate_during_sleep)

    summary = source_scanner.scan_image_readiness(tmp_path, stability_seconds=0.1)

    assert summary["ready"] == []
    assert summary["skipped_unready"] == 1


def test_scan_image_readiness_waits_once_for_a_batch(monkeypatch, tmp_path):
    image_paths = [tmp_path / f"image-{index}.jpg" for index in range(3)]
    for image_path in image_paths:
        make_image(image_path)
    sleep_calls = []
    monkeypatch.setattr(source_scanner.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    summary = source_scanner.scan_image_readiness(image_paths, stability_seconds=0.1)

    assert summary["ready"] == image_paths
    assert sleep_calls == [0.1]


def test_scan_ready_images_skips_invalid_images(tmp_path):
    invalid_image = tmp_path / "partial.jpg"
    invalid_image.write_text("not an image", encoding="utf-8")

    summary = source_scanner.scan_image_readiness(tmp_path, stability_seconds=0)

    assert summary["ready"] == []
    assert summary["skipped_unready"] == 1


def test_process_source_once_skips_duplicate_insert_race(monkeypatch, tmp_path):
    image_path = tmp_path / "image.jpg"
    make_image(image_path)
    existing_image = {"id": "image-existing", "model_id": "model-1", "session_id": "session-1", "content_hash": "same-hash"}
    image_where_calls = 0

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            nonlocal image_where_calls
            if self.table_name != "images":
                return []
            image_where_calls += 1
            if image_where_calls == 1:
                return []
            return [existing_image] if all(existing_image.get(key) == value for key, value in filters.items()) else []

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            return []

        def insert(self, payload):
            if self.table_name == "images":
                raise Exception('duplicate key value violates unique constraint "idx_images_model_session_content_hash"')
            return {"id": f"{self.table_name}-1", **payload}

    monkeypatch.setattr(session_manager, "Repository", FakeRepository)
    monkeypatch.setattr(session_manager, "scan_image_readiness", lambda path: {"ready": [image_path], "total": 1, "skipped_unready": 0})
    monkeypatch.setattr(
        session_manager,
        "extract_image_metadata",
        lambda path: ImageMetadata(path, path.name, None, "same-hash", "JPEG", "RGB", 3, None, 8, 8, 1.0, 10),
    )
    monkeypatch.setattr(
        session_manager,
        "extract_image_features",
        lambda path: ImageFeatures(0.5, 0.1, 0.1, 1.0, 0.2, 0.1, 0.05, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.5] * 12),
    )

    summary = session_manager.process_source_once(
        object(),
        {"id": "model-1", "selected_task_type": "classification"},
        {},
        {"source_uri": str(tmp_path), "source_type": "watched_folder"},
        {"id": "session-1"},
        [],
    )

    assert summary["processed"] == 0
    assert summary["skipped_duplicates"] == 1
    assert summary["insert_errors"] == 0


def test_process_source_once_reports_non_duplicate_insert_error(monkeypatch, tmp_path):
    image_path = tmp_path / "image.jpg"
    make_image(image_path)

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            return []

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            return []

        def insert(self, payload):
            if self.table_name == "images":
                raise RuntimeError("permission denied for table images")
            return {"id": f"{self.table_name}-1", **payload}

    monkeypatch.setattr(session_manager, "Repository", FakeRepository)
    monkeypatch.setattr(session_manager, "scan_image_readiness", lambda path: {"ready": [image_path], "total": 1, "skipped_unready": 0})
    monkeypatch.setattr(
        session_manager,
        "extract_image_metadata",
        lambda path: ImageMetadata(path, path.name, None, "new-hash", "JPEG", "RGB", 3, None, 8, 8, 1.0, 10),
    )
    monkeypatch.setattr(
        session_manager,
        "extract_image_features",
        lambda path: ImageFeatures(0.5, 0.1, 0.1, 1.0, 0.2, 0.1, 0.05, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.5] * 12),
    )

    summary = session_manager.process_source_once(
        object(),
        {"id": "model-1", "selected_task_type": "classification"},
        {},
        {"source_uri": str(tmp_path), "source_type": "watched_folder"},
        {"id": "session-1"},
        [],
    )

    assert summary["processed"] == 0
    assert summary["insert_errors"] == 1
    assert summary["reason"] == "image_insert_failed"
    assert "permission denied" in summary["error"]


def test_process_source_once_marks_image_complete_after_predictions(monkeypatch, tmp_path):
    image_path = tmp_path / "image.jpg"
    make_image(image_path)
    stores: dict[str, list[dict]] = {"images": [], "predictions": [], "monitoring_sessions": [{"id": "session-1", "status": "running_live_monitoring"}], "drift_results": [], "alerts": []}
    image_inserts = []
    image_updates = []

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def get(self, row_id):
            return next((row for row in stores.get(self.table_name, []) if row.get("id") == row_id), None)

        def where(self, **filters):
            return [row for row in stores.get(self.table_name, []) if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            rows = self.where(**filters)
            return rows[:limit] if limit else rows

        def insert(self, payload):
            row = {"id": f"{self.table_name}-{len(stores.setdefault(self.table_name, [])) + 1}", **payload}
            stores[self.table_name].append(row)
            if self.table_name == "images":
                image_inserts.append(payload)
            return row

        def update(self, row_id, payload):
            row = self.get(row_id)
            if row:
                row.update(payload)
            if self.table_name == "images":
                image_updates.append(payload)
            return row

    monkeypatch.setattr(session_manager, "Repository", FakeRepository)
    monkeypatch.setattr(session_manager, "scan_image_readiness", lambda path: {"ready": [image_path], "total": 1, "skipped_unready": 0})
    monkeypatch.setattr(
        session_manager,
        "extract_image_metadata",
        lambda path: ImageMetadata(path, path.name, None, "new-hash", "JPEG", "RGB", 3, None, 8, 8, 1.0, 10),
    )
    monkeypatch.setattr(
        session_manager,
        "extract_image_features",
        lambda path: ImageFeatures(0.5, 0.1, 0.1, 1.0, 0.2, 0.1, 0.05, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.5] * 12),
    )
    monkeypatch.setattr(
        session_manager,
        "run_yolo_inference",
        lambda artifact_path, image_path, task_type: [{"task_type": "object_detection", "predicted_class_name": "Good", "confidence": 0.9}],
    )

    summary = session_manager.process_source_once(
        object(),
        {"id": "model-1", "selected_task_type": "object_detection"},
        {"id": "artifact-1", "local_path": "model.pt"},
        {"source_uri": str(tmp_path), "source_type": "watched_folder"},
        {"id": "session-1"},
        [{"profile_type": "feature_rows", "metrics": {}}, {"profile_type": "class_distribution", "metrics": {}}],
    )

    assert summary["processed"] == 1
    assert summary["last_processed_image_id"] == "images-1"
    assert image_inserts[0]["inference_status"] == "pending"
    assert image_updates[-1]["inference_status"] == "complete"
    assert image_updates[-1]["prediction_count"] == 1


def test_process_source_once_stops_after_prediction_drift_pause(monkeypatch, tmp_path):
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    make_image(first_path)
    make_image(second_path)
    stores: dict[str, list[dict]] = {
        "images": [],
        "predictions": [],
        "monitoring_sessions": [{"id": "session-1", "status": "running_live_monitoring"}],
        "drift_results": [],
        "alerts": [],
    }
    inferred = []

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def get(self, row_id):
            return next((row for row in stores.get(self.table_name, []) if row.get("id") == row_id), None)

        def where(self, **filters):
            return [row for row in stores.get(self.table_name, []) if all(row.get(key) == value for key, value in filters.items())]

        def insert(self, payload):
            row = {"id": f"{self.table_name}-{len(stores.setdefault(self.table_name, [])) + 1}", **payload}
            stores[self.table_name].append(row)
            return row

        def update(self, row_id, payload):
            row = self.get(row_id)
            if row:
                row.update(payload)
            return row

    monkeypatch.setattr(session_manager, "Repository", FakeRepository)
    monkeypatch.setattr(session_manager, "scan_image_readiness", lambda path: {"ready": [first_path, second_path], "total": 2, "skipped_unready": 0})
    monkeypatch.setattr(
        session_manager,
        "extract_image_metadata",
        lambda path: ImageMetadata(path, path.name, None, path.name, "JPEG", "RGB", 3, None, 8, 8, 1.0, 10),
    )
    monkeypatch.setattr(
        session_manager,
        "extract_image_features",
        lambda path: ImageFeatures(0.5, 0.1, 0.1, 1.0, 0.2, 0.1, 0.05, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.5] * 12),
    )

    def fake_inference(artifact_path, image_path, task_type):
        inferred.append(image_path.name)
        return [{"task_type": "classification", "predicted_class_name": "new_class", "confidence": 0.9}]

    monkeypatch.setattr(session_manager, "run_yolo_inference", fake_inference)
    monkeypatch.setattr(session_manager, "persist_drift_for_completed_images", lambda *args, **kwargs: True)

    summary = session_manager.process_source_once(
        object(),
        {"id": "model-1", "selected_task_type": "classification"},
        {"id": "artifact-1", "local_path": "model.pt"},
        {"source_uri": str(tmp_path), "source_type": "watched_folder"},
        {"id": "session-1"},
        [],
    )

    assert summary["processed"] == 1
    assert summary["prediction_drift_paused"] is True
    assert inferred == ["first.jpg"]

def test_process_source_once_marks_image_failed_when_inference_fails(monkeypatch, tmp_path):
    image_path = tmp_path / "image.jpg"
    make_image(image_path)
    stores: dict[str, list[dict]] = {"images": [], "predictions": [], "monitoring_sessions": [{"id": "session-1", "status": "running_live_monitoring"}], "drift_results": [], "alerts": []}
    image_updates = []

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def get(self, row_id):
            return next((row for row in stores.get(self.table_name, []) if row.get("id") == row_id), None)

        def where(self, **filters):
            return [row for row in stores.get(self.table_name, []) if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            return self.where(**filters)

        def insert(self, payload):
            row = {"id": f"{self.table_name}-{len(stores.setdefault(self.table_name, [])) + 1}", **payload}
            stores[self.table_name].append(row)
            return row

        def update(self, row_id, payload):
            row = self.get(row_id)
            if row:
                row.update(payload)
            if self.table_name == "images":
                image_updates.append(payload)
            return row

    def fail_inference(*args, **kwargs):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(session_manager, "Repository", FakeRepository)
    monkeypatch.setattr(session_manager, "scan_image_readiness", lambda path: {"ready": [image_path], "total": 1, "skipped_unready": 0})
    monkeypatch.setattr(
        session_manager,
        "extract_image_metadata",
        lambda path: ImageMetadata(path, path.name, None, "new-hash", "JPEG", "RGB", 3, None, 8, 8, 1.0, 10),
    )
    monkeypatch.setattr(
        session_manager,
        "extract_image_features",
        lambda path: ImageFeatures(0.5, 0.1, 0.1, 1.0, 0.2, 0.1, 0.05, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.5] * 12),
    )
    monkeypatch.setattr(session_manager, "run_yolo_inference", fail_inference)

    summary = session_manager.process_source_once(
        object(),
        {"id": "model-1", "selected_task_type": "object_detection"},
        {"id": "artifact-1", "local_path": "model.pt"},
        {"source_uri": str(tmp_path), "source_type": "watched_folder"},
        {"id": "session-1"},
        [{"profile_type": "feature_rows", "metrics": {}}, {"profile_type": "class_distribution", "metrics": {}}],
    )

    assert summary["processed"] == 1
    assert summary["inference_errors"] == 1
    assert summary["last_processed_image_id"] == "images-1"
    assert image_updates[-1]["inference_status"] == "failed"
    assert image_updates[-1]["prediction_count"] == 0
    assert "model crashed" in image_updates[-1]["inference_error"]


def test_process_source_once_normalizes_source_path(monkeypatch, tmp_path):
    seen_paths = []

    class EmptyRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            return []

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            return []

        def insert(self, payload):
            return {"id": f"{self.table_name}-1", **payload}

    def fake_scan_images(path):
        seen_paths.append(path)
        return []

    monkeypatch.setattr(session_manager, "Repository", EmptyRepository)
    monkeypatch.setattr(session_manager, "scan_images", fake_scan_images)
    monkeypatch.setattr(session_manager, "scan_image_readiness", lambda paths: {"ready": [], "total": 0, "skipped_unready": 0})

    model = {"id": "model-1", "selected_task_type": "classification"}
    artifact = {}
    source_values = [str(tmp_path), f'"{tmp_path}"', f"  '{tmp_path}'  "]
    for index, source_uri in enumerate(source_values, start=1):
        summary = session_manager.process_source_once(
            object(),
            model,
            artifact,
            {"source_uri": source_uri, "source_type": "watched_folder"},
            {"id": f"session-{index}"},
            [],
        )
        assert summary["processed"] == 0

    assert seen_paths == [tmp_path.resolve(), tmp_path.resolve(), tmp_path.resolve()]


def test_process_source_once_rejects_missing_source_path(tmp_path):
    model = {"id": "model-1", "selected_task_type": "classification"}
    source = {"source_uri": str(tmp_path / "missing"), "source_type": "watched_folder"}

    summary = session_manager.process_source_once(object(), model, {}, source, {"id": "session-1"}, [])

    assert summary == {"processed": 0, "reason": "source_path_missing"}


def test_process_source_once_rejects_file_source_path(tmp_path):
    source_file = tmp_path / "image-source.txt"
    source_file.write_text("not a directory")
    model = {"id": "model-1", "selected_task_type": "classification"}
    source = {"source_uri": str(source_file), "source_type": "watched_folder"}

    summary = session_manager.process_source_once(object(), model, {}, source, {"id": "session-1"}, [])

    assert summary == {"processed": 0, "reason": "source_path_missing"}


def test_yolo_box_to_pixel_rect_clips_normalized_detection():
    prediction = {"x_center": 0.5, "y_center": 0.5, "width": 0.25, "height": 0.5}

    assert yolo_box_to_pixel_rect(prediction, 200, 100) == (75, 25, 125, 75)


def test_industrial_prediction_and_concept_metrics():
    training = class_distribution(["ok", "ok", "bad"])
    production = class_distribution(["ok", "bad", "bad"])
    prediction_drift = chi_square_prediction_drift(training, production)
    summary = review_summary(["approve", "false_positive", "wrong_class", "approve"])

    assert training == {"ok": 2, "bad": 1}
    assert production == {"ok": 1, "bad": 2}
    assert round(prediction_drift["statistic"], 2) == 1.5
    assert prediction_drift["p_value"] > 0.05
    assert summary["accuracy_percent"] == 50.0
    assert summary["counts"]["good_response"] == 2
    assert summary["counts"]["false_positive"] == 1


def test_prediction_drift_calculates_without_two_feature_rows():
    baseline_profiles = [
        {
            "profile_type": "class_distribution",
            "metrics": {"training_counts": {"ok": 2, "bad": 1}},
        }
    ]

    results = session_manager.calculate_drift_for_window(baseline_profiles, [], ["ok", "bad", "bad"])

    prediction_result = next(row for row in results if row.metric_name == "prediction_class_distribution")
    assert round(prediction_result.metric_value, 2) == 1.5
    assert prediction_result.threshold == 0.05
    assert prediction_result.status == DriftStatus.OK
    assert prediction_result.details["calculation"] == "chi_square_goodness_of_fit"
    assert prediction_result.details["p_value"] > 0.05
    assert prediction_result.details["expected_counts"] == {"bad": 1.0, "ok": 2.0}
    assert prediction_result.details["observed_counts"] == {"bad": 2, "ok": 1}
    assert prediction_result.details["production_counts"] == {"ok": 1, "bad": 2}


def test_prediction_drift_records_missing_training_counts_warning():
    baseline_profiles = [{"profile_type": "class_distribution", "metrics": {"training_counts": {}}}]

    results = session_manager.calculate_drift_for_window(baseline_profiles, [], ["ok", "bad"])

    prediction_result = next(row for row in results if row.metric_name == "prediction_class_distribution")
    assert prediction_result.metric_value is None
    assert prediction_result.status == DriftStatus.WARNING
    assert "Training class distribution is empty" in prediction_result.details["reason"]
    assert prediction_result.details["production_counts"] == {"ok": 1, "bad": 1}


def test_prediction_drift_marks_unseen_production_classes_critical():
    baseline_profiles = [
        {
            "profile_type": "class_distribution",
            "metrics": {"training_counts": {"ok": 2, "bad": 1}},
        }
    ]

    results = session_manager.calculate_drift_for_window(baseline_profiles, [], ["ok", "new_class"])

    prediction_result = next(row for row in results if row.metric_name == "prediction_class_distribution")
    assert prediction_result.status == DriftStatus.CRITICAL
    assert prediction_result.details["unseen_classes"] == ["new_class"]
    assert "absent from the training distribution" in prediction_result.details["reason"]


def test_prediction_drift_ui_formatting_handles_unavailable_metric():
    row = with_drift_score({"metric_value": None, "status": "warning"})

    assert row["score"] == 60.0
    assert format_score(row["metric_value"]) == "N/A"


def test_drift_alert_overview_uses_unresolved_linked_alerts():
    alerts = [
        {"severity": "warning", "is_resolved": False, "drift_result_id": "drift-1"},
        {"severity": "critical", "is_resolved": True, "drift_result_id": "drift-2"},
        {"severity": "critical", "is_resolved": False, "drift_result_id": None},
    ]

    unresolved = unresolved_drift_alerts(alerts)

    assert unresolved == [alerts[0]]
    assert alert_overall_status(unresolved) == "Warning"


def test_drift_alert_wording_is_operator_friendly():
    result = DriftResult(DriftType.DATA, "brightness", 22.4, 0.0, DriftStatus.WARNING, {})

    alert = alerts_from_drift([result])[0]

    assert alert["title"] == "Brightness drift warning"
    assert "Production brightness differs from the training baseline" in alert["message"]
    assert "Latest value: 22.4000" in alert["message"]


def test_concept_drift_status_uses_accuracy_thresholds():
    assert session_manager.concept_status_from_accuracy(85.0) == DriftStatus.OK
    assert session_manager.concept_status_from_accuracy(75.0) == DriftStatus.WARNING
    assert session_manager.concept_status_from_accuracy(55.0) == DriftStatus.CRITICAL


def test_manual_threshold_config_preserves_defaults_and_derives_critical_values():
    config = session_manager.threshold_config_from_slider_values(0.05, 0.1, 80.0)

    assert config == session_manager.DEFAULT_THRESHOLD_CONFIG


def test_graph_window_defaults_and_is_bounded_in_session_config():
    assert session_manager.normalized_threshold_config({})["display"]["graph_window"] == 25
    assert session_manager.normalized_threshold_config({"display": {"graph_window": 2}})["display"]["graph_window"] == 5
    assert session_manager.normalized_threshold_config({"display": {"graph_window": 80}})["display"]["graph_window"] == 50
    assert session_manager.threshold_config_from_slider_values(0.05, 0.1, 80.0, graph_window=12)["display"]["graph_window"] == 12


def test_manual_prediction_threshold_changes_status_deterministically():
    strict = session_manager.threshold_config_from_slider_values(0.03, 0.1, 80.0)

    assert session_manager.status_from_chi_square(0.04, False, True) == DriftStatus.WARNING
    assert session_manager.status_from_chi_square(0.04, False, True, strict["prediction"]) == DriftStatus.OK


def test_manual_data_threshold_changes_status_deterministically():
    tolerant = session_manager.threshold_config_from_slider_values(0.2, 0.2, 80.0)

    assert session_manager.status_from_mmd(0.12) == DriftStatus.WARNING
    assert session_manager.status_from_mmd(0.12, tolerant["data"]) == DriftStatus.OK


def test_manual_concept_threshold_changes_status_deterministically():
    tolerant = session_manager.threshold_config_from_slider_values(0.05, 0.1, 70.0)

    assert session_manager.concept_status_from_accuracy(75.0) == DriftStatus.WARNING
    assert session_manager.concept_status_from_accuracy(75.0, tolerant["concept"]) == DriftStatus.OK


def test_prediction_drift_result_records_effective_threshold_config():
    baseline_profiles = [{"profile_type": "class_distribution", "metrics": {"training_counts": {"ok": 2, "bad": 1}}}]
    config = session_manager.threshold_config_from_slider_values(0.03, 0.1, 80.0)

    results = session_manager.calculate_prediction_drift_for_window(baseline_profiles, ["ok", "bad", "bad"], config)
    prediction_result = next(row for row in results if row.metric_name == "prediction_class_distribution")

    assert prediction_result.threshold == 0.03
    assert prediction_result.status == DriftStatus.OK
    assert prediction_result.details["threshold_config"] == config


def test_upsert_unresolved_alert_updates_existing_alert():
    stores = {
        "alerts": [
            {
                "id": "alert-1",
                "session_id": "session-1",
                "title": "Brightness drift warning",
                "severity": "warning",
                "is_resolved": False,
                "message": "old",
            }
        ]
    }

    class FakeAlerts:
        def where(self, **filters):
            return [row for row in stores["alerts"] if all(row.get(key) == value for key, value in filters.items())]

        def update(self, row_id, payload):
            row = next(row for row in stores["alerts"] if row["id"] == row_id)
            row.update(payload)
            return row

        def insert(self, payload):
            stores["alerts"].append({"id": "new", **payload})
            return stores["alerts"][-1]

    session_manager.upsert_unresolved_alert(
        FakeAlerts(),
        "model-1",
        "session-1",
        {"title": "Brightness drift warning", "severity": "warning", "message": "new"},
        {"id": "drift-1"},
    )

    assert len(stores["alerts"]) == 1
    assert stores["alerts"][0]["message"] == "new"
    assert stores["alerts"][0]["drift_result_id"] == "drift-1"


def test_latest_feedback_by_prediction_uses_latest_decision():
    rows = [
        {"id": "old", "prediction_id": "pred-1", "feedback_type": "reject", "created_at": "2026-01-01T00:00:00Z"},
        {"id": "new", "prediction_id": "pred-1", "feedback_type": "approve", "created_at": "2026-01-01T00:01:00Z"},
    ]

    latest = session_manager.latest_feedback_by_prediction(rows)

    assert latest == [rows[1]]


def test_persist_prediction_drift_for_completed_images_advances_window(monkeypatch):
    image_rows = [
        {"id": "image-1", "session_id": "session-1", "inference_status": "complete", "created_at": "2026-01-01T00:00:00Z", "feature_payload": {"brightness_mean": 1.0}},
        {"id": "image-2", "session_id": "session-1", "inference_status": "complete", "created_at": "2026-01-01T00:01:00Z", "feature_payload": {"brightness_mean": 2.0}},
        {"id": "image-3", "session_id": "session-1", "inference_status": "complete", "created_at": "2026-01-01T00:02:00Z", "feature_payload": {"brightness_mean": 3.0}},
    ]
    prediction_rows = {
        "image-1": [{"predicted_class_name": "ok"}],
        "image-2": [{"predicted_class_name": "ok"}],
        "image-3": [{"predicted_class_name": "bad"}],
    }
    persisted = []

    class FakeImages:
        def where(self, **filters):
            return [row for row in image_rows if all(row.get(key) == value for key, value in filters.items())]

    class FakePredictions:
        def where(self, **filters):
            return prediction_rows.get(filters.get("image_id"), [])

    monkeypatch.setattr(
        session_manager,
        "calculate_prediction_drift_for_window",
        lambda baselines, predictions, threshold_config=None: [DriftResult(DriftType.PREDICTION, "prediction_class_distribution", 0.0, 0.05, DriftStatus.OK, {})],
    )
    monkeypatch.setattr(
        session_manager,
        "persist_drift_results",
        lambda drift_repo, alerts_repo, model_id, session_id, num_images, results, window_start, window_end: persisted.append(
            {"num_images": num_images, "window_start": window_start, "window_end": window_end, "results": results}
        ),
    )

    paused = session_manager.persist_drift_for_completed_images(
        FakeImages(),
        FakePredictions(),
        object(),
        object(),
        "model-1",
        "session-1",
        [],
        ["image-2", "image-3"],
        25,
    )

    assert paused is False
    assert [row["window_end"] for row in persisted] == ["2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z"]
    assert [row["num_images"] for row in persisted] == [2, 3]
    assert all(result.drift_type == DriftType.PREDICTION for row in persisted for result in row["results"])


def test_critical_prediction_drift_pauses_and_backfills_data_history(monkeypatch):
    image_rows = [
        {
            "id": f"image-{index}",
            "session_id": "session-1",
            "inference_status": "complete",
            "created_at": f"2026-01-01T00:0{index}:00Z",
            "feature_payload": {"brightness_mean": float(index)},
        }
        for index in range(1, 4)
    ]
    session_updates = []
    persisted = []

    class FakeImages:
        def where(self, **filters):
            return [row for row in image_rows if all(row.get(key) == value for key, value in filters.items())]

    class FakePredictions:
        def where(self, **filters):
            return [{"predicted_class_name": "new_class"}]

    class FakeSessions:
        def update(self, row_id, payload):
            session_updates.append(payload)
            return {"id": row_id, **payload}

    prediction_result = DriftResult(DriftType.PREDICTION, "prediction_class_distribution", 10.0, 0.05, DriftStatus.CRITICAL, {})
    data_result = DriftResult(DriftType.DATA, "total_data_drift", 0.2, 0.0, DriftStatus.WARNING, {})
    monkeypatch.setattr(session_manager, "calculate_prediction_drift_for_window", lambda baselines, predictions, threshold_config=None: [prediction_result])
    monkeypatch.setattr(session_manager, "calculate_data_drift_for_window", lambda baselines, features, threshold_config=None: [data_result])
    monkeypatch.setattr(
        session_manager,
        "persist_drift_results",
        lambda drift_repo, alerts_repo, model_id, session_id, num_images, results, window_start, window_end: persisted.append(
            {"num_images": num_images, "results": results, "window_start": window_start, "window_end": window_end}
        ),
    )

    paused = session_manager.persist_drift_for_completed_images(
        FakeImages(),
        FakePredictions(),
        object(),
        object(),
        "model-1",
        "session-1",
        [],
        ["image-3"],
        25,
        sessions_repo=FakeSessions(),
    )

    assert paused is True
    assert session_updates[0]["status"] == "paused"
    assert session_updates[0]["diagnostic_status"] == "running"
    assert session_updates[-1]["diagnostic_status"] == "completed"
    assert [row["results"][0].drift_type for row in persisted] == [DriftType.PREDICTION, DriftType.DATA, DriftType.DATA]
    assert [row["window_end"] for row in persisted[1:]] == ["2026-01-01T00:02:00Z", "2026-01-01T00:03:00Z"]
    assert [row["num_images"] for row in persisted[1:]] == [2, 3]


def test_historical_data_drift_windows_cap_at_25_and_stop_at_trigger():
    image_rows = [
        {
            "id": f"image-{index:02d}",
            "session_id": "session-1",
            "inference_status": "complete",
            "created_at": f"2026-01-01T00:{index:02d}:00Z",
            "feature_payload": {"brightness_mean": float(index)},
        }
        for index in range(1, 31)
    ]

    class FakeImages:
        def where(self, **filters):
            return [row for row in image_rows if all(row.get(key) == value for key, value in filters.items())]

    windows = session_manager.historical_data_drift_windows(FakeImages(), "session-1", "image-28", 25)

    assert len(windows) == 27
    assert windows[0]["num_images"] == 2
    assert windows[0]["window_end"] == "2026-01-01T00:02:00Z"
    assert windows[-1]["num_images"] == 25
    assert windows[-1]["window_start"] == "2026-01-01T00:04:00Z"
    assert windows[-1]["window_end"] == "2026-01-01T00:28:00Z"
    assert all(window["window_end"] != "2026-01-01T00:29:00Z" for window in windows)


def test_trigger_prediction_diagnostics_marks_failed_backfill(monkeypatch):
    image_rows = [
        {"id": "image-1", "session_id": "session-1", "created_at": "2026-01-01T00:01:00Z", "feature_payload": {"brightness_mean": 1.0}},
        {"id": "image-2", "session_id": "session-1", "created_at": "2026-01-01T00:02:00Z", "feature_payload": {"brightness_mean": 2.0}},
    ]
    session_updates = []

    class FakeImages:
        def where(self, **filters):
            return image_rows

    class FakeSessions:
        def update(self, row_id, payload):
            session_updates.append(payload)
            return {"id": row_id, **payload}

    monkeypatch.setattr(session_manager, "calculate_data_drift_for_window", lambda *args: (_ for _ in ()).throw(RuntimeError("backfill failed")))

    session_manager.trigger_prediction_diagnostics(
        FakeSessions(),
        FakeImages(),
        object(),
        object(),
        "model-1",
        "session-1",
        [],
        "image-2",
        25,
        {"window_start": "2026-01-01T00:01:00Z", "window_end": "2026-01-01T00:02:00Z"},
    )

    assert session_updates[0]["diagnostic_status"] == "running"
    assert session_updates[-1]["diagnostic_status"] == "failed"
    assert session_updates[-1]["diagnostic_error"] == "backfill failed"


def test_insert_drift_idempotent_returns_existing_window_row():
    existing = {"id": "drift-1", "session_id": "session-1", "metric_name": "brightness", "window_end": "2026-01-01T00:02:00Z"}

    class FakeDrift:
        def insert(self, payload):
            raise RuntimeError("duplicate key")

        def where(self, **filters):
            return [existing] if all(existing.get(key) == value for key, value in filters.items()) else []

    result = session_manager.insert_drift_idempotent(
        FakeDrift(),
        {"session_id": "session-1", "metric_name": "brightness", "window_end": "2026-01-01T00:02:00Z"},
    )

    assert result == existing

def test_cleanup_completed_session_data_preserves_unresolved_alerts():
    stores = {
        "monitoring_sessions": [{"id": "session-old", "status": "completed", "ended_at": "2000-01-01T00:00:00+00:00"}],
        "images": [{"id": "image-1", "session_id": "session-old"}],
        "predictions": [{"id": "prediction-1", "image_id": "image-1"}],
        "feedback": [{"id": "feedback-1", "image_id": "image-1"}],
        "drift_results": [{"id": "drift-1", "session_id": "session-old"}],
        "alerts": [
            {"id": "alert-resolved", "session_id": "session-old", "is_resolved": True},
            {"id": "alert-open", "session_id": "session-old", "is_resolved": False},
        ],
    }

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def delete_where(self, **filters):
            rows = self.where(**filters)
            stores[self.table_name] = [row for row in stores[self.table_name] if row not in rows]
            return rows

        def delete(self, row_id):
            row = next((row for row in stores[self.table_name] if row.get("id") == row_id), None)
            if row:
                stores[self.table_name].remove(row)
            return row

        def update(self, row_id, payload):
            row = next(row for row in stores[self.table_name] if row["id"] == row_id)
            row.update(payload)
            return row

    from background_worker import retention

    original_repository = retention.Repository
    retention.Repository = FakeRepository
    try:
        summary = cleanup_completed_session_data(object(), retention_days=30)
    finally:
        retention.Repository = original_repository

    assert summary["cleaned_sessions"] == 1
    assert stores["images"] == []
    assert stores["predictions"] == []
    assert stores["feedback"] == []
    assert stores["drift_results"] == []
    assert stores["alerts"] == [{"id": "alert-open", "session_id": "session-old", "is_resolved": False}]
    assert stores["monitoring_sessions"][0]["archived_at"]

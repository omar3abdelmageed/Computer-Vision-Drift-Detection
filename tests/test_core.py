from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.tabs.live_sessions import alert_overall_status, format_score, source_for_session, unresolved_drift_alerts, with_drift_score
from app.components.image_viewer import yolo_box_to_pixel_rect
from backend.common import ImageFeatures, ImageMetadata
from backend.baseline import build_baseline as baseline_builder
from backend.alerts.alert_rules import alerts_from_drift
from backend.common import DriftResult, DriftStatus, DriftType, TaskType, ValidationStatus
from backend.database.repositories import SupabaseRepository
from backend.baseline.persistence import save_baseline_profiles
from backend.drift.industrial_metrics import chi_square_prediction_drift, class_distribution, review_summary, rbf_mmd
from backend.ingestion.classification_validator import validate_classification_dataset
from backend.ingestion.detection_validator import parse_detection_label_file, validate_detection_dataset
from backend.monitoring import session_manager
from backend.registration.local_files import resolve_dataset_root, resolve_model_artifact


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
    assert [row["profile_type"] for row in repository.inserted] == ["feature_rows", "class_distribution", "predictions"]
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

    monkeypatch.setattr(session_manager, "SupabaseRepository", FakeRepository)
    monkeypatch.setattr(session_manager, "scan_images", lambda path: [image_path])
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

    monkeypatch.setattr(session_manager, "SupabaseRepository", EmptyRepository)
    monkeypatch.setattr(session_manager, "scan_images", fake_scan_images)

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
        assert summary == {"processed": 0}

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

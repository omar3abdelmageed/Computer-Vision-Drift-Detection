from pathlib import Path

from dashboard.views import live_sessions


def test_quoted_windows_style_source_path_is_normalized_and_valid(tmp_path):
    quoted = f'"{tmp_path}"'

    assert live_sessions.normalize_source_uri(quoted) == str(tmp_path)
    assert live_sessions.source_path_is_valid({"source_uri": quoted})


def test_session_start_blockers_explain_invalid_source_path(tmp_path):
    model = {
        "baseline_status": "completed",
        "_baseline_profiles": [{"profile_type": "feature_rows"}],
        "_latest_dataset": {"id": "dataset-1", "validation_status": "valid"},
        "_latest_artifact": {"local_path": "model.pt", "compatibility_status": "valid"},
    }
    missing = tmp_path / "missing"

    blockers = live_sessions.session_start_blockers({"source_uri": f'"{missing}"'}, model, None)

    assert blockers == [f"Choose an existing production image folder; the saved path is not valid: {missing}"]


def test_session_start_has_no_blockers_for_ready_local_configuration(tmp_path):
    model = {
        "baseline_status": "completed",
        "_baseline_profiles": [{"profile_type": "feature_rows"}],
        "_latest_dataset": {"id": "dataset-1", "validation_status": "valid"},
        "_latest_artifact": {"local_path": "model.pt", "compatibility_status": "valid"},
    }

    assert live_sessions.session_start_blockers({"source_uri": f'"{tmp_path}"'}, model, None) == []


def test_confidence_gauge_uses_native_altair_chart(monkeypatch):
    charts = []
    captions = []

    class FakeStreamlit:
        @staticmethod
        def altair_chart(chart, **kwargs):
            charts.append((chart.to_dict(), kwargs))

        @staticmethod
        def caption(message):
            captions.append(message)

    monkeypatch.setattr(live_sessions, "st", FakeStreamlit)

    live_sessions.render_confidence_gauge(0.767, 0.5, 0.75)

    assert len(charts) == 1
    spec = charts[0][0]
    assert len(spec["layer"]) == 4
    assert spec["height"] == 75
    assert spec["padding"] == {"top": 28, "right": 6, "bottom": 0, "left": 6}
    assert spec["layer"][0]["mark"]["size"] == 24
    assert spec["layer"][3]["mark"]["fontSize"] == 14
    assert charts[0][1] == {"width": "stretch"}
    assert captions == ["Good confidence · 76.7%"]


def test_confidence_gauge_labels_align_inward_at_scale_edges(monkeypatch):
    charts = []

    class FakeStreamlit:
        @staticmethod
        def altair_chart(chart, **kwargs):
            charts.append(chart.to_dict())

        @staticmethod
        def caption(message):
            pass

    monkeypatch.setattr(live_sessions, "st", FakeStreamlit)

    expected = [(0.0, "left", 5), (0.5, "center", 0), (0.75, "center", 0), (1.0, "right", -5)]
    for value, align, dx in expected:
        live_sessions.render_confidence_gauge(value, 0.5, 0.75)
        label_mark = charts[-1]["layer"][3]["mark"]
        assert label_mark["align"] == align
        assert label_mark["dx"] == dx


def test_live_sessions_dashboard_does_not_call_worker_processing():
    source = Path("dashboard/views/live_sessions.py").read_text(encoding="utf-8")

    assert "run_yolo_inference" not in source
    assert "process_source_once(" not in source
    assert "from background_worker.session_processor import record_feedback_concept_drift" in source


def test_live_sessions_dashboard_initializes_and_displays_worker_status():
    source = Path("dashboard/views/live_sessions.py").read_text(encoding="utf-8")

    assert '"worker_status": "idle"' in source
    assert "def render_worker_status" in source
    assert "last_worker_error" in source
    assert "Worker error:" in source


def test_live_sessions_dashboard_reruns_after_session_state_changes():
    source = Path("dashboard/views/live_sessions.py").read_text(encoding="utf-8")

    assert "Session paused." in source
    assert "Session resumed." in source
    assert "Session completed." in source
    assert source.count("st.rerun()") >= 5


def test_live_sessions_dashboard_has_refresh_only_polling():
    source = Path("dashboard/views/live_sessions.py").read_text(encoding="utf-8")

    assert "def auto_refresh_fragment" in source
    assert "session_change_token_" in source
    assert "session_change_token(client, session" in source
    assert "Refresh session data" in source


def test_live_sessions_dashboard_starts_worker_after_session_insert():
    source = Path("dashboard/views/live_sessions.py").read_text(encoding="utf-8")

    assert "from background_worker.supervisor import ensure_worker_running" in source
    assert "stop_worker" in source
    assert "worker_result = ensure_worker_running" in source
    assert 'reason="session_start_reset"' in source
    assert 'reason="session_paused"' in source
    assert 'reason="session_resume_reset"' in source
    assert 'reason="session_stopped"' in source
    assert "Session started, but the background worker could not be started" in source
    assert "def recover_idle_worker" in source


def test_active_monitoring_session_is_global_by_default():
    stores = {
        "monitoring_sessions": [
            {"id": "old", "model_id": "model-1", "status": "running_live_monitoring", "created_at": "2026-01-01T00:00:00Z"},
            {"id": "new", "model_id": "model-2", "status": "paused", "created_at": "2026-01-01T00:01:00Z"},
        ]
    }

    class FakeSessions:
        def where_ordered(self, **filters):
            return [row for row in stores["monitoring_sessions"] if all(row.get(key) == value for key, value in filters.items())]

    assert live_sessions.active_monitoring_session(FakeSessions())["id"] == "new"
    assert live_sessions.active_monitoring_session(FakeSessions(), "model-1")["id"] == "old"


def test_session_change_token_changes_when_latest_rows_change(monkeypatch):
    stores = {
        "monitoring_sessions": [
            {
                "id": "session-1",
                "status": "running_live_monitoring",
                "worker_status": "caught_up",
                "last_worker_error": None,
                "last_processed_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ],
        "images": [{"id": "image-1", "session_id": "session-1", "created_at": "2026-01-01T00:00:00Z"}],
        "predictions": [],
        "drift_results": [],
        "alerts": [],
    }

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def get(self, row_id):
            return next((row for row in stores[self.table_name] if row.get("id") == row_id), None)

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            rows = [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]
            rows = sorted(rows, key=lambda row: str(row.get(order_by) or ""), reverse=desc)
            return rows[:limit] if limit else rows

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_in(self, column, values):
            return [row for row in stores[self.table_name] if row.get(column) in values]

    monkeypatch.setattr(live_sessions, "Repository", FakeRepository)

    first = live_sessions.session_change_token(object(), "session-1")
    stores["images"].append({"id": "image-2", "session_id": "session-1", "created_at": "2026-01-01T00:01:00Z"})
    second = live_sessions.session_change_token(object(), "session-1")

    assert first != second


def test_session_change_token_ignores_updated_at_churn(monkeypatch):
    stores = {
        "monitoring_sessions": [
            {
                "id": "session-1",
                "status": "running_live_monitoring",
                "worker_status": "caught_up",
                "last_worker_error": None,
                "last_processed_at": "2026-01-01T00:00:00Z",
                "last_processed_image_id": "image-1",
                "worker_heartbeat_at": "2026-01-01T00:00:00Z",
                "worker_lease_expires_at": "2026-01-01T00:01:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ],
        "images": [{"id": "image-1", "session_id": "session-1", "created_at": "2026-01-01T00:00:00Z"}],
        "predictions": [],
        "drift_results": [],
        "alerts": [],
    }

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def get(self, row_id):
            return next((row for row in stores[self.table_name] if row.get("id") == row_id), None)

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            rows = [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]
            rows = sorted(rows, key=lambda row: str(row.get(order_by) or ""), reverse=desc)
            return rows[:limit] if limit else rows

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_in(self, column, values):
            return [row for row in stores[self.table_name] if row.get(column) in values]

    monkeypatch.setattr(live_sessions, "Repository", FakeRepository)

    first = live_sessions.session_change_token(object(), "session-1")
    stores["monitoring_sessions"][0]["updated_at"] = "2026-01-01T00:01:00Z"
    stores["monitoring_sessions"][0]["worker_heartbeat_at"] = "2026-01-01T00:02:00Z"
    stores["monitoring_sessions"][0]["worker_lease_expires_at"] = "2026-01-01T00:03:00Z"
    second = live_sessions.session_change_token(object(), "session-1")

    assert first == second


def test_session_change_token_ignores_worker_liveness_churn(monkeypatch):
    stores = {
        "monitoring_sessions": [
            {
                "id": "session-1",
                "status": "running_live_monitoring",
                "worker_status": "running",
                "worker_pid": 111,
                "worker_heartbeat_at": "2026-01-01T00:00:00Z",
                "worker_lease_expires_at": "2026-01-01T00:01:00Z",
                "last_worker_error": None,
                "last_processed_image_id": "image-1",
            }
        ],
        "images": [{"id": "image-1", "session_id": "session-1", "inference_status": "complete", "created_at": "2026-01-01T00:00:00Z"}],
        "predictions": [{"id": "prediction-1", "image_id": "image-1", "created_at": "2026-01-01T00:00:01Z"}],
        "drift_results": [],
        "alerts": [],
    }

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def get(self, row_id):
            return next((row for row in stores[self.table_name] if row.get("id") == row_id), None)

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            rows = self.where(**filters)
            rows = sorted(rows, key=lambda row: str(row.get(order_by) or ""), reverse=desc)
            return rows[:limit] if limit else rows

        def where_in(self, column, values):
            return [row for row in stores[self.table_name] if row.get(column) in values]

    monkeypatch.setattr(live_sessions, "Repository", FakeRepository)

    first = live_sessions.session_change_token(object(), "session-1")
    stores["monitoring_sessions"][0].update(
        {
            "worker_status": "caught_up",
            "worker_pid": 222,
            "worker_heartbeat_at": "2026-01-01T00:02:00Z",
            "worker_lease_expires_at": "2026-01-01T00:03:00Z",
        }
    )
    second = live_sessions.session_change_token(object(), "session-1")

    assert first == second


def test_session_change_token_changes_when_predictions_arrive(monkeypatch):
    stores = {
        "monitoring_sessions": [{"id": "session-1", "status": "running_live_monitoring", "worker_status": "running"}],
        "images": [{"id": "image-1", "session_id": "session-1", "created_at": "2026-01-01T00:00:00Z"}],
        "predictions": [],
        "drift_results": [],
        "alerts": [],
    }

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def get(self, row_id):
            return next((row for row in stores[self.table_name] if row.get("id") == row_id), None)

        def where_ordered(self, order_by="created_at", desc=True, limit=None, **filters):
            rows = [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]
            rows = sorted(rows, key=lambda row: str(row.get(order_by) or ""), reverse=desc)
            return rows[:limit] if limit else rows

        def where_in(self, column, values):
            return [row for row in stores[self.table_name] if row.get(column) in values]

    monkeypatch.setattr(live_sessions, "Repository", FakeRepository)

    first = live_sessions.session_change_token(object(), "session-1")
    stores["predictions"].append({"id": "prediction-1", "image_id": "image-1", "created_at": "2026-01-01T00:00:01Z"})
    second = live_sessions.session_change_token(object(), "session-1")

    assert first != second


def test_rerun_cooldown_respects_interval(monkeypatch):
    now = live_sessions.datetime.now(live_sessions.timezone.utc)

    class FakeDateTime:
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return now if cls.calls == 1 else now + live_sessions.timedelta(seconds=2)

    class FakeStreamlit:
        session_state = {}

    monkeypatch.setattr(live_sessions, "st", FakeStreamlit)
    monkeypatch.setattr(live_sessions, "datetime", FakeDateTime)

    assert live_sessions.rerun_cooldown_elapsed("session-1", 5)
    assert not live_sessions.rerun_cooldown_elapsed("session-1", 5)


def test_build_session_summary_counts_predictions_only_for_complete_images():
    image_rows = [
        {"id": "complete-image", "session_id": "session-1", "inference_status": "complete"},
        {"id": "pending-image", "session_id": "session-1", "inference_status": "pending"},
        {"id": "failed-image", "session_id": "session-1", "inference_status": "failed"},
    ]
    prediction_rows = [
        {"id": "complete-prediction", "image_id": "complete-image", "confidence": 0.8},
        {"id": "pending-prediction", "image_id": "pending-image", "confidence": 0.1},
        {"id": "failed-prediction", "image_id": "failed-image", "confidence": 0.2},
    ]
    feedback_rows = [{"id": "feedback-1", "prediction_id": "complete-prediction", "feedback_type": "approve"}]

    class FakeImages:
        def where(self, **filters):
            return [row for row in image_rows if all(row.get(key) == value for key, value in filters.items())]

    class FakePredictions:
        def where(self, **filters):
            return [row for row in prediction_rows if all(row.get(key) == value for key, value in filters.items())]

    class FakeFeedback:
        def where_ordered(self, **filters):
            return [row for row in feedback_rows if all(row.get(key) == value for key, value in filters.items() if key != "limit")]

    summary = live_sessions.build_session_summary(
        FakeImages(),
        FakePredictions(),
        FakeFeedback(),
        {"id": "session-1"},
        {"source_uri": "source"},
        "2026-01-01T00:00:00Z",
    )

    assert summary["processed_image_count"] == 3
    assert summary["completed_image_count"] == 1
    assert summary["failed_image_count"] == 1
    assert summary["pending_image_count"] == 1
    assert summary["prediction_count"] == 1
    assert summary["average_confidence"] == 0.8
    assert summary["reviewed_prediction_count"] == 1


def test_available_drift_sections_follow_persisted_results():
    prediction_only = [{"drift_type": "prediction"}]
    with_data = [*prediction_only, {"drift_type": "data"}]
    with_concept = [*with_data, {"drift_type": "concept"}]

    assert live_sessions.available_drift_sections(prediction_only) == {"prediction": True, "data": False, "concept": False}
    assert live_sessions.available_drift_sections(with_data) == {"prediction": True, "data": True, "concept": False}
    assert live_sessions.available_drift_sections(with_concept) == {"prediction": True, "data": True, "concept": True}

def test_drift_chart_time_prefers_window_end():
    row = {"created_at": "2026-01-01T00:00:00Z", "window_end": "2026-01-01T00:05:00Z"}

    assert live_sessions.drift_chart_time(row) == "2026-01-01T00:05:00Z"
    assert live_sessions.drift_chart_time({"created_at": "old"}) == "old"


def test_windowed_rows_keeps_latest_configured_samples_in_time_order():
    rows = [
        {"window_end": f"2026-01-01T00:0{index}:00Z", "value": index}
        for index in [7, 2, 6, 1, 5, 3, 4]
    ]

    visible_rows = live_sessions.windowed_rows(rows, 5)

    assert [row["value"] for row in visible_rows] == [3, 4, 5, 6, 7]


def test_all_line_frames_use_numbered_configured_window():
    total_rows = [
        {
            "window_end": f"2026-01-01T00:0{index}:00Z",
            "metric_value": float(index),
            "num_images": index,
        }
        for index in range(1, 8)
    ]
    property_rows = [
        {
            "window_end": f"2026-01-01T00:0{index}:00Z",
            "details": {"training_average": 10.0, "production_average": float(index)},
        }
        for index in range(1, 8)
    ]
    prediction_rows = [
        {
            "window_end": f"2026-01-01T00:0{index}:00Z",
            "metric_value": float(index),
            "num_images": index,
            "status": "ok",
            "details": {"p_value": index / 10},
        }
        for index in range(1, 8)
    ]
    concept_rows = [
        {
            "created_at": f"2026-01-01T00:0{index}:00Z",
            "status": "ok",
            "details": {"accuracy_percent": 90.0 - index, "total_reviews": index},
        }
        for index in range(1, 8)
    ]

    total_frame = live_sessions.total_data_drift_frame(total_rows, 5)
    property_frame = live_sessions.property_line_frame(property_rows, 5)
    prediction_frame = live_sessions.prediction_drift_history_frame(prediction_rows, 5)
    concept_frame = live_sessions.concept_drift_history_frame(concept_rows, 5)

    assert total_frame["Sample"].tolist() == [1, 2, 3, 4, 5]
    assert total_frame["MMD"].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert property_frame["Sample"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert property_frame[property_frame["Series"] == "Production average"]["Value"].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert prediction_frame["Sample"].tolist() == [1, 2, 3, 4, 5]
    assert prediction_frame["PValue"].tolist() == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert concept_frame["Sample"].tolist() == [1, 2, 3, 4, 5]
    assert concept_frame["Reviews"].tolist() == [3, 4, 5, 6, 7]


def test_graph_window_comes_from_normalized_session_configuration():
    assert live_sessions.graph_window_from_session(None) == 25
    assert live_sessions.graph_window_from_session({"threshold_config": {"display": {"graph_window": 12}}}) == 12


def test_session_controls_preserve_active_window_and_default_new_session_window():
    configured = {"display": {"graph_window": 12}}

    assert live_sessions.session_control_config({"status": "paused", "threshold_config": configured})["display"]["graph_window"] == 12
    assert live_sessions.session_control_config({"status": "running_live_monitoring", "threshold_config": configured})["display"]["graph_window"] == 12
    assert live_sessions.session_control_config({"status": "completed", "threshold_config": configured})["display"]["graph_window"] == 25


def test_drift_threshold_helpers_use_session_config_before_persisted_row_config():
    row = {
        "threshold": 0.2,
        "details": {
            "accuracy_percent": 75.0,
            "threshold_config": {
                "data": {"mmd_warning": 0.2, "mmd_critical": 0.5},
                "concept": {"accuracy_warning_percent": 70.0, "accuracy_critical_percent": 50.0},
            },
        },
    }
    session_config = {"data": {"mmd_warning": 0.3, "mmd_critical": 0.75}}

    assert live_sessions.mmd_thresholds_from_rows([row], session_config) == (0.3, 0.75)
    assert live_sessions.mmd_thresholds_from_rows([row]) == (0.2, 0.5)
    assert live_sessions.is_concept_drift_high(row) is False


class FakeColumn:
    def __init__(self):
        self.messages = []

    def image(self, *args, **kwargs):
        self.messages.append(("image", args, kwargs))

    def caption(self, message):
        self.messages.append(("caption", message))

    def markdown(self, message):
        self.messages.append(("markdown", message))

    def write(self, message):
        self.messages.append(("write", message))

    def dataframe(self, rows, **kwargs):
        self.messages.append(("dataframe", rows, kwargs))


class FakeStreamlitForPredictions:
    infos = []
    warnings = []
    columns_created = []

    @classmethod
    def columns(cls, *args, **kwargs):
        cols = [FakeColumn(), FakeColumn()]
        cls.columns_created.append(cols)
        return cols

    @classmethod
    def info(cls, message):
        cls.infos.append(message)

    @classmethod
    def warning(cls, message):
        cls.warnings.append(message)


def test_pending_prediction_row_hides_feedback(monkeypatch):
    called = []
    FakeStreamlitForPredictions.infos = []
    FakeStreamlitForPredictions.warnings = []
    FakeStreamlitForPredictions.columns_created = []
    monkeypatch.setattr(live_sessions, "st", FakeStreamlitForPredictions)
    monkeypatch.setattr(live_sessions, "render_feedback_table", lambda *args, **kwargs: called.append(True))

    live_sessions.render_image_review_group(
        object(),
        {"selected_task_type": "object_detection"},
        {"id": "session-1"},
        {"id": "image-1", "filename": "image.jpg", "inference_status": "pending"},
        [],
        object(),
    )

    assert "Prediction pending" in FakeStreamlitForPredictions.infos
    assert called == []


def test_failed_prediction_row_hides_feedback(monkeypatch):
    called = []
    FakeStreamlitForPredictions.infos = []
    FakeStreamlitForPredictions.warnings = []
    FakeStreamlitForPredictions.columns_created = []
    monkeypatch.setattr(live_sessions, "st", FakeStreamlitForPredictions)
    monkeypatch.setattr(live_sessions, "render_feedback_table", lambda *args, **kwargs: called.append(True))

    live_sessions.render_image_review_group(
        object(),
        {"selected_task_type": "object_detection"},
        {"id": "session-1"},
        {"id": "image-1", "filename": "image.jpg", "inference_status": "failed", "inference_error": "boom"},
        [],
        object(),
    )

    assert "Inference failed" in FakeStreamlitForPredictions.warnings
    assert called == []


def test_complete_zero_detection_row_hides_feedback(monkeypatch):
    called = []
    FakeStreamlitForPredictions.infos = []
    FakeStreamlitForPredictions.warnings = []
    FakeStreamlitForPredictions.columns_created = []
    monkeypatch.setattr(live_sessions, "st", FakeStreamlitForPredictions)
    monkeypatch.setattr(live_sessions, "render_feedback_table", lambda *args, **kwargs: called.append(True))

    live_sessions.render_image_review_group(
        object(),
        {"selected_task_type": "object_detection"},
        {"id": "session-1"},
        {"id": "image-1", "filename": "image.jpg", "inference_status": "complete", "prediction_count": 0},
        [],
        object(),
    )

    assert "No detections" in FakeStreamlitForPredictions.infos
    assert called == []


def test_complete_prediction_row_shows_feedback(monkeypatch):
    called = []
    FakeStreamlitForPredictions.infos = []
    FakeStreamlitForPredictions.warnings = []
    FakeStreamlitForPredictions.columns_created = []
    monkeypatch.setattr(live_sessions, "st", FakeStreamlitForPredictions)
    monkeypatch.setattr(live_sessions, "render_feedback_table", lambda *args, **kwargs: called.append(True))

    live_sessions.render_image_review_group(
        object(),
        {"selected_task_type": "object_detection"},
        {"id": "session-1"},
        {"id": "image-1", "filename": "image.jpg", "inference_status": "complete", "prediction_count": 1},
        [{"id": "prediction-1", "task_type": "object_detection", "predicted_class_name": "Good", "confidence": 0.9}],
        object(),
    )

    assert called == [True]


def test_recover_idle_worker_marks_session_failed_when_worker_start_fails(monkeypatch):
    updates = []

    class FakeSessions:
        def update(self, row_id, payload):
            updates.append((row_id, payload))
            return {"id": row_id, **payload}

    class FakeStreamlit:
        session_state = {}

        @staticmethod
        def caption(message):
            pass

        @staticmethod
        def warning(message):
            pass

        @staticmethod
        def error(message):
            FakeStreamlit.last_error = message

    result = type("WorkerResult", (), {"ok": False, "error": "worker boom"})()

    monkeypatch.setattr(live_sessions, "st", FakeStreamlit)
    monkeypatch.setattr(live_sessions, "ensure_worker_running", lambda interval_seconds, client=None: result)

    session = {"id": "session-1", "status": "running_live_monitoring", "worker_status": "idle"}
    live_sessions.recover_idle_worker(FakeSessions(), session, {"polling_interval_seconds": 5})

    assert updates == [("session-1", {"worker_status": "failed", "last_worker_error": "worker boom"})]
    assert session["worker_status"] == "failed"
    assert "worker boom" in FakeStreamlit.last_error


def test_live_sessions_dashboard_has_manual_threshold_controls():
    source = Path("dashboard/views/live_sessions.py").read_text(encoding="utf-8")
    schema = Path("database/migrations.py").read_text(encoding="utf-8")

    assert "def render_threshold_controls" in source
    assert "threshold_config" in source
    assert "Save session configuration" in source
    assert '"Graph window"' in source
    assert "min_value=GRAPH_WINDOW_MIN" in source
    assert "max_value=GRAPH_WINDOW_MAX" in source
    assert "threshold_help_" in source
    assert "session_control_config" in source
    assert 'clear_threshold_widget_state(latest_session["id"])' in source
    assert "use_container_width" not in source
    assert "threshold_config TEXT NOT NULL DEFAULT '{}'" in schema


def test_session_tab_returns_edited_threshold_config_for_drift_display(monkeypatch):
    calls = []

    class FakeStreamlitForSession:
        @staticmethod
        def subheader(message):
            pass

        @staticmethod
        def text_input(label, value=""):
            return value

        @staticmethod
        def columns(count, **kwargs):
            return [FakeButtonColumn() for _ in range(count)]

        @staticmethod
        def info(message):
            pass

        @staticmethod
        def caption(message):
            pass

        @staticmethod
        def warning(message):
            pass

        @staticmethod
        def button(*args, **kwargs):
            return False

    class FakeButtonColumn:
        def button(self, *args, **kwargs):
            return False

    edited_config = live_sessions.threshold_config_from_slider_values(0.03, 0.2, 70.0)
    monkeypatch.setattr(live_sessions, "st", FakeStreamlitForSession)
    monkeypatch.setattr(live_sessions, "render_threshold_controls", lambda sessions, session, model_id: edited_config)
    monkeypatch.setattr(live_sessions, "source_path_is_valid", lambda source: True)
    monkeypatch.setattr(live_sessions, "recover_idle_worker", lambda *args, **kwargs: calls.append("recover"))
    monkeypatch.setattr(live_sessions, "render_worker_status", lambda session: calls.append(session["threshold_config"]))
    monkeypatch.setattr(live_sessions, "render_session_confidence_metrics", lambda *args, **kwargs: None)

    returned = live_sessions.render_session_tab(
        object(),
        object(),
        object(),
        object(),
        object(),
        {"id": "model-1", "_live_eligible": True, "_latest_artifact": {"local_path": "model.pt"}, "_latest_dataset": {"id": "dataset-1"}},
        {"id": "source-1", "source_uri": "source", "polling_interval_seconds": 1},
        {"id": "session-1", "model_id": "model-1", "name": "Session", "status": "paused", "threshold_config": live_sessions.normalized_threshold_config({})},
        active_session={"id": "session-1", "status": "paused"},
    )

    assert returned["threshold_config"] == edited_config
    assert edited_config in calls

def test_clear_threshold_widget_state_removes_session_slider_values(monkeypatch):
    class FakeStreamlitState:
        session_state = {
            "threshold_prediction_session-1": 0.03,
            "threshold_data_session-1": 0.2,
            "threshold_concept_session-1": 70.0,
            "graph_window_session-1": 12,
            "other": "keep",
        }

    monkeypatch.setattr(live_sessions, "st", FakeStreamlitState)

    live_sessions.clear_threshold_widget_state("session-1")

    assert FakeStreamlitState.session_state == {"other": "keep"}


def test_threshold_widget_key_suffix_uses_model_after_session_stop():
    assert live_sessions.threshold_widget_key_suffix({"id": "session-1", "status": "paused"}, "model-1") == "session-1"
    assert live_sessions.threshold_widget_key_suffix({"id": "session-1", "status": "running_live_monitoring"}, "model-1") == "session-1"
    assert live_sessions.threshold_widget_key_suffix({"id": "session-1", "status": "completed"}, "model-1") == "model-1"
    assert live_sessions.threshold_widget_key_suffix(None, "model-1") == "model-1"

import pytest

from background_worker import session_processor
from background_worker import supervisor
from background_worker import worker


def test_worker_processes_running_sessions_and_updates_status(monkeypatch):
    stores = {
        "models": [{"id": "model-1"}],
        "model_artifacts": [{"id": "artifact-1", "model_id": "model-1"}],
        "production_sources": [{"id": "source-1"}],
        "monitoring_sessions": [
            {
                "id": "session-1",
                "model_id": "model-1",
                "artifact_id": "artifact-1",
                "source_id": "source-1",
                "status": "running_live_monitoring",
            }
        ],
        "baseline_profiles": [
            {"id": "baseline-1", "model_id": "model-1", "artifact_id": "artifact-1", "profile_type": "feature_rows"},
            {"id": "baseline-2", "model_id": "model-1", "artifact_id": "artifact-1", "profile_type": "class_distribution"},
        ],
    }
    updates = []

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, **filters):
            return self.where(**filters)

        def get(self, row_id):
            return next((row for row in stores[self.table_name] if row.get("id") == row_id), None)

        def update(self, row_id, payload):
            updates.append((self.table_name, row_id, payload))
            row = self.get(row_id)
            if row:
                row.update(payload)
            return row

    monkeypatch.setattr(worker, "Repository", FakeRepository)
    monkeypatch.setattr(worker, "process_source_once", lambda client, model, artifact, source, session, baselines, **kwargs: {"processed": 2})

    summary = worker.process_running_sessions(object(), worker_owner_id="owner")

    assert summary["sessions"] == 1
    assert summary["processed_images"] == 2
    assert summary["failed_sessions"] == 0
    assert summary["details"] == [{"session_id": "session-1", "processed": 2, "status": "caught_up"}]
    assert summary["cleanup"] == {}
    assert updates[0][2]["worker_status"] == "running"
    assert updates[-1][2]["worker_status"] == "caught_up"


def test_worker_marks_missing_baseline_as_failed(monkeypatch):
    stores = {
        "models": [{"id": "model-1"}],
        "model_artifacts": [{"id": "artifact-1", "model_id": "model-1"}],
        "production_sources": [{"id": "source-1"}],
        "monitoring_sessions": [
            {
                "id": "session-1",
                "model_id": "model-1",
                "artifact_id": "artifact-1",
                "source_id": "source-1",
                "status": "running_live_monitoring",
            }
        ],
        "baseline_profiles": [],
    }
    updates = []

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, **filters):
            return self.where(**filters)

        def get(self, row_id):
            return next((row for row in stores[self.table_name] if row.get("id") == row_id), None)

        def update(self, row_id, payload):
            updates.append((self.table_name, row_id, payload))
            row = self.get(row_id)
            if row:
                row.update(payload)
            return row

    monkeypatch.setattr(worker, "Repository", FakeRepository)

    summary = worker.process_running_sessions(object(), worker_owner_id="owner")

    assert summary["failed_sessions"] == 1
    assert summary["details"][0]["reason"].startswith("Missing baseline profiles for running session")
    assert updates[-1][2]["worker_status"] == "failed"


def test_worker_skips_session_owned_by_live_other_worker():
    future = "2999-01-01T00:00:00+00:00"
    session = {"worker_owner_id": "other-owner", "worker_lease_expires_at": future}

    assert not worker.worker_can_process_session(session, "local-owner")


def test_worker_can_reacquire_expired_other_worker_session():
    past = "2000-01-01T00:00:00+00:00"
    session = {"worker_owner_id": "other-owner", "worker_lease_expires_at": past}

    assert worker.worker_can_process_session(session, "local-owner")


def test_worker_does_not_update_caught_up_session_when_no_new_data(monkeypatch):
    stores = {
        "models": [{"id": "model-1"}],
        "model_artifacts": [{"id": "artifact-1", "model_id": "model-1"}],
        "production_sources": [{"id": "source-1"}],
        "monitoring_sessions": [
            {
                "id": "session-1",
                "model_id": "model-1",
                "artifact_id": "artifact-1",
                "source_id": "source-1",
                "status": "running_live_monitoring",
                "worker_status": "caught_up",
            }
        ],
        "baseline_profiles": [
            {"id": "baseline-1", "model_id": "model-1", "artifact_id": "artifact-1", "profile_type": "feature_rows"},
            {"id": "baseline-2", "model_id": "model-1", "artifact_id": "artifact-1", "profile_type": "class_distribution"},
        ],
    }
    updates = []

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, **filters):
            return self.where(**filters)

        def get(self, row_id):
            return next((row for row in stores[self.table_name] if row.get("id") == row_id), None)

        def update(self, row_id, payload):
            updates.append((self.table_name, row_id, payload))
            row = self.get(row_id)
            if row:
                row.update(payload)
            return row

    monkeypatch.setattr(worker, "Repository", FakeRepository)
    monkeypatch.setattr(worker, "process_source_once", lambda client, model, artifact, source, session, baselines, **kwargs: {"processed": 0})

    summary = worker.process_running_sessions(object(), worker_owner_id="owner")

    assert summary["processed_images"] == 0
    assert summary["failed_sessions"] == 0
    assert updates
    assert all("last_processed_at" not in payload for _, _, payload in updates)


def test_worker_does_not_fail_when_session_stops_during_processing(monkeypatch):
    stores = {
        "models": [{"id": "model-1"}],
        "model_artifacts": [{"id": "artifact-1", "model_id": "model-1"}],
        "production_sources": [{"id": "source-1"}],
        "monitoring_sessions": [
            {
                "id": "session-1",
                "model_id": "model-1",
                "artifact_id": "artifact-1",
                "source_id": "source-1",
                "status": "running_live_monitoring",
            }
        ],
        "baseline_profiles": [
            {"id": "baseline-1", "model_id": "model-1", "artifact_id": "artifact-1", "profile_type": "feature_rows"},
            {"id": "baseline-2", "model_id": "model-1", "artifact_id": "artifact-1", "profile_type": "class_distribution"},
        ],
    }
    updates = []

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def where_ordered(self, **filters):
            return self.where(**filters)

        def get(self, row_id):
            return next((row for row in stores[self.table_name] if row.get("id") == row_id), None)

        def update(self, row_id, payload):
            updates.append((self.table_name, row_id, payload))
            row = self.get(row_id)
            if row:
                row.update(payload)
            return row

    monkeypatch.setattr(worker, "Repository", FakeRepository)
    monkeypatch.setattr(worker, "process_source_once", lambda client, model, artifact, source, session, baselines, **kwargs: {"processed": 1, "reason": "session_not_running"})

    summary = worker.process_running_sessions(object(), worker_owner_id="owner")

    assert summary["failed_sessions"] == 0
    assert summary["details"][-1]["status"] == "stopped"
    assert all(payload.get("worker_status") != "failed" for _, _, payload in updates)


def test_source_processing_does_not_persist_drift_when_no_new_images(monkeypatch):
    inserts = {"drift_results": [], "alerts": []}
    image_rows = [
        {"id": "image-1", "session_id": "session-1", "feature_payload": {"brightness_mean": 10.0}},
        {"id": "image-2", "session_id": "session-1", "feature_payload": {"brightness_mean": 12.0}},
    ]

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            if self.table_name == "images":
                return [row for row in image_rows if all(row.get(key) == value for key, value in filters.items())]
            return []

        def where_ordered(self, **filters):
            return self.where(**filters)

        def insert(self, payload):
            inserts.setdefault(self.table_name, []).append(payload)
            return {"id": f"{self.table_name}-1", **payload}

    monkeypatch.setattr(session_processor, "Repository", FakeRepository)
    monkeypatch.setattr(session_processor, "resolve_source_path", lambda raw_path: object())
    monkeypatch.setattr(session_processor, "scan_images", lambda source_path: [])
    monkeypatch.setattr(session_processor, "scan_image_readiness", lambda source_path: {"ready": [], "total": len(image_rows), "skipped_unready": 0})
    monkeypatch.setattr(session_processor, "calculate_drift_for_window", lambda *args, **kwargs: [object()])

    result = session_processor.process_source_once(
        object(),
        {"id": "model-1", "selected_task_type": "object_detection"},
        {"id": "artifact-1", "local_path": "model.pt"},
        {"source_uri": "source", "source_type": "watched_folder"},
        {"id": "session-1"},
        [{"profile_type": "feature_rows", "metrics": {}}],
    )

    assert result["processed"] == 0
    assert inserts["drift_results"] == []
    assert inserts["alerts"] == []


def test_supervisor_starts_worker_when_not_running(monkeypatch, tmp_path):
    started = {}

    class FakeProcess:
        pid = 12345
        returncode = None

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        started["command"] = command
        started["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", tmp_path / "worker.pid")
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "STARTUP_CHECK_SECONDS", 0)
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: False)
    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)

    result = supervisor.ensure_worker_running(interval_seconds=7)

    assert result.started
    assert result.pid == 12345
    assert started["command"][-4:-2] == ["--interval-seconds", "7"]
    assert started["command"][-2] == "--worker-owner-id"
    assert started["command"][:3] == [supervisor.worker_python_executable(), "-m", supervisor.WORKER_MODULE]
    assert (tmp_path / "worker.pid").read_text(encoding="utf-8") == "12345"
    assert result.log_path


def test_supervisor_does_not_start_duplicate_worker(monkeypatch, tmp_path):
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", pid_file)
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: True)
    monkeypatch.setattr(supervisor, "heartbeat_is_stale", lambda heartbeat, interval_seconds: False)

    result = supervisor.ensure_worker_running(interval_seconds=7)

    assert result.already_running
    assert result.pid == 12345


def test_supervisor_handles_stale_pid_file(monkeypatch, tmp_path):
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text("22222", encoding="utf-8")

    class FakeProcess:
        pid = 33333

        def poll(self):
            return None

    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", pid_file)
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "STARTUP_CHECK_SECONDS", 0)
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: False)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    result = supervisor.ensure_worker_running(interval_seconds=7)

    assert result.started
    assert result.pid == 33333
    assert pid_file.read_text(encoding="utf-8") == "33333"


def test_supervisor_reports_launch_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", tmp_path / "worker.pid")
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: False)

    def fail_popen(*args, **kwargs):
        raise OSError("launch failed")

    monkeypatch.setattr(supervisor.subprocess, "Popen", fail_popen)

    result = supervisor.ensure_worker_running(interval_seconds=7)

    assert not result.ok
    assert result.error == "launch failed"


def test_supervisor_reports_worker_that_exits_during_startup(monkeypatch, tmp_path):
    class FakeProcess:
        pid = 44444
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", tmp_path / "worker.pid")
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "STARTUP_CHECK_SECONDS", 0)
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: False)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    result = supervisor.ensure_worker_running(interval_seconds=7)

    assert not result.ok
    assert result.pid == 44444
    assert "exited during startup" in result.error
    assert not (tmp_path / "worker.pid").exists()


def test_supervisor_clears_pid_when_liveness_check_errors(monkeypatch, tmp_path):
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text("30136", encoding="utf-8")

    class FakeProcess:
        pid = 55555

        def poll(self):
            return None

    def broken_liveness(pid):
        raise SystemError("broken Windows pid check")

    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", pid_file)
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "STARTUP_CHECK_SECONDS", 0)
    monkeypatch.setattr(supervisor, "is_worker_process", broken_liveness)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    result = supervisor.ensure_worker_running(interval_seconds=7)

    assert result.started
    assert result.pid == 55555
    assert pid_file.read_text(encoding="utf-8") == "55555"


def test_supervisor_rejects_non_worker_pid(monkeypatch, tmp_path):
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text("12345", encoding="utf-8")

    class FakeProcess:
        pid = 67890
        stdin = type("Stdin", (), {"write": lambda self, value: None, "close": lambda self: None})()

        def poll(self):
            return None

    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", pid_file)
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "STARTUP_CHECK_SECONDS", 0)
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: False)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    result = supervisor.ensure_worker_running(interval_seconds=7)

    assert result.started
    assert result.pid == 67890
    assert pid_file.read_text(encoding="utf-8") == "67890"


def test_supervisor_marks_owned_sessions_stale_when_pid_dead(monkeypatch, tmp_path):
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text("12345", encoding="utf-8")
    stores = {
        "monitoring_sessions": [
            {
                "id": "session-1",
                "status": "running_live_monitoring",
                "worker_owner_id": "owner-1",
                "worker_status": "running",
            }
        ]
    }

    class FakeRepository:
        def __init__(self, client, table_name):
            self.table_name = table_name

        def where(self, **filters):
            return [row for row in stores[self.table_name] if all(row.get(key) == value for key, value in filters.items())]

        def update(self, row_id, payload):
            row = next(row for row in stores[self.table_name] if row["id"] == row_id)
            row.update(payload)
            return row

    class FakeProcess:
        pid = 67890
        stdin = type("Stdin", (), {"write": lambda self, value: None, "close": lambda self: None})()

        def poll(self):
            return None

    monkeypatch.setattr(supervisor, "Repository", FakeRepository)
    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", pid_file)
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "STARTUP_CHECK_SECONDS", 0)
    monkeypatch.setattr(supervisor, "worker_owner_id", lambda: "owner-1")
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: False)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    client = object()
    result = supervisor.ensure_worker_running(interval_seconds=7, client=client)

    assert result.started
    assert stores["monitoring_sessions"][0]["worker_status"] == "stale"


def test_stop_worker_terminates_valid_worker_and_clears_pid(monkeypatch, tmp_path):
    pid_file = tmp_path / "worker.pid"
    heartbeat_file = tmp_path / "heartbeat.json"
    pid_file.write_text("12345", encoding="utf-8")
    stopped = []

    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", pid_file)
    monkeypatch.setattr(supervisor, "HEARTBEAT_FILE", heartbeat_file)
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "worker_owner_id", lambda: "owner-1")
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: True)
    monkeypatch.setattr(supervisor, "stop_worker_process", lambda pid, timeout_seconds=3.0: stopped.append(pid) or True)

    result = supervisor.stop_worker(reason="test_stop")

    assert result.ok
    assert result.stopped
    assert stopped == [12345]
    assert not pid_file.exists()
    assert '"status": "stopped"' in heartbeat_file.read_text(encoding="utf-8")
    assert '"reason": "test_stop"' in heartbeat_file.read_text(encoding="utf-8")


def test_stop_worker_handles_pid_heartbeat_mismatch(monkeypatch, tmp_path):
    pid_file = tmp_path / "worker.pid"
    heartbeat_file = tmp_path / "heartbeat.json"
    pid_file.write_text("11111", encoding="utf-8")
    heartbeat_file.write_text('{"pid": 22222}', encoding="utf-8")
    stopped = []

    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", pid_file)
    monkeypatch.setattr(supervisor, "HEARTBEAT_FILE", heartbeat_file)
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "worker_owner_id", lambda: "owner-1")
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: True)
    monkeypatch.setattr(supervisor, "stop_worker_process", lambda pid, timeout_seconds=3.0: stopped.append(pid) or True)

    result = supervisor.stop_worker()

    assert result.ok
    assert stopped == [11111, 22222]
    assert result.attempted_pids == (11111, 22222)


def test_stop_worker_refuses_non_worker_pid(monkeypatch, tmp_path):
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text("12345", encoding="utf-8")

    def fail_if_called(pid, timeout_seconds=3.0):
        raise AssertionError("non-worker pid must not be terminated")

    monkeypatch.setattr(supervisor, "WORKER_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", pid_file)
    monkeypatch.setattr(supervisor, "HEARTBEAT_FILE", tmp_path / "heartbeat.json")
    monkeypatch.setattr(supervisor, "OWNER_FILE", tmp_path / "worker_owner.json")
    monkeypatch.setattr(supervisor, "worker_owner_id", lambda: "owner-1")
    monkeypatch.setattr(supervisor, "is_worker_process", lambda pid: False)
    monkeypatch.setattr(supervisor, "stop_worker_process", fail_if_called)

    result = supervisor.stop_worker()

    assert result.ok
    assert not result.stopped
    assert not pid_file.exists()

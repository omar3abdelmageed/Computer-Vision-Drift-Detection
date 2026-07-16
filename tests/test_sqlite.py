from __future__ import annotations

import sqlite3
import threading

import pytest

from database.local import LocalDatabase
from database.repositories import Repository


def test_local_database_enables_wal_foreign_keys_and_integrity(tmp_path):
    database = LocalDatabase(tmp_path / "monitor.sqlite3")

    with database.connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert database.integrity_check()


def test_repository_round_trips_json_booleans_yaml_and_ordering(tmp_path):
    database = LocalDatabase(tmp_path / "monitor.sqlite3")
    models = Repository(database, "models")
    datasets = Repository(database, "datasets")
    sources = Repository(database, "production_sources")
    model = models.insert({"name": "Detector", "selected_task_type": "object_detection", "tags": ["local"]})
    dataset = datasets.insert({"model_id": model["id"], "yaml_content": "names:\n  0: car"})
    sources.insert({"id": "old", "model_id": model["id"], "source_type": "watched_folder", "mode": "live_monitoring", "is_active": False, "created_at": "2025-01-01T00:00:00+00:00"})
    sources.insert({"id": "new", "model_id": model["id"], "source_type": "watched_folder", "mode": "live_monitoring", "is_active": True, "config": {"stride": 3}, "created_at": "2026-01-01T00:00:00+00:00"})

    assert models.get(model["id"])["tags"] == ["local"]
    assert datasets.get(dataset["id"])["yaml_content"] == "names:\n  0: car"
    assert sources.get("old")["is_active"] is False
    assert sources.get("new")["config"] == {"stride": 3}
    assert sources.latest_where(model_id=model["id"])["id"] == "new"


def test_repository_rejects_unknown_identifiers_and_invalid_source_type(tmp_path):
    database = LocalDatabase(tmp_path / "monitor.sqlite3")
    with pytest.raises(ValueError, match="Unknown repository table"):
        Repository(database, "models; DROP TABLE models")
    models = Repository(database, "models")
    model = models.insert({"name": "Safe", "selected_task_type": "classification"})
    with pytest.raises(ValueError, match="Unknown models columns"):
        models.update(model["id"], {"name = 'broken'": "value"})
    with pytest.raises(sqlite3.IntegrityError):
        Repository(database, "production_sources").insert({"model_id": model["id"], "source_type": "rtsp", "mode": "live_monitoring"})


def test_wal_allows_reads_while_writer_commits(tmp_path):
    database = LocalDatabase(tmp_path / "monitor.sqlite3")
    model = Repository(database, "models").insert({"name": "Concurrent", "selected_task_type": "classification"})
    failures = []

    def write_rows():
        try:
            images = Repository(database, "images")
            for index in range(50):
                images.insert({"model_id": model["id"], "filename": f"{index}.jpg"})
        except Exception as exc:
            failures.append(exc)

    writer = threading.Thread(target=write_rows)
    writer.start()
    reader = Repository(database, "images")
    while writer.is_alive():
        reader.count_where(model_id=model["id"])
    writer.join()

    assert not failures
    assert reader.count_where(model_id=model["id"]) == 50

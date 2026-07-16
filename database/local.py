from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from core.runtime_paths import runtime_paths
from database.migrations import MIGRATIONS, SCHEMA_VERSION


class LocalDatabase:
    """Connection factory with transactional, versioned SQLite migrations."""

    def __init__(self, path: Path | None = None):
        paths = runtime_paths().create()
        self.path = Path(path) if path else paths.data / "cv_monitor.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migration_lock = RLock()
        self.initialize()

    def connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=5.0)
        else:
            connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        if not readonly:
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                connection.close()
                raise RuntimeError("SQLite WAL mode could not be enabled.")
            connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def initialize(self) -> None:
        with self._migration_lock:
            with self.connect() as connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"Database schema {version} is newer than this application supports.")
            if 0 < version < SCHEMA_VERSION:
                if not self.integrity_check():
                    raise RuntimeError("Database integrity check failed before migration.")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                self.backup(runtime_paths().create().backups / f"pre-migration-v{version}-{stamp}.sqlite3")
            for target_version in range(version + 1, SCHEMA_VERSION + 1):
                with self.connect() as connection:
                    script = (
                        "BEGIN IMMEDIATE;\n"
                        + MIGRATIONS[target_version - 1]
                        + f"\nPRAGMA user_version={target_version};\nCOMMIT;"
                    )
                    try:
                        connection.executescript(script)
                    except Exception:
                        connection.rollback()
                        raise

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def integrity_check(self) -> bool:
        with self.connect(readonly=True) as connection:
            return connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    def backup(self, destination: Path | None = None) -> Path:
        paths = runtime_paths().create()
        automatic = destination is None
        if destination is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination = paths.backups / f"cv_monitor-{stamp}.sqlite3"
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connect(readonly=True) as source, sqlite3.connect(destination) as target:
            source.backup(target)
        if automatic:
            regular = sorted(paths.backups.glob("cv_monitor-*.sqlite3"), key=lambda item: item.stat().st_mtime, reverse=True)
            for old in regular[3:]:
                old.unlink(missing_ok=True)
        return destination

_default_database: LocalDatabase | None = None


def create_local_database(path: Path | None = None) -> LocalDatabase:
    global _default_database
    if path is not None:
        return LocalDatabase(path)
    if _default_database is None:
        _default_database = LocalDatabase()
    return _default_database

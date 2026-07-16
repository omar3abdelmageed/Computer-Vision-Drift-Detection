from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from database.local import LocalDatabase
from database.tables import (
    ALERTS, BASELINE_PROFILES, DATASET_PATHS, DATASETS, DRIFT_RESULTS, FEEDBACK,
    IMAGES, MODEL_ARTIFACTS, MODELS, MONITORING_SESSIONS, PREDICTIONS, PRODUCTION_SOURCES,
)


ALLOWED_TABLES = {
    MODELS, DATASETS, DATASET_PATHS, MODEL_ARTIFACTS, PRODUCTION_SOURCES,
    MONITORING_SESSIONS, IMAGES, PREDICTIONS, BASELINE_PROFILES, DRIFT_RESULTS,
    FEEDBACK, ALERTS, "ground_truth_labels",
}
JSON_COLUMNS = {
    MODELS: {"tags", "retention_policy"},
    DATASETS: {"yaml_candidates", "class_names", "validation_errors"},
    MODEL_ARTIFACTS: {"class_names", "compatibility_details", "raw_metadata"},
    PRODUCTION_SOURCES: {"config"},
    MONITORING_SESSIONS: {"threshold_config", "summary"},
    IMAGES: {"feature_payload"},
    PREDICTIONS: {"top_k", "raw_prediction"},
    BASELINE_PROFILES: {"metrics"},
    DRIFT_RESULTS: {"details"},
    FEEDBACK: {"corrected_payload"},
}
BOOLEAN_COLUMNS = {
    DATASETS: {"has_test_split", "test_has_labels"},
    DATASET_PATHS: {"has_images", "has_labels"},
    MODEL_ARTIFACTS: {"is_compatible"},
    PRODUCTION_SOURCES: {"is_active"},
    ALERTS: {"is_resolved"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, database: LocalDatabase | None, table_name: str):
        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"Unknown repository table: {table_name}")
        self.database = database
        self.table_name = table_name
        self._columns: set[str] | None = None

    @property
    def columns(self) -> set[str]:
        if self._columns is None:
            if self.database is None:
                return set()
            with self.database.connect(readonly=True) as connection:
                self._columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({self.table_name})")}
        return self._columns

    def list(self, order_by: str = "created_at", desc: bool = True, limit: int | None = None) -> list[dict[str, Any]]:
        return self.where_ordered(order_by=order_by, desc=desc, limit=limit)

    def get(self, row_id: str) -> dict[str, Any] | None:
        rows = self._select("id = ?", [row_id], limit=1)
        return rows[0] if rows else None

    def insert(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.database is None:
            return None
        prepared = self._prepare_insert(payload)
        names = list(prepared)
        sql = f"INSERT INTO {self.table_name} ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})"
        with self.database.transaction() as connection:
            connection.execute(sql, [prepared[name] for name in names])
        return self.get(str(prepared["id"]))

    def update(self, row_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        rows = self.update_where(payload, id=row_id)
        return rows[0] if rows else None

    def update_where(self, payload: dict[str, Any], **filters) -> list[dict[str, Any]]:
        if self.database is None or not payload:
            return []
        before = self.where(**filters)
        if not before:
            return []
        prepared = self._prepare_payload(payload)
        if "updated_at" in self.columns and "updated_at" not in prepared:
            prepared["updated_at"] = utc_now()
        where, params = self._where_clause(filters)
        assignments = ", ".join(f"{name} = ?" for name in prepared)
        with self.database.transaction() as connection:
            connection.execute(
                f"UPDATE {self.table_name} SET {assignments} WHERE {where}",
                [prepared[name] for name in prepared] + params,
            )
        return [row for item in before if (row := self.get(item["id"]))]

    def delete(self, row_id: str) -> dict[str, Any] | None:
        rows = self.delete_where(id=row_id)
        return rows[0] if rows else None

    def delete_where(self, **filters) -> list[dict[str, Any]]:
        if self.database is None:
            return []
        rows = self.where(**filters)
        if not rows:
            return []
        where, params = self._where_clause(filters)
        with self.database.transaction() as connection:
            connection.execute(f"DELETE FROM {self.table_name} WHERE {where}", params)
        return rows

    def upsert(self, payload: dict[str, Any], on_conflict: str | None = None) -> dict[str, Any] | None:
        if not on_conflict:
            row_id = payload.get("id")
            return self.update(str(row_id), payload) if row_id and self.get(str(row_id)) else self.insert(payload)
        conflict_columns = [part.strip() for part in on_conflict.split(",") if part.strip()]
        if not conflict_columns or any(name not in self.columns for name in conflict_columns):
            raise ValueError("Invalid upsert conflict columns.")
        existing = self.where(**{name: payload.get(name) for name in conflict_columns})
        return self.update(existing[0]["id"], payload) if existing else self.insert(payload)

    def where(self, **filters) -> list[dict[str, Any]]:
        where, params = self._where_clause(filters)
        return self._select(where, params)

    def where_in(self, column: str, values: list[Any]) -> list[dict[str, Any]]:
        self._validate_column(column)
        if not values:
            return []
        placeholders = ", ".join("?" for _ in values)
        return self._select(f"{column} IN ({placeholders})", [self._encode(column, value) for value in values])

    def count_where(self, **filters) -> int:
        if self.database is None:
            return 0
        where, params = self._where_clause(filters)
        with self.database.connect(readonly=True) as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE {where}", params).fetchone()[0])

    def where_ordered(self, order_by: str = "created_at", desc: bool = True, limit: int | None = None, **filters) -> list[dict[str, Any]]:
        self._validate_column(order_by)
        where, params = self._where_clause(filters)
        order = "DESC" if desc else "ASC"
        return self._select(where, params, order_by=f"{order_by} {order}, id {order}", limit=limit)

    def latest_stable_where(self, order_by: str = "created_at", **filters) -> dict[str, Any] | None:
        rows = self.where_ordered(order_by=order_by, desc=True, limit=1, **filters)
        return rows[0] if rows else None

    def latest_where(self, order_by: str = "created_at", **filters) -> dict[str, Any] | None:
        return self.latest_stable_where(order_by=order_by, **filters)

    def _prepare_insert(self, payload: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(payload)
        prepared.setdefault("id", str(uuid.uuid4()))
        now = utc_now()
        if "created_at" in self.columns:
            prepared.setdefault("created_at", now)
        if "updated_at" in self.columns:
            prepared.setdefault("updated_at", now)
        if "started_at" in self.columns:
            prepared.setdefault("started_at", now)
        return self._prepare_payload(prepared)

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        unknown = set(payload) - self.columns
        if unknown:
            raise ValueError(f"Unknown {self.table_name} columns: {', '.join(sorted(unknown))}")
        return {name: self._encode(name, value) for name, value in payload.items()}

    def _encode(self, column: str, value: Any) -> Any:
        if column in JSON_COLUMNS.get(self.table_name, set()):
            return json.dumps(value if value is not None else {}, separators=(",", ":"), sort_keys=True)
        if column in BOOLEAN_COLUMNS.get(self.table_name, set()) and value is not None:
            return int(bool(value))
        return value

    def _decode(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for column in JSON_COLUMNS.get(self.table_name, set()):
            value = result.get(column)
            if isinstance(value, str):
                try:
                    result[column] = json.loads(value)
                except json.JSONDecodeError:
                    result[column] = {}
        for column in BOOLEAN_COLUMNS.get(self.table_name, set()):
            if result.get(column) is not None:
                result[column] = bool(result[column])
        return result

    def _where_clause(self, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        if not filters:
            return "1 = 1", []
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in filters.items():
            self._validate_column(column)
            if value is None:
                clauses.append(f"{column} IS NULL")
            else:
                clauses.append(f"{column} = ?")
                params.append(self._encode(column, value))
        return " AND ".join(clauses), params

    def _select(self, where: str, params: list[Any], *, order_by: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        if self.database is None:
            return []
        sql = f"SELECT * FROM {self.table_name} WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, max(0, int(limit))]
        with self.database.connect(readonly=True) as connection:
            return [self._decode(row) for row in connection.execute(sql, params).fetchall()]

    def _validate_column(self, column: str) -> None:
        if column not in self.columns:
            raise ValueError(f"Unknown {self.table_name} column: {column}")
LocalRepository = Repository

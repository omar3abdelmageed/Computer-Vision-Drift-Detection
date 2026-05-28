from __future__ import annotations

from typing import Any


class SupabaseRepository:
    def __init__(self, client, table_name: str):
        self.client = client
        self.table_name = table_name

    def list(self, order_by: str = "created_at", desc: bool = True, limit: int | None = None) -> list[dict[str, Any]]:
        if self.client is None:
            return []
        query = self.client.table(self.table_name).select("*").order(order_by, desc=desc)
        if limit:
            query = query.limit(limit)
        return query.execute().data or []

    def get(self, row_id: str) -> dict[str, Any] | None:
        if self.client is None:
            return None
        data = self.client.table(self.table_name).select("*").eq("id", row_id).limit(1).execute().data
        return data[0] if data else None

    def insert(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.client is None:
            return None
        data = self.client.table(self.table_name).insert(payload).execute().data
        return data[0] if data else None

    def update(self, row_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.client is None:
            return None
        data = self.client.table(self.table_name).update(payload).eq("id", row_id).execute().data
        return data[0] if data else None

    def delete(self, row_id: str) -> dict[str, Any] | None:
        if self.client is None:
            return None
        data = self.client.table(self.table_name).delete().eq("id", row_id).execute().data
        return data[0] if data else None

    def upsert(self, payload: dict[str, Any], on_conflict: str | None = None) -> dict[str, Any] | None:
        if self.client is None:
            return None
        query = self.client.table(self.table_name).upsert(payload, on_conflict=on_conflict) if on_conflict else self.client.table(self.table_name).upsert(payload)
        data = query.execute().data
        return data[0] if data else None

    def where(self, **filters) -> list[dict[str, Any]]:
        if self.client is None:
            return []
        query = self.client.table(self.table_name).select("*")
        for key, value in filters.items():
            query = query.eq(key, value)
        return query.execute().data or []

    def where_ordered(
        self,
        order_by: str = "created_at",
        desc: bool = True,
        limit: int | None = None,
        **filters,
    ) -> list[dict[str, Any]]:
        if self.client is None:
            return []
        query = self.client.table(self.table_name).select("*")
        for key, value in filters.items():
            query = query.eq(key, value)
        query = query.order(order_by, desc=desc)
        if limit:
            query = query.limit(limit)
        return query.execute().data or []

    def latest_where(self, order_by: str = "created_at", **filters) -> dict[str, Any] | None:
        rows = self.where_ordered(order_by=order_by, desc=True, limit=1, **filters)
        return rows[0] if rows else None

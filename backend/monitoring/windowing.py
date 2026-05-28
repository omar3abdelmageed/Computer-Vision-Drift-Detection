from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class MonitoringWindow:
    items: list
    window_start: datetime | None = None
    window_end: datetime | None = None


def count_windows(items: Sequence[T], size: int = 25) -> Iterable[MonitoringWindow]:
    for start in range(0, len(items), size):
        chunk = list(items[start:start + size])
        if chunk:
            yield MonitoringWindow(items=chunk)


def time_windows(items: Sequence[T], timestamp_getter, duration: timedelta) -> Iterable[MonitoringWindow]:
    sorted_items = sorted(items, key=timestamp_getter)
    if not sorted_items:
        return
    start = timestamp_getter(sorted_items[0])
    end = start + duration
    chunk: list[T] = []
    for item in sorted_items:
        ts = timestamp_getter(item)
        if ts >= end and chunk:
            yield MonitoringWindow(items=chunk, window_start=start, window_end=end)
            start = ts
            end = start + duration
            chunk = []
        chunk.append(item)
    if chunk:
        yield MonitoringWindow(items=chunk, window_start=start, window_end=end)

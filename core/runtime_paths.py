from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "CVMonitor"


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    logs: Path
    worker: Path
    models: Path
    data: Path | None = None
    backups: Path | None = None

    def create(self) -> "RuntimePaths":
        for path in (self.root, self.logs, self.worker, self.models, self.data, self.backups):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
        return self


def default_app_data_dir() -> Path:
    configured = os.getenv("APP_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def runtime_paths() -> RuntimePaths:
    root = default_app_data_dir()
    return RuntimePaths(
        root=root,
        logs=root / "logs",
        worker=root / "runtime" / "worker",
        models=root / "models",
        data=root / "data",
        backups=root / "backups",
    )

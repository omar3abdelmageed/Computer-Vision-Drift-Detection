from __future__ import annotations

import shutil
from pathlib import Path

from backend.ingestion.image_scanner import scan_images


def read_test_folder(path: Path) -> list[Path]:
    return scan_images(path)


def read_watched_folder(path: Path) -> list[Path]:
    return scan_images(path)


def save_manual_upload(uploaded_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / uploaded_file.name
    with destination.open("wb") as handle:
        shutil.copyfileobj(uploaded_file, handle)
    return destination

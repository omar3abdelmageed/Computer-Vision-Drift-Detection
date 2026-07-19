from __future__ import annotations

import time
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FILE_STABILITY_CHECK_SECONDS = 0.2


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not path.name.startswith(".")


def scan_images(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if is_supported_image(p))


def scan_image_readiness(path: Path | list[Path] | None, stability_seconds: float = FILE_STABILITY_CHECK_SECONDS) -> dict:
    image_paths = list(path) if isinstance(path, list) else scan_images(path)
    before_states = {image_path: file_state(image_path) for image_path in image_paths}
    if image_paths and stability_seconds > 0:
        time.sleep(stability_seconds)
    ready_paths = [
        image_path
        for image_path in image_paths
        if before_states[image_path] is not None
        and before_states[image_path] == file_state(image_path)
        and validate_image(image_path)
    ]
    return {
        "ready": ready_paths,
        "total": len(image_paths),
        "skipped_unready": len(image_paths) - len(ready_paths),
    }


def scan_ready_images(path: Path | None, stability_seconds: float = FILE_STABILITY_CHECK_SECONDS) -> list[Path]:
    return scan_image_readiness(path, stability_seconds)["ready"]


def file_is_ready(path: Path, stability_seconds: float = FILE_STABILITY_CHECK_SECONDS) -> bool:
    before = file_state(path)
    if before is None:
        return False
    if stability_seconds > 0:
        time.sleep(stability_seconds)
    if before != file_state(path):
        return False
    return validate_image(path)


def file_state(path: Path) -> tuple[int, int] | None:
    try:
        state = path.stat()
    except OSError:
        return None
    return state.st_size, state.st_mtime_ns


def validate_image(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

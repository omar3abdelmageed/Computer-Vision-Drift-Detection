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


def scan_image_readiness(path: Path | None, stability_seconds: float = FILE_STABILITY_CHECK_SECONDS) -> dict:
    image_paths = scan_images(path)
    ready_paths = [image_path for image_path in image_paths if file_is_ready(image_path, stability_seconds)]
    return {
        "ready": ready_paths,
        "total": len(image_paths),
        "skipped_unready": len(image_paths) - len(ready_paths),
    }


def scan_ready_images(path: Path | None, stability_seconds: float = FILE_STABILITY_CHECK_SECONDS) -> list[Path]:
    return scan_image_readiness(path, stability_seconds)["ready"]


def file_is_ready(path: Path, stability_seconds: float = FILE_STABILITY_CHECK_SECONDS) -> bool:
    try:
        before = path.stat()
    except OSError:
        return False
    if stability_seconds > 0:
        time.sleep(stability_seconds)
    try:
        after = path.stat()
    except OSError:
        return False
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        return False
    return validate_image(path)


def validate_image(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

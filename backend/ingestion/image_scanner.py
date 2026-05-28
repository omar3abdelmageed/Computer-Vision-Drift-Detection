from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not path.name.startswith(".")


def scan_images(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if is_supported_image(p))


def validate_image(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

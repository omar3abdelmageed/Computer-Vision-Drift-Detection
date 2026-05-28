from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def extract_dataset_archive(archive_path: Path, workspace_dir: Path, model_id: str) -> Path:
    extract_dir = workspace_dir / "datasets" / model_id
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
    elif suffix == ".7z":
        try:
            import py7zr
        except ImportError as exc:
            raise RuntimeError("Install py7zr to extract 7z dataset archives.") from exc
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=extract_dir)
    else:
        raise ValueError(f"Unsupported dataset archive type: {suffix}")
    return find_dataset_root(extract_dir)


def extract_dataset_zip(zip_path: Path, workspace_dir: Path, model_id: str) -> Path:
    return extract_dataset_archive(zip_path, workspace_dir, model_id)


def find_dataset_root(extraction_dir: Path) -> Path:
    visible_items = [p for p in extraction_dir.iterdir() if not p.name.startswith(".")]
    folders = [p for p in visible_items if p.is_dir()]
    files = [p for p in visible_items if p.is_file()]
    if len(folders) == 1 and not files:
        return folders[0]
    return extraction_dir

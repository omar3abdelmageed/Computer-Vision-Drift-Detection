from __future__ import annotations

from pathlib import Path


def resolve_local_path(raw_path: str) -> Path:
    path_text = raw_path.strip().strip('"').strip("'")
    if not path_text:
        raise ValueError("Enter a local path.")
    return Path(path_text).expanduser().resolve(strict=False)


def resolve_dataset_root(raw_path: str) -> Path:
    path = resolve_local_path(raw_path)
    if not path.exists():
        raise ValueError(f"Dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Dataset path must be a directory: {path}")
    return path


def resolve_model_artifact(raw_path: str) -> Path:
    path = resolve_local_path(raw_path)
    if not path.exists():
        raise ValueError(f"Model artifact does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Model artifact must be a file: {path}")
    if path.suffix.lower() != ".pt":
        raise ValueError("Model artifact must be a .pt file.")
    return path

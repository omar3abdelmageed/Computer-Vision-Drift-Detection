from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from core.hashing import file_sha256
from core.local_files import resolve_model_artifact
from core.runtime_paths import RuntimePaths, runtime_paths


@dataclass(frozen=True)
class TrustedModelArtifact:
    path: Path
    sha256: str
    size_bytes: int
    original_name: str


def import_trusted_model(raw_path: str, paths: RuntimePaths | None = None) -> TrustedModelArtifact:
    source = resolve_model_artifact(raw_path)
    digest = file_sha256(source)
    size_bytes = source.stat().st_size
    app_paths = (paths or runtime_paths()).create()
    destination_dir = app_paths.models / digest
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if not destination.exists() or file_sha256(destination) != digest:
        shutil.copy2(source, destination)
    return TrustedModelArtifact(destination, digest, size_bytes, source.name)

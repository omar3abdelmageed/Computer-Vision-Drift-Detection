from __future__ import annotations

from pathlib import Path


class StorageUploadError(RuntimeError):
    def __init__(self, message: str, *, status: int | str | None = None) -> None:
        super().__init__(message)
        self.status = status


def _is_payload_too_large(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    return str(status) == "413"


def upload_file(client, bucket: str, storage_path: str, local_path: Path) -> str:
    if client is None:
        return storage_path
    with local_path.open("rb") as handle:
        try:
            client.storage.from_(bucket).upload(storage_path, handle, {"upsert": "true"})
        except Exception as exc:
            if _is_payload_too_large(exc):
                size_mb = local_path.stat().st_size / (1024 * 1024)
                message = (
                    f"{local_path.name} is {size_mb:.1f} MB, which exceeds the "
                    f"Supabase Storage size limit for bucket '{bucket}'."
                )
                raise StorageUploadError(message, status=getattr(exc, "status", None)) from exc
            raise
    return storage_path

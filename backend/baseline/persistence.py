from __future__ import annotations

from typing import Any

from backend.database.repositories import SupabaseRepository


BASELINE_PROFILE_TYPES = ("dataset_profile", "image_stats", "feature_rows", "class_distribution")


def save_baseline_profiles(
    baselines: SupabaseRepository,
    model_id: str,
    dataset_id: str,
    artifact_id: str,
    profile: dict[str, Any],
) -> None:
    existing_rows = baselines.where(model_id=model_id, dataset_id=dataset_id, artifact_id=artifact_id)
    existing_by_type = {row.get("profile_type"): row for row in existing_rows}

    for profile_type in BASELINE_PROFILE_TYPES:
        payload = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "artifact_id": artifact_id,
            "profile_type": profile_type,
            "metrics": profile.get(profile_type, {}),
        }
        existing = existing_by_type.get(profile_type)
        if existing:
            baselines.update(existing["id"], {"metrics": payload["metrics"], "dataset_id": dataset_id, "artifact_id": artifact_id})
        else:
            baselines.insert(payload)

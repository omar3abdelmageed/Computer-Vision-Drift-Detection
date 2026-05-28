from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.components.status_cards import status_card
from backend.baseline.build_baseline import build_baseline_profile
from backend.baseline.persistence import save_baseline_profiles
from backend.common import SplitName, TaskType
from backend.database.repositories import SupabaseRepository
from backend.ingestion.dataset_validation import validate_dataset
from backend.model.compatibility import check_compatibility
from backend.model.yolo import load_yolo_metadata
from backend.registration.local_files import resolve_dataset_root, resolve_model_artifact
from backend.utils.serialization import to_jsonable


def render_model_registration_tab(client) -> None:
    st.header("Model Management")
    models = SupabaseRepository(client, "models")
    datasets = SupabaseRepository(client, "datasets")
    dataset_paths = SupabaseRepository(client, "dataset_paths")
    artifacts = SupabaseRepository(client, "model_artifacts")
    baselines = SupabaseRepository(client, "baseline_profiles")

    all_models = models.list()
    selected_model = select_or_create_model(models, all_models)
    if not selected_model:
        return

    model_id = selected_model["id"]
    task_type = render_task_type(models, selected_model)
    dataset_record = render_dataset_registration(datasets, models, model_id, task_type)
    artifact_record = render_model_registration(artifacts, models, model_id)
    dataset_metadata = render_dataset_validation(datasets, dataset_paths, models, dataset_record, task_type)
    artifact_metadata = render_compatibility(artifacts, models, artifact_record, dataset_metadata)
    render_baseline_builder(models, baselines, selected_model, dataset_record, artifact_record, dataset_metadata, artifact_metadata)
    render_baseline_report(baselines, model_id)


def select_or_create_model(models: SupabaseRepository, all_models: list[dict]) -> dict | None:
    st.subheader("1. Model Selection")

    create_option = "__create_new_model__"
    models_by_id = {model["id"]: model for model in all_models}
    option_ids = [create_option, *models_by_id.keys()]
    remembered_id = st.session_state.get("selected_model_id", create_option)
    selected_index = option_ids.index(remembered_id) if remembered_id in option_ids else 0
    selected_id = st.selectbox(
        "Model",
        option_ids,
        index=selected_index,
        format_func=lambda option_id: "Add new model" if option_id == create_option else format_model_option(models_by_id[option_id]),
    )
    st.session_state["selected_model_id"] = selected_id
    selected = models_by_id.get(selected_id)

    if selected is None:
        render_create_model_form(models)
        return None

    render_selected_model_metadata(models, selected)
    return selected


def format_model_option(model: dict) -> str:
    name = model.get("name") or "Untitled model"
    status = model.get("registration_status") or "draft"
    model_id = str(model.get("id") or "")
    return f"{name} ({status}) - {model_id[:8]}"


def render_create_model_form(models: SupabaseRepository) -> None:
    st.divider()
    st.markdown("#### Create New Model")
    with st.form("create_registration"):
        name = st.text_input("Registration name")
        description = st.text_area("Description")
        tags = st.text_input("Tags", help="Comma-separated")
        submitted = st.form_submit_button("Create registration")
        if submitted and name:
            created = models.insert({"name": name, "description": description, "tags": [tag.strip() for tag in tags.split(",") if tag.strip()], "selected_task_type": "object_detection"})
            if created:
                st.session_state["selected_model_id"] = created["id"]
            st.success("Model created.")
            st.rerun()


def render_selected_model_metadata(models: SupabaseRepository, selected: dict) -> None:
    st.divider()
    title_col, delete_col = st.columns([0.88, 0.12], vertical_alignment="center")
    title_col.markdown(f"#### {selected.get('name') or 'Untitled model'}")
    title_col.caption(f"Status: {selected.get('registration_status') or 'draft'}")
    if delete_col.button("Delete", icon=":material/delete:", help="Delete selected model", key=f"delete_model_{selected['id']}"):
        render_delete_model_dialog(models, selected)

    with st.expander("Technical details"):
        st.code(selected["id"])

    st.markdown("#### Edit Metadata")
    with st.form("update_metadata"):
        name = st.text_input("Name", value=selected.get("name") or "")
        description = st.text_area("Description", value=selected.get("description") or "")
        tags_text = ", ".join(selected.get("tags") or [])
        tags = st.text_input("Tags", value=tags_text)
        if st.form_submit_button("Update metadata"):
            models.update(selected["id"], {"name": name, "description": description, "tags": [tag.strip() for tag in tags.split(",") if tag.strip()]})
            st.success("Metadata updated.")
            st.rerun()


@st.dialog("Delete Model")
def render_delete_model_dialog(models: SupabaseRepository, selected: dict) -> None:
    st.write(f"You are about to delete `{selected.get('name') or 'Untitled model'}`.")
    st.warning("Deleting this model removes its related datasets, artifacts, baseline profiles, sessions, predictions, drift results, feedback, and alerts.")
    cancel_col, delete_col = st.columns(2)
    if cancel_col.button("Cancel", use_container_width=True):
        st.rerun()
    if delete_col.button("Delete model", icon=":material/delete:", type="primary", use_container_width=True):
        models.delete(selected["id"])
        st.session_state["selected_model_id"] = "__create_new_model__"
        st.success("Model deleted.")
        st.rerun()


def render_task_type(models: SupabaseRepository, selected_model: dict) -> str:
    st.subheader("2. Task Type Selection")
    selected = st.radio("Task type", [TaskType.OBJECT_DETECTION.value, TaskType.CLASSIFICATION.value], index=0 if selected_model.get("selected_task_type") != TaskType.CLASSIFICATION.value else 1, horizontal=True)
    if selected != selected_model.get("selected_task_type"):
        models.update(
            selected_model["id"],
            {
                "selected_task_type": selected,
                "detected_task_type": None,
                "task_validation_status": "pending",
                "baseline_status": "not_started",
                "registration_status": "task_type_changed",
            },
        )
        st.rerun()
    cols = st.columns(2)
    cols[0].metric("Detected task", selected_model.get("detected_task_type") or "pending")
    cols[1].metric("Task validation", selected_model.get("task_validation_status") or "pending")
    return selected


def render_dataset_registration(datasets: SupabaseRepository, models: SupabaseRepository, model_id: str, task_type: str) -> dict | None:
    st.subheader("3. Dataset Registration")
    dataset_record = datasets.latest_where(model_id=model_id)
    with st.form("register_dataset_path"):
        dataset_path = st.text_input("YOLO dataset directory")
        submitted = st.form_submit_button("Register dataset")
    if submitted:
        try:
            dataset_root = resolve_dataset_root(dataset_path)
        except ValueError as exc:
            st.error(str(exc))
            return dataset_record
        payload = {
            "model_id": model_id,
            "storage_backend": "local",
            "source_uri": str(dataset_root),
            "storage_path": None,
            "dataset_root_path": str(dataset_root),
            "dataset_type": "yolo_detection" if task_type == "object_detection" else "yolo_classification",
        }
        dataset_record = datasets.insert(payload)
        models.update(model_id, {"registration_status": "dataset_uploaded", "baseline_status": "not_started"})
        st.success("Dataset registered.")
        st.rerun()
    if dataset_record:
        status_card("Dataset", dataset_record.get("validation_status") or "uploaded", dataset_record.get("dataset_root_path") or "")
    return dataset_record


def render_model_registration(artifacts: SupabaseRepository, models: SupabaseRepository, model_id: str) -> dict | None:
    st.subheader("4. Model Registration")
    artifact_record = artifacts.latest_where(model_id=model_id)
    with st.form("register_model_path"):
        artifact_path = st.text_input("YOLO .pt artifact path")
        submitted = st.form_submit_button("Register model")
    if submitted:
        try:
            local_path = resolve_model_artifact(artifact_path)
        except ValueError as exc:
            st.error(str(exc))
            return artifact_record
        metadata = load_yolo_metadata(local_path)
        payload = {
            "model_id": model_id,
            "storage_backend": "local",
            "source_uri": str(local_path),
            "storage_path": None,
            "local_path": str(local_path),
            "artifact_name": local_path.name,
            "artifact_type": "yolo_pt",
            "model_task": metadata.model_task,
            "class_names": metadata.class_names,
            "num_classes": metadata.num_classes,
            "input_size": metadata.input_size,
            "raw_metadata": metadata.raw_metadata,
        }
        artifact_record = artifacts.insert(payload)
        models.update(model_id, {"registration_status": "artifact_uploaded", "baseline_status": "not_started"})
        st.success("Model artifact registered.")
        st.rerun()
    if artifact_record:
        status_card("Artifact", artifact_record.get("compatibility_status") or "uploaded", artifact_record.get("artifact_name") or "")
        with st.expander("Raw artifact metadata"):
            st.json(artifact_record.get("raw_metadata") or {})
    return artifact_record


def render_dataset_validation(datasets: SupabaseRepository, dataset_paths: SupabaseRepository, models: SupabaseRepository, dataset_record: dict | None, task_type: str):
    st.subheader("5. Dataset Validation")
    if not dataset_record:
        st.info("Register a dataset first.")
        return None
    if st.button("Refresh validation"):
        metadata = validate_dataset(Path(dataset_record["dataset_root_path"]), task_type)
        persist_dataset_validation(datasets, dataset_paths, models, dataset_record, metadata)
        st.success("Dataset validation complete.")
        st.rerun()
    try:
        metadata = validate_dataset(Path(dataset_record["dataset_root_path"]), task_type)
    except Exception as exc:
        st.warning(f"Validation preview unavailable: {exc}")
        return None
    if persist_dataset_validation(datasets, dataset_paths, models, dataset_record, metadata):
        st.rerun()
    status_card("Validation", metadata.validation_status.value, f"{metadata.supported_image_count} supported images")
    st.dataframe([to_jsonable(split) for split in metadata.splits.values()], use_container_width=True)
    with st.expander("Validation issues"):
        st.json(to_jsonable(metadata.issues))
    return metadata


def persist_dataset_validation(datasets: SupabaseRepository, dataset_paths: SupabaseRepository, models: SupabaseRepository, dataset_record: dict, metadata) -> bool:
    payload = {
        "dataset_layout": metadata.layout.value,
        "selected_yaml_path": str(metadata.selected_yaml_path) if metadata.selected_yaml_path else None,
        "yaml_filename": metadata.selected_yaml_path.name if metadata.selected_yaml_path else None,
        "yaml_candidates": to_jsonable(metadata.yaml_candidates),
        "class_source": metadata.class_source,
        "num_classes": len(metadata.classes),
        "class_names": to_jsonable(metadata.classes),
        "num_train_images": metadata.splits[SplitName.TRAIN].image_count,
        "num_val_images": metadata.splits[SplitName.VAL].image_count,
        "num_test_images": metadata.splits[SplitName.TEST].image_count,
        "has_test_split": metadata.has_test_split,
        "test_has_labels": metadata.test_has_labels,
        "supported_image_count": metadata.supported_image_count,
        "unsupported_file_count": metadata.unsupported_file_count,
        "validation_status": metadata.validation_status.value,
        "validation_errors": to_jsonable(metadata.issues),
    }
    if any(dataset_record.get(key) != value for key, value in payload.items()):
        datasets.update(dataset_record["id"], payload)
        dataset_record.update(payload)
        for split_name, split in metadata.splits.items():
            dataset_paths.upsert(
                {
                    "dataset_id": dataset_record["id"],
                    "split": split_name.value,
                    "images_path": str(split.images_path) if split.images_path else None,
                    "labels_path": str(split.labels_path) if split.labels_path else None,
                    "has_images": split.has_images,
                    "has_labels": split.has_labels,
                    "image_count": split.image_count,
                    "label_count": split.label_count,
                },
                on_conflict="dataset_id,split",
            )
        models.update(dataset_record["model_id"], {"detected_task_type": metadata.detected_task_type.value, "task_validation_status": metadata.validation_status.value, "registration_status": "dataset_validated" if metadata.validation_status.value != "invalid" else "failed"})
        return True
    return False


def render_compatibility(artifacts: SupabaseRepository, models: SupabaseRepository, artifact_record: dict | None, dataset_metadata):
    st.subheader("6. Model Compatibility Check")
    if not artifact_record or not dataset_metadata:
        st.info("Register a model and validate a dataset first.")
        return None
    artifact_metadata = type("Artifact", (), {
        "model_task": artifact_record.get("model_task") or "unknown",
        "class_names": artifact_record.get("class_names") or [],
        "num_classes": artifact_record.get("num_classes") or 0,
    })()
    if st.button("Refresh compatibility"):
        result = check_compatibility(dataset_metadata, artifact_metadata)
        persist_compatibility_result(artifacts, models, artifact_record, result)
        st.success("Compatibility check complete.")
        st.rerun()
    result = check_compatibility(dataset_metadata, artifact_metadata)
    if persist_compatibility_result(artifacts, models, artifact_record, result):
        st.rerun()
    status_card("Compatibility", result["compatibility_status"], "Ready for baseline" if result["is_compatible"] else "Blocked")
    with st.expander("Compatibility details"):
        st.json(result["compatibility_details"])
    return artifact_metadata if result["is_compatible"] else None


def persist_compatibility_result(artifacts: SupabaseRepository, models: SupabaseRepository, artifact_record: dict, result: dict) -> bool:
    if any(artifact_record.get(key) != value for key, value in result.items()):
        artifacts.update(artifact_record["id"], result)
        artifact_record.update(result)
        models.update(artifact_record["model_id"], {"registration_status": "compatible" if result["is_compatible"] else "failed"})
        return True
    return False


def render_baseline_builder(
    models: SupabaseRepository,
    baselines: SupabaseRepository,
    selected_model: dict,
    dataset_record: dict | None,
    artifact_record: dict | None,
    dataset_metadata,
    artifact_metadata,
) -> None:
    st.subheader("7. Baseline Builder")
    enabled = dataset_record is not None and artifact_record is not None and dataset_metadata is not None and artifact_metadata is not None and dataset_metadata.validation_status.value != "invalid"
    if st.button("Build baseline", disabled=not enabled):
        models.update(selected_model["id"], {"baseline_status": "running", "registration_status": "baseline_running"})
        try:
            profile = build_baseline_profile(dataset_metadata)
            save_baseline_profiles(baselines, selected_model["id"], dataset_record["id"], artifact_record["id"], profile)
            models.update(selected_model["id"], {"baseline_status": "completed", "registration_status": "baseline_completed"})
            st.success("Baseline completed.")
        except Exception as exc:
            models.update(selected_model["id"], {"baseline_status": "failed", "registration_status": "failed"})
            st.error(f"Baseline failed: {exc}")
    status_card("Baseline", selected_model.get("baseline_status") or "not_started", selected_model.get("registration_status") or "")


def render_baseline_report(baselines: SupabaseRepository, model_id: str) -> None:
    st.subheader("8. Baseline Report")
    rows = baselines.where_ordered(model_id=model_id)
    if not rows:
        st.info("No baseline profiles yet.")
        return
    st.dataframe(rows, use_container_width=True)
    with st.expander("Raw baseline JSON"):
        st.code(json.dumps(rows, indent=2, default=str), language="json")

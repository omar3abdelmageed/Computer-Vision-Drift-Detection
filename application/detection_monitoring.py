from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DETECTION_CLUSTER_IOU_THRESHOLD = 0.80


@dataclass(frozen=True)
class DetectionCluster:
    index: int
    representative: dict
    members: tuple[dict, ...]
    is_ambiguous: bool
    is_same_class_duplicate: bool


def cluster_detection_predictions(
    predictions: Iterable[dict],
    iou_threshold: float = DETECTION_CLUSTER_IOU_THRESHOLD,
) -> list[DetectionCluster]:
    threshold = max(0.0, min(1.0, float(iou_threshold)))
    builders: list[dict] = []

    for prediction in predictions:
        rect = normalized_box_rect(prediction)
        best_builder = None
        best_iou = -1.0
        if rect is not None:
            for builder in builders:
                representative_rect = normalized_box_rect(builder["representative"])
                if representative_rect is None:
                    continue
                overlap = box_iou(rect, representative_rect)
                if overlap >= threshold and overlap > best_iou:
                    best_builder = builder
                    best_iou = overlap

        if best_builder is None:
            builders.append({"representative": prediction, "members": [prediction]})
            continue

        best_builder["members"].append(prediction)
        if prediction_confidence(prediction) > prediction_confidence(best_builder["representative"]):
            best_builder["representative"] = prediction

    clusters = []
    for index, builder in enumerate(builders, start=1):
        members = tuple(builder["members"])
        class_keys = {prediction_class_key(member) for member in members}
        clusters.append(
            DetectionCluster(
                index=index,
                representative=builder["representative"],
                members=members,
                is_ambiguous=len(members) > 1 and len(class_keys) > 1,
                is_same_class_duplicate=len(members) > 1 and len(class_keys) == 1,
            )
        )
    return clusters


def summarize_detection_overlaps(prediction_groups: Iterable[Iterable[dict]]) -> dict[str, int]:
    summary = {
        "location_count": 0,
        "raw_prediction_count": 0,
        "overlapping_location_count": 0,
        "ambiguous_location_count": 0,
        "same_class_duplicate_count": 0,
        "affected_image_count": 0,
    }
    for predictions in prediction_groups:
        detection_rows = [row for row in predictions if row.get("task_type") == "object_detection"]
        if not detection_rows:
            continue
        clusters = cluster_detection_predictions(detection_rows)
        overlapping = [cluster for cluster in clusters if len(cluster.members) > 1]
        summary["location_count"] += len(clusters)
        summary["raw_prediction_count"] += len(detection_rows)
        summary["overlapping_location_count"] += len(overlapping)
        summary["ambiguous_location_count"] += sum(cluster.is_ambiguous for cluster in clusters)
        summary["same_class_duplicate_count"] += sum(cluster.is_same_class_duplicate for cluster in clusters)
        summary["affected_image_count"] += bool(overlapping)
    return summary


def normalized_box_rect(prediction: dict) -> tuple[float, float, float, float] | None:
    values = [prediction.get(key) for key in ("x_center", "y_center", "width", "height")]
    if any(not isinstance(value, (int, float)) for value in values):
        return None
    x_center, y_center, width, height = (float(value) for value in values)
    if width <= 0 or height <= 0:
        return None
    left = max(0.0, min(1.0, x_center - width / 2))
    top = max(0.0, min(1.0, y_center - height / 2))
    right = max(0.0, min(1.0, x_center + width / 2))
    bottom = max(0.0, min(1.0, y_center + height / 2))
    return (left, top, right, bottom) if right > left and bottom > top else None


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def prediction_confidence(prediction: dict) -> float:
    confidence = prediction.get("confidence")
    return float(confidence) if isinstance(confidence, (int, float)) else -1.0


def prediction_class_key(prediction: dict) -> tuple[str, object]:
    class_id = prediction.get("predicted_class_id")
    if class_id is not None:
        return "id", class_id
    return "name", str(prediction.get("predicted_class_name") or "")


def cluster_member_label(cluster: DetectionCluster, member_index: int) -> str:
    if len(cluster.members) == 1:
        return str(cluster.index)
    return f"{cluster.index}{alphabetic_suffix(member_index)}"


def alphabetic_suffix(index: int) -> str:
    value = max(0, int(index)) + 1
    suffix = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        suffix = chr(ord("a") + remainder) + suffix
    return suffix

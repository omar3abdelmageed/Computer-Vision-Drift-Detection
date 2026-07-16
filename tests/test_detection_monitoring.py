from __future__ import annotations

import pytest

from application.detection_monitoring import (
    DETECTION_CLUSTER_IOU_THRESHOLD,
    box_iou,
    cluster_detection_predictions,
    cluster_member_label,
    normalized_box_rect,
    summarize_detection_overlaps,
)
from dashboard.views import live_sessions


def detection(index: int, x: float, y: float, class_name: str = "Good Weld", confidence: float = 0.8, **overrides) -> dict:
    row = {
        "id": f"prediction-{index}",
        "image_id": "image-1",
        "task_type": "object_detection",
        "predicted_class_name": class_name,
        "confidence": confidence,
        "x_center": x,
        "y_center": y,
        "width": 0.04,
        "height": 0.05,
    }
    row.update(overrides)
    return row


def test_clusters_competing_classes_and_uses_highest_confidence_representative():
    lower = detection(1, 0.5, 0.5, "Good Weld", 0.4)
    higher = detection(2, 0.5, 0.5, "Missing Weld", 0.9)

    clusters = cluster_detection_predictions([lower, higher])

    assert len(clusters) == 1
    assert clusters[0].members == (lower, higher)
    assert clusters[0].representative is higher
    assert clusters[0].is_ambiguous is True
    assert clusters[0].is_same_class_duplicate is False
    assert [cluster_member_label(clusters[0], index) for index in range(2)] == ["1a", "1b"]


def test_clusters_same_class_duplicates_separately_from_ambiguity():
    clusters = cluster_detection_predictions(
        [detection(1, 0.5, 0.5), detection(2, 0.5, 0.5, confidence=0.6)]
    )

    assert len(clusters) == 1
    assert clusters[0].is_ambiguous is False
    assert clusters[0].is_same_class_duplicate is True


def test_iou_threshold_is_inclusive_and_boxes_below_it_remain_separate():
    first = detection(1, 0.5, 0.5, width=0.2, height=0.2)
    second = detection(2, 0.5222222222222223, 0.5, width=0.2, height=0.2)
    overlap = box_iou(normalized_box_rect(first), normalized_box_rect(second))

    assert round(overlap, 8) == DETECTION_CLUSTER_IOU_THRESHOLD
    assert len(cluster_detection_predictions([first, second], iou_threshold=overlap)) == 1
    assert len(cluster_detection_predictions([first, second], iou_threshold=overlap + 0.000001)) == 2


def test_invalid_boxes_are_preserved_as_independent_locations():
    invalid = detection(1, 0.5, 0.5, width=None)
    valid = detection(2, 0.5, 0.5)

    clusters = cluster_detection_predictions([invalid, valid])

    assert len(clusters) == 2
    assert [cluster.index for cluster in clusters] == [1, 2]
    assert [cluster_member_label(cluster, 0) for cluster in clusters] == ["1", "2"]


def test_supplied_prediction_patterns_produce_expected_location_counts():
    image_five = [
        detection(index, x, y, class_name, confidence)
        for index, (x, y, class_name, confidence) in enumerate(
            [
                (0.653, 0.635, "Good Weld", 0.876),
                (0.512, 0.289, "Good Weld", 0.865),
                (0.502, 0.525, "Missing Weld", 0.785),
                (0.656, 0.407, "Burnt Weld", 0.778),
                (0.502, 0.752, "Incomplete Weld", 0.716),
                (0.351, 0.414, "Incomplete Weld", 0.666),
                (0.351, 0.634, "Missing Weld", 0.628),
                (0.656, 0.407, "Good Weld", 0.354),
            ],
            start=1,
        )
    ]
    image_ten = [detection(index, 0.1 * index, 0.5) for index in range(1, 8)]
    image_two = [
        detection(1, 0.504, 0.758),
        detection(2, 0.356, 0.649),
        detection(3, 0.660, 0.420, "Missing Weld"),
        detection(4, 0.354, 0.413),
        detection(5, 0.507, 0.295),
        detection(6, 0.656, 0.648),
        detection(7, 0.504, 0.532, "Missing Weld"),
        detection(8, 0.504, 0.532),
        detection(9, 0.657, 0.648, "Missing Weld"),
    ]

    assert len(cluster_detection_predictions(image_five)) == 7
    assert len(cluster_detection_predictions(image_ten)) == 7
    assert len(cluster_detection_predictions(image_two)) == 7
    assert [cluster.index for cluster in cluster_detection_predictions(image_two)] == list(range(1, 8))


def test_overlap_summary_keeps_raw_prediction_count():
    group = [detection(1, 0.5, 0.5), detection(2, 0.5, 0.5, "Missing Weld")]

    summary = summarize_detection_overlaps([group])

    assert summary == {
        "location_count": 1,
        "raw_prediction_count": 2,
        "overlapping_location_count": 1,
        "ambiguous_location_count": 1,
        "same_class_duplicate_count": 0,
        "affected_image_count": 1,
    }


def test_session_metrics_remain_raw_prediction_based_when_locations_are_clustered():
    rows = [
        detection(1, 0.5, 0.5, confidence=0.8),
        detection(2, 0.5, 0.5, "Missing Weld", confidence=0.4),
    ]

    class Images:
        @staticmethod
        def where(**filters):
            return [{"id": "image-1", "inference_status": "complete"}]

    class Predictions:
        @staticmethod
        def where_in(column, values):
            assert column == "image_id"
            assert values == ["image-1"]
            return rows

    stats = live_sessions.session_prediction_stats(Images(), Predictions(), {"id": "session-1"})

    assert stats["prediction_count"] == 2
    assert stats["average_confidence"] == pytest.approx(0.6)
    assert stats["low_confidence_count"] == 1
    assert stats["detection_location_count"] == 1
    assert stats["ambiguous_location_count"] == 1
    assert stats["overlap_affected_image_count"] == 1


def test_feedback_table_renders_cluster_subrows_with_original_predictions(monkeypatch):
    clusters = cluster_detection_predictions(
        [detection(1, 0.5, 0.5), detection(2, 0.5, 0.5, "Missing Weld")]
    )
    rendered = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def caption(self, *args, **kwargs):
            pass

    class FakeStreamlit:
        @staticmethod
        def container(**kwargs):
            return Context()

        @staticmethod
        def caption(*args, **kwargs):
            pass

        @staticmethod
        def columns(widths):
            return [Context() for _ in widths]

    monkeypatch.setattr(live_sessions, "st", FakeStreamlit)
    monkeypatch.setattr(
        live_sessions,
        "render_prediction_feedback_form",
        lambda client, model, session, image, prediction, feedback, label: rendered.append((prediction["id"], label)),
    )

    live_sessions.render_feedback_table(
        object(),
        {"selected_task_type": "object_detection"},
        {"id": "session-1"},
        {"id": "image-1"},
        list(clusters[0].members),
        object(),
        Context(),
        clusters,
    )

    assert rendered == [("prediction-1", "1a"), ("prediction-2", "1b")]

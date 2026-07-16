from __future__ import annotations

from PIL import Image

from application.detection_monitoring import cluster_detection_predictions
from dashboard.components.image_viewer import (
    DETECTION_COLORS,
    detection_badge_rect,
    detection_badge_scale,
    draw_detection_overlays,
)


def detection(index: int = 1, **overrides) -> dict:
    row = {
        "id": f"prediction-{index}",
        "task_type": "object_detection",
        "predicted_class_name": f"class-{index}",
        "confidence": 0.75,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.2,
        "height": 0.2,
    }
    row.update(overrides)
    return row


def test_detection_badge_scale_grows_with_resolution_and_stays_bounded():
    small = detection_badge_scale(320, 240)
    hd = detection_badge_scale(1920, 1080)
    large = detection_badge_scale(7680, 4320)

    assert small[0] == 18
    assert small[0] < hd[0] < large[0]
    assert large == (96, 18, 8, 12)


def test_detection_badge_rect_uses_above_or_inside_placement_and_clamps_to_image():
    above = detection_badge_rect((80, 70, 140, 130), 200, 150, 40, 30, 5, 4)
    top_edge = detection_badge_rect((0, 0, 60, 60), 200, 150, 40, 30, 5, 4)
    right_edge = detection_badge_rect((190, 140, 200, 150), 200, 150, 40, 30, 5, 4)
    tiny_image = detection_badge_rect((0, 0, 8, 8), 8, 8, 40, 30, 5, 4)

    assert above == (80, 35, 120, 65)
    assert top_edge == (0, 4, 40, 34)
    assert right_edge == (160, 105, 200, 135)
    assert tiny_image == (0, 0, 8, 8)
    for rect in (above, top_edge, right_edge):
        assert 0 <= rect[0] < rect[2] <= 200
        assert 0 <= rect[1] < rect[3] <= 150


def test_draw_detection_overlays_handles_dense_edge_detections_and_double_digit_badges(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (640, 480), "white").save(image_path)
    predictions = [
        detection(
            index,
            x_center=0.02 if index % 2 else 0.98,
            y_center=0.02 if index <= 6 else 0.98,
            width=0.04,
            height=0.04,
        )
        for index in range(1, 13)
    ]

    rendered = draw_detection_overlays(image_path, cluster_detection_predictions(predictions))

    assert rendered.size == (640, 480)
    assert rendered.mode == "RGB"
    assert rendered.tobytes() != Image.open(image_path).convert("RGB").tobytes()
    assert DETECTION_COLORS[0] == DETECTION_COLORS[6 % len(DETECTION_COLORS)]

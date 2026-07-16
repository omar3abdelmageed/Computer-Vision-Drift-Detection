from __future__ import annotations

from pathlib import Path

from application.detection_monitoring import DetectionCluster


DETECTION_COLORS = ("#00E5FF", "#FF1744", "#FFD600", "#00E676", "#FF9100", "#D500F9")


def yolo_box_to_pixel_rect(prediction: dict, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    values = [prediction.get(key) for key in ("x_center", "y_center", "width", "height")]
    if any(not isinstance(value, (int, float)) for value in values):
        return None
    x_center, y_center, box_width, box_height = [float(value) for value in values]
    left = int(round((x_center - box_width / 2) * image_width))
    top = int(round((y_center - box_height / 2) * image_height))
    right = int(round((x_center + box_width / 2) * image_width))
    bottom = int(round((y_center + box_height / 2) * image_height))
    left = max(0, min(image_width, left))
    right = max(0, min(image_width, right))
    top = max(0, min(image_height, top))
    bottom = max(0, min(image_height, bottom))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def draw_detection_overlays(path: str | Path, clusters: list[DetectionCluster]):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_size, badge_padding, badge_border, badge_gap = detection_badge_scale(image.width, image.height)
    font = ImageFont.load_default(size=font_size)
    for cluster in clusters:
        index = cluster.index
        prediction = cluster.representative
        rect = yolo_box_to_pixel_rect(prediction, image.width, image.height)
        if rect is None:
            continue
        rect = padded_rect(rect, image.width, image.height)
        color = DETECTION_COLORS[(index - 1) % len(DETECTION_COLORS)]
        line_width = max(4, image.width // 180)
        draw.rectangle(rect, outline="black", width=line_width + 2)
        draw.rectangle(rect, outline=color, width=line_width)
        draw_box_corners(draw, rect, color, line_width)
        badge_text = str(index)
        text_bbox = draw.textbbox((0, 0), badge_text, font=font, stroke_width=1)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        badge_width = min(image.width, text_width + badge_padding * 2)
        badge_height = min(image.height, text_height + badge_padding * 2)
        badge_rect = detection_badge_rect(
            rect,
            image.width,
            image.height,
            badge_width,
            badge_height,
            badge_gap,
            line_width,
        )
        draw.rounded_rectangle(
            badge_rect,
            radius=max(3, min(badge_width, badge_height) // 4),
            fill="black",
            outline=color,
            width=badge_border,
        )
        text_left = badge_rect[0] + (badge_width - text_width) / 2 - text_bbox[0]
        text_top = badge_rect[1] + (badge_height - text_height) / 2 - text_bbox[1]
        draw.text(
            (text_left, text_top),
            badge_text,
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )
    return image


def detection_badge_scale(image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Return resolution-aware font size, padding, border, and box gap."""
    short_edge = max(1, min(image_width, image_height))
    font_size = max(18, min(96, round(short_edge * 0.05)))
    padding = max(5, min(18, round(short_edge * 0.012)))
    border = max(2, min(8, round(short_edge * 0.004)))
    gap = max(3, min(12, round(short_edge * 0.006)))
    return font_size, padding, border, gap


def detection_badge_rect(
    detection_rect: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    badge_width: int,
    badge_height: int,
    gap: int,
    line_width: int,
) -> tuple[int, int, int, int]:
    """Place a badge above a box when possible and otherwise inside it."""
    left, top, _, _ = detection_rect
    badge_width = max(1, min(badge_width, image_width))
    badge_height = max(1, min(badge_height, image_height))
    badge_left = max(0, min(left, image_width - badge_width))
    if top >= badge_height + gap:
        badge_top = top - badge_height - gap
    else:
        badge_top = top + line_width
    badge_top = max(0, min(badge_top, image_height - badge_height))
    return badge_left, badge_top, badge_left + badge_width, badge_top + badge_height


def padded_rect(rect: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = rect
    pad = max(4, min(image_width, image_height) // 80)
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(image_width, right + pad),
        min(image_height, bottom + pad),
    )


def draw_box_corners(draw, rect: tuple[int, int, int, int], color: str, line_width: int) -> None:
    left, top, right, bottom = rect
    corner = max(14, min(right - left, bottom - top) // 5)
    segments = [
        ((left, top), (left + corner, top)),
        ((left, top), (left, top + corner)),
        ((right, top), (right - corner, top)),
        ((right, top), (right, top + corner)),
        ((left, bottom), (left + corner, bottom)),
        ((left, bottom), (left, bottom - corner)),
        ((right, bottom), (right - corner, bottom)),
        ((right, bottom), (right, bottom - corner)),
    ]
    for start, end in segments:
        draw.line((start, end), fill=color, width=line_width + 2)

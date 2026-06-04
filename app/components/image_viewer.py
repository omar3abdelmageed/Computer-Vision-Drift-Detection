from __future__ import annotations

from pathlib import Path
from typing import Iterable

import streamlit as st


def render_image(path: str | Path | None, caption: str | None = None) -> None:
    if not path:
        st.info("No image selected.")
        return
    path = Path(path)
    if not path.exists():
        st.warning(f"Image not found: {path}")
        return
    st.image(str(path), caption=caption or path.name, use_container_width=True)


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


def draw_detection_overlays(path: str | Path, predictions: Iterable[dict]):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ["#00E5FF", "#FF1744", "#FFD600", "#00E676", "#FF9100", "#D500F9"]
    font = ImageFont.load_default()
    for index, prediction in enumerate(predictions, start=1):
        rect = yolo_box_to_pixel_rect(prediction, image.width, image.height)
        if rect is None:
            continue
        rect = padded_rect(rect, image.width, image.height)
        color = colors[(index - 1) % len(colors)]
        label = f"{index}. {prediction.get('predicted_class_name') or 'object'}"
        confidence = prediction.get("confidence")
        if isinstance(confidence, (int, float)):
            label = f"{label} {confidence:.2f}"
        line_width = max(4, image.width // 180)
        draw.rectangle(rect, outline="black", width=line_width + 2)
        draw.rectangle(rect, outline=color, width=line_width)
        draw_box_corners(draw, rect, color, line_width)
        text_bbox = draw.textbbox((rect[0], rect[1]), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        label_top = max(0, rect[1] - text_height - 10)
        label_rect = (rect[0], label_top, rect[0] + text_width + 12, label_top + text_height + 8)
        draw.rectangle(label_rect, fill="black")
        draw.rectangle(label_rect, outline=color, width=2)
        draw.text((rect[0] + 6, label_top + 4), label, fill="white", font=font)
    return image


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

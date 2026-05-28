from __future__ import annotations

import time
from pathlib import Path


def run_yolo_inference(model_path: Path, image_path: Path, task_type: str) -> list[dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for YOLO inference.") from exc
    model = YOLO(str(model_path))
    started = time.perf_counter()
    results = model(str(image_path))
    inference_ms = (time.perf_counter() - started) * 1000
    if task_type == "classification":
        return parse_classification_results(results, inference_ms)
    return parse_detection_results(results, inference_ms)


def parse_classification_results(results, inference_ms: float) -> list[dict]:
    parsed: list[dict] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        probs = getattr(result, "probs", None)
        if probs is None:
            continue
        top1 = int(probs.top1)
        top5 = [int(i) for i in getattr(probs, "top5", [])]
        confs = [float(c) for c in getattr(probs, "top5conf", [])]
        parsed.append(
            {
                "task_type": "classification",
                "predicted_class_id": top1,
                "predicted_class_name": str(names.get(top1, top1)),
                "confidence": float(probs.top1conf),
                "top_k": [{"class_id": cls, "class_name": str(names.get(cls, cls)), "confidence": confs[idx] if idx < len(confs) else None} for idx, cls in enumerate(top5)],
                "inference_time_ms": inference_ms,
                "raw_prediction": {},
            }
        )
    return parsed


def parse_detection_results(results, inference_ms: float) -> list[dict]:
    parsed: list[dict] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xywhn = boxes.xywhn.cpu().tolist() if hasattr(boxes.xywhn, "cpu") else boxes.xywhn.tolist()
        cls_values = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.tolist()
        conf_values = boxes.conf.cpu().tolist() if hasattr(boxes.conf, "cpu") else boxes.conf.tolist()
        for coords, class_id, conf in zip(xywhn, cls_values, conf_values):
            cid = int(class_id)
            parsed.append(
                {
                    "task_type": "object_detection",
                    "predicted_class_id": cid,
                    "predicted_class_name": str(names.get(cid, cid)),
                    "confidence": float(conf),
                    "top_k": None,
                    "x_center": float(coords[0]),
                    "y_center": float(coords[1]),
                    "width": float(coords[2]),
                    "height": float(coords[3]),
                    "inference_time_ms": inference_ms,
                    "raw_prediction": {},
                }
            )
    return parsed

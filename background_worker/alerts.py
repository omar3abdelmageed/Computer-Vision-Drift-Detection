from __future__ import annotations

from core.types import DriftResult, DriftStatus


METRIC_LABELS = {
    "total_data_drift": "Total data drift",
    "brightness": "Brightness drift",
    "contrast": "Contrast drift",
    "colour_distribution": "Colour distribution drift",
    "noise": "Noise drift",
    "prediction_class_distribution": "Prediction drift",
    "human_review_summary": "Review accuracy drift",
}

METRIC_GUIDANCE = {
    "total_data_drift": "Production images look different from the training baseline. Check source changes, camera setup, lighting, blur, or image preprocessing.",
    "brightness": "Production brightness differs from the training baseline. Check lighting, camera exposure, or source changes.",
    "contrast": "Production contrast differs from the training baseline. Check lighting conditions, focus, camera settings, or preprocessing.",
    "colour_distribution": "Production colour distribution differs from the training baseline. Check camera white balance, lighting, or image source changes.",
    "noise": "Production noise differs from the training baseline. Check blur, compression, low-light capture, or camera quality.",
    "prediction_class_distribution": "The model is predicting a different class mix than expected. Check whether the incoming batch or class balance has changed.",
    "human_review_summary": "Human review suggests model quality has dropped. Review labels, inspect recent mistakes, or consider retraining.",
}


def alerts_from_drift(results: list[DriftResult]) -> list[dict]:
    alerts: list[dict] = []
    for result in results:
        if result.status == DriftStatus.OK:
            continue
        severity = "critical" if result.status == DriftStatus.CRITICAL else "warning"
        value = f"{result.metric_value:.4f}" if isinstance(result.metric_value, (int, float)) else "not available"
        metric_label = METRIC_LABELS.get(result.metric_name, result.metric_name.replace("_", " ").title())
        guidance = METRIC_GUIDANCE.get(result.metric_name, "Review this drift metric and compare production inputs with the training baseline.")
        if result.metric_name == "prediction_class_distribution" and result.status == DriftStatus.CRITICAL:
            guidance = f"{guidance} Production monitoring has been automatically paused while data drift diagnostics are calculated."
        alerts.append(
            {
                "severity": severity,
                "title": f"{metric_label} {severity}",
                "message": f"{guidance} Latest value: {value}.",
                "details": result.details,
            }
        )
    return alerts

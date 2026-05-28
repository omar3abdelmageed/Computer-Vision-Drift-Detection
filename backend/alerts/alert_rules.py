from __future__ import annotations

from backend.common import DriftResult, DriftStatus


def alerts_from_drift(results: list[DriftResult]) -> list[dict]:
    alerts: list[dict] = []
    for result in results:
        if result.status == DriftStatus.OK:
            continue
        severity = "critical" if result.status == DriftStatus.CRITICAL else "warning"
        alerts.append(
            {
                "severity": severity,
                "title": f"{result.metric_name} {severity}",
                "message": f"{result.drift_type.value} drift metric {result.metric_name} is {result.metric_value:.4f} against threshold {result.threshold:.4f}.",
                "details": result.details,
            }
        )
    return alerts

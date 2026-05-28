from __future__ import annotations

from typing import Sequence

from backend.common import DriftResult, DriftStatus, DriftType
from backend.drift.statistical_tests import (
    TestOutput,
    categorical_distribution_drift,
    centroid_distance,
    domain_classifier_drift,
    energy_distance_test,
    jensen_shannon_numeric,
    ks_test,
    nearest_neighbor_distance,
    population_stability_index,
    rbf_mmd,
    wasserstein_test,
)


def status_from_test(output: TestOutput) -> DriftStatus:
    if not output.drift_detected:
        return DriftStatus.OK
    if output.p_value is not None and output.p_value <= 0.01:
        return DriftStatus.CRITICAL
    if output.statistic >= output.threshold * 2:
        return DriftStatus.CRITICAL
    return DriftStatus.WARNING


def numeric_feature_drift(metric_name: str, reference: Sequence[float], current: Sequence[float]) -> list[DriftResult]:
    outputs = [
        ks_test(reference, current),
        wasserstein_test(reference, current),
        population_stability_index(reference, current),
        jensen_shannon_numeric(reference, current),
        energy_distance_test(reference, current),
    ]
    return [to_drift_result(DriftType.DATA, f"{metric_name}_{output.name}", output) for output in outputs]


def categorical_prediction_drift(metric_name: str, reference: Sequence[str | int], current: Sequence[str | int]) -> DriftResult:
    return to_drift_result(DriftType.PREDICTION, metric_name, categorical_distribution_drift(reference, current))


def embedding_drift(reference_embeddings: Sequence[Sequence[float]], current_embeddings: Sequence[Sequence[float]]) -> list[DriftResult]:
    outputs = [
        centroid_distance(reference_embeddings, current_embeddings),
        nearest_neighbor_distance(reference_embeddings, current_embeddings),
        rbf_mmd(reference_embeddings, current_embeddings),
        domain_classifier_drift(reference_embeddings, current_embeddings),
    ]
    return [to_drift_result(DriftType.DATA, output.name, output) for output in outputs]


def to_drift_result(drift_type: DriftType, metric_name: str, output: TestOutput) -> DriftResult:
    return DriftResult(
        drift_type=drift_type,
        metric_name=metric_name,
        metric_value=output.statistic,
        threshold=output.threshold,
        status=status_from_test(output),
        details={"test_name": output.name, "p_value": output.p_value, **output.details},
    )

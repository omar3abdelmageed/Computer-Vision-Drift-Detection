from __future__ import annotations

from typing import Sequence

from background_worker.data_drift_tests.image_property_drift import property_value


def feature_vector(row: dict) -> list[float]:
    return [
        float(row.get("brightness_mean") or 0.0),
        float(row.get("contrast") or 0.0),
        property_value("colour_distribution", row),
        property_value("noise", row),
    ]


def feature_vectors(feature_rows: Sequence[dict]) -> list[list[float]]:
    return [feature_vector(row) for row in feature_rows]


def rbf_mmd(reference_vectors: Sequence[Sequence[float]], current_vectors: Sequence[Sequence[float]]) -> float:
    if not reference_vectors or not current_vectors:
        return 0.0
    import numpy as np

    ref = np.asarray(reference_vectors, dtype=float)
    cur = np.asarray(current_vectors, dtype=float)
    combined = np.vstack([ref, cur])
    variance = float(np.var(combined))
    gamma = 1.0 / (combined.shape[1] * variance + 1e-8)
    k_xx = rbf_kernel_mean(ref, ref, gamma)
    k_yy = rbf_kernel_mean(cur, cur, gamma)
    k_xy = rbf_kernel_mean(ref, cur, gamma)
    return float(max(0.0, k_xx + k_yy - 2 * k_xy))


def rbf_kernel_mean(left, right, gamma: float) -> float:
    import numpy as np

    diff = left[:, None, :] - right[None, :, :]
    squared_distances = np.sum(diff * diff, axis=2)
    return float(np.exp(-gamma * squared_distances).mean())

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TestOutput:
    name: str
    statistic: float
    p_value: float | None
    threshold: float
    drift_detected: bool
    details: dict


def _as_floats(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def ks_test(reference: Sequence[float], current: Sequence[float], alpha: float = 0.05) -> TestOutput:
    ref, cur = _as_floats(reference), _as_floats(current)
    from scipy.stats import ks_2samp
    result = ks_2samp(ref, cur, alternative="two-sided", method="auto")
    return TestOutput("ks_2samp", float(result.statistic), float(result.pvalue), alpha, float(result.pvalue) <= alpha, {})


def wasserstein_test(reference: Sequence[float], current: Sequence[float], threshold: float = 0.1) -> TestOutput:
    from scipy.stats import wasserstein_distance
    value = float(wasserstein_distance(_as_floats(reference), _as_floats(current)))
    return TestOutput("wasserstein_distance", value, None, threshold, value >= threshold, {})


def energy_distance_test(reference: Sequence[float], current: Sequence[float], threshold: float = 0.1) -> TestOutput:
    from scipy.stats import energy_distance
    value = float(energy_distance(_as_floats(reference), _as_floats(current)))
    return TestOutput("energy_distance", value, None, threshold, value >= threshold, {})


def population_stability_index(reference: Sequence[float], current: Sequence[float], bins: int = 10, threshold: float = 0.2) -> TestOutput:
    import numpy as np
    ref, cur = np.asarray(_as_floats(reference)), np.asarray(_as_floats(current))
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        return TestOutput("psi", 0.0, None, threshold, False, {"reason": "constant_reference"})
    ref_counts, _ = np.histogram(ref, bins=quantiles)
    cur_counts, _ = np.histogram(cur, bins=quantiles)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-8)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-8)
    psi = float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())
    return TestOutput("population_stability_index", psi, None, threshold, psi >= threshold, {"bins": quantiles.tolist()})


def jensen_shannon_numeric(reference: Sequence[float], current: Sequence[float], bins: int = 20, threshold: float = 0.1) -> TestOutput:
    import numpy as np
    from scipy.spatial.distance import jensenshannon
    ref, cur = np.asarray(_as_floats(reference)), np.asarray(_as_floats(current))
    edges = np.histogram_bin_edges(ref, bins=bins)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = ref_counts / max(ref_counts.sum(), 1)
    cur_p = cur_counts / max(cur_counts.sum(), 1)
    value = float(jensenshannon(ref_p, cur_p, base=2.0))
    return TestOutput("jensen_shannon_numeric", value, None, threshold, value >= threshold, {})


def categorical_distribution_drift(reference: Sequence[str | int], current: Sequence[str | int], threshold: float = 0.1) -> TestOutput:
    from collections import Counter
    import numpy as np
    from scipy.spatial.distance import jensenshannon
    labels = sorted(set(reference) | set(current), key=str)
    ref_counts = Counter(reference)
    cur_counts = Counter(current)
    ref = np.asarray([ref_counts[label] for label in labels], dtype=float)
    cur = np.asarray([cur_counts[label] for label in labels], dtype=float)
    ref = ref / max(ref.sum(), 1.0)
    cur = cur / max(cur.sum(), 1.0)
    value = float(jensenshannon(ref, cur, base=2.0))
    deltas = {str(label): float(cur[i] - ref[i]) for i, label in enumerate(labels)}
    return TestOutput("jensen_shannon_categorical", value, None, threshold, value >= threshold, {"labels": [str(v) for v in labels], "proportion_deltas": deltas})


def centroid_distance(reference_embeddings: Sequence[Sequence[float]], current_embeddings: Sequence[Sequence[float]], threshold: float = 0.1) -> TestOutput:
    import numpy as np
    ref = np.asarray(reference_embeddings, dtype=float)
    cur = np.asarray(current_embeddings, dtype=float)
    value = float(np.linalg.norm(ref.mean(axis=0) - cur.mean(axis=0)))
    return TestOutput("centroid_distance", value, None, threshold, value >= threshold, {})


def nearest_neighbor_distance(reference_embeddings: Sequence[Sequence[float]], current_embeddings: Sequence[Sequence[float]], threshold: float = 0.1) -> TestOutput:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    ref = np.asarray(reference_embeddings, dtype=float)
    cur = np.asarray(current_embeddings, dtype=float)
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    distances, _ = nn.kneighbors(cur)
    value = float(distances.mean())
    return TestOutput("nearest_neighbor_distance", value, None, threshold, value >= threshold, {})


def rbf_mmd(reference_embeddings: Sequence[Sequence[float]], current_embeddings: Sequence[Sequence[float]], threshold: float = 0.05, gamma: float | None = None) -> TestOutput:
    import numpy as np
    from sklearn.metrics.pairwise import rbf_kernel
    ref = np.asarray(reference_embeddings, dtype=float)
    cur = np.asarray(current_embeddings, dtype=float)
    combined = np.vstack([ref, cur])
    if gamma is None:
        variance = np.var(combined)
        gamma = 1.0 / (combined.shape[1] * variance + 1e-8)
    k_xx = rbf_kernel(ref, ref, gamma=gamma).mean()
    k_yy = rbf_kernel(cur, cur, gamma=gamma).mean()
    k_xy = rbf_kernel(ref, cur, gamma=gamma).mean()
    value = float(k_xx + k_yy - 2 * k_xy)
    return TestOutput("rbf_mmd", value, None, threshold, value >= threshold, {"gamma": gamma})


def domain_classifier_drift(reference_embeddings: Sequence[Sequence[float]], current_embeddings: Sequence[Sequence[float]], threshold: float = 0.6) -> TestOutput:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    x = np.vstack([reference_embeddings, current_embeddings])
    y = np.asarray([0] * len(reference_embeddings) + [1] * len(current_embeddings))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    score = float(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
    return TestOutput("domain_classifier_roc_auc", score, None, threshold, score >= threshold, {})


def online_kswin(values: Sequence[float], alpha: float = 0.005, window_size: int = 100, stat_size: int = 30) -> TestOutput:
    try:
        from river.drift import KSWIN
    except ImportError as exc:
        raise RuntimeError("river is required for KSWIN online drift detection.") from exc
    detector = KSWIN(alpha=alpha, window_size=window_size, stat_size=stat_size)
    drift_indexes: list[int] = []
    for idx, value in enumerate(values):
        in_drift, _ = detector.update(float(value))
        if in_drift:
            drift_indexes.append(idx)
    return TestOutput("kswin", float(len(drift_indexes)), None, 1.0, bool(drift_indexes), {"drift_indexes": drift_indexes})


def online_adwin(values: Sequence[float], delta: float = 0.002) -> TestOutput:
    try:
        from river.drift import ADWIN
    except ImportError as exc:
        raise RuntimeError("river is required for ADWIN online drift detection.") from exc
    detector = ADWIN(delta=delta)
    drift_indexes: list[int] = []
    for idx, value in enumerate(values):
        detector.update(float(value))
        if detector.drift_detected:
            drift_indexes.append(idx)
    return TestOutput("adwin", float(len(drift_indexes)), None, 1.0, bool(drift_indexes), {"drift_indexes": drift_indexes})

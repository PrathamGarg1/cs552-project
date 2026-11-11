"""Kolmogorov-Smirnov feature drift monitoring (paper Sec. 3.4, Alg. 1 Step 2).

For every incoming mini-batch (B = 500) the empirical distribution of each
feature is compared against a reference window (the rolling buffer, size 1500)
with a two-sample KS test:

    D_j = sup_z | F_ref,j(z) - F_cur,j(z) |

A feature has drifted when the test's p-value is below delta = 0.05
(Table 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class DriftReport:
    statistic: np.ndarray
    p_value: np.ndarray
    drifted: np.ndarray  # bool per feature


@dataclass
class KSDriftMonitor:
    feature_names: list[str]
    delta: float = 0.05  # paper Table 4
    reference_size: int = 1500  # paper Table 4 buffer size

    reference_: np.ndarray | None = field(default=None, init=False, repr=False)
    drift_counts_: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.drift_counts_ = np.zeros(len(self.feature_names), dtype=int)

    def set_reference(self, X: np.ndarray) -> None:
        self.reference_ = np.asarray(X, dtype=np.float64)[-self.reference_size :].copy()

    def test(self, X_batch: np.ndarray) -> DriftReport:
        if self.reference_ is None:
            raise RuntimeError("call set_reference() first")
        X_batch = np.asarray(X_batch, dtype=np.float64)
        d = X_batch.shape[1]
        stat = np.zeros(d)
        p = np.ones(d)
        for j in range(d):
            ref, cur = self.reference_[:, j], X_batch[:, j]
            if ref.min() == ref.max() == cur.min() == cur.max():
                continue  # identical constant feature: no drift, avoid degenerate test
            res = stats.ks_2samp(ref, cur, alternative="two-sided", method="asymp")
            stat[j], p[j] = res.statistic, res.pvalue
        drifted = p < self.delta
        self.drift_counts_ += drifted.astype(int)
        return DriftReport(stat, p, drifted)

    @property
    def drift_counts(self) -> dict[str, int]:
        return {n: int(c) for n, c in zip(self.feature_names, self.drift_counts_)}

"""Adaptive feature selection - paper Sec. 3.4, Algorithm 1.

Step 1 (initial selection)
    w  = |Lasso.coef|                      (eq. 1, lambda = 0.0001, Table 4)
    I  = mutual_info(X_B, y_B)             (eq. 2)
    S  = w + I                             (combined importance, Alg. 1 line 7)
    tau = Percentile(S, 75);  F* = {f : S(f) >= tau}   (Alg. 1 lines 16-17)

Step 2 (on drift)
    Features whose KS test flagged drift are marked obsolete (removed from F*),
    LASSO + MI are re-executed on the recent buffer, and the threshold is
    re-applied over *all* candidate features, so obsolete features can be
    dropped and previously unselected ones are added ("feature space
    expansion"). Scores are blended with the stored importance history H
    (Alg. 1 line 19) with weight ``history_weight`` to damp noise from small
    buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


@dataclass
class DynamicFeatureSelector:
    feature_names: list[str]
    lasso_alpha: float = 1e-4  # paper Table 4: lambda = 0.0001
    lasso_max_iter: int = 1000  # paper Table 4
    percentile: float = 75.0  # paper Alg. 1: tau = Percentile(S, 75)
    history_weight: float = 0.5
    min_features: int = 5
    mi_max_samples: int = 20000
    random_state: int = 42

    selected_: np.ndarray = field(init=False)  # boolean mask over all features
    scores_: np.ndarray = field(init=False)
    lasso_w_: np.ndarray = field(init=False)
    mi_: np.ndarray = field(init=False)
    tau_: float = field(init=False, default=0.0)
    history_: list[np.ndarray] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------
    def _score(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int).ravel()
        if len(X) > self.mi_max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), self.mi_max_samples, replace=False)
            X, y = X[idx], y[idx]
        Xs = StandardScaler().fit_transform(X)
        lasso = Lasso(alpha=self.lasso_alpha, max_iter=self.lasso_max_iter)
        lasso.fit(Xs, y.astype(float))
        w = np.abs(lasso.coef_)
        mi = mutual_info_classif(Xs, y, random_state=self.random_state)
        return w, mi

    def _apply_threshold(self, S: np.ndarray) -> np.ndarray:
        tau = float(np.percentile(S, self.percentile))
        mask = S >= tau
        if mask.sum() < self.min_features:
            mask = np.zeros_like(mask)
            mask[np.argsort(S)[::-1][: self.min_features]] = True
        self.tau_ = tau
        return mask

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DynamicFeatureSelector":
        """Alg. 1 Step 1: initial LASSO + MI selection."""
        w, mi = self._score(X, y)
        self.lasso_w_, self.mi_ = w, mi
        self.scores_ = w + mi
        self.history_ = [self.scores_.copy()]
        self.selected_ = self._apply_threshold(self.scores_)
        return self

    def update(
        self, X: np.ndarray, y: np.ndarray, drifted: np.ndarray | None = None
    ) -> dict[str, list[str]]:
        """Alg. 1 Step 2: re-select on the recent buffer after a drift event."""
        prev = self.selected_.copy()
        drifted = np.zeros_like(prev) if drifted is None else np.asarray(drifted, dtype=bool)
        obsolete = prev & drifted
        w, mi = self._score(X, y)
        S_new = w + mi
        S = (1.0 - self.history_weight) * S_new + self.history_weight * self.history_[-1]
        self.lasso_w_, self.mi_, self.scores_ = w, mi, S
        self.history_.append(S.copy())
        self.selected_ = self._apply_threshold(S)
        names = np.asarray(self.feature_names)
        return {
            "obsolete": names[obsolete].tolist(),
            "removed": names[prev & ~self.selected_].tolist(),
            "added": names[~prev & self.selected_].tolist(),
        }

    # ------------------------------------------------------------------
    @property
    def selected_indices_(self) -> np.ndarray:
        return np.flatnonzero(self.selected_)

    @property
    def selected_names(self) -> list[str]:
        return [self.feature_names[i] for i in self.selected_indices_]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.selected_indices_]

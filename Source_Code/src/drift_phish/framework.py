"""End-to-end adaptive drift-aware phishing URL detector (paper Fig. 1-3).

                    +-- offline phase (batch) ---------------------------+
 labelled URLs ---> | lexical features -> LASSO+MI (F*) -> SGD (Table 4)  |
                    +----------------------------------------------------+
                    +-- online phase (per mini-batch, B = 500) -----------+
 URL stream ------> | 1. predict batch (prequential / test-then-train)    |
                    | 2. KS test of every feature vs. reference window    |
                    | 3. drift in F*  -> re-run LASSO+MI on the 1500-row  |
                    |    buffer (obsolete features out, new ones in) and  |
                    |    partially retrain the SGD weights on the buffer  |
                    | 4. SGD partial_fit on the batch                     |
                    +----------------------------------------------------+

``mode`` selects the system variant, which is what the ablation / baselines
in the experiments compare:

  adaptive        proposed framework (KS drift + adaptive F* + online SGD)
  online_static   online SGD but F* frozen at its initial value
  frozen          offline model only, never updated
  periodic_batch  conventional batch baseline: every ``retrain_every`` URLs
                  re-run selection + full SGD training from scratch on the
                  last ``retrain_window`` labelled URLs
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .drift.ks_monitor import KSDriftMonitor
from .learning.online_sgd import OnlineSGDClassifier
from .selection.dynamic import DynamicFeatureSelector

MODES = ("adaptive", "online_static", "frozen", "periodic_batch")


@dataclass
class FrameworkConfig:
    """Defaults = paper Table 4."""

    mode: str = "adaptive"
    loss: str = "hinge"  # SGD-SVM
    eta0: float = 0.01
    lasso_alpha: float = 1e-4
    ks_delta: float = 0.05
    buffer_size: int = 1500
    batch_size: int = 500
    random_state: int = 42
    # --- design choices the paper leaves open --------------------------------
    drift_min_fraction: float = 0.25  # fraction of F* that must drift to trigger
    min_buffer_for_reselect: int = 100
    replay_epochs: int = 3
    history_weight: float = 0.5
    # --- periodic batch baseline --------------------------------------------
    retrain_every: int = 5000
    retrain_window: int = 20000


@dataclass
class DriftAwarePhishingDetector:
    feature_names: list[str]
    config: FrameworkConfig = field(default_factory=FrameworkConfig)

    def __post_init__(self) -> None:
        assert self.config.mode in MODES, self.config.mode
        c = self.config
        self.selector = DynamicFeatureSelector(
            feature_names=list(self.feature_names),
            lasso_alpha=c.lasso_alpha,
            history_weight=c.history_weight,
            random_state=c.random_state,
        )
        self.monitor = KSDriftMonitor(
            feature_names=list(self.feature_names),
            delta=c.ks_delta,
            reference_size=c.buffer_size,
        )
        self.model = OnlineSGDClassifier(loss=c.loss, eta0=c.eta0, random_state=c.random_state)
        self.buf_X = np.empty((0, len(self.feature_names)))
        self.buf_y = np.empty((0,), dtype=int)
        self.events: list[dict] = []
        self._since_retrain = 0
        self._hist_X: list[np.ndarray] = []
        self._hist_y: list[np.ndarray] = []
        self.timings = {"offline_s": 0.0, "update_s": 0.0, "reselect_s": 0.0, "ks_s": 0.0}

    # ------------------------------------------------------------- offline
    def fit_offline(self, X: np.ndarray, y: np.ndarray) -> "DriftAwarePhishingDetector":
        """Initial LASSO+MI selection and offline SGD training (Table 4)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        t0 = time.perf_counter()
        self.selector.fit(X, y)
        self.model.fit_offline(X, y, mask=self.selector.selected_.astype(float))
        self.timings["offline_s"] = time.perf_counter() - t0
        rng = np.random.default_rng(self.config.random_state)
        ref_idx = rng.choice(len(X), min(self.config.buffer_size, len(X)), replace=False)
        self.monitor.set_reference(X[ref_idx])
        self._last_offline = (X, y)
        self.events.append(
            {"t": 0, "type": "initial", "features": self.selector.selected_names}
        )
        return self

    # -------------------------------------------------------------- online
    def _push_buffer(self, Xb: np.ndarray, yb: np.ndarray) -> None:
        self.buf_X = np.vstack([self.buf_X, Xb])[-self.config.buffer_size :]
        self.buf_y = np.concatenate([self.buf_y, yb])[-self.config.buffer_size :]

    def _trigger(self, drifted: np.ndarray) -> bool:
        sel = self.selector.selected_
        n_sel = int(sel.sum())
        need = max(1, math.ceil(self.config.drift_min_fraction * n_sel))
        return int((drifted & sel).sum()) >= need

    def process_batch(self, Xb: np.ndarray, yb: np.ndarray, t: int = 0):
        """Prequential step: returns (predictions, scores) made BEFORE learning."""
        c = self.config
        Xb = np.asarray(Xb, dtype=np.float64)
        yb = np.asarray(yb).astype(int)
        scores = self.model.decision_function(Xb)
        pred = (scores > 0).astype(int)

        if c.mode == "frozen":
            return pred, scores

        self._push_buffer(Xb, yb)

        if c.mode == "periodic_batch":
            self._hist_X.append(Xb)
            self._hist_y.append(yb)
            self._since_retrain += len(Xb)
            if self._since_retrain >= c.retrain_every:
                t0 = time.perf_counter()
                HX = np.vstack(self._hist_X)[-c.retrain_window :]
                Hy = np.concatenate(self._hist_y)[-c.retrain_window :]
                self.selector.fit(HX, Hy)
                self.model.fit_offline(HX, Hy, mask=self.selector.selected_.astype(float))
                self.timings["update_s"] += time.perf_counter() - t0
                self._since_retrain = 0
                self.events.append(
                    {"t": t, "type": "batch_retrain", "features": self.selector.selected_names}
                )
            return pred, scores

        if c.mode == "adaptive":
            t0 = time.perf_counter()
            rep = self.monitor.test(Xb)
            self.timings["ks_s"] += time.perf_counter() - t0
            if self._trigger(rep.drifted) and len(self.buf_X) >= c.min_buffer_for_reselect:
                t0 = time.perf_counter()
                info = self.selector.update(self.buf_X, self.buf_y, drifted=rep.drifted)
                self.model.set_mask(self.selector.selected_.astype(float))
                # partial retraining: refresh weights on the rolling buffer
                self.model.partial_fit(self.buf_X, self.buf_y, epochs=c.replay_epochs)
                self.monitor.set_reference(self.buf_X)
                self.timings["reselect_s"] += time.perf_counter() - t0
                self.events.append(
                    {
                        "t": t,
                        "type": "drift",
                        "n_drifted_features": int(rep.drifted.sum()),
                        "drifted_in_Fstar": int((rep.drifted & self.selector.selected_).sum()),
                        "ks_stat": rep.statistic.tolist(),
                        "p_value": rep.p_value.tolist(),
                        "features": self.selector.selected_names,
                        **info,
                    }
                )

        t0 = time.perf_counter()
        self.model.partial_fit(Xb, yb)
        self.timings["update_s"] += time.perf_counter() - t0
        return pred, scores

    def process_stream(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Prequential (test-then-train) evaluation over mini-batches of B."""
        B = self.config.batch_size
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        preds, scores, nsel = [], [], []
        for s in range(0, len(X), B):
            p, sc = self.process_batch(X[s : s + B], y[s : s + B], t=s)
            preds.append(p)
            scores.append(sc)
            nsel.append(int(self.selector.selected_.sum()))
        return {
            "y_true": y,
            "y_pred": np.concatenate(preds),
            "score": np.concatenate(scores),
            "n_selected": nsel,
        }

    # ------------------------------------------------------------- predict
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(np.asarray(X, dtype=np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float64))

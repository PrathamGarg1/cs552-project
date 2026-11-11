"""SGD linear classifiers: offline batch phase + online mini-batch phase.

Paper Sec. 3.5 / 4 / Table 4
  * offline phase : SGD, adaptive learning rate, eta0 = 0.01, max 1000 iters,
                    early stopping with a 10 % validation split, balanced
                    class weights, seed 42.
  * online phase  : the same linear model is updated with ``partial_fit`` on
                    mini-batches (B = 500) taken from the rolling buffer.
Loss selects the SGD variant compared in the paper's Tables 5-7:
  hinge -> SGD-SVM (best), log_loss -> SGD-LR, perceptron -> SGD-Perceptron.

The model always lives in the FULL feature space; the currently selected
feature subset F* is applied as a 0/1 mask on the standardised inputs. This
lets the feature set change over time while keeping the learned weights
(partial retraining) instead of rebuilding the classifier from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

LOSS_NAMES = {"hinge": "SGD-SVM", "log_loss": "SGD-LR", "perceptron": "SGD-Perceptron"}


@dataclass
class OnlineSGDClassifier:
    loss: str = "hinge"
    eta0: float = 0.01
    alpha: float = 1e-4
    learning_rate: str = "adaptive"
    max_iter: int = 1000
    validation_fraction: float = 0.1
    random_state: int = 42
    clip: float = 10.0  # robustness: clip standardised features to +-clip

    scaler_: StandardScaler = field(init=False, repr=False)
    clf_: SGDClassifier = field(init=False, repr=False)
    mask_: np.ndarray = field(init=False, repr=False)
    fitted_: bool = field(default=False, init=False)

    def _make(self, **kw) -> SGDClassifier:
        return SGDClassifier(
            loss=self.loss,
            penalty="l2",
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            random_state=self.random_state,
            **kw,
        )

    def _prep(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        np.clip(Xs, -self.clip, self.clip, out=Xs)
        return Xs * self.mask_

    # ---------------------------------------------------------------- offline
    def fit_offline(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        self.mask_ = (
            np.ones(X.shape[1]) if mask is None else np.asarray(mask, dtype=np.float64)
        )
        self.scaler_ = StandardScaler().fit(X)
        self.clf_ = self._make(
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=5,
            class_weight="balanced",
        )
        self.clf_.fit(self._prep(X), np.asarray(y).astype(int))
        self.fitted_ = True
        return self

    # ----------------------------------------------------------------- online
    def set_mask(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=np.float64)
        self.clf_.coef_[:, mask == 0] = 0.0  # features leaving F* lose their weight
        self.mask_ = mask

    def partial_fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1):
        y = np.asarray(y).astype(int).ravel()
        Xs = self._prep(X)
        n1 = int(y.sum())
        n0 = len(y) - n1
        if n0 and n1:
            sw = np.where(y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * n0))
        else:
            sw = np.ones(len(y))
        for _ in range(max(1, epochs)):
            self.clf_.partial_fit(Xs, y, sample_weight=sw)
        return self

    # ---------------------------------------------------------------- predict
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.decision_function(self._prep(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)

    @property
    def coef_(self) -> np.ndarray:
        return self.clf_.coef_.ravel()

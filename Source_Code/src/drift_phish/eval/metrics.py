"""Evaluation helpers.

Convention used everywhere in this package: label ``1`` = PHISHING (positive
class), label ``0`` = legitimate. FPR = legitimate URLs flagged as phishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Accuracy / precision / recall / F1 / FPR / FNR for the phishing class."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y_true)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


@dataclass
class StreamMetrics:
    """Accumulates (y_true, y_pred) pairs of a prequential evaluation."""

    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)

    def extend(self, y_true, y_pred) -> None:
        self.y_true.extend(np.asarray(y_true).astype(int).tolist())
        self.y_pred.extend(np.asarray(y_pred).astype(int).tolist())

    def summary(self) -> dict[str, float]:
        return classification_report_dict(np.asarray(self.y_true), np.asarray(self.y_pred))

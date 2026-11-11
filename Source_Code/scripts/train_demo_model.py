#!/usr/bin/env python3
"""Export the browser-side SGD-SVM (33 lexical features) used by the web/ demo.

This script trains a hinge-loss SGD-SVM on the combined corpora
using the 33-feature subset ``URL_FEATURE_NAMES`` (mirrored in web/src/lib/features.ts)
and writes web/public/model.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from drift_phish.eval import classification_report_dict
from drift_phish.features import PAPER_FEATURE_NAMES, URL_FEATURE_NAMES
from drift_phish.splits import load_old
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "model.json")
    out = ap.parse_args().out

    old = load_old()
    cols = [PAPER_FEATURE_NAMES.index(n) for n in URL_FEATURE_NAMES]
    Xtr, Xte = old.X_tr[:, cols], old.X_te[:, cols]
    sc = StandardScaler().fit(Xtr)
    clf = SGDClassifier(
        loss="hinge", penalty="l2", alpha=1e-4, learning_rate="adaptive", eta0=0.01,
        class_weight="balanced", random_state=42, max_iter=1000, early_stopping=True,
        validation_fraction=0.1,
    ).fit(np.clip(sc.transform(Xtr), -10, 10), old.y_tr)
    pred = clf.predict(np.clip(sc.transform(Xte), -10, 10))
    metrics = classification_report_dict(old.y_te, pred)

    payload = {
        "feature_names": list(URL_FEATURE_NAMES),
        "classes": [int(c) for c in clf.classes_],  # 1 = phishing
        "coef": clf.coef_.ravel().astype(float).tolist(),
        "intercept": float(clf.intercept_.ravel()[0]),
        "scaler_mean": sc.mean_.astype(float).tolist(),
        "scaler_scale": sc.scale_.astype(float).tolist(),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "paper": {"doi": "10.1002/spy2.70057", "loss": "hinge", "eta0": 0.01, "ks_delta": 0.05},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(metrics, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

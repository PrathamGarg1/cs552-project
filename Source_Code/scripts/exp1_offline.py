#!/usr/bin/env python3
"""Experiment 1 - offline phase (paper Tables 5-7, Fig. 6).

LASSO + MI selected features -> SGD-SVM / SGD-LR / SGD-Perceptron, per corpus
and on the combined corpus, 80/20 split, seed 42, Table 4 hyper-parameters.
Also reports "no feature selection" and classical batch baselines.
"""

from __future__ import annotations

import time

import _common as C
import numpy as np
from drift_phish.data import load_features, load_iscx
from drift_phish.eval import classification_report_dict as R
from drift_phish.features import PAPER_FEATURE_NAMES as F
from drift_phish.learning import LOSS_NAMES, OnlineSGDClassifier
from drift_phish.selection import DynamicFeatureSelector
from drift_phish.splits import OLD_SOURCES, load_old
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def evaluate(Xtr, ytr, Xte, yte, names=None):
    out = {}
    names = list(F) if names is None else names
    t0 = time.perf_counter()
    sel = DynamicFeatureSelector(feature_names=names).fit(Xtr, ytr)
    out["selection_s"] = time.perf_counter() - t0
    out["selected_features"] = sel.selected_names
    mask = sel.selected_.astype(float)
    for loss, name in LOSS_NAMES.items():
        t0 = time.perf_counter()
        m = OnlineSGDClassifier(loss=loss, random_state=42).fit_offline(Xtr, ytr, mask)
        tt = time.perf_counter() - t0
        out[name] = {**R(yte, m.predict(Xte), m.decision_function(Xte)), "train_s": tt}
    # no feature selection (all lexical features)
    t0 = time.perf_counter()
    m = OnlineSGDClassifier(loss="hinge", random_state=42).fit_offline(Xtr, ytr, None)
    out["SGD-SVM (all features, no selection)"] = {
        **R(yte, m.predict(Xte), m.decision_function(Xte)),
        "train_s": time.perf_counter() - t0,
    }
    # classical batch baselines on the selected features (context only)
    idx = sel.selected_indices_
    base = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "DecisionTree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
        ),
        "GaussianNB": make_pipeline(StandardScaler(), GaussianNB()),
    }
    for name, mdl in base.items():
        t0 = time.perf_counter()
        mdl.fit(Xtr[:, idx], ytr)
        out[name + " (batch, same F*)"] = {
            **R(yte, mdl.predict(Xte[:, idx])),
            "train_s": time.perf_counter() - t0,
        }
    return out


def main() -> None:
    results = {}
    for name in OLD_SOURCES:
        X, y, _ = load_features(name)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[{name}] train={len(ytr)} test={len(yte)}")
        results[name] = {"n_train": len(ytr), "n_test": len(yte), **evaluate(Xtr, ytr, Xte, yte)}
        print({k: round(v["accuracy"], 4) for k, v in results[name].items() if isinstance(v, dict)})

    # ISCX-URL-2016: authors' own 79 lexical feature vectors (paper Table 5)
    X, y, names = load_iscx()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[iscx] train={len(ytr)} test={len(yte)} features={len(names)}")
    results["iscx"] = {
        "n_train": len(ytr), "n_test": len(yte), **evaluate(Xtr, ytr, Xte, yte, names)
    }
    print({k: round(v["accuracy"], 4) for k, v in results["iscx"].items() if isinstance(v, dict)})

    old = load_old()
    print(f"[combined] train={len(old.y_tr)} test={len(old.y_te)}")
    results["combined"] = {
        "n_train": len(old.y_tr),
        "n_test": len(old.y_te),
        **evaluate(old.X_tr, old.y_tr, old.X_te, old.y_te),
    }
    # per-source accuracy of the combined SGD-SVM
    sel = DynamicFeatureSelector(feature_names=list(F)).fit(old.X_tr, old.y_tr)
    m = OnlineSGDClassifier(random_state=42).fit_offline(old.X_tr, old.y_tr, sel.selected_.astype(float))
    results["combined"]["per_source_SGD-SVM"] = {
        s: R(old.y_te[old.src_te == s], m.predict(old.X_te[old.src_te == s]))
        for s in OLD_SOURCES
    }
    print({k: round(v["accuracy"], 4) for k, v in results["combined"].items() if isinstance(v, dict) and "accuracy" in v})
    C.save_json("exp1_offline.json", results)


if __name__ == "__main__":
    main()

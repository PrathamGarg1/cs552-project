#!/usr/bin/env python3
"""Experiment 4 - runtime (paper Tables 10-11).

Table 10: end-to-end per-URL latency (raw URL string -> lexical features ->
          selected features -> SGD-SVM score), single-threaded, one URL at a time.
Table 11: incremental update vs. conventional batch retraining.
"""

from __future__ import annotations

import copy
import time

import _common as C
import numpy as np
from drift_phish import DriftAwarePhishingDetector, FrameworkConfig
from drift_phish.data import load_raw
from drift_phish.features import PAPER_FEATURE_NAMES as F
from drift_phish.features import paper_feature_vector
from drift_phish.splits import load_old, url_key

REPEATS = 10


def main() -> None:
    old = load_old()
    det = DriftAwarePhishingDetector(F, FrameworkConfig(mode="frozen")).fit_offline(old.X_tr, old.y_tr)
    rng = np.random.default_rng(42)

    # ---------------- Table 10: per-URL latency --------------------------------
    lat = {}
    pools = {}
    for name in ("phiusiil", "kaggle", "hannousse", "phishtank"):
        urls = load_raw(name)["url"].to_numpy()
        pools[name] = urls[rng.choice(len(urls), min(2000, len(urls)), replace=False)]
    pools["combined"] = np.concatenate([pools[k] for k in ("phiusiil", "kaggle", "hannousse")])
    for name, urls in pools.items():
        ts = []
        for u in urls:
            t0 = time.perf_counter()
            x = paper_feature_vector(u)[None, :]
            det.model.decision_function(x)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.asarray(ts)
        lat[name] = {
            "avg_ms": float(ts.mean()),
            "median_ms": float(np.median(ts)),
            "p95_ms": float(np.percentile(ts, 95)),
            "n_urls": int(len(ts)),
        }
        print("latency", name, {k: round(v, 3) for k, v in lat[name].items()})

    # ---------------- Table 11: incremental vs batch ---------------------------
    legit = np.flatnonzero(old.y_te == 0)
    phish = np.flatnonzero(old.y_te == 1)
    rows = {}
    for n in (100, 500, 1000):
        inc, bat = [], []
        for r in range(REPEATS):
            idx = np.r_[rng.choice(phish, n // 2, replace=False), rng.choice(legit, n - n // 2, replace=False)]
            Xn, yn = old.X_te[idx], old.y_te[idx]

            d1 = copy.deepcopy(det)
            d1.config.mode = "online_static"
            t0 = time.perf_counter()
            d1.model.partial_fit(Xn, yn)
            inc.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            d2 = DriftAwarePhishingDetector(F, FrameworkConfig(mode="frozen"))
            d2.fit_offline(np.vstack([old.X_tr, Xn]), np.r_[old.y_tr, yn])
            bat.append((time.perf_counter() - t0) * 1e3)
        rows[f"Model update ({n} URLs)"] = {
            "incremental_ms": float(np.mean(inc)),
            "batch_ms": float(np.mean(bat)),
            "gain_x": float(np.mean(bat) / np.mean(inc)),
        }
        print(n, rows[f"Model update ({n} URLs)"])

    # feature adaptation: re-run selection on the 1500-URL buffer vs on all data
    inc, bat = [], []
    for r in range(REPEATS):
        idx = rng.choice(len(old.y_te), 1500, replace=False)
        d = copy.deepcopy(det)
        t0 = time.perf_counter()
        d.selector.update(old.X_te[idx], old.y_te[idx])
        inc.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        d.selector.fit(old.X_tr, old.y_tr)
        bat.append((time.perf_counter() - t0) * 1e3)
    rows["Feature adaptation"] = {
        "incremental_ms": float(np.mean(inc)),
        "batch_ms": float(np.mean(bat)),
        "gain_x": float(np.mean(bat) / np.mean(inc)),
    }
    print("feature adaptation", rows["Feature adaptation"])
    C.save_json("exp4_runtime.json", {"latency": lat, "incremental_vs_batch": rows,
                                     "n_offline_train": int(len(old.y_tr)), "repeats": REPEATS})


if __name__ == "__main__":
    main()

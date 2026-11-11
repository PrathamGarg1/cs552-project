#!/usr/bin/env python3
"""Experiment 3 - ablation: adaptive vs static feature selection (paper Table 9).

An offline model is adapted with N = 100 / 500 / 1000 labelled new URLs and
evaluated on held-out new URLs:
  static   : online SGD on the new URLs, initial feature set F* kept
  adaptive : same, plus KS drift detection + LASSO+MI re-selection + partial
             retraining on the buffer (the proposed framework)
Three targets: PhishTank (paper setting, adapt on the early part of the
chronological stream, test on its later part), Kaggle and Hannousse (offline
trained on PhiUSIIL only). 10 seeds; mean +- std.
"""

from __future__ import annotations

import copy

import _common as C
import numpy as np
from drift_phish import DriftAwarePhishingDetector, FrameworkConfig
from drift_phish.data import load_features
from drift_phish.eval import classification_report_dict as R
from drift_phish.features import PAPER_FEATURE_NAMES as F
from drift_phish.splits import load_new_phish, load_old
from sklearn.model_selection import train_test_split

NS = [100, 500, 1000]
SEEDS = list(range(42, 52))


def adapt_and_eval(base, Xa, ya, Xe, ye, mode):
    det = copy.deepcopy(base)
    det.config.mode = mode
    if len(Xa):
        det.process_stream(Xa, ya)
    r = R(ye, det.predict(Xe))
    r["n_features"] = int(det.selector.selected_.sum())
    r["n_drift_events"] = sum(e["type"] == "drift" for e in det.events)
    return r


def run_target(name, make_offline, make_pools, seeds):
    table = {n: {"static": [], "adaptive": []} for n in [0] + NS}
    for seed in seeds:
        Xtr, ytr = make_offline()
        base = DriftAwarePhishingDetector(F, FrameworkConfig(random_state=seed)).fit_offline(Xtr, ytr)
        rng = np.random.default_rng(seed)
        (Xp, yp), (Xe, ye) = make_pools(rng)
        table[0]["static"].append(adapt_and_eval(base, Xp[:0], yp[:0], Xe, ye, "frozen"))
        for n in NS:
            idx = rng.permutation(len(yp))[:n]
            for mode, key in (("online_static", "static"), ("adaptive", "adaptive")):
                table[n][key].append(adapt_and_eval(base, Xp[idx], yp[idx], Xe, ye, mode))
        print(
            f"  [{name}] seed {seed}",
            {n: (round(table[n]["static"][-1]["f1"], 4), round(table[n]["adaptive"][-1]["f1"], 4)) for n in NS},
        )
    summary = {}
    for n in [0] + NS:
        summary[n] = {
            k: {**C.agg(v), "n_features": float(np.mean([r["n_features"] for r in v])),
                "n_drift_events": float(np.mean([r["n_drift_events"] for r in v]))}
            for k, v in table[n].items() if v
        }
    return summary, table


def main() -> None:
    old = load_old()
    Xn, _, _ = load_new_phish(old.keys)
    cut = int(0.4 * len(Xn))  # chronological split: adapt on early, test on later URLs
    legit = np.flatnonzero(old.y_te == 0)

    def offline_old():
        return old.X_tr, old.y_tr

    def pools_phishtank(rng):
        L = rng.permutation(legit)
        la, le = L[: len(L) // 2], L[len(L) // 2 :]
        Xa = np.vstack([Xn[:cut], old.X_te[la[:cut]]])
        ya = np.r_[np.ones(cut, int), np.zeros(cut, int)]
        ne = len(Xn) - cut
        Xe = np.vstack([Xn[cut:], old.X_te[le[:ne]]])
        ye = np.r_[np.ones(ne, int), np.zeros(ne, int)]
        return (Xa, ya), (Xe, ye)

    Xp, yp, _ = load_features("phiusiil")
    Xp_tr, _, yp_tr, _ = train_test_split(Xp, yp, test_size=0.2, random_state=42, stratify=yp)

    def offline_phi():
        return Xp_tr, yp_tr

    def pools_corpus(name):
        X, y, _ = load_features(name)

        def f(rng):
            Xa, Xe, ya, ye = train_test_split(
                X, y, test_size=0.5, random_state=int(rng.integers(1 << 30)), stratify=y
            )
            return (Xa, ya), (Xe, ye)

        return f

    out = {}
    for name, off, pools in [
        ("PhishTank Jun2024-Jan2025 (offline: combined)", offline_old, pools_phishtank),
        ("Kaggle (offline: PhiUSIIL)", offline_phi, pools_corpus("kaggle")),
        ("Hannousse (offline: PhiUSIIL)", offline_phi, pools_corpus("hannousse")),
    ]:
        print("target:", name)
        summary, raw = run_target(name, off, pools, SEEDS)
        out[name] = {"summary": summary, "raw": raw}
    C.save_json("exp3_ablation.json", out)


if __name__ == "__main__":
    main()

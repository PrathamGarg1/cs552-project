#!/usr/bin/env python3
"""Experiment 2 - online phase: prequential evaluation under drift.

Scenario S1 (paper setting)
    offline : combined old corpora (PhiUSIIL + Kaggle + Hannousse), 80 % split
    stream  : PhishTank phishing URLs from June 2024 - Jan 2025 in
              chronological order, interleaved with held-out benign URLs.
Scenario S2 (strong, cross-corpus shift - stress test)
    offline : PhiUSIIL only
    stream  : held-out PhiUSIIL (no drift) -> Kaggle -> Hannousse

Compared systems (same offline model, same stream, Table 4 settings):
    frozen          offline model never updated
    online_static   online SGD, feature set F* frozen   (paper "without adaptive features")
    adaptive        proposed: KS drift -> LASSO+MI re-selection + partial retrain
    periodic_batch  conventional batch retraining every 5000 URLs (window 20k)
5 seeds; mean +- std reported. The metric is prequential (test-then-train).
"""

from __future__ import annotations

import time

import _common as C
import numpy as np
from drift_phish import DriftAwarePhishingDetector, FrameworkConfig
from drift_phish.data import load_features
from drift_phish.eval import classification_report_dict as R
from drift_phish.features import PAPER_FEATURE_NAMES as F
from drift_phish.splits import build_stream, load_new_phish, load_old
from sklearn.model_selection import train_test_split

MODES = ["frozen", "online_static", "adaptive", "periodic_batch"]


def run_modes(Xtr, ytr, Xs, ys, phases, seed, keep_curve=False):
    out = {}
    B = 500
    for mode in MODES:
        det = DriftAwarePhishingDetector(
            F, FrameworkConfig(mode=mode, random_state=seed)
        ).fit_offline(Xtr, ytr)
        t0 = time.perf_counter()
        res = det.process_stream(Xs, ys)
        wall = time.perf_counter() - t0
        pred = res["y_pred"]
        rec = {
            "overall": R(ys, pred, res["score"]),
            "phases": {n: R(ys[a:b], pred[a:b]) for n, (a, b) in phases.items()},
            "stream_s": wall,
            "n_drift_events": sum(e["type"] == "drift" for e in det.events),
            "n_batch_retrains": sum(e["type"] == "batch_retrain" for e in det.events),
            "timings": det.timings,
        }
        if keep_curve:
            rec["batch_acc"] = [
                float(np.mean(pred[s : s + B] == ys[s : s + B])) for s in range(0, len(ys), B)
            ]
            rec["batch_fpr"] = [
                float(np.mean(pred[s : s + B][ys[s : s + B] == 0] == 1))
                if (ys[s : s + B] == 0).any()
                else float("nan")
                for s in range(0, len(ys), B)
            ]
            rec["events"] = [
                {k: v for k, v in e.items() if k not in ("ks_stat", "p_value")}
                for e in det.events
            ]
            rec["n_selected"] = res["n_selected"]
        out[mode] = rec
    return out


def s1(seeds):
    old = load_old()
    Xn, _, _ = load_new_phish(old.keys)
    print(f"[S1] offline train {old.X_tr.shape}, new phishing URLs {Xn.shape}")
    runs = []
    for i, seed in enumerate(seeds):
        Xs, ys, _ = build_stream(old, Xn, seed=seed)
        n = len(ys)
        q = {f"Q{k+1}": (k * n // 4, (k + 1) * n // 4) for k in range(4)}
        r = run_modes(old.X_tr, old.y_tr, Xs, ys, q, seed, keep_curve=(i == 0))
        runs.append(r)
        print(f"  seed {seed}:", {m: round(r[m]["overall"]["accuracy"], 4) for m in MODES})
    return {"n_stream": int(n), "n_new_phish": int(len(Xn)), "runs": runs}


def s2(seeds):
    Xp, yp, _ = load_features("phiusiil")
    Xk, yk, _ = load_features("kaggle")
    Xh, yh, _ = load_features("hannousse")
    Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=0.2, random_state=42, stratify=yp)
    print(f"[S2] offline train {Xtr.shape}; stream kaggle={len(yk)} hannousse={len(yh)}")
    runs = []
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        a = rng.permutation(len(yte))[:5000]
        k = rng.permutation(len(yk))
        h = rng.permutation(len(yh))
        Xs = np.vstack([Xte[a], Xk[k], Xh[h]])
        ys = np.concatenate([yte[a], yk[k], yh[h]])
        b1, b2 = 5000, 5000 + len(yk)
        phases = {
            "PhiUSIIL (no drift)": (0, b1),
            "Kaggle (drift 1)": (b1, b2),
            "Hannousse (drift 2)": (b2, len(ys)),
        }
        r = run_modes(Xtr, ytr, Xs, ys, phases, seed, keep_curve=(i == 0))
        runs.append(r)
        print(f"  seed {seed}:", {m: round(r[m]["overall"]["accuracy"], 4) for m in MODES})
    return {"n_stream": int(len(ys)), "phase_bounds": [0, b1, b2, int(len(ys))], "runs": runs}


def main() -> None:
    res = {"S1_paper_setting": s1(C.SEEDS), "S2_cross_corpus_shift": s2(C.SEEDS)}
    C.save_json("exp2_stream.json", res)


if __name__ == "__main__":
    main()

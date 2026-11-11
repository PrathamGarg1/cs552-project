"""Train / test / stream splits shared by all experiments.

OLD data  = PhiUSIIL + Kaggle + Hannousse (the paper trains on ISCX + Kaggle +
            PhiUSIIL; Hannousse replaces ISCX). Cross-corpus duplicates are
            removed, then a stratified 80/20 split (seed 42, Table 4).
NEW data  = verified PhishTank phishing URLs submitted between June 2024 and
            January 2025 - the exact collection window of the paper - ordered
            chronologically. URLs that also occur in the old corpora are
            removed. PhishTank is phishing-only; the benign side of every
            stream is drawn from the *held-out* old test legitimate URLs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_features, load_raw

OLD_SOURCES = ("phiusiil", "kaggle", "hannousse")
NEW_START, NEW_END = "2024-06-01", "2025-02-01"  # paper: June 2024 - January 2025

_KEY = re.compile(r"^(?:[a-z]+://)?(?:www\.)?", re.I)


def url_key(u: str) -> str:
    return _KEY.sub("", u.strip().lower()).rstrip("/")


@dataclass
class OldSplit:
    X_tr: np.ndarray
    y_tr: np.ndarray
    src_tr: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    src_te: np.ndarray
    keys: set


def load_old(seed: int = 42, test_size: float = 0.2) -> OldSplit:
    Xs, ys, ss, ks = [], [], [], []
    seen: set[str] = set()
    for name in OLD_SOURCES:
        X, y, urls = load_features(name)
        keep = []
        for i, u in enumerate(urls):
            k = url_key(u)
            if k in seen:
                continue
            seen.add(k)
            keep.append(i)
        keep = np.asarray(keep)
        Xs.append(X[keep])
        ys.append(y[keep])
        ss.append(np.full(len(keep), name))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    strat = np.char.add(s.astype(str), y.astype(str))
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    return OldSplit(X[tr], y[tr], s[tr], X[te], y[te], s[te], seen)


def load_new_phish(old_keys: set[str]):
    """Chronological PhishTank phishing features inside the paper's window."""
    X, y, urls = load_features("phishtank")
    df = load_raw("phishtank")
    t = df["submission_time"].to_numpy()
    mask = (df["submission_time"] >= pd.Timestamp(NEW_START, tz="UTC")) & (
        df["submission_time"] < pd.Timestamp(NEW_END, tz="UTC")
    )
    mask = mask.to_numpy() & np.array([url_key(u) not in old_keys for u in urls])
    return X[mask], urls[mask], t[mask]


def build_stream(old: OldSplit, X_new: np.ndarray, seed: int = 42, n_legit: int | None = None):
    """Chronological phishing stream with held-out legitimate URLs interleaved.

    Returns X, y and a 'is_new_phish' helper through the label order. Phishing
    order is preserved; legitimate URLs are placed at random positions.
    """
    rng = np.random.default_rng(seed)
    legit_idx = np.flatnonzero(old.y_te == 0)
    n_legit = n_legit or len(X_new)
    n_legit = min(n_legit, len(legit_idx))
    chosen = rng.choice(legit_idx, n_legit, replace=False)
    n = len(X_new) + n_legit
    kind = np.zeros(n, dtype=int)  # 1 = phishing slot
    kind[rng.choice(n, len(X_new), replace=False)] = 1
    X = np.empty((n, X_new.shape[1]), dtype=np.float32)
    y = kind.copy()
    X[kind == 1] = X_new  # chronological order preserved
    X[kind == 0] = old.X_te[chosen]
    return X, y, chosen

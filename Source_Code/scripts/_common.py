"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "Experimental_Results"
FIGS = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]


def save_json(name: str, obj) -> None:
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=conv))
    print("wrote", RESULTS / name)


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    v = np.asarray([r[key] for r in rows], dtype=float)
    return float(v.mean()), float(v.std())


def agg(rows: list[dict], keys=("accuracy", "precision", "recall", "f1", "fpr")) -> dict:
    out = {}
    for k in keys:
        m, s = mean_std(rows, k)
        out[k] = m
        out[k + "_std"] = s
    return out


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}%"

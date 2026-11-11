#!/usr/bin/env python3
"""Experiment 5 - feature-distribution analysis (paper Fig. 4, Fig. 5, Fig. 7).

Fig. 4 : distribution of 18 URL features, OLD phishing (PhiUSIIL/Kaggle/
         Hannousse) vs NEW phishing (PhishTank, Jun 2024 - Jan 2025).
Fig. 7 : two-sample KS statistic (and significance at delta = 0.05) per feature
         between the old reference window (1500) and a new batch (500).
Fig. 5 : feature co-occurrence networks of the selected features for
         malicious and benign URLs.
"""

from __future__ import annotations

import _common as C
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from drift_phish.features import PAPER_FEATURE_NAMES as F
from drift_phish.selection import DynamicFeatureSelector
from drift_phish.splits import load_new_phish, load_old
from scipy import stats

FIG4 = [
    "url_length", "domain_length", "path_length", "token_count", "entropy", "num_dots",
    "num_slashes", "special_char_count", "num_hyphens", "number_count", "num_params",
    "longest_path_token_length", "num_delimiters_path", "num_digits_path", "num_tlds",
    "has_free", "has_secure", "url_encoded",
]


def fig4(Xold, Xnew):
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(3, 6, figsize=(20, 9))
    for ax, name in zip(axes.ravel(), FIG4):
        j = F.index(name)
        a, b = Xold[:, j].astype(float), Xnew[:, j].astype(float)
        if len(np.unique(np.r_[a[:2000], b[:2000]])) <= 2:
            ax.bar([-0.2, 0.8], [a.mean(), a.mean()], width=0.0)  # axis scaffolding
            ax.cla()
            ax.bar(["old", "new"], [a.mean(), b.mean()], color=["#1F4E79", "#C44900"])
            ax.set_ylabel("fraction = 1")
        else:
            hi = np.percentile(np.r_[a, b], 99)
            sns.kdeplot(np.clip(a, 0, hi), ax=ax, label="Old", color="#1F4E79", fill=True, alpha=0.3, cut=0, warn_singular=False)
            sns.kdeplot(np.clip(b, 0, hi), ax=ax, label="New", color="#C44900", fill=True, alpha=0.3, cut=0, warn_singular=False)
            ax.set_ylabel("density")
        ax.set_title(name)
    axes[0, 0].legend()
    fig.suptitle("Phishing URL feature distributions: Old (PhiUSIIL+Kaggle+Hannousse) vs New (PhishTank Jun 2024-Jan 2025)")
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig4_feature_distributions.png", dpi=130)
    plt.close(fig)


def fig7(Xold, Xnew, rng):
    full = np.array([stats.ks_2samp(Xold[:, j], Xnew[:, j]).statistic for j in range(len(F))])
    ref = Xold[rng.choice(len(Xold), 1500, replace=False)]
    cur = Xnew[rng.choice(len(Xnew), 500, replace=False)]
    win = [stats.ks_2samp(ref[:, j], cur[:, j], method="asymp") for j in range(len(F))]
    p = np.array([w.pvalue for w in win])
    ks = np.array([w.statistic for w in win])
    order = np.argsort(ks)[::-1]
    fig, ax = plt.subplots(figsize=(15, 5))
    colors = ["#C44900" if p[j] < 0.05 else "#7F8C8D" for j in order]
    ax.bar(range(len(F)), ks[order], color=colors)
    ax.set_xticks(range(len(F)))
    ax.set_xticklabels([F[j] for j in order], rotation=90, fontsize=8)
    ax.set_ylabel("KS statistic (1500 old vs 500 new)")
    ax.set_title("KS drift statistic per feature, old vs new phishing (orange: p < 0.05 = drift)")
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig7_ks_statistics.png", dpi=130)
    plt.close(fig)
    return {
        "n_features": len(F),
        "n_drifted_at_0.05": int((p < 0.05).sum()),
        "features": {
            F[j]: {"ks_window": float(ks[j]), "p_window": float(p[j]), "ks_full": float(full[j])}
            for j in range(len(F))
        },
    }


def fig5(old):
    sel = DynamicFeatureSelector(feature_names=list(F)).fit(old.X_tr, old.y_tr)
    names = sel.selected_names
    idx = sel.selected_indices_
    X = old.X_tr[:, idx]
    thr = np.median(X, axis=0)
    B = (X > thr).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    pos = np.c_[np.cos(ang), np.sin(ang)]
    for ax, (cls, title) in zip(axes, [(1, "(a) Malicious URLs"), (0, "(b) Benign URLs")]):
        Bc = B[old.y_tr == cls]
        co = (Bc.T @ Bc) / len(Bc)
        np.fill_diagonal(co, 0)
        thresh = np.percentile(co[np.triu_indices(len(names), 1)], 70)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if co[i, j] >= thresh:
                    ax.plot(*zip(pos[i], pos[j]), color="#1F4E79", lw=0.5 + 6 * co[i, j], alpha=0.5)
        ax.scatter(pos[:, 0], pos[:, 1], s=250, color="#0B6E4F", zorder=3)
        for (x, y), n in zip(pos, names):
            ax.text(x * 1.12, y * 1.12, n, ha="center", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.4, 1.4)
        ax.axis("off")
    fig.suptitle("Feature co-occurrence networks of selected features (edge width = co-occurrence)")
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig5_cooccurrence.png", dpi=130)
    plt.close(fig)
    return names


def main() -> None:
    old = load_old()
    Xn, _, _ = load_new_phish(old.keys)
    rng = np.random.default_rng(42)
    ph = old.X_tr[old.y_tr == 1]
    Xold = ph[rng.choice(len(ph), min(12000, len(ph)), replace=False)]
    fig4(Xold, Xn)
    ks = fig7(Xold, Xn, rng)
    names = fig5(old)
    C.save_json("exp5_analysis.json", {"ks": ks, "cooccurrence_features": names})


if __name__ == "__main__":
    main()

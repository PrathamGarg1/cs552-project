#!/usr/bin/env python3
"""Generate every chart used in Presentation_Slides from the real result JSONs.

Run:  python make_assets.py      (needs numpy, scipy, matplotlib, pillow)
Outputs go to ../assets/.  Charts marked ILLUSTRATION use synthetic toy data
purely to explain a concept; everything else is read from
Experimental_Results/*.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "Experimental_Results"
OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(exist_ok=True)

# palette (kept in sync with slides.css)
NAVY, INK, BLUE = "#0B1B3A", "#14213D", "#2F6BFF"
RED, GREEN, AMBER, GREY, LINE = "#E5484D", "#12A06B", "#F5A524", "#8A97AD", "#E3E8F2"
PURPLE = "#7C5CFF"
MUTED = "#5B6B86"

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 15,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": "#EDF0F7",
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.transparent": True,
    }
)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


def load(name):
    return json.loads((RES / name).read_text())


exp1, exp2, exp3, exp4, exp5 = (load(f) for f in
    ["exp1_offline.json", "exp2_stream.json", "exp3_ablation.json", "exp4_runtime.json", "exp5_analysis.json"])

# --------------------------------------------------------------------------- #
# 1. offline SGD variants (paper Tables 5-7 / Fig 6)
# --------------------------------------------------------------------------- #
def offline_variants():
    corp = [("phiusiil", "PhiUSIIL"), ("kaggle", "Kaggle"), ("iscx", "ISCX\n(79 feats)"),
            ("hannousse", "Hannousse"), ("combined", "Combined")]
    models = [("SGD-SVM", "SGD-SVM", BLUE), ("SGD-LR", "SGD-LR", AMBER), ("SGD-Perceptron", "SGD-Perceptron", GREEN)]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    w = 0.26
    x = np.arange(len(corp))
    for i, (k, lab, c) in enumerate(models):
        v = [exp1[cc][k]["accuracy"] * 100 for cc, _ in corp]
        b = ax.bar(x + (i - 1) * w, v, w, label=lab, color=c, zorder=3)
        if k == "SGD-SVM":
            for xi, vi in zip(x, v):
                ax.text(xi - w, vi + 1.2, f"{vi:.1f}", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.set_xticks(x, [l for _, l in corp])
    ax.set_ylim(50, 104)
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=13)
    ax.grid(axis="x", visible=False)
    save(fig, "c_offline_variants")


# --------------------------------------------------------------------------- #
# 2. why is the level lower: F* vs all features vs RF vs paper
# --------------------------------------------------------------------------- #
def gap_chart():
    corp = [("iscx", "ISCX"), ("kaggle", "Kaggle"), ("hannousse", "Hannousse"), ("combined", "Combined")]
    paper = {"iscx": 99.02, "kaggle": 99.00, "hannousse": None, "combined": 99.85}
    series = [
        ("SGD-SVM on F*  (paper recipe)", "SGD-SVM", BLUE),
        ("SGD-SVM, all features", "SGD-SVM (all features, no selection)", PURPLE),
        ("Random Forest on same F*", "RandomForest (batch, same F*)", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    n = len(series) + 1
    w = 0.2
    x = np.arange(len(corp))
    for i, (lab, key, c) in enumerate(series):
        v = [exp1[cc][key]["accuracy"] * 100 for cc, _ in corp]
        ax.bar(x + (i - 1.5) * w, v, w, label=lab, color=c, zorder=3)
    pv = [paper[cc] if paper[cc] else np.nan for cc, _ in corp]
    ax.bar(x + 1.5 * w, pv, w, label="Paper reports", color="white", edgecolor=RED, hatch="////",
           linewidth=1.6, zorder=3)
    for xi, p in zip(x, pv):
        if not np.isnan(p):
            ax.text(xi + 1.5 * w, p + 0.8, f"{p:g}", ha="center", fontsize=11.5, color=RED, fontweight="bold")
    ax.set_xticks(x, [l for _, l in corp])
    ax.set_ylim(60, 104)
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=12.5)
    ax.grid(axis="x", visible=False)
    save(fig, "c_gap")


# --------------------------------------------------------------------------- #
# 3. per-source breakdown inside the combined corpus
# --------------------------------------------------------------------------- #
def per_source():
    src = ["PhiUSIIL", "Kaggle", "Hannousse"]
    acc = [98.52, 59.59, 52.68]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    b = ax.barh(src[::-1], acc[::-1], color=[RED, AMBER, GREEN], height=0.55, zorder=3)
    for r, a in zip(b, acc[::-1]):
        ax.text(a - 2, r.get_y() + r.get_height() / 2, f"{a:.1f}%", va="center", ha="right",
                color="white", fontweight="bold", fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Combined-model accuracy on that source (%)", fontsize=11.5)
    ax.grid(axis="y", visible=False)
    save(fig, "c_per_source")


# --------------------------------------------------------------------------- #
# 4. KS statistic bars (paper Fig 7 analogue)
# --------------------------------------------------------------------------- #
def ks_bars():
    feats = exp5["ks"]["features"]
    items = sorted(feats.items(), key=lambda kv: -kv[1]["ks_window"])[:20]
    names = [k for k, _ in items][::-1]
    vals = [v["ks_window"] for _, v in items][::-1]
    cols = [RED if feats[n]["p_window"] < 0.05 else GREY for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ax.barh(names, vals, color=cols, height=0.7, zorder=3)
    ax.set_xlabel("KS statistic D  (0 = identical, 1 = completely different)", fontsize=12)
    ax.tick_params(axis="y", labelsize=12.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.75)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=11, color=MUTED)
    save(fig, "c_ks_bars")


# --------------------------------------------------------------------------- #
# 5/6. prequential curves
# --------------------------------------------------------------------------- #
SYS = [("frozen", "Frozen offline model", GREY, 2.2),
       ("online_static", "Online SGD, static F*", BLUE, 2.2),
       ("adaptive", "Proposed (KS + adaptive F* + SGD)", GREEN, 3.4),
       ("periodic_batch", "Periodic batch retraining", AMBER, 2.2)]


def stream_curve(scn, name, ylim, phases=None, size=(11.2, 4.1)):
    run = exp2[scn]["runs"][0]
    fig, ax = plt.subplots(figsize=size)
    B = 500
    n = len(run["adaptive"]["batch_acc"])
    xs = np.arange(n) * B
    if phases:
        cols = ["#F2F6FF", "#FFF4E5", "#FDECEC"]
        bounds = exp2[scn]["phase_bounds"]
        labels = ["PhiUSIIL\n(no drift)", "Kaggle  (drift 1)", "Hannousse  (drift 2)"]
        for i in range(3):
            ax.axvspan(bounds[i], bounds[i + 1], color=cols[i], zorder=0)
            ax.text((bounds[i] + bounds[i + 1]) / 2 + (1300 if i == 0 else 0), ylim[1] - 1.2, labels[i], ha="center", va="top",
                    fontsize=12.5, color=MUTED, fontweight="bold")
    for key, lab, c, lw in SYS:
        y = np.array(run[key]["batch_acc"]) * 100
        ax.plot(xs, y, color=c, lw=lw, label=lab, zorder=3 if key != "adaptive" else 4)
    ev = [e["t"] for e in run["adaptive"]["events"] if e["type"] == "drift"]
    ax.plot(ev, [ylim[0] + 0.6] * len(ev), "|", color=GREEN, ms=12, mew=2, zorder=5,
            label=f"KS drift event ({len(ev)} in this run)")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, xs[-1] + B)
    ax.set_xlabel("Position in the URL stream (number of URLs seen)")
    ax.set_ylabel("Accuracy on the next batch (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=12.5, columnspacing=1.4,
              handlelength=1.8)
    save(fig, name)


def frozen_decay():
    run = exp2["S2_cross_corpus_shift"]["runs"][0]["frozen"]["batch_acc"]
    B = 500
    xs = np.arange(len(run)) * B
    bounds = exp2["S2_cross_corpus_shift"]["phase_bounds"]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.axvspan(bounds[0], bounds[1], color="#E8F7F0", zorder=0)
    ax.axvspan(bounds[1], bounds[3], color="#FDECEC", zorder=0)
    ax.plot(xs, np.array(run) * 100, color=RED, lw=3.2, zorder=3)
    ax.text(bounds[1] / 2, 70, "trained\nfor this\ndata", ha="center", fontsize=12, color=GREEN, fontweight="bold")
    ax.text((bounds[1] + bounds[3]) / 2 + 2500, 86, "Data has changed (different URL source)", ha="center", fontsize=12.5,
            color=RED, fontweight="bold")
    ax.text(2500, 89, "99 %", ha="center", fontsize=15, fontweight="bold", color=GREEN)
    ax.text(21000, 63, "≈ 48 %  (coin-flip)", ha="center", fontsize=16, fontweight="bold", color=RED)
    ax.set_ylim(40, 103)
    ax.set_xlim(0, xs[-1])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("URLs seen (frozen model, never updated)")
    ax.grid(axis="x", visible=False)
    save(fig, "c_frozen_decay")


def stream_bars():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, (scn, title) in zip(axes, [("S1_paper_setting", "S1 · PhishTank stream (paper's setting)"),
                                       ("S2_cross_corpus_shift", "S2 · strong cross-corpus shift")]):
        runs = exp2[scn]["runs"]
        for i, (k, lab, c, _) in enumerate(SYS):
            v = np.array([r[k]["overall"]["accuracy"] for r in runs]) * 100
            ax.bar(i, v.mean(), color=c, yerr=v.std(), capsize=5, zorder=3, width=0.65,
                   error_kw=dict(ecolor=INK, lw=1.5))
            ax.text(i, 50.8, f"{v.mean():.1f}", ha="center", color="white", fontweight="bold", fontsize=15)
        ax.set_xticks(range(4), ["Frozen", "Online\nstatic", "Proposed\n(adaptive)", "Periodic\nbatch"], fontsize=12.5)
        ax.set_ylim(50, 100)
        ax.set_title(title, fontsize=14, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean accuracy over stream (%), 5 seeds")
    fig.tight_layout()
    save(fig, "c_stream_bars")


# --------------------------------------------------------------------------- #
# 7. ablation (paper Table 9)
# --------------------------------------------------------------------------- #
def ablation():
    key = "PhishTank Jun2024-Jan2025 (offline: combined)"
    S = exp3[key]["summary"]
    Ns = [100, 500, 1000]
    st = [S[str(n)]["static"]["f1"] * 100 for n in Ns]
    ad = [S[str(n)]["adaptive"]["f1"] * 100 for n in Ns]
    sd = [S[str(n)]["adaptive"]["f1_std"] * 100 for n in Ns]
    ssd = [S[str(n)]["static"]["f1_std"] * 100 for n in Ns]
    p_st, p_ad = [92.31, 93.89, 93.73], [96.22, 97.15, 98.30]
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    x = np.arange(3)
    ax.plot(x, p_ad, "--o", color=GREEN, lw=2, ms=9, mfc="white", mew=2, label="Paper · adaptive")
    ax.plot(x, p_st, "--o", color=GREY, lw=2, ms=9, mfc="white", mew=2, label="Paper · static")
    ax.errorbar(x, ad, sd, fmt="-o", color=GREEN, lw=3.2, ms=9, capsize=4, label="This study · adaptive")
    ax.errorbar(x, st, ssd, fmt="-o", color=INK, lw=3.2, ms=9, capsize=4, label="This study · static")
    ax.set_xticks(x, ["100", "500", "1000"])
    ax.set_xlabel("Labelled new URLs used to adapt (N)")
    ax.set_ylabel("F1 on held-out new URLs (%)")
    ax.set_ylim(91, 99.2)
    ax.legend(loc="lower right", fontsize=11.5, ncol=1, facecolor="white", frameon=True, edgecolor=LINE)
    ax.grid(axis="x", visible=False)
    save(fig, "c_ablation_f1")

    # accuracy / FPR panel for the three targets at N=500
    tg = [("PhishTank Jun2024-Jan2025 (offline: combined)", "PhishTank\n(small shift)"),
          ("Kaggle (offline: PhiUSIIL)", "Kaggle\n(big shift)"),
          ("Hannousse (offline: PhiUSIIL)", "Hannousse\n(big shift)")]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    w = 0.34
    x = np.arange(3)
    a_st = [exp3[k]["summary"]["500"]["static"]["accuracy"] * 100 for k, _ in tg]
    a_ad = [exp3[k]["summary"]["500"]["adaptive"]["accuracy"] * 100 for k, _ in tg]
    ax.bar(x - w / 2, a_st, w, color=INK, label="Static F*", zorder=3)
    ax.bar(x + w / 2, a_ad, w, color=GREEN, label="Adaptive F*", zorder=3)
    for xi, s, a in zip(x, a_st, a_ad):
        ax.text(xi - w / 2, s + 0.8, f"{s:.1f}", ha="center", fontsize=12, color=INK, fontweight="bold")
        ax.text(xi + w / 2, a + 0.8, f"{a:.1f}", ha="center", fontsize=12, color=GREEN, fontweight="bold")
        ax.text(xi, 52, f"+{a - s:.1f}", ha="center", fontsize=13, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=GREEN, ec="none"))
    ax.set_xticks(x, [l for _, l in tg])
    ax.set_ylim(50, 102)
    ax.set_ylabel("Accuracy after N = 500 labelled URLs (%)")
    ax.legend(loc="upper right", ncol=2, fontsize=12, bbox_to_anchor=(1, 1.1))
    ax.grid(axis="x", visible=False)
    save(fig, "c_ablation_acc")


# --------------------------------------------------------------------------- #
# 8. runtime (paper Tables 10-11)
# --------------------------------------------------------------------------- #
def runtime():
    d = exp4["incremental_vs_batch"]
    ops = ["Model update (100 URLs)", "Model update (500 URLs)", "Model update (1000 URLs)", "Feature adaptation"]
    labs = ["Update, 100 URLs", "Update, 500 URLs", "Update, 1000 URLs", "Feature re-selection"]
    inc = [d[o]["incremental_ms"] for o in ops]
    bat = [d[o]["batch_ms"] for o in ops]
    gain = [d[o]["gain_x"] for o in ops]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(ops))[::-1]
    h = 0.34
    ax.barh(y + h / 2, bat, h, color=AMBER, label="Batch retraining (from scratch)", zorder=3)
    ax.barh(y - h / 2, inc, h, color=GREEN, label="Incremental (this framework)", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 40000)
    for yi, i, b, g in zip(y, inc, bat, gain):
        ax.text(i * 1.15, yi - h / 2, f"{i:,.1f} ms", va="center", fontsize=12, color=GREEN, fontweight="bold")
        ax.text(b * 1.15, yi + h / 2, f"{b / 1000:,.1f} s" if b >= 1000 else f"{b:,.0f} ms", va="center",
                fontsize=12, color="#B77700", fontweight="bold")
        ax.text(0.9, yi + 0.0, "", fontsize=1)
    ax.set_yticks(y, labs, fontsize=13)
    ax.set_xlabel("Time (milliseconds, log scale)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=12.5)
    ax.grid(axis="y", visible=False)
    save(fig, "c_runtime")


# --------------------------------------------------------------------------- #
# ILLUSTRATIONS (synthetic toy data, concept only)
# --------------------------------------------------------------------------- #
def ill_boundary():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True, sharex=True)
    a0 = rng.normal([-1.3, -1.0], 0.75, (70, 2))
    a1 = rng.normal([1.3, 1.1], 0.75, (70, 2))
    for ax, title, shift in zip(axes, ["① Offline: learn a dividing line", "② Attackers change style → line adapts"], [0, 1]):
        b = a1 + (np.array([-1.7, 0.9]) if shift else 0)
        ax.scatter(*a0.T, s=44, c=GREEN, alpha=0.85, label="legitimate", zorder=3, edgecolor="white", lw=0.6)
        ax.scatter(*b.T, s=44, c=RED, alpha=0.85, label="phishing", zorder=3, edgecolor="white", lw=0.6)
        xs = np.linspace(-4, 4, 10)
        ax.plot(xs, -xs * 0.95 + 0.0, color=INK, lw=3, zorder=4, label="decision line" if not shift else "old line", ls="-" if not shift else "--", alpha=1 if not shift else 0.45)
        if shift:
            ax.plot(xs, -xs * 0.2 + 0.25, color=BLUE, lw=3.4, zorder=5, label="updated line")
            ax.annotate("", (-1.3, 2.0), (1.0, 1.1), arrowprops=dict(arrowstyle="->", color=RED, lw=2, ls="--"))
        ax.set_title(title, fontsize=13.5, loc="left")
        ax.set_xlim(-4, 4); ax.set_ylim(-3, 3.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(loc="lower left", fontsize=11, ncol=2, bbox_to_anchor=(0, -0.02))
    fig.tight_layout()
    save(fig, "i_boundary")


def ill_ks():
    rng = np.random.default_rng(7)
    ref = np.sort(rng.normal(0, 1, 1500))
    cur = np.sort(rng.normal(0.9, 1.0, 500))
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    ax.step(ref, np.arange(1, len(ref) + 1) / len(ref), color=BLUE, lw=3, label="Reference buffer (1,500 URLs)", where="post")
    ax.step(cur, np.arange(1, len(cur) + 1) / len(cur), color=RED, lw=3, label="New mini-batch (500 URLs)", where="post")
    grid = np.linspace(-3, 4, 800)
    Fr = np.searchsorted(ref, grid, side="right") / len(ref)
    Fc = np.searchsorted(cur, grid, side="right") / len(cur)
    i = np.argmax(np.abs(Fr - Fc))
    ax.annotate("", (grid[i], Fr[i]), (grid[i], Fc[i]), arrowprops=dict(arrowstyle="<->", color=INK, lw=2.4))
    ax.text(grid[i] + 0.12, (Fr[i] + Fc[i]) / 2, "D = biggest\ngap between\nthe two curves", fontsize=12.5,
            color=INK, fontweight="bold", va="center")
    ax.set_xlabel("value of one feature (e.g. host entropy)")
    ax.set_ylabel("fraction of URLs ≤ value")
    ax.set_xlim(-3, 4.2)
    ax.legend(loc="upper left", fontsize=11.5)
    ax.set_xticks([]); ax.grid(False)
    save(fig, "i_ks")


def ill_selection():
    rng = np.random.default_rng(11)
    s = np.sort(rng.gamma(1.4, 0.25, 58))[::-1]
    tau = np.percentile(s, 75)
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    cols = [BLUE if v >= tau else "#CBD3E3" for v in s]
    ax.bar(range(58), s, color=cols, zorder=3, width=0.8)
    ax.axhline(tau, color=RED, lw=2.2, ls="--", zorder=4)
    ax.text(57, tau + 0.04, "τ = 75th percentile of S", color=RED, ha="right", fontsize=12.5, fontweight="bold")
    ax.text(7, s[0] * 0.92, "kept → F*\n(top 25 % = 15 features)", color=BLUE, fontsize=12.5, fontweight="bold", ha="left", va="top")
    ax.set_xlabel("the 58 features, ranked by combined score S = w + I", fontsize=12)
    ax.set_ylabel("score S")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    save(fig, "i_selection")


# --------------------------------------------------------------------------- #
# crops of our KDE figure to mirror the paper's six panels
# --------------------------------------------------------------------------- #
def crop_kde():
    im = Image.open(RES / "figures" / "fig4_feature_distributions.png").convert("RGB")
    W, H = im.size
    top, row_h = 54, (H - 54) / 3
    col_w = W / 6
    # (row, col) of: domain_length, entropy, has_free, has_secure, longest_path_token_length, url_encoded
    cells = [(0, 1), (0, 4), (2, 3), (2, 4), (1, 5), (2, 5)]
    tiles = []
    for r, c in cells:
        box = (int(c * col_w + 2), int(top + r * row_h), int((c + 1) * col_w - 2), int(top + (r + 1) * row_h - 4))
        tiles.append(im.crop(box))
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (tw * 2, th * 3), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t.resize((tw, th)), ((i % 2) * tw, (i // 2) * th))
    sheet.save(OUT / "ours_kde_6.png")
    print("wrote ours_kde_6.png", sheet.size)


def copy_paper_figs():
    p = next(ROOT.glob("Security and Privacy*"))
    for n, name in [(2, "paper_fig1_arch"), (3, "paper_fig2_adaptive"), (4, "paper_fig3_online"),
                    (5, "paper_kde_a"), (6, "paper_kde_b"), (8, "paper_fig5_cooc"),
                    (10, "paper_fig6_perf"), (11, "paper_fig7_ks"), (13, "paper_fig9_updates"),
                    (14, "paper_fig10_compare")]:
        Image.open(p / f"{n}.png").convert("RGB").save(OUT / f"{name}.png", optimize=True)
    for f in ["fig5_cooccurrence", "fig4_feature_distributions"]:
        Image.open(RES / "figures" / f"{f}.png").convert("RGB").save(OUT / f"ours_{f}.png", optimize=True)
    print("copied paper figures")


if __name__ == "__main__":
    offline_variants(); gap_chart(); per_source(); ks_bars()
    stream_curve("S1_paper_setting", "c_stream_s1", (91.5, 99.4))
    stream_curve("S2_cross_corpus_shift", "c_stream_s2", (40, 113), phases=True)
    frozen_decay(); stream_bars(); ablation(); runtime()
    ill_boundary(); ill_ks(); ill_selection()
    crop_kde(); copy_paper_figs()

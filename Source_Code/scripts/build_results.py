#!/usr/bin/env python3
"""Assemble RESULTS.md + figures from the experiment JSON files."""

from __future__ import annotations

import json

import _common as C
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

R = C.RESULTS
pct = C.pct
MODES = ["frozen", "online_static", "adaptive", "periodic_batch"]
MODE_LABEL = {
    "frozen": "Frozen offline model",
    "online_static": "Online SGD, static F* (no adaptation)",
    "adaptive": "**Proposed: KS + adaptive F* + online SGD**",
    "periodic_batch": "Periodic batch retraining (5000 URLs, window 20k)",
}
MODE_SHORT = {"frozen": "Frozen", "online_static": "Online static", "adaptive": "Proposed (adaptive)",
              "periodic_batch": "Periodic batch"}
COLORS = {"frozen": "#7F8C8D", "online_static": "#1F4E79", "adaptive": "#0B6E4F", "periodic_batch": "#C44900"}

PAPER_T5_7 = {  # SGD-SVM rows of paper Tables 5-7
    "iscx": ("ISCX (paper Table 5)", 0.990150, 0.983947, 0.996278, 0.990073),
    "kaggle": ("Kaggle (paper Table 6)", 0.990000, 0.989234, 0.987674, 0.988454),
    "phiusiil": ("PhiUSIIL (paper Table 7)", 0.9878, 0.9784, 0.9873, 0.9828),
}
PAPER_T9 = {100: (92.31, 96.22), 500: (93.89, 97.15), 1000: (93.73, 98.30)}
PAPER_T11 = {"Model update (100 URLs)": 15.9, "Model update (500 URLs)": 29.7,
             "Model update (1000 URLs)": 33.7, "Feature adaptation": 30.9}


def ms(d, k, nd=2):
    return f"{100 * d[k]:.{nd}f} ± {100 * d[k + '_std']:.{nd}f}"


def exp1(L):
    e = json.loads((R / "exp1_offline.json").read_text())
    L += ["## 1. Offline phase (paper Tables 5-7, Fig. 6)", "",
          "LASSO (λ=0.0001) + MI → top-quartile feature set F* (Alg. 1) → SGD (Table 4). 80/20 stratified split, seed 42. "
          "Positive class = phishing.", ""]
    for ds, title in [("phiusiil", "PhiUSIIL"), ("kaggle", "Kaggle (21k subset)"),
                      ("iscx", "ISCX-URL-2016 (authors' 79 lexical features)"), ("hannousse", "Hannousse 2021 (URL strings)"), ("combined", "Combined (PhiUSIIL+Kaggle+Hannousse, de-duplicated)")]:
        d = e[ds]
        L += [f"### {title}  (train {d['n_train']:,} / test {d['n_test']:,}; |F*| = {len(d['selected_features'])})", "",
              "| Model | Accuracy | Precision | Recall | F1 | FPR | Train time (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in d.items():
            if isinstance(v, dict) and "accuracy" in v:
                bold = "**" if k == "SGD-SVM" else ""
                L.append(f"| {bold}{k}{bold} | {pct(v['accuracy'])} | {pct(v['precision'])} | {pct(v['recall'])} | "
                         f"{pct(v['f1'])} | {pct(v['fpr'], 3)} | {v['train_s']:.2f} |")
        L.append("")
    L += ["### SGD-SVM: this study vs. paper-reported values", "",
          "| Corpus | This study Acc | This study Prec | This study Rec | This study F1 | Paper Acc | Paper Prec | Paper Rec | Paper F1 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, (nm, a, p, r, f) in PAPER_T5_7.items():
        v = e[ds]["SGD-SVM"]
        L.append(f"| {nm} | {pct(v['accuracy'])} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | "
                 f"{pct(a)} | {pct(p)} | {pct(r)} | {pct(f)} |")
    v = e["combined"]["SGD-SVM"]
    L.append(f"| Combined (paper Fig. 6: 99.85 % / FPR 0.067 %) | {pct(v['accuracy'])} | {pct(v['precision'])} | "
             f"{pct(v['recall'])} | {pct(v['f1'])} | 99.85% | – | – | – |")
    L += ["", "Combined SGD-SVM per source corpus:", "", "| Source | Accuracy | F1 | FPR |", "|---|---:|---:|---:|"]
    for s, v in e["combined"]["per_source_SGD-SVM"].items():
        L.append(f"| {s} | {pct(v['accuracy'])} | {pct(v['f1'])} | {pct(v['fpr'], 2)} |")
    L += ["", "Selected features F* (combined): `" + "`, `".join(e["combined"]["selected_features"]) + "`", ""]

    sns.set_theme(style="whitegrid", context="talk")
    rows = []
    for ds in ("phiusiil", "kaggle", "iscx", "hannousse", "combined"):
        for k in ("SGD-SVM", "SGD-LR", "SGD-Perceptron"):
            rows.append((ds, k, e[ds][k]["accuracy"]))
    df = pd.DataFrame(rows, columns=["corpus", "model", "accuracy"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="corpus", y="accuracy", hue="model", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Offline SGD variants on LASSO+MI features (paper Tables 5-7)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11)
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig6_offline_sgd_variants.png", dpi=140)
    plt.close(fig)


def stream_tables(L, name, key, caption, phases):
    e = json.loads((R / "exp2_stream.json").read_text())[key]
    runs = e["runs"]
    L += [f"### {name}", "", caption, "",
          f"Stream length {e['n_stream']:,} URLs, mini-batch B=500, 5 seeds (mean ± std, %).", "",
          "| System | Accuracy | Precision | Recall | F1 | FPR | Drift events | Stream time (s) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MODES:
        ov = [r[m]["overall"] for r in runs]
        a = C.agg(ov)
        ev = np.mean([r[m]["n_drift_events"] + r[m]["n_batch_retrains"] for r in runs])
        t = np.mean([r[m]["stream_s"] for r in runs])
        L.append(f"| {MODE_LABEL[m]} | {ms(a,'accuracy')} | {ms(a,'precision')} | {ms(a,'recall')} | {ms(a,'f1')} | "
                 f"{ms(a,'fpr')} | {ev:.1f} | {t:.1f} |")
    L += ["", "Per-phase prequential accuracy (%):", "",
          "| System | " + " | ".join(phases) + " |", "|---|" + "---:|" * len(phases)]
    for m in MODES:
        cells = []
        for p in phases:
            v = [r[m]["phases"][p] for r in runs]
            cells.append(ms(C.agg(v, ("accuracy",)), "accuracy"))
        L.append(f"| {MODE_LABEL[m]} | " + " | ".join(cells) + " |")
    L.append("")
    return e


def curve_fig(e, fname, title, bounds=None):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 5))
    r0 = e["runs"][0]
    for m in MODES:
        acc = np.asarray(r0[m]["batch_acc"])
        sm = pd.Series(acc).rolling(3, center=True, min_periods=1).mean().to_numpy()
        ax.plot(np.arange(len(acc)) * 500, sm, label=MODE_SHORT[m], color=COLORS[m], lw=2.2 if m == "adaptive" else 1.6)
    for ev in r0["adaptive"]["events"]:
        if ev["type"] == "drift":
            ax.axvline(ev["t"], color="#0B6E4F", alpha=0.15, lw=1)
    if bounds:
        for b in bounds[1:-1]:
            ax.axvline(b, color="black", ls="--", lw=1.3)
    ax.set_xlabel("Stream position (URLs)")
    ax.set_ylabel("Prequential accuracy")
    ax.set_title(title + "  (green ticks: KS drift events)")
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(C.FIGS / fname, dpi=140)
    plt.close(fig)


def evolution_table(L, e, label):
    r0 = e["runs"][0]["adaptive"]["events"]
    L += [f"**{label}** – feature set F* over time (adaptive run, seed 42):", "",
          "| Time instance | URLs seen | Δ vs previous | Selected feature set |", "|---|---:|---|---|"]
    prev: set[str] = set()
    k = 0
    for ev in r0:
        if ev["type"] == "initial":
            cur = set(ev["features"])
            L.append(f"| Initial stage | 0 | – | {', '.join(sorted(cur))} |")
            prev = cur
        elif ev["type"] == "drift":
            k += 1
            if k > 3:
                continue
            cur = set(ev["features"])
            add, rem = sorted(cur - prev), sorted(prev - cur)
            L.append(f"| Drift stage {k} | {ev['t']:,} | +{', '.join(add) or '∅'} / −{', '.join(rem) or '∅'} | {', '.join(sorted(cur))} |")
            prev = cur
    total = sum(1 for ev in r0 if ev["type"] == "drift")
    L += ["", f"Total KS-triggered re-selections in this run: {total}.", ""]


def exp2(L):
    L += ["## 2. Online phase: prequential evaluation under drift (paper Sec. 4-5.3)", ""]
    s1 = stream_tables(
        L, "S1 - paper setting: PhishTank Jun 2024 - Jan 2025 (chronological)", "S1_paper_setting",
        "Offline model on the combined old corpora; stream = 12,888 chronologically ordered verified PhishTank phishing URLs "
        "(the paper's window) interleaved with an equal number of held-out benign URLs.", ["Q1", "Q2", "Q3", "Q4"])
    s2 = stream_tables(
        L, "S2 - stress test: strong cross-corpus shift", "S2_cross_corpus_shift",
        "Offline model on PhiUSIIL only; stream = held-out PhiUSIIL (no drift) → Kaggle (drift 1) → Hannousse (drift 2).",
        ["PhiUSIIL (no drift)", "Kaggle (drift 1)", "Hannousse (drift 2)"])
    curve_fig(s1, "fig8_stream_S1_phishtank.png", "S1: PhishTank stream")
    curve_fig(s2, "fig9_stream_S2_cross_corpus.png", "S2: cross-corpus shift", s2_bounds := json.loads((R / "exp2_stream.json").read_text())["S2_cross_corpus_shift"]["phase_bounds"])
    # bar chart of overall accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)
    for ax, (e, t) in zip(axes, [(s1, "S1: PhishTank (paper setting)"), (s2, "S2: cross-corpus shift")]):
        means = [np.mean([r[m]["overall"]["accuracy"] for r in e["runs"]]) for m in MODES]
        stds = [np.std([r[m]["overall"]["accuracy"] for r in e["runs"]]) for m in MODES]
        ax.bar([MODE_SHORT[m] for m in MODES], means, yerr=stds, color=[COLORS[m] for m in MODES], capsize=4)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(means):
            ax.text(i, v + 0.01, f"{100*v:.1f}", ha="center", fontsize=10)
    axes[0].set_ylabel("Prequential accuracy")
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig10_stream_accuracy_bars.png", dpi=140)
    plt.close(fig)

    L += ["### Evolution of the adaptive feature set (paper Table 8)", ""]
    evolution_table(L, s1, "S1 (PhishTank)")
    evolution_table(L, s2, "S2 (cross-corpus)")


def exp3(L):
    e = json.loads((R / "exp3_ablation.json").read_text())
    L += ["## 3. Ablation: effect of adaptive feature selection (paper Table 9)", "",
          "Offline model adapted with N labelled new URLs, tested on held-out new URLs (10 seeds, mean ± std, %). "
          "'Static' = online SGD with F* frozen; 'Adaptive' = full framework. A frozen model that flags almost every URL as phishing "
          "(FPR ≈ 99 %) still reaches F1 ≈ 65 %, so compare accuracy and FPR as well as F1.", ""]
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, (tgt, d) in zip(axes, e.items()):
        s = d["summary"]
        L += [f"### Target: {tgt}", "",
              "| Model type | Training URLs | Accuracy | Precision | Recall | F1 | FPR | mean \\|F*\\| | drift events |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        v0 = s["0"]["static"]
        L.append(f"| No adaptation (frozen offline) | 0 | {ms(v0,'accuracy')} | {ms(v0,'precision')} | {ms(v0,'recall')} | "
                 f"{ms(v0,'f1')} | {ms(v0,'fpr')} | {v0['n_features']:.0f} | – |")
        for n in ("100", "500", "1000"):
            for key, lab in (("static", "Without adaptive features"), ("adaptive", "With adaptive features")):
                v = s[n][key]
                L.append(f"| {lab} | {n} | {ms(v,'accuracy')} | {ms(v,'precision')} | {ms(v,'recall')} | {ms(v,'f1')} | "
                         f"{ms(v,'fpr')} | {v['n_features']:.1f} | {v['n_drift_events']:.2f} |")
        L.append("")
        ns = [100, 500, 1000]
        for key, lab, c in (("static", "Without adaptive features", "#7F8C8D"), ("adaptive", "With adaptive features", "#0B6E4F")):
            mu = [s[str(n)][key]["f1"] for n in ns]
            sd = [s[str(n)][key]["f1_std"] for n in ns]
            ax.errorbar(ns, mu, yerr=sd, marker="o", label=lab, color=c, capsize=4)
        ax.axhline(v0["f1"], color="black", ls=":", label="frozen")
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels(ns)
        ax.set_title(tgt.split(" (")[0], fontsize=14)
        ax.set_xlabel("Training URLs")
    axes[0].set_ylabel("F1 (phishing)")
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig11_ablation_table9.png", dpi=140)
    plt.close(fig)
    L += ["Paper Table 9 (F1 %, static → adaptive): " + "; ".join(f"{n} URLs: {a} → {b}" for n, (a, b) in PAPER_T9.items()), ""]


def exp4(L):
    e = json.loads((R / "exp4_runtime.json").read_text())
    L += ["## 4. Runtime (paper Tables 10-11)", "", "### Per-URL latency (raw URL → features → SGD-SVM score, 1 CPU thread)", "",
          "| Dataset | Avg (ms) | Median (ms) | 95th pct (ms) | URLs |", "|---|---:|---:|---:|---:|"]
    for k, v in e["latency"].items():
        L.append(f"| {k} | {v['avg_ms']:.3f} | {v['median_ms']:.3f} | {v['p95_ms']:.3f} | {v['n_urls']:,} |")
    L += ["", "Paper Table 10 reports 12.4-15.2 ms per URL (different hardware/implementation).", "",
          f"### Incremental update vs. batch retraining (mean of {e['repeats']} runs; batch = LASSO+MI + full SGD fit on "
          f"{e['n_offline_train']:,} + N URLs)", "",
          "| Operation | Incremental (ms) | Batch retraining (ms) | Efficiency gain (×) | Paper gain (×) |", "|---|---:|---:|---:|---:|"]
    for k, v in e["incremental_vs_batch"].items():
        L.append(f"| {k} | {v['incremental_ms']:.2f} | {v['batch_ms']:.1f} | {v['gain_x']:.1f} | {PAPER_T11[k]} |")
    L.append("")


def exp5(L):
    e = json.loads((R / "exp5_analysis.json").read_text())["ks"]
    L += ["## 5. Old vs. new feature distributions (paper Fig. 4, 5, 7)", "",
          f"KS test between 1500 old phishing URLs and 500 new PhishTank phishing URLs at δ = 0.05: "
          f"**{e['n_drifted_at_0.05']} of {e['n_features']} features drifted.** Top features by KS statistic:", "",
          "| Feature | KS stat (window) | p-value | KS stat (full samples) |", "|---|---:|---:|---:|"]
    top = sorted(e["features"].items(), key=lambda kv: -kv[1]["ks_window"])[:12]
    for k, v in top:
        L.append(f"| {k} | {v['ks_window']:.3f} | {v['p_window']:.2g} | {v['ks_full']:.3f} |")
    L += ["", "Figures: `figures/fig4_feature_distributions.png`, `figures/fig5_cooccurrence.png`, "
          "`figures/fig7_ks_statistics.png`.", ""]


def main() -> None:
    L = ["# Experimental results", "",
         "Re-implementation of Krishnan & Naik, *Security and Privacy* 2025 (DOI 10.1002/spy2.70057). "
         "Every number below is produced by the scripts in `Source_Code/scripts` (see `README.md`); "
         "JSON with all raw values is in `results/`. Label convention: 1 = phishing (positive class).", ""]
    exp1(L)
    exp2(L)
    exp3(L)
    exp4(L)
    exp5(L)
    (R / "RESULTS.md").write_text("\n".join(L))
    print("wrote", R / "RESULTS.md")


if __name__ == "__main__":
    main()

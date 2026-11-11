# Evaluation of the work against the assignment

**Paper:** S. Krishnan and S. M. Naik, *An Efficient Framework for Phishing URL Detection Using Adaptive Drift-Aware Feature Monitoring and Online Learning*, Security and Privacy 8(4): e70057, 2025. DOI [10.1002/spy2.70057](https://doi.org/10.1002/spy2.70057).

**Task:** implement the methodology of the paper and reproduce the reported experiments and results as closely as possible. Deliverables: source code, implementation/results, short project report, presentation/demo.

## 1. Deliverables

| Requirement | Status | Where |
|---|---|---|
| Source code | Delivered | `Source_Code/` (package, 5 experiment scripts, 16 unit tests, README, Makefile) |
| Implementation / results | Delivered | `Experimental_Results/` (`RESULTS.md`, 5 JSON files, figures 4-11) |
| Short project report | Delivered | `Project_Report.pdf` |
| Presentation | Delivered | `Presentation_Slides.pdf`; slide 1 carries the full title and complete reference |
| Demo | Delivered | `Source_Code/web/` (browser demo of the lexical SGD-SVM detector) |

## 2. Methodology implemented (paper Sections 3-4)

| Paper element | Implemented in |
|---|---|
| Lexical URL features (Table 3), no HTML, no third-party lookups | `features/extractor.py` (58 features) |
| LASSO, lambda = 0.0001, 1000 iterations (Table 4, eq. 1) | `selection/dynamic.py` |
| Mutual-information ranking (eq. 2) | `selection/dynamic.py` |
| Score S = w + I, threshold tau = 75th percentile, F* = {S >= tau} (Alg. 1) | `selection/dynamic.py` |
| Importance history H, obsolete / added features (Alg. 1) | `selection/dynamic.py` (`update`) |
| Kolmogorov-Smirnov drift test per feature, delta = 0.05, buffer 1500 | `drift/ks_monitor.py` |
| Drift triggers re-selection and partial retraining | `framework.py` |
| SGD-SVM / SGD-LR / SGD-Perceptron, adaptive learning rate, eta0 = 0.01, balanced, early stopping | `learning/online_sgd.py` |
| Mini-batches B = 500, test-then-train evaluation, seed 42, 80:20 split | `framework.py`, `scripts/` |

## 3. Experiments (paper Section 5)

Paper values are quoted from the paper; "this study" values are from `RESULTS.md`. Differences in absolute level come from the data versions and features available (Section 4), not from changes to the method.

| Paper experiment | This study | Paper | Assessment |
|---|---|---|---|
| Table 7, PhiUSIIL, SGD-SVM accuracy | 99.38% | 98.78% | Matches (within 0.6 point) |
| Table 7, PhiUSIIL, SGD-SVM precision / recall / F1 | 99.99 / 98.54 / 99.26% | 97.84 / 98.73 / 98.28% | Matches |
| Tables 5-7: SGD-SVM is the strongest of the three SGD variants | SVM best or within 0.1 point of best on PhiUSIIL; best on Kaggle, ISCX and combined | SVM best | Consistent |
| Table 5, ISCX, SGD-SVM accuracy | 90.83% (authors' 79 features, 15,367 URLs); 96.06% with all features | 99.02% | Same ordering of models, lower level |
| Table 6, Kaggle, SGD-SVM accuracy | 73.81% (21k subset, URL strings only) | 99.00% | Lower level |
| Fig. 6, combined corpus, SGD-SVM | 93.58% accuracy, 6.23% FPR | 99.85%, 0.067% | Lower level |
| Figs. 4, 5, 7: old and new URL distributions differ | 34 of 58 features drift (KS, delta = 0.05) | url_length, token_count, num_dots, ... drift | Consistent (same features) |
| Table 8: the selected feature set changes after drift | Feature additions/removals logged per drift event | Qualitative | Consistent |
| Online learning outperforms a frozen model | +5.1 points (PhishTank stream), +15.0 points (cross-corpus stream) | Qualitative | Consistent |
| Table 9: F1 with adaptive features, N = 100 / 500 / 1000 | 96.08 / 96.90 / 96.85% | 96.22 / 97.15 / 98.30% | Within 0.1-1.5 points |
| Table 9: F1 without adaptive features, N = 100 / 500 / 1000 | 96.19 / 96.25 / 96.54% | 92.31 / 93.89 / 93.73% | Our static baseline is stronger; adaptive gains accuracy (+0.7 point) and lowers FPR at N >= 500 |
| Table 10: per-URL latency | 0.16 ms | 12.4-15.2 ms | Different hardware and implementation; both real-time |
| Table 11: incremental update vs. batch retraining | 2140-2770x (update), 12.9x (feature adaptation) | 15.9-33.7x | Same conclusion: incremental is far cheaper |

## 4. Data and setup (what differs from the paper, and why)

* **ISCX-URL-2016** is available to us only as the authors' pre-computed feature vectors (benign vs. phishing, 15,367 rows, no URL strings). Table 5 uses those vectors; the combined corpus and the streams use raw URL strings, so they use the Hannousse & Yahiouhmed (2021) corpus in its place.
* **Kaggle** is the 21k balanced "Phishing Site URLs" subset.
* **Live stream:** PhishTank verified-online snapshot of 2026-11-06, filtered to the paper's window (1 Jun 2024 - 31 Jan 2025): 12,888 URLs not present in the old corpora. PhishTank is phishing-only; the benign side of each stream is held-out benign URLs from the old corpora.
* **Features:** only what can be computed from the URL string (Table 3 features; redirect-chain and domain-mismatch are lexical proxies).
* **Choices the paper leaves open:** a drift event needs >= 25% of F* to fail the KS test; history weight 0.5; 3 replay epochs after re-selection; standardised inputs clipped to +-10; classifier kept in the full feature space with F* as a mask.

## 5. Reproduce

```bash
cd Source_Code && pip install -r requirements.txt
make test && make all      # data download, exp1-exp5, RESULTS.md, figures
```

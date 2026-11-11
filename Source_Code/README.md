# Drift-aware phishing URL detection — re-implementation

Re-implementation and experimental study of

> S. Krishnan and S. M. Naik, *An Efficient Framework for Phishing URL Detection Using Adaptive
> Drift-Aware Feature Monitoring and Online Learning*, Security and Privacy 8(4): e70057, 2025.
> DOI [10.1002/spy2.70057](https://doi.org/10.1002/spy2.70057)

The implementation follows the paper's description.

## Pipeline (paper Fig. 1–3, Alg. 1, Table 4)

| Paper component | Code |
|---|---|
| Lexical URL features, Table 3 (+ Table 8 names), no HTML / no third-party lookups | `src/drift_phish/features/extractor.py` (`extract_paper_features`, 58 features) |
| LASSO (λ = 0.0001) + Mutual Information, `S = w + I`, `τ = P75(S)`, history `H` | `src/drift_phish/selection/dynamic.py` |
| Kolmogorov–Smirnov drift monitor, δ = 0.05, reference buffer 1500 | `src/drift_phish/drift/ks_monitor.py` |
| SGD-SVM / LR / Perceptron, adaptive η₀ = 0.01, balanced, early stopping, online `partial_fit` | `src/drift_phish/learning/online_sgd.py` |
| End-to-end loop (prequential, B = 500, drift → re-selection → partial retraining) | `src/drift_phish/framework.py` |

`FrameworkConfig(mode=...)` gives the compared systems: `adaptive` (proposed), `online_static`
(paper's "without adaptive features"), `frozen`, `periodic_batch`.

## Run

```bash
pip install -r requirements.txt
make test          # 16 unit tests
make data          # downloads PhiUSIIL (UCI 967), Kaggle subset, ISCX-URL-2016 features, Hannousse 2021, PhishTank
make all           # = data, test, exp1..exp5, results  (≈ 20 min on a laptop CPU)
```

Individual experiments: `make exp1` … `make exp5`, `make results`. Outputs are written to
`../Experimental_Results/` (`*.json`, `RESULTS.md`, `figures/`).

| Script | Reproduces |
|---|---|
| `scripts/exp1_offline.py` | Tables 5–7, Fig. 6 (offline SGD-SVM/LR/Perceptron on LASSO+MI features; ISCX from the authors' 79 feature vectors) |
| `scripts/exp2_stream.py` | Sec. 5 online phase: prequential evaluation under drift (S1 PhishTank, S2 cross-corpus) |
| `scripts/exp3_ablation.py` | Table 9 (adaptive vs. static feature selection, N = 100/500/1000) |
| `scripts/exp4_runtime.py` | Tables 10–11 (per-URL latency, incremental vs. batch) |
| `scripts/exp5_analysis.py` + `build_results.py` | Table 8, Figs. 4, 5, 7 |

## Data and setup

* **ISCX-URL-2016** is distributed as pre-computed feature vectors (79 lexical features, benign vs. phishing, no URL
  strings). Table 5 uses those vectors. The combined corpus and the streams need raw URLs, so they use
  **Hannousse & Yahiouhmed (2021)** (11,430 URLs) in ISCX's place. **Kaggle** is the 21k balanced subset of
  "Phishing Site URLs" mirrored on Hugging Face.
* **PhishTank**: `data/raw/phishtank_20261106.csv` is a snapshot of the live verified-online feed. URLs submitted
  1 Jun 2024 – 31 Jan 2025 (the paper's window; 12,888 not in the old corpora) form the stream. PhishTank is
  phishing-only; the benign side of every stream is held-out old benign URLs. Keep the snapshot file for exact
  reproduction.
* **Features**: URL-string experiments use only features computed from the URL (`PAPER_FEATURE_NAMES`);
  redirect chain / domain-mismatch are lexical proxies (no HTTP fetches).
* Label convention everywhere: **1 = phishing** (positive class); FPR = benign flagged as phishing.
* Choices the paper leaves open (documented in `framework.py`): a drift event requires ≥ 25 % of F* to fail the KS
  test, history weight 0.5, 3 replay epochs over the buffer after re-selection, standardised inputs clipped to ±10.

Paper-vs-this-study comparison: `../Experimental_Results/EVALUATION.md`.

`scripts/train_demo_model.py` exports the model used by the `web/` browser demo.

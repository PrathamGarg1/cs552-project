# Experimental results

Re-implementation of Krishnan & Naik, *Security and Privacy* 2025 (DOI 10.1002/spy2.70057). Every number below is produced by the scripts in `Source_Code/scripts` (see `README.md`); JSON with all raw values is in `results/`. Label convention: 1 = phishing (positive class).

## 1. Offline phase (paper Tables 5-7, Fig. 6)

LASSO (λ=0.0001) + MI → top-quartile feature set F* (Alg. 1) → SGD (Table 4). 80/20 stratified split, seed 42. Positive class = phishing.

### PhiUSIIL  (train 188,296 / test 47,074; |F*| = 15)

| Model | Accuracy | Precision | Recall | F1 | FPR | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **SGD-SVM** | 99.38% | 99.99% | 98.54% | 99.26% | 0.004% | 1.36 |
| SGD-LR | 99.39% | 99.95% | 98.62% | 99.28% | 0.037% | 3.15 |
| SGD-Perceptron | 99.22% | 99.27% | 98.90% | 99.09% | 0.541% | 2.14 |
| SGD-SVM (all features, no selection) | 99.46% | 99.92% | 98.82% | 99.37% | 0.059% | 1.59 |
| LogisticRegression (batch, same F*) | 99.48% | 99.92% | 98.86% | 99.39% | 0.056% | 0.44 |
| DecisionTree (batch, same F*) | 99.64% | 99.89% | 99.25% | 99.57% | 0.078% | 0.16 |
| RandomForest (batch, same F*) | 99.66% | 99.89% | 99.31% | 99.60% | 0.082% | 2.25 |
| GaussianNB (batch, same F*) | 99.26% | 99.81% | 98.46% | 99.13% | 0.141% | 0.12 |

### Kaggle (21k subset)  (train 16,588 / test 4,147; |F*| = 15)

| Model | Accuracy | Precision | Recall | F1 | FPR | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **SGD-SVM** | 73.81% | 81.84% | 60.33% | 69.46% | 13.048% | 0.12 |
| SGD-LR | 73.14% | 75.62% | 67.27% | 71.20% | 21.143% | 0.13 |
| SGD-Perceptron | 61.71% | 70.96% | 37.96% | 49.46% | 15.143% | 0.08 |
| SGD-SVM (all features, no selection) | 79.79% | 87.85% | 68.54% | 77.00% | 9.238% | 0.11 |
| LogisticRegression (batch, same F*) | 73.19% | 76.34% | 66.19% | 70.91% | 20.000% | 0.02 |
| DecisionTree (batch, same F*) | 82.86% | 85.68% | 78.36% | 81.86% | 12.762% | 0.06 |
| RandomForest (batch, same F*) | 86.13% | 89.78% | 81.14% | 85.25% | 9.000% | 0.24 |
| GaussianNB (batch, same F*) | 69.30% | 84.62% | 46.21% | 59.78% | 8.190% | 0.01 |

### ISCX-URL-2016 (authors' 79 lexical features)  (train 12,293 / test 3,074; |F*| = 20)

| Model | Accuracy | Precision | Recall | F1 | FPR | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **SGD-SVM** | 90.83% | 90.49% | 90.97% | 90.73% | 9.313% | 0.07 |
| SGD-LR | 90.27% | 89.44% | 91.03% | 90.23% | 10.469% | 0.12 |
| SGD-Perceptron | 90.11% | 89.61% | 90.44% | 90.03% | 10.212% | 0.07 |
| SGD-SVM (all features, no selection) | 96.06% | 96.29% | 95.72% | 96.00% | 3.597% | 0.07 |
| LogisticRegression (batch, same F*) | 90.76% | 90.59% | 90.71% | 90.65% | 9.184% | 0.03 |
| DecisionTree (batch, same F*) | 97.40% | 97.87% | 96.84% | 97.35% | 2.055% | 0.08 |
| RandomForest (batch, same F*) | 98.31% | 99.00% | 97.56% | 98.27% | 0.963% | 0.26 |
| GaussianNB (batch, same F*) | 90.83% | 92.44% | 88.66% | 90.51% | 7.065% | 0.01 |

### Hannousse 2021 (URL strings)  (train 9,143 / test 2,286; |F*| = 15)

| Model | Accuracy | Precision | Recall | F1 | FPR | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **SGD-SVM** | 77.82% | 82.85% | 70.17% | 75.98% | 14.523% | 0.05 |
| SGD-LR | 77.73% | 81.96% | 71.13% | 76.16% | 15.661% | 0.09 |
| SGD-Perceptron | 72.57% | 70.06% | 78.83% | 74.19% | 33.683% | 0.05 |
| SGD-SVM (all features, no selection) | 84.34% | 87.78% | 79.79% | 83.59% | 11.111% | 0.07 |
| LogisticRegression (batch, same F*) | 77.91% | 82.29% | 71.13% | 76.30% | 15.311% | 0.02 |
| DecisionTree (batch, same F*) | 80.84% | 81.22% | 80.23% | 80.72% | 18.548% | 0.03 |
| RandomForest (batch, same F*) | 86.83% | 89.13% | 83.90% | 86.44% | 10.236% | 0.17 |
| GaussianNB (batch, same F*) | 72.40% | 88.10% | 51.79% | 65.23% | 6.999% | 0.01 |

### Combined (PhiUSIIL+Kaggle+Hannousse, de-duplicated)  (train 210,598 / test 52,650; |F*| = 15)

| Model | Accuracy | Precision | Recall | F1 | FPR | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **SGD-SVM** | 93.58% | 91.78% | 93.33% | 92.55% | 6.232% | 1.38 |
| SGD-LR | 92.97% | 91.96% | 91.53% | 91.75% | 5.960% | 2.15 |
| SGD-Perceptron | 93.00% | 90.68% | 93.19% | 91.92% | 7.137% | 1.44 |
| SGD-SVM (all features, no selection) | 95.07% | 93.35% | 95.24% | 94.28% | 5.059% | 1.45 |
| LogisticRegression (batch, same F*) | 92.57% | 92.70% | 89.65% | 91.15% | 5.261% | 0.32 |
| DecisionTree (batch, same F*) | 96.29% | 95.69% | 95.62% | 95.66% | 3.212% | 0.37 |
| RandomForest (batch, same F*) | 96.98% | 95.92% | 97.06% | 96.49% | 3.080% | 2.33 |
| GaussianNB (batch, same F*) | 86.12% | 89.67% | 76.28% | 82.43% | 6.550% | 0.06 |

### SGD-SVM: this study vs. paper-reported values

| Corpus | This study Acc | This study Prec | This study Rec | This study F1 | Paper Acc | Paper Prec | Paper Rec | Paper F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ISCX (paper Table 5) | 90.83% | 90.49% | 90.97% | 90.73% | 99.02% | 98.39% | 99.63% | 99.01% |
| Kaggle (paper Table 6) | 73.81% | 81.84% | 60.33% | 69.46% | 99.00% | 98.92% | 98.77% | 98.85% |
| PhiUSIIL (paper Table 7) | 99.38% | 99.99% | 98.54% | 99.26% | 98.78% | 97.84% | 98.73% | 98.28% |
| Combined (paper Fig. 6: 99.85 % / FPR 0.067 %) | 93.58% | 91.78% | 93.33% | 92.55% | 99.85% | – | – | – |

Combined SGD-SVM per source corpus:

| Source | Accuracy | F1 | FPR |
|---|---:|---:|---:|
| phiusiil | 98.52% | 98.20% | 0.01% |
| kaggle | 59.59% | 59.83% | 41.83% |
| hannousse | 52.68% | 66.84% | 90.25% |

Selected features F* (combined): `url_length`, `domain_length`, `path_length`, `token_count`, `num_subdirectories`, `special_char_count`, `number_count`, `num_dots`, `num_slashes`, `num_hyphens`, `non_alphanumeric_chars`, `has_https`, `scheme_http`, `num_delimiters_path`, `digit_ratio`

## 2. Online phase: prequential evaluation under drift (paper Sec. 4-5.3)

### S1 - paper setting: PhishTank Jun 2024 - Jan 2025 (chronological)

Offline model on the combined old corpora; stream = 12,888 chronologically ordered verified PhishTank phishing URLs (the paper's window) interleaved with an equal number of held-out benign URLs.

Stream length 25,776 URLs, mini-batch B=500, 5 seeds (mean ± std, %).

| System | Accuracy | Precision | Recall | F1 | FPR | Drift events | Stream time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen offline model | 92.63 ± 6.43 | 93.49 ± 0.91 | 91.51 ± 13.01 | 92.03 ± 7.81 | 6.25 ± 0.24 | 0.0 | 0.0 |
| Online SGD, static F* (no adaptation) | 97.77 ± 0.16 | 96.96 ± 0.33 | 98.64 ± 0.09 | 97.79 ± 0.16 | 3.10 ± 0.35 | 0.0 | 0.1 |
| **Proposed: KS + adaptive F* + online SGD** | 97.80 ± 0.05 | 96.90 ± 0.07 | 98.76 ± 0.14 | 97.82 ± 0.05 | 3.16 ± 0.07 | 28.6 | 7.0 |
| Periodic batch retraining (5000 URLs, window 20k) | 97.31 ± 0.61 | 96.38 ± 0.05 | 98.31 ± 1.22 | 97.33 ± 0.62 | 3.69 ± 0.03 | 5.0 | 9.9 |

Per-phase prequential accuracy (%):

| System | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Frozen offline model | 93.91 ± 3.20 | 91.22 ± 9.88 | 91.87 ± 8.35 | 93.53 ± 4.28 |
| Online SGD, static F* (no adaptation) | 96.70 ± 0.16 | 98.13 ± 0.22 | 98.40 ± 0.17 | 97.86 ± 0.30 |
| **Proposed: KS + adaptive F* + online SGD** | 96.96 ± 0.15 | 98.27 ± 0.10 | 98.14 ± 0.14 | 97.83 ± 0.12 |
| Periodic batch retraining (5000 URLs, window 20k) | 94.75 ± 2.31 | 98.15 ± 0.16 | 98.37 ± 0.13 | 97.95 ± 0.07 |

### S2 - stress test: strong cross-corpus shift

Offline model on PhiUSIIL only; stream = held-out PhiUSIIL (no drift) → Kaggle (drift 1) → Hannousse (drift 2).

Stream length 37,164 URLs, mini-batch B=500, 5 seeds (mean ± std, %).

| System | Accuracy | Precision | Recall | F1 | FPR | Drift events | Stream time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen offline model | 55.51 ± 0.10 | 52.29 ± 0.04 | 97.73 ± 0.23 | 68.13 ± 0.08 | 84.52 ± 0.35 | 0.0 | 0.0 |
| Online SGD, static F* (no adaptation) | 70.49 ± 0.22 | 71.77 ± 0.88 | 64.93 ± 1.86 | 68.15 ± 0.66 | 24.25 ± 1.75 | 0.0 | 0.1 |
| **Proposed: KS + adaptive F* + online SGD** | 73.21 ± 1.13 | 74.87 ± 2.27 | 67.82 ± 0.52 | 71.14 ± 0.83 | 21.66 ± 2.54 | 7.0 | 3.3 |
| Periodic batch retraining (5000 URLs, window 20k) | 69.97 ± 0.18 | 68.47 ± 0.28 | 71.00 ± 1.36 | 69.70 ± 0.55 | 31.00 ± 1.01 | 7.0 | 15.8 |

Per-phase prequential accuracy (%):

| System | PhiUSIIL (no drift) | Kaggle (drift 1) | Hannousse (drift 2) |
|---|---:|---:|---:|
| Frozen offline model | 99.30 ± 0.13 | 48.06 ± 0.17 | 49.87 ± 0.01 |
| Online SGD, static F* (no adaptation) | 99.32 ± 0.12 | 62.56 ± 0.31 | 72.25 ± 0.64 |
| **Proposed: KS + adaptive F* + online SGD** | 99.32 ± 0.12 | 66.47 ± 2.25 | 74.03 ± 0.62 |
| Periodic batch retraining (5000 URLs, window 20k) | 99.30 ± 0.13 | 62.24 ± 0.31 | 71.18 ± 0.16 |

### Evolution of the adaptive feature set (paper Table 8)

**S1 (PhishTank)** – feature set F* over time (adaptive run, seed 42):

| Time instance | URLs seen | Δ vs previous | Selected feature set |
|---|---:|---|---|
| Initial stage | 0 | – | digit_ratio, domain_length, has_https, non_alphanumeric_chars, num_delimiters_path, num_dots, num_hyphens, num_slashes, num_subdirectories, number_count, path_length, scheme_http, special_char_count, token_count, url_length |
| Drift stage 1 | 0 | +entropy, host_entropy, longest_host_label_length / −has_https, num_dots, scheme_http | digit_ratio, domain_length, entropy, host_entropy, longest_host_label_length, non_alphanumeric_chars, num_delimiters_path, num_hyphens, num_slashes, num_subdirectories, number_count, path_length, special_char_count, token_count, url_length |
| Drift stage 2 | 500 | +num_ampersand / −num_hyphens | digit_ratio, domain_length, entropy, host_entropy, longest_host_label_length, non_alphanumeric_chars, num_ampersand, num_delimiters_path, num_slashes, num_subdirectories, number_count, path_length, special_char_count, token_count, url_length |
| Drift stage 3 | 1,000 | +num_digits_query / −num_ampersand | digit_ratio, domain_length, entropy, host_entropy, longest_host_label_length, non_alphanumeric_chars, num_delimiters_path, num_digits_query, num_slashes, num_subdirectories, number_count, path_length, special_char_count, token_count, url_length |

Total KS-triggered re-selections in this run: 28.

**S2 (cross-corpus)** – feature set F* over time (adaptive run, seed 42):

| Time instance | URLs seen | Δ vs previous | Selected feature set |
|---|---:|---|---|
| Initial stage | 0 | – | digit_ratio, domain_length, has_https, longest_host_label_length, non_alphanumeric_chars, num_delimiters_path, num_equals, num_hyphens, num_percent, num_slashes, num_subdirectories, path_length, scheme_http, special_char_count, token_count |
| Drift stage 1 | 5,000 | +num_dots, url_length / −num_equals, num_percent | digit_ratio, domain_length, has_https, longest_host_label_length, non_alphanumeric_chars, num_delimiters_path, num_dots, num_hyphens, num_slashes, num_subdirectories, path_length, scheme_http, special_char_count, token_count, url_length |
| Drift stage 2 | 5,500 | +num_subdomains / −url_length | digit_ratio, domain_length, has_https, longest_host_label_length, non_alphanumeric_chars, num_delimiters_path, num_dots, num_hyphens, num_slashes, num_subdirectories, num_subdomains, path_length, scheme_http, special_char_count, token_count |
| Drift stage 3 | 6,000 | +num_digits_query, num_equals, num_percent, number_count / −has_https, num_delimiters_path, num_subdirectories, scheme_http | digit_ratio, domain_length, longest_host_label_length, non_alphanumeric_chars, num_digits_query, num_dots, num_equals, num_hyphens, num_percent, num_slashes, num_subdomains, number_count, path_length, special_char_count, token_count |

Total KS-triggered re-selections in this run: 7.

## 3. Ablation: effect of adaptive feature selection (paper Table 9)

Offline model adapted with N labelled new URLs, tested on held-out new URLs (10 seeds, mean ± std, %). 'Static' = online SGD with F* frozen; 'Adaptive' = full framework. A frozen model that flags almost every URL as phishing (FPR ≈ 99 %) still reaches F1 ≈ 65 %, so compare accuracy and FPR as well as F1.

### Target: PhishTank Jun2024-Jan2025 (offline: combined)

| Model type | Training URLs | Accuracy | Precision | Recall | F1 | FPR | mean \|F*\| | drift events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No adaptation (frozen offline) | 0 | 94.21 ± 5.27 | 93.64 ± 0.91 | 94.75 ± 10.58 | 93.90 ± 6.47 | 6.33 ± 0.32 | 15 | – |
| Without adaptive features | 100 | 96.07 ± 0.17 | 93.42 ± 0.25 | 99.13 ± 0.17 | 96.19 ± 0.16 | 6.99 ± 0.29 | 15.0 | 0.00 |
| With adaptive features | 100 | 95.95 ± 1.15 | 93.42 ± 1.43 | 98.89 ± 0.73 | 96.08 ± 1.09 | 6.98 ± 1.61 | 15.0 | 1.00 |
| Without adaptive features | 500 | 96.14 ± 0.35 | 93.58 ± 0.54 | 99.08 ± 0.26 | 96.25 ± 0.33 | 6.80 ± 0.61 | 15.0 | 0.00 |
| With adaptive features | 500 | 96.84 ± 0.31 | 94.88 ± 0.70 | 99.02 ± 0.26 | 96.90 ± 0.29 | 5.35 ± 0.78 | 15.0 | 1.00 |
| Without adaptive features | 1000 | 96.45 ± 0.41 | 94.11 ± 0.68 | 99.10 ± 0.25 | 96.54 ± 0.39 | 6.20 ± 0.76 | 15.0 | 0.00 |
| With adaptive features | 1000 | 96.78 ± 0.32 | 94.89 ± 0.52 | 98.88 ± 0.34 | 96.85 ± 0.31 | 5.33 ± 0.57 | 15.0 | 1.00 |

### Target: Kaggle (offline: PhiUSIIL)

| Model type | Training URLs | Accuracy | Precision | Recall | F1 | FPR | mean \|F*\| | drift events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No adaptation (frozen offline) | 0 | 48.23 ± 0.25 | 48.78 ± 0.13 | 97.07 ± 0.52 | 64.93 ± 0.19 | 99.37 ± 0.53 | 15 | – |
| Without adaptive features | 100 | 52.20 ± 2.35 | 51.44 ± 2.10 | 64.75 ± 4.03 | 57.19 ± 0.96 | 60.03 ± 8.14 | 15.0 | 0.00 |
| With adaptive features | 100 | 61.01 ± 2.49 | 61.79 ± 3.02 | 56.23 ± 8.70 | 58.42 ± 4.66 | 34.32 ± 8.05 | 15.0 | 1.00 |
| Without adaptive features | 500 | 60.08 ± 1.58 | 64.49 ± 5.34 | 47.75 ± 15.68 | 52.81 ± 7.92 | 27.90 ± 14.95 | 15.0 | 0.00 |
| With adaptive features | 500 | 64.17 ± 1.50 | 67.76 ± 6.21 | 57.94 ± 16.72 | 60.43 ± 6.22 | 29.75 ± 17.51 | 15.0 | 1.00 |
| Without adaptive features | 1000 | 61.26 ± 1.43 | 65.09 ± 6.03 | 51.18 ± 13.07 | 55.64 ± 6.90 | 28.91 ± 12.67 | 15.0 | 0.00 |
| With adaptive features | 1000 | 64.86 ± 1.19 | 67.28 ± 5.12 | 59.44 ± 12.12 | 61.86 ± 5.78 | 29.85 ± 11.09 | 15.0 | 1.10 |

### Target: Hannousse (offline: PhiUSIIL)

| Model type | Training URLs | Accuracy | Precision | Recall | F1 | FPR | mean \|F*\| | drift events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No adaptation (frozen offline) | 0 | 49.90 ± 0.06 | 49.94 ± 0.03 | 99.20 ± 0.10 | 66.44 ± 0.05 | 99.39 ± 0.04 | 15 | – |
| Without adaptive features | 100 | 50.81 ± 0.88 | 50.41 ± 0.46 | 98.73 ± 0.53 | 66.74 ± 0.33 | 97.09 ± 2.16 | 15.0 | 0.00 |
| With adaptive features | 100 | 62.85 ± 4.71 | 63.31 ± 7.88 | 70.38 ± 13.20 | 65.15 ± 3.06 | 44.68 ± 21.02 | 15.0 | 1.00 |
| Without adaptive features | 500 | 66.59 ± 2.71 | 64.54 ± 3.42 | 74.99 ± 7.48 | 69.05 ± 3.06 | 41.80 ± 8.60 | 15.0 | 0.00 |
| With adaptive features | 500 | 73.62 ± 1.59 | 75.81 ± 3.33 | 70.04 ± 6.59 | 72.49 ± 2.92 | 22.81 ± 5.56 | 15.0 | 1.00 |
| Without adaptive features | 1000 | 72.06 ± 1.09 | 72.34 ± 3.22 | 72.28 ± 5.66 | 72.04 ± 1.73 | 28.16 ± 6.56 | 15.0 | 0.00 |
| With adaptive features | 1000 | 74.00 ± 1.00 | 77.62 ± 4.30 | 68.40 ± 6.07 | 72.35 ± 1.82 | 20.40 ± 6.73 | 15.0 | 1.00 |

Paper Table 9 (F1 %, static → adaptive): 100 URLs: 92.31 → 96.22; 500 URLs: 93.89 → 97.15; 1000 URLs: 93.73 → 98.3

## 4. Runtime (paper Tables 10-11)

### Per-URL latency (raw URL → features → SGD-SVM score, 1 CPU thread)

| Dataset | Avg (ms) | Median (ms) | 95th pct (ms) | URLs |
|---|---:|---:|---:|---:|
| phiusiil | 0.154 | 0.149 | 0.180 | 2,000 |
| kaggle | 0.162 | 0.156 | 0.196 | 2,000 |
| hannousse | 0.169 | 0.162 | 0.208 | 2,000 |
| phishtank | 0.164 | 0.154 | 0.211 | 2,000 |
| combined | 0.160 | 0.154 | 0.192 | 6,000 |

Paper Table 10 reports 12.4-15.2 ms per URL (different hardware/implementation).

### Incremental update vs. batch retraining (mean of 10 runs; batch = LASSO+MI + full SGD fit on 210,598 + N URLs)

| Operation | Incremental (ms) | Batch retraining (ms) | Efficiency gain (×) | Paper gain (×) |
|---|---:|---:|---:|---:|
| Model update (100 URLs) | 1.45 | 4030.1 | 2770.6 | 15.9 |
| Model update (500 URLs) | 1.62 | 4021.7 | 2476.6 | 29.7 |
| Model update (1000 URLs) | 1.89 | 4040.5 | 2140.2 | 33.7 |
| Feature adaptation | 213.09 | 2744.1 | 12.9 | 30.9 |

## 5. Old vs. new feature distributions (paper Fig. 4, 5, 7)

KS test between 1500 old phishing URLs and 500 new PhishTank phishing URLs at δ = 0.05: **34 of 58 features drifted.** Top features by KS statistic:

| Feature | KS stat (window) | p-value | KS stat (full samples) |
|---|---:|---:|---:|
| host_entropy | 0.671 | 0 | 0.647 |
| longest_host_label_length | 0.641 | 0 | 0.622 |
| domain_length | 0.610 | 0 | 0.591 |
| has_https | 0.563 | 0 | 0.563 |
| longest_path_token_length | 0.560 | 0 | 0.562 |
| path_length | 0.549 | 0 | 0.551 |
| scheme_http | 0.472 | 0 | 0.472 |
| url_length | 0.463 | 0 | 0.431 |
| num_slashes | 0.393 | 0 | 0.402 |
| num_dots | 0.373 | 0 | 0.366 |
| num_digits_path | 0.354 | 1.5e-42 | 0.349 |
| num_subdirectories | 0.347 | 9.1e-41 | 0.354 |

Figures: `figures/fig4_feature_distributions.png`, `figures/fig5_cooccurrence.png`, `figures/fig7_ks_statistics.png`.

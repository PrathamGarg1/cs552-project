# Speaker script — 20 minutes (29 slides: 26 main + 3 appendix)

Split: **Pratham** = slides 1–9 (problem, idea, features) · **Sumit** = 10–18 (SVM, KS, loop, data, offline results, drift analysis) · **Vivaan** = 19–26 (streams, ablation, runtime, demo, wrap-up). Appendix slides are only for Q&A.

| # | Slide | Time | Say this |
|---|---|---|---|
| 1 | Title | 0:20 | Introduce the paper (Krishnan & Naik, 2025) and say we implemented and evaluated it from scratch. |
| 2 | Roadmap | 0:30 | Five parts: problem → idea → how it works → what we built → results. |
| 3 | Problem | 1:15 | Real vs fake URL. Blocklists are always one step behind, so we judge a link from its text alone. |
| 4 | Concept drift | 1:30 | Same model: 99 % on the data it was trained for, ~48 % (coin flip) after the URL source changed. 34 of 58 features shifted. |
| 5 | Paper idea | 1:15 | Fig. 1 in 3 steps: pick clues, watch for change, learn on the fly. Paper claims 99.85 % accuracy, 0.067 % FPR. |
| 6 | Our task | 1:00 | Each box of Fig. 1 maps to one file in `drift_phish`. `make all` reproduces everything. |
| 7 | URL → numbers | 1:15 | Walk through the anatomy and the real feature values (wikipedia vs secure-paypal-login.tk). |
| 8 | 58 features | 0:45 | Six families. Redirect and domain-mismatch features are lexical proxies since we use no HTTP. |
| 9 | LASSO + MI | 1:15 | LASSO = "drop clues that don't help", MI = "how much does this clue tell us". Score S = w + I, keep the top 25 % (15 features). |
| 10 | SGD-SVM | 1:00 | Draw a line; when attackers change style, nudge it with small batches (`partial_fit`). |
| 11 | KS test | 1:15 | Compare the CDF of old vs new data; D = biggest gap. p < 0.05 → feature drifted. Our rule: ≥ 25 % of F* drifted → drift event. |
| 12 | Online loop | 1:15 | Predict first, then KS-test, then (if drift) re-select + blend history + replay, then learn. |
| 13 | Data & experiments | 0:45 | Five experiments, datasets, seeds. |
| 14 | Offline results | 1:00 | SVM best, like the paper. 99.38 % on PhiUSIIL vs paper 98.78 %. |
| 15 | Why gap | 0:45 | Lower on Kaggle/ISCX/combined because the paper doesn't publish feature definitions / cleaning. Honest. |
| 16 | KDE | 0:45 | Old vs new URLs: same shape of shift as the paper (shorter domains, fewer bait words). |
| 17 | KS bars | 0:30 | Biggest shifts are length/structure features, like the paper. |
| 18 | Table 8 | 0:45 | F* changes after drift: host-entropy, query features come in. |
| 19 | Stream S1 | 1:00 | Online 97.8 % vs frozen 92.6 %; +5 points. Adaptive ≈ static here because the shift is gentle. |
| 20 | Stream S2 | 1:00 | Big shift: frozen 55 %, adaptive 73 %, +2.7 over static. |
| 21 | Ablation | 0:45 | Adaptive F* helps more under big shifts. Caveat: our static baseline is stronger than the paper's. |
| 22 | Runtime | 0:45 | 0.16 ms per URL; incremental updates ~2,000× cheaper than retraining. |
| 23 | Scorecard | 0:45 | What reproduced, what is lower-level. |
| 24 | Demo | 1:00 | Live PhishCheck demo (or screenshots). Be upfront about wikipedia.org's 51.8 % false alarm. |
| 25 | Takeaways | 0:30 | Learned / limitations / future work. |
| 26 | Q&A | — | Leave on screen. |

Total ≈ 22 min if nothing is cut. If short on time, speed up slides 8, 13, 16–18 and 21.

## Notes / likely questions
- **Why are our absolute numbers lower than the paper's on Kaggle/ISCX/Combined?** The paper doesn't publish exact feature definitions, data versions or pre-processing. Ordering (SVM best, adaptive > frozen) is what reproduces.
- **Paper inconsistencies (only if asked):** Table 5 has LR (0.9957) above SVM (0.9901) although the text says SVM is best; Fig. 7 draws a 0.2 threshold while Table 4 says δ=0.05; the domain_length KDE shows shorter domains for new URLs but the caption says "longer".
- **Why adaptive ≈ static on PhishTank?** The shift is mild; online SGD alone already tracks it. Adaptive selection matters on the cross-corpus stream.
- **Where does "≥ 25 % of F* drifted" come from?** Our design choice; the paper leaves the drift rule open.
- **Demo false alarm:** bare domains with little context sit near the decision boundary; a URL-only model has this limitation.
- **Caveat on the demo model:** check that `web/public/model.json` label convention matches `scripts/train_demo_model.py` before re-training.
- **Speed:** 0.16 ms/URL feature extraction + scoring on 1 CPU thread; paper reports 12.4–15.2 ms on different hardware.

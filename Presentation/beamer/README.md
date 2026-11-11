# LaTeX Beamer presentation

CS 552 group presentation for Krishnan & Naik, *An Efficient Framework for Phishing URL Detection Using Adaptive Drift-Aware Feature Monitoring and Online Learning* (Security and Privacy, 2025).

## Build

```bash
cd Presentation/beamer
# figures/ already contains PNG + PDF exports of Presentation/assets
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # second pass for page numbers / outlines
```

Output: `main.pdf` (also copied to `Presentation/Presentation_Beamer.pdf` and repo root `Presentation_Beamer.pdf`).

## Contents (~26 main slides + 3 appendix)

1. Title & paper reference  
2. Roadmap  
3–4. Problem & concept drift  
5–6. Paper idea & our implementation map  
7–12. Method: features → LASSO+MI → SGD-SVM → KS → online loop  
13. Data & experiments  
14–22. Results (offline, gap, drift evidence, streams, ablation, runtime)  
23–25. Scorecard, demo, takeaways  
26. Q&A  
A–C. Appendix figures for Q&A  

Speaker timing notes: see `../Speaker_Script.md`.

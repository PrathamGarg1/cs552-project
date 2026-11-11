PY ?= python
export PYTHONPATH := src:scripts

.PHONY: install data test exp1 exp2 exp3 exp4 exp5 results all clean-cache

install:
	$(PY) -m pip install -r requirements.txt

data:            ## download PhiUSIIL, Kaggle, Hannousse, PhishTank into data/raw
	$(PY) scripts/download_data.py

test:
	$(PY) -m pytest -q

exp1:            ## offline phase (paper Tables 5-7, Fig. 6)
	$(PY) scripts/exp1_offline.py
exp2:            ## online phase under drift (S1 PhishTank, S2 cross-corpus)
	$(PY) scripts/exp2_stream.py
exp3:            ## ablation: adaptive vs static feature selection (Table 9)
	$(PY) scripts/exp3_ablation.py
exp4:            ## latency + incremental vs batch (Tables 10-11)
	$(PY) scripts/exp4_runtime.py
exp5:            ## KDE / KS / co-occurrence figures (Figs. 4, 5, 7)
	$(PY) scripts/exp5_analysis.py
results:         ## RESULTS.md + remaining figures
	$(PY) scripts/build_results.py

all: data test exp1 exp2 exp3 exp4 exp5 results

clean-cache:
	rm -rf data/processed

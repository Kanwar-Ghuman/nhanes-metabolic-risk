PYTHON ?= python

.PHONY: all data qc table1 analysis figures ml r clean

all: data qc table1 analysis figures ml r

data:
	$(PYTHON) scripts/01_download.py

qc:
	$(PYTHON) scripts/02_clean_qc.py

table1:
	$(PYTHON) scripts/03_table1.py

analysis:
	$(PYTHON) scripts/04_analysis.py

figures:
	$(PYTHON) scripts/05_figures.py

ml:
	$(PYTHON) scripts/06_ml.py

r:
	Rscript R/svyglm_check.R

clean:
	rm -rf data/processed/*

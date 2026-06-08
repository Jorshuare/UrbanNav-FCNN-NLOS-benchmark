# Reproduction pipeline for the FCNN GNSS multipath/NLOS benchmark.
# Each target is a thin wrapper around a script in scripts/.

PYTHON := python
VENV := .venv

.PHONY: help setup data train eval figures test lint clean

help:
	@echo "setup    - create venv (Python 3.11/3.12) and install pinned deps"
	@echo "data     - build model-ready features from data/raw/*.xlsx"
	@echo "train    - train the 5 FCNN variants for both tasks"
	@echo "eval     - reproduce Table 8 + Table 9 (validation) into reports/tables/"
	@echo "figures  - regenerate Fig. 13 + training curves into reports/figures/"
	@echo "test     - run pytest"
	@echo "lint     - run ruff format + check"

setup:
	uv venv $(VENV) --python=3.12 || $(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -e ".[dev]"

data:
	$(PYTHON) scripts/01_prepare_data.py

train:
	$(PYTHON) scripts/02_train_classification.py
	$(PYTHON) scripts/03_train_regression.py

eval:
	$(PYTHON) scripts/04_evaluate.py

figures:
	$(PYTHON) scripts/05_make_figures.py

test:
	$(PYTHON) -m pytest

lint:
	ruff format src scripts tests
	ruff check src scripts tests

clean:
	rm -rf data/interim/* data/processed/* results/models/* results/metrics/* results/logs/*

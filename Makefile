BOOTSTRAP_PYTHON ?= python3
PYTHON ?= .venv/bin/python

.PHONY: setup demo test benchmark cold-start eval-extraction offline-check import-private

setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

demo:
	@echo "Demo server is reserved for the API agent."

test:
	$(PYTHON) -m pytest

benchmark:
	$(PYTHON) -m dough.forecast.evaluate benchmark

cold-start:
	$(PYTHON) -m dough.forecast.evaluate cold-start

eval-extraction:
	PYTHONPATH=src $(PYTHON) -m dough.voice.evaluate

offline-check:
	@echo "Offline check is reserved for the quality agent."

import-private:
	@echo "Private-data import is reserved for the data/API agents. FILE=$(FILE)"
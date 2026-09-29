.PHONY: help install install-hpo test lint fmt demo run clean

PY ?= python

help:
	@echo "FeatureForge targets:"
	@echo "  make install      install runtime deps"
	@echo "  make install-hpo  install runtime + Optuna (HPO backend)"
	@echo "  make test         run pytest"
	@echo "  make lint         ruff check + format check"
	@echo "  make fmt          ruff format (write)"
	@echo "  make demo         run benchmark and dump benchmark.json"
	@echo "  make run          single-dataset demo (interaction DGP)"

install:
	$(PY) -m pip install -r requirements.txt

install-hpo:
	$(PY) -m pip install -r requirements.txt optuna

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff format .

demo:
	$(PY) -m featureforge.examples.run_demo

run:
	$(PY) -m featureforge.cli run --dgp interaction --seed 1

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache

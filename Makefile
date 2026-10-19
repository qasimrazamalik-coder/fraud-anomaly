PY ?= python

.PHONY: install lint format test smoke data train eval app clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	$(PY) -m pytest -q

# End-to-end run on a SYNTHETIC fixture. Proves the pipeline works; numbers are not results.
smoke:
	$(PY) -m fraud_anomaly.train --config configs/smoke.yaml
	$(PY) -m fraud_anomaly.evaluate --config configs/smoke.yaml

data:
	$(PY) scripts/download_data.py

train:
	$(PY) -m fraud_anomaly.train --config configs/default.yaml

# Renders tables and figures from results/run.json and rewrites the README results block.
eval:
	$(PY) -m fraud_anomaly.evaluate --config configs/default.yaml --update-readme

app:
	$(PY) app/app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info

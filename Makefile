VENV ?= .venvs/test
PYTHON ?= python3
VENV_PYTHON = $(VENV)/bin/python
JOBS ?= 4
export CORPUS JOBS

.PHONY: venv setup-data test
venv:
	"$(PYTHON)" -m venv "$(VENV)"
	"$(VENV_PYTHON)" -m pip install -e '.[test]'

setup-data:
	@set -a; . ./.env && "$(VENV_PYTHON)" scripts/setup_data.py

test:
	"$(VENV_PYTHON)" -m pytest

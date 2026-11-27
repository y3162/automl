EXAMPLES_VENV ?= .venvs/examples
MLFLOW_VENV ?= .venvs/mlflow
SPEECH_VENV ?= .venvs/speech-v0
VENV ?= $(EXAMPLES_VENV)
PYTHON ?= python3
VENV_PYTHON = $(VENV)/bin/python
JOBS ?= 4
export CORPUS JOBS

.PHONY: venv venv-examples venv-mlflow setup-data test clean-mock
venv: venv-examples

venv-examples:
	"$(PYTHON)" -m venv "$(EXAMPLES_VENV)"
	"$(EXAMPLES_VENV)/bin/python" -m pip install -e '.[test]'

venv-mlflow:
	"$(PYTHON)" -m venv "$(MLFLOW_VENV)"
	"$(MLFLOW_VENV)/bin/python" -m pip install -e '.[test,mlflow]'

setup-data:
	@set -a; . ./.env && "$(VENV_PYTHON)" scripts/setup_data.py

test:
	"$(EXAMPLES_VENV)/bin/python" -c 'import pytest'
	"$(MLFLOW_VENV)/bin/python" -c 'import pytest, mlflow, sqlalchemy'
	"$(SPEECH_VENV)/bin/python" -c 'import pytest, torch, torchaudio, nnAudio, nemo.collections.asr.parts.submodules.subsampling, nemo.collections.asr.parts.preprocessing.features'
	"$(EXAMPLES_VENV)/bin/python" -m pytest tests/automl tests/corpora tests/examples --ignore=tests/automl/test_mlflow.py
	"$(MLFLOW_VENV)/bin/python" -m pytest tests/automl/test_mlflow.py
	"$(SPEECH_VENV)/bin/python" -m pytest tests/methods

clean-mock:
	rm -rfv data/mock

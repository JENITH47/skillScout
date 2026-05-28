.PHONY: setup run lint

PYTHON ?= python
VENV := .venv

ifeq ($(OS),Windows_NT)
PY := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip
else
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
endif

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

run:
	$(PY) -m streamlit run app/app.py

lint:
	$(PY) -m compileall app src

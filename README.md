# EstNLTK Dependency Chains Tagger

A small library and tooling to detect and tag dependency-chain relations in Estonian dependency trees. The project contains a compact tagger implementation, pattern/matcher utilities and an orchestration layer for applying rules to parsed sentences.

## Features

- Rule-based dependency-chain matching and tagging
- Reusable pattern and condition building blocks
- Lightweight orchestration to run matchers over parsed trees
- Unit tests for core logic

## Requirements

- Python 3.8+
- pip
- (recommended) A virtual environment for isolation

## Installation

1. Clone the repository and enter it:

```bash
git clone <repo-url>
cd EstNLTK_DependencyChains
```

2. Create and activate a virtual environment (recommended):

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the package in editable mode:

```bash
pip install -e .
```

4. If the project provides a `requirements.txt`, install additional dependencies:

```bash
pip install -r requirements.txt
```

## Running tests

Run the test suite with `pytest`:

```bash
python -m pytest -q
# or
python run_tests.py
```

## Project structure

- `pyproject.toml`, `setup.cfg`, `pytest.ini` — build/packaging and test configuration
- `run_tests.py` — convenience runner for the test suite
- `scripts/` — main package code (see `scripts/README.md` for details)
- `tests/` — unit tests for the modules in `scripts/`

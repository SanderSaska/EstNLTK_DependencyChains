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
- `estnltk_depchain_tagger.egg-info/` — packaging metadata (generated)
- `scripts/` — main package code (see `scripts/README.md` for details)
- `tests/` — unit tests for the modules in `scripts/`

<!-- ## Usage (example)

After installing the package, the main tagger components are importable from the `DepChainTagger` package. A minimal usage example:

```python
from DepChainTagger import tagger

# Construct and use the tagger on your parsed document object
# (replace `Doc` with your tree/document representation)
# t = tagger.DepChainTagger()
# results = t.tag(doc)
```

See `scripts/README.md` for module-level details and example call patterns.

## Contributing

1. Create an issue describing the feature or bug.
2. Create a branch, implement your change and add tests under `tests/`.
3. Run `pytest` and ensure all tests pass.
4. Open a pull request.

## License

See the package metadata in `estnltk_depchain_tagger.egg-info/PKG-INFO` for licensing information. -->

# `scripts/` — DepChainTagger package

This directory contains the `DepChainTagger` implementation: matcher and tagger modules, pattern definitions, helper utilities and configuration used to identify dependency-chain relations in parsed sentences.

Core modules

- `child_matcher.py` — Utilities to match and select child nodes in a dependency tree according to simple constraints.
- `child_tagger.py` — Logic to apply tags or annotations to matched child relations.
- `conditions.py` — Reusable predicate functions and condition helpers used in patterns.
- `config.py` — Configuration constants and defaults used across the package.
- `decorator.py` — Small decorators used to register patterns or wrap tagger functions.
- `graph.py` — Graph utilities to traverse and manipulate dependency graphs and subtrees.
- `matcher.py` — Core matching engine that evaluates patterns against trees.
- `orchestrator.py` — High-level orchestration: apply matchers over a corpus or document and collect results.
- `patterns.py` — Pattern definitions and pattern-building helpers.
- `tagger.py` — Main tagger interface exposing the API to run tagging over documents/trees.
- `types.py` — Shared types, dataclasses and type aliases.

### Extending patterns

Patterns and conditions are intentionally modular: add new conditions in `conditions.py`, define patterns in `patterns.py` and register or use them through `decorator.py` or `orchestrator.py` as appropriate.

### Tests

Unit tests that cover the modules in this package are available in the repository `tests/` folder. Run them with `pytest` from the repository root.

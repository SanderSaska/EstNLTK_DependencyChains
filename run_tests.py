"""Simple test runner wrapper that executes pytest.

Run with:

    python run_tests.py

Or use `pytest` directly after installing the dev requirements.
"""

import sys

import pytest


def main() -> int:
    # Collect pytest args from the command line, allow passing extra flags
    args = sys.argv[1:]
    if not args:
        args = ["-q"]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())

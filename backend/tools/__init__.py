"""Developer tools. Not part of the application.

Nothing under ``app/`` may import from here: these modules exist to measure
runs after the fact (parse the logs a job leaves behind, compare two runs,
decide whether a change made things worse), and they are excluded from the
wheel by ``[tool.setuptools] packages = find(include=["app*"])``.

The dependency arrow points one way — tools import app, never the reverse —
and ``tests/test_compare_runs.py`` keeps it that way.

Run from ``backend/``:

    .venv/bin/python -m tools.compare_runs --help
"""

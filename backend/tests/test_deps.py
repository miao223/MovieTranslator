"""The startup dependency check, and the thing that keeps it honest.

Updating this app means replacing files, so the code routinely runs against
an environment installed several releases ago. Pillow became a dependency in
0.10.0; a machine set up before that answered `python -m app.main` with
`No module named 'PIL'` and exited — the fix was one pip command, and
nothing on screen said so.
"""

import re
import sys
from pathlib import Path

import pytest

from app.core import deps

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def declared() -> set[str]:
    """The base dependencies, as pyproject declares them."""
    text = PYPROJECT.read_text(encoding="utf-8")
    # not just the first "]" — "uvicorn[standard]" has one of its own
    block = text.split("dependencies = [", 1)[1].split("\n]", 1)[0]
    names = set()
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'"([A-Za-z0-9_.-]+)', line)
        if m:
            names.add(m.group(1).lower())
    return names


def test_every_declared_dependency_is_checked():
    """A dependency added to pyproject but not here would go back to
    failing the way PIL did — this test is the whole point of the table."""
    assert declared() == set(deps.REQUIRED)


def test_the_check_passes_in_a_complete_environment():
    assert deps.missing() == []
    deps.require_dependencies()  # must not raise


def test_a_missing_package_is_named_along_with_the_fix(monkeypatch):
    monkeypatch.setitem(deps.REQUIRED, "pillow", "PIL_definitely_not_here")
    assert deps.missing() == ["pillow"]
    with pytest.raises(SystemExit) as exc:
        deps.require_dependencies()
    assert exc.value.code == 1


def test_the_message_says_which_package_and_which_command():
    text = deps.report(["pillow", "openai"])
    assert "pillow" in text and "openai" in text
    assert "pip install -e ." in text
    assert ("Scripts" in text) == (sys.platform == "win32")


def test_the_guard_imports_nothing_it_is_meant_to_check():
    """It runs before the packages it looks for, so it may only use the
    standard library."""
    source = Path(deps.__file__).read_text(encoding="utf-8")
    for module in deps.REQUIRED.values():
        assert f"import {module}" not in source

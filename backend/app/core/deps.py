"""Check the third-party packages are actually installed, before anything
tries to import one.

The app is distributed as a zip and updated by replacing files, so the code
moves ahead of the environment whenever a release adds a dependency. What
that looks like on the user's machine is a traceback ending in
``No module named 'PIL'`` and an immediate exit — no hint that the fix is
one pip command, and no hint which command.

Nothing here may import a third-party module: it runs before them.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec

# distribution name (as written in pyproject) -> the name you import.
# `test_deps.py` keeps this in step with pyproject's base dependencies;
# a new dependency with no entry here fails that test.
REQUIRED = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "av": "av",
    "faster-whisper": "faster_whisper",
    "mcp": "mcp",
    "openai": "openai",
    "pillow": "PIL",
    "platformdirs": "platformdirs",
    "pydantic": "pydantic",
}


def missing() -> list[str]:
    """Which distributions cannot be imported, in the order declared."""
    absent = []
    for dist, module in REQUIRED.items():
        try:
            found = find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            absent.append(dist)
    return absent


def report(absent: list[str]) -> str:
    pip = (r"  .venv\Scripts\pip install -e ."
           if sys.platform == "win32" else "  .venv/bin/pip install -e .")
    return "\n".join([
        "缺少运行所需的组件：" + "、".join(absent),
        "",
        "这通常是更新时只替换了程序文件、没有重装依赖。",
        "请在 backend 目录下执行：",
        pip,
        "",
        "装完后再启动：python -m app.main",
    ])


def require_dependencies() -> None:
    """Stop with something the user can act on, rather than a traceback."""
    absent = missing()
    if not absent:
        return
    print(report(absent), file=sys.stderr)
    raise SystemExit(1)

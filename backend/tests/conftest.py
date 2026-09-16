"""Shared test fixtures."""

from __future__ import annotations

import pytest

from app.core import config
from app.models.schemas import AppSettings


@pytest.fixture(scope="session", autouse=True)
def _contained_cache(tmp_path_factory):
    """Keep the whole test session out of the developer's real cache dir.

    cache._base_dir() falls back to platformdirs' user cache whenever
    work_dir is unset, so any test that builds a Job writes into the real
    ~/.cache/MovieTranslator. That stayed invisible while the only thing
    landing there was jobs/, which the app wipes on every startup — but
    checkpoints/ is deliberately never wiped, so test leftovers accumulated
    and stayed.

    Done by setting XDG_CACHE_HOME rather than by patching _base_dir,
    because monkeypatch is undone when a test ends and the threads a test
    started are not: a job thread outliving its test wrote one directory
    into the real cache every run. The environment variable holds for the
    whole session, threads included.

    Session-scoped and autouse: the rule is about every test, not the ones
    that remember to ask.
    """
    import os

    base = tmp_path_factory.mktemp("appcache")
    old = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(base)
    yield base
    if old is None:
        os.environ.pop("XDG_CACHE_HOME", None)
    else:
        os.environ["XDG_CACHE_HOME"] = old


def local_client(app, **kwargs):
    """A TestClient that looks like the browser on this machine.

    Starlette's default peer address is the literal string "testclient",
    which is not an IP and so reads as remote to the access guard — every
    request would come back 403. Real traffic always carries an address, so
    the guard is right and the test client is the thing to correct.
    """
    from fastapi.testclient import TestClient

    kwargs.setdefault("client", ("127.0.0.1", 40000))
    return TestClient(app, **kwargs)


def remote_client(app, host: str = "192.168.1.50", **kwargs):
    """A TestClient that looks like another device on the LAN."""
    from fastapi.testclient import TestClient

    kwargs.setdefault("client", (host, 40000))
    return TestClient(app, **kwargs)


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Point the settings store at a temp file and hand back a writer.

    Anything that reads settings goes through config.load_settings(), which
    caches on the file's stat — so tests must write through save_settings
    rather than poking the object, or the cache will hand back the old one.
    """
    path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "settings_path", lambda: path)
    monkeypatch.setattr(config, "_cache", None, raising=False)

    def write(**kwargs) -> AppSettings:
        """write(server__lan_access=True) -> settings with that field set."""
        settings = AppSettings()
        for key, value in kwargs.items():
            section, _, field = key.partition("__")
            setattr(getattr(settings, section) if field else settings,
                    field or section, value)
        config.save_settings(settings)
        return settings

    yield write
    config._cache = None

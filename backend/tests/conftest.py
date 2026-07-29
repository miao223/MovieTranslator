"""Shared test fixtures."""

from __future__ import annotations

import pytest

from app.core import config
from app.models.schemas import AppSettings


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

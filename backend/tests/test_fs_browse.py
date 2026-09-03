"""The server-side file picker.

The page has no extension table of its own — what it lists, and which icon it
draws, come from these two endpoints. Nothing else checks them.
"""

import pytest

from app.main import app
from tests.conftest import local_client


@pytest.fixture
def client():
    with local_client(app) as c:
        yield c


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "film.mkv").write_bytes(b"x")
    (tmp_path / "ep1.mp3").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden.mp4").write_bytes(b"x")
    (tmp_path / "season").mkdir()
    return tmp_path


def test_browse_lists_both_kinds_and_says_which_is_which(client, folder):
    r = client.get("/api/fs/browse", params={"path": str(folder)}).json()
    assert r["dirs"] == ["season"]
    assert {f["name"]: f["kind"] for f in r["files"]} == {
        "film.mkv": "video", "ep1.mp3": "audio",
    }  # .txt is not media, the dot-file is hidden


def test_resolve_tells_the_page_what_it_was_handed(client, folder):
    def resolve(name):
        return client.get(
            "/api/fs/resolve", params={"path": str(folder / name)}
        ).json()

    audio = resolve("ep1.mp3")
    assert audio["type"] == "file" and audio["is_media"] and audio["is_audio"]

    video = resolve("film.mkv")
    assert video["is_media"] and not video["is_audio"]

    other = resolve("notes.txt")
    assert other["type"] == "file" and not other["is_media"]

    assert resolve("nope.mkv")["type"] == "missing"


def test_a_quoted_path_still_resolves(client, folder):
    """Windows' "copy as path" wraps the whole thing in quotes."""
    r = client.get(
        "/api/fs/resolve", params={"path": f'"{folder / "ep1.mp3"}"'}
    ).json()
    assert r["type"] == "file" and r["is_audio"]

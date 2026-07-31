from pathlib import Path

import pytest

from app.core.media import scan_videos


def make_tree(root: Path):
    (root / "a.mkv").write_bytes(b"x")
    (root / "b.mp4").write_bytes(b"x")
    (root / "b.srt").write_text("1", encoding="utf-8")  # b has subs already
    (root / "notes.txt").write_text("x", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.avi").write_bytes(b"x")
    hidden = root / ".hidden"
    hidden.mkdir()
    (hidden / "d.mkv").write_bytes(b"x")


def test_scan_recursive_with_skip(tmp_path):
    make_tree(tmp_path)
    videos, skipped = scan_videos(tmp_path, recursive=True, skip_existing_srt=True)
    names = [v.name for v in videos]
    assert names == ["a.mkv", "c.avi"]  # b skipped (srt), d hidden, txt ignored
    assert [s.name for s in skipped] == ["b.mp4"]


def test_scan_non_recursive_no_skip(tmp_path):
    make_tree(tmp_path)
    videos, skipped = scan_videos(tmp_path, recursive=False, skip_existing_srt=False)
    assert [v.name for v in videos] == ["a.mkv", "b.mp4"]
    assert skipped == []


def test_scan_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_videos(tmp_path / "nope", True, True)


def test_batch_endpoints(tmp_path):
    from tests.conftest import local_client

    from app.main import app

    make_tree(tmp_path)
    c = local_client(app)

    scan = c.get(
        "/api/batch/scan",
        params={"path": str(tmp_path), "recursive": True, "skip_existing": True},
    ).json()
    assert scan["total"] == 2 and len(scan["skipped"]) == 1

    r = c.post("/api/batch", json={"directory": str(tmp_path)})
    assert r.status_code == 200
    b = r.json()
    assert b["total"] == 2
    batch_id = b["id"]

    # cancel immediately; fake videos would fail extraction anyway — either
    # way every job must reach a terminal state without blocking the batch
    c.post(f"/api/batch/{batch_id}/cancel")
    import time

    for _ in range(100):
        st = c.get(f"/api/batch/{batch_id}").json()
        if st["pending"] + st["running"] == 0:
            break
        time.sleep(0.1)
    assert st["pending"] + st["running"] == 0
    assert st["done"] + st["failed"] + st["cancelled"] == 2

    # empty dir → 400
    empty = tmp_path / "empty"
    empty.mkdir()
    assert c.post("/api/batch", json={"directory": str(empty)}).status_code == 400


def test_batch_passes_audio_language_to_each_job(tmp_path, monkeypatch):
    """Batches select tracks by language tag, so every job must carry it."""
    from app.models.schemas import BatchRequest, JobStatus
    from app.services import batch as batch_mod

    make_tree(tmp_path)
    seen = []
    fakes = {}

    class FakeJob:
        def __init__(self, req):
            seen.append(req)
            self.id = f"job{len(seen)}"
            self.status = JobStatus(id=self.id)
            fakes[self.id] = self

    monkeypatch.setattr(batch_mod.job_manager, "create", FakeJob)
    monkeypatch.setattr(batch_mod.job_manager, "get", lambda jid: fakes[jid])
    batch_mod.batch_manager.create(
        BatchRequest(directory=str(tmp_path), audio_language="jpn")
    )
    assert len(seen) == 2
    assert {r.audio_language for r in seen} == {"jpn"}
    assert all(r.audio_track is None for r in seen)


def _fake_batch(tmp_path, monkeypatch, **kwargs):
    """Create a batch without running anything, returning (status, requests)."""
    from app.models.schemas import BatchRequest, JobStatus
    from app.services import batch as batch_mod

    seen = []
    fakes = {}

    class FakeJob:
        def __init__(self, req):
            seen.append(req)
            self.id = f"job{len(seen)}"
            self.status = JobStatus(id=self.id)
            fakes[self.id] = self

    monkeypatch.setattr(batch_mod.job_manager, "create", FakeJob)
    monkeypatch.setattr(batch_mod.job_manager, "get", lambda jid: fakes[jid])
    status = batch_mod.batch_manager.create(
        BatchRequest(directory=str(tmp_path), **kwargs)
    )
    return status, seen


def test_series_mode_gives_every_episode_the_same_glossary(tmp_path, monkeypatch):
    """The point of the mode: one table, shared. A per-job table would be
    what the batch already did."""
    from app.services import series

    make_tree(tmp_path)
    status, seen = _fake_batch(tmp_path, monkeypatch, series_mode=True)

    ids = {r.series_id for r in seen}
    assert len(ids) == 1 and ids != {""}
    shared = series.get(seen[0].series_id)
    assert shared is not None and len(shared) == 0  # nothing learned yet

    # what the first episode settles on is what the batch reports
    shared.learn("タカシ → 隆")
    from app.services.batch import batch_manager

    assert batch_manager.status(status.id).glossary == "タカシ → 隆"


def test_without_series_mode_nothing_is_shared(tmp_path, monkeypatch):
    """A directory of unrelated films must not cross-contaminate names, so
    the default carries no series id and no table exists to write to."""
    from app.services import series

    make_tree(tmp_path)
    status, seen = _fake_batch(tmp_path, monkeypatch)

    assert {r.series_id for r in seen} == {""}
    assert series.get("") is None
    assert status.glossary == ""


def test_saving_a_series_glossary_merges_into_the_settings_table(
    tmp_path, monkeypatch, settings_file
):
    """The accumulated table dies with the batch unless the user keeps it,
    and keeping it must not disturb what is already there."""
    from tests.conftest import local_client

    from app.core import config
    from app.main import app
    from app.services import series

    settings_file(llm__api_key="sk-real", prompts__glossary="タカシ → 塔卡西")
    make_tree(tmp_path)
    status, seen = _fake_batch(tmp_path, monkeypatch, series_mode=True)
    shared = series.get(seen[0].series_id)
    shared.learn("タカシ → 隆\nユキ → 雪")  # タカシ is already the user's

    with local_client(app) as client:
        r = client.post(f"/api/batch/{status.id}/glossary/save")
        assert r.status_code == 200 and r.json() == {"added": 1, "total": 2}

        saved = config.load_settings()
        assert saved.prompts.glossary == "タカシ → 塔卡西\nユキ → 雪"
        assert saved.llm.api_key == "sk-real"  # untouched by a glossary save

        # saving twice must not duplicate the entry
        again = client.post(f"/api/batch/{status.id}/glossary/save")
        assert again.json()["added"] == 0
        assert config.load_settings().prompts.glossary.count("ユキ") == 1

        assert client.post("/api/batch/nosuch/glossary/save").status_code == 404


def test_saving_an_empty_glossary_is_refused(tmp_path, monkeypatch, settings_file):
    """A batch that ran without series mode has nothing to keep, and
    silently answering ok would look like it worked."""
    from tests.conftest import local_client

    from app.main import app

    settings_file()
    make_tree(tmp_path)
    status, _ = _fake_batch(tmp_path, monkeypatch)
    with local_client(app) as client:
        assert client.post(f"/api/batch/{status.id}/glossary/save").status_code == 400

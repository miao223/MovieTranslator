from pathlib import Path

import pytest

from app.core.media import scan_media


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


def add_audio(root: Path):
    """Audio files on top of make_tree; kept apart so the counts the other
    tests assert on stay put."""
    (root / "e.mp3").write_bytes(b"x")
    (root / "g.m4a").write_bytes(b"x")
    (root / "g.srt").write_text("1", encoding="utf-8")  # g has subs already
    (root / "cover.jpg").write_bytes(b"x")
    (root / "sub" / "f.flac").write_bytes(b"x")


def test_scan_recursive_with_skip(tmp_path):
    make_tree(tmp_path)
    videos, skipped, shadowed, _ = scan_media(
        tmp_path, recursive=True, skip_existing_srt=True
    )
    names = [v.name for v in videos]
    assert names == ["a.mkv", "c.avi"]  # b skipped (srt), d hidden, txt ignored
    assert [s.name for s in skipped] == ["b.mp4"]
    assert shadowed == []


def test_scan_non_recursive_no_skip(tmp_path):
    make_tree(tmp_path)
    videos, skipped, _, _ = scan_media(
        tmp_path, recursive=False, skip_existing_srt=False
    )
    assert [v.name for v in videos] == ["a.mkv", "b.mp4"]
    assert skipped == []


def test_scan_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_media(tmp_path / "nope", True, True)


def test_audio_files_are_scanned_like_videos(tmp_path):
    """A release shared as its audio track alone is the whole point of the
    feature; the .srt skip has to apply to it the same way."""
    make_tree(tmp_path)
    add_audio(tmp_path)
    found, skipped, shadowed, _ = scan_media(
        tmp_path, recursive=True, skip_existing_srt=True
    )
    assert [f.name for f in found] == ["a.mkv", "e.mp3", "c.avi", "f.flac"]
    assert [s.name for s in skipped] == ["b.mp4", "g.m4a"]
    assert shadowed == []


def test_an_audio_twin_of_a_video_steps_aside(tmp_path):
    """film.mkv and film.mp3 both write film.srt: translating each would
    burn an hour of GPU to overwrite the other's subtitle."""
    make_tree(tmp_path)
    (tmp_path / "a.mp3").write_bytes(b"x")
    (tmp_path / "solo.mp3").write_bytes(b"x")
    found, _, shadowed, _ = scan_media(tmp_path, recursive=True, skip_existing_srt=True)
    names = [f.name for f in found]
    assert "a.mkv" in names and "a.mp3" not in names
    assert "solo.mp3" in names  # no video of that name, so it stands
    assert [s.name for s in shadowed] == ["a.mp3"]


def test_a_video_already_subtitled_still_shadows_its_audio(tmp_path):
    """The grouping runs before the .srt check on purpose: the other order
    lets b.mp3 live on and overwrite the subtitle b.mp4 already has."""
    make_tree(tmp_path)
    (tmp_path / "b.mp3").write_bytes(b"x")
    found, skipped, shadowed, _ = scan_media(
        tmp_path, recursive=True, skip_existing_srt=True
    )
    assert "b.mp3" not in [f.name for f in found]
    assert [s.name for s in shadowed] == ["b.mp3"]
    assert [s.name for s in skipped] == ["b.mp4"]


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
    assert scan["audio"] == [] and scan["audio_count"] == 0
    assert scan["shadowed"] == []

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
        def __init__(self, req, settings=None):
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
        def __init__(self, req, settings=None):
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


def test_batch_passes_the_container_and_codec_to_each_job(tmp_path, monkeypatch):
    """Every field a batch shares has to be forwarded by hand — both in
    BatchManager and in the page's startBatch(). A field that is added to
    the single-file form and forgotten here goes missing without a word."""
    make_tree(tmp_path)
    _, seen = _fake_batch(
        tmp_path, monkeypatch,
        embed_subtitle=True,
        embed={"container": "mp4", "video_codec": "libx265", "quality": 30},
    )
    assert len(seen) == 2
    assert {r.embed.container for r in seen} == {"mp4"}
    assert {r.embed.video_codec for r in seen} == {"libx265"}
    assert {r.embed.quality for r in seen} == {30}


def test_the_scan_endpoint_names_the_audio_files(tmp_path):
    """The confirm dialog warns that these only ever produce a subtitle
    file, so it needs to know which of them they are."""
    from tests.conftest import local_client

    from app.main import app

    make_tree(tmp_path)
    add_audio(tmp_path)
    (tmp_path / "a.mp3").write_bytes(b"x")
    with local_client(app) as c:
        scan = c.get("/api/batch/scan", params={"path": str(tmp_path)}).json()
    assert [Path(a).name for a in scan["audio"]] == ["e.mp3", "f.flac"]
    assert scan["audio_count"] == 2
    assert [Path(s).name for s in scan["shadowed"]] == ["a.mp3"]


def test_a_directory_of_only_audio_can_be_batched(tmp_path):
    """The error a moment ago said "no video files" and stopped there."""
    from tests.conftest import local_client

    from app.main import app

    (tmp_path / "ep1.mp3").write_bytes(b"x")
    (tmp_path / "ep2.m4a").write_bytes(b"x")
    with local_client(app) as c:
        r = c.post("/api/batch", json={"directory": str(tmp_path)})
        assert r.status_code == 200
        batch_id = r.json()["id"]
        assert r.json()["total"] == 2
        c.post(f"/api/batch/{batch_id}/cancel")

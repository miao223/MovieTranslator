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
    scan = scan_media(tmp_path, recursive=True, skip_existing_srt=True)
    videos, skipped, shadowed = scan.to_translate, scan.skipped, scan.shadowed
    names = [v.name for v in videos]
    assert names == ["a.mkv", "c.avi"]  # b skipped (srt), d hidden, txt ignored
    assert [s.name for s in skipped] == ["b.mp4"]
    assert shadowed == []


def test_scan_non_recursive_no_skip(tmp_path):
    make_tree(tmp_path)
    scan = scan_media(tmp_path, recursive=False, skip_existing_srt=False)
    # 关掉跳过时 b 这一组还在，而组内字幕优先于视频——b.srt 就是那份现成的原文
    assert [v.name for v in scan.to_translate] == ["a.mkv", "b.srt"]
    assert scan.skipped == []
    assert [r.name for r in scan.replaced] == ["b.mp4"]


def test_scan_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_media(tmp_path / "nope", True, True)


def test_audio_files_are_scanned_like_videos(tmp_path):
    """A release shared as its audio track alone is the whole point of the
    feature; the .srt skip has to apply to it the same way."""
    make_tree(tmp_path)
    add_audio(tmp_path)
    scan = scan_media(tmp_path, recursive=True, skip_existing_srt=True)
    found, skipped, shadowed = scan.to_translate, scan.skipped, scan.shadowed
    assert [f.name for f in found] == ["a.mkv", "e.mp3", "c.avi", "f.flac"]
    assert [s.name for s in skipped] == ["b.mp4", "g.m4a"]
    assert shadowed == []


def test_an_audio_twin_of_a_video_steps_aside(tmp_path):
    """film.mkv and film.mp3 both write film.srt: translating each would
    burn an hour of GPU to overwrite the other's subtitle."""
    make_tree(tmp_path)
    (tmp_path / "a.mp3").write_bytes(b"x")
    (tmp_path / "solo.mp3").write_bytes(b"x")
    scan = scan_media(tmp_path, recursive=True, skip_existing_srt=True)
    found, shadowed = scan.to_translate, scan.shadowed
    names = [f.name for f in found]
    assert "a.mkv" in names and "a.mp3" not in names
    assert "solo.mp3" in names  # no video of that name, so it stands
    assert [s.name for s in shadowed] == ["a.mp3"]


def test_a_video_already_subtitled_still_shadows_its_audio(tmp_path):
    """The grouping runs before the .srt check on purpose: the other order
    lets b.mp3 live on and overwrite the subtitle b.mp4 already has."""
    make_tree(tmp_path)
    (tmp_path / "b.mp3").write_bytes(b"x")
    scan = scan_media(tmp_path, recursive=True, skip_existing_srt=True)
    found, skipped, shadowed = scan.to_translate, scan.skipped, scan.shadowed
    assert "b.mp3" not in [f.name for f in found]
    assert [s.name for s in shadowed] == ["b.mp3"]
    assert [s.name for s in skipped] == ["b.mp4"]


def _srt(path: Path, text: str = "Hello there") -> Path:
    path.write_text(f"1\n00:00:01,000 --> 00:00:03,000\n{text}\n", encoding="utf-8")
    return path


def test_a_directory_of_subtitles_is_a_batch(tmp_path):
    """整季字幕一次翻完——目录里一个视频都没有也照样成立。"""
    for n in ("ep01.en.srt", "ep02.en.srt", "ep03.en.srt"):
        _srt(tmp_path / n)

    scan = scan_media(tmp_path, target_language="简体中文")
    assert [f.name for f in scan.to_translate] == [
        "ep01.en.srt", "ep02.en.srt", "ep03.en.srt"]
    assert not scan.skipped and not scan.replaced


def test_a_subtitle_takes_the_place_of_its_film(tmp_path):
    """有现成字幕就翻它：十几秒读完、不占 GPU。代价（拿不到内嵌视频、不走
    语音识别）由 replaced 说出来，调用方必须转达。"""
    (tmp_path / "film.mkv").write_bytes(b"x")
    _srt(tmp_path / "film.en.srt")

    scan = scan_media(tmp_path, target_language="简体中文")
    assert [f.name for f in scan.to_translate] == ["film.en.srt"]
    assert [f.name for f in scan.replaced] == ["film.mkv"]
    assert not scan.shadowed          # 它不是「让路」，是被顶替


def test_only_one_subtitle_per_film_is_translated(tmp_path):
    """同一部片旁边好几种语言时只翻一份，且可以用「字幕轨语言」指定哪一份。"""
    _srt(tmp_path / "film.en.srt")
    _srt(tmp_path / "film.fr.srt", "Bonjour")

    both = scan_media(tmp_path, target_language="简体中文")
    assert [f.name for f in both.to_translate] == ["film.en.srt"]   # 按名字

    picked = scan_media(tmp_path, target_language="简体中文", prefer_language="fre")
    assert [f.name for f in picked.to_translate] == ["film.fr.srt"]


def test_a_text_subtitle_beats_a_graphic_one(tmp_path):
    """OCR 要按分钟计，而文字是免费且精确的——与 pick_track 同一条排序。"""
    _srt(tmp_path / "film.en.srt")
    (tmp_path / "film.sup").write_bytes(b"PG")

    scan = scan_media(tmp_path, target_language="简体中文")
    assert [f.name for f in scan.to_translate] == ["film.en.srt"]


def test_a_vobsub_pair_is_one_job_not_two(tmp_path):
    """.idx + .sub 是一对，两个都算就会有两个任务去写同一份产物。"""
    (tmp_path / "film.idx").write_text("# VobSub index file\n", encoding="utf-8")
    (tmp_path / "film.sub").write_bytes(b"\x00" * 16)

    scan = scan_media(tmp_path, target_language="简体中文")
    assert [f.name for f in scan.to_translate] == ["film.idx"]


def test_a_finished_pair_of_subtitles_is_not_translated_again(tmp_path):
    """翻完之后再扫一次，源和产物都在那儿——这一组已经做完了。"""
    _srt(tmp_path / "film.en.srt")
    _srt(tmp_path / "film.zh.srt", "你好啊")

    scan = scan_media(tmp_path, target_language="简体中文")
    assert not scan.to_translate
    assert [f.name for f in scan.skipped] == ["film.en.srt"]


def test_a_stepped_aside_copy_is_never_picked_as_a_source(tmp_path):
    """film.zh.2.srt 是防覆盖留下的副本，不是原料也不是成品。"""
    _srt(tmp_path / "film.zh.2.srt", "你好啊")

    scan = scan_media(tmp_path, target_language="简体中文")
    assert not scan.to_translate and not scan.skipped


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

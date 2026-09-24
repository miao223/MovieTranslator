"""The 原盘 page's endpoints and the queue that runs its one button.

The page only ever does two things: ask what a disc holds (GET
/api/disc/scan) and add a remux to the queue (POST /api/queue/disc). A
queued disc then runs as a real pipeline job — so the last tests here are
end to end: a synthetic Blu-ray in, MKV files next to it out.
"""

from __future__ import annotations

import time
from pathlib import Path

import av
import pytest

from app.core.media import scan_media
from app.services.jobqueue import QueueManager
from tests import discgen as g
from tests.discgen import Item
from tests.test_mux import _frames


def film_disc(tmp_path, name="Film (2001)", extras=True, encrypted=False):
    """A Blu-ray folder with real streams: a two-clip film, its two
    segments as playlists of their own, and one extra."""
    a = g.make_m2ts(1.5, tmp=tmp_path)
    b = g.make_m2ts(1.5, start=100.0, tmp=tmp_path)
    x = g.make_m2ts(1.0, start=50.0, tmp=tmp_path)
    clips = {"00001": a, "00002": b}
    pls = {
        "00800": g.mpls([Item("00001", a.start, a.end), Item("00002", b.start, b.end, connection=5)],
                        marks=[(1, 0, a.start), (1, 1, b.start)]),
        "00001": g.mpls([Item("00001", a.start, a.end)]),
        "00002": g.mpls([Item("00002", b.start, b.end)]),
    }
    if extras:
        clips["00003"] = x
        pls["00003"] = g.mpls([Item("00003", x.start, x.end)])
    streams = {k: c.data for k, c in clips.items()}
    if encrypted:
        streams["00001"] = streams["00001"][:192 + 16] + bytes(range(256)) * 24 + streams["00001"][6352:]
    info = {k: g.clpi(c.start, c.end) for k, c in clips.items()}
    root = g.write_bd(tmp_path / name, pls, info, streams=streams)
    return root, clips


def series_volume(root, tmp_path):
    """One volume of a box set with real streams: three episodes, their
    play-all, one extra. The episodes last seconds — see short_episodes."""
    episode = g.make_m2ts(1.5, tmp=tmp_path)
    extra = g.make_m2ts(1.1, tmp=tmp_path)     # under 1 s would be "empty"
    clips = {"00011": episode, "00012": episode, "00013": episode, "00090": extra}
    pls = {f"0000{i}": g.mpls([Item(c, episode.start, episode.end)])
           for i, c in enumerate(("00011", "00012", "00013"), start=1)}
    pls["00000"] = g.mpls([Item(c, episode.start, episode.end)
                           for c in ("00011", "00012", "00013")])
    pls["00050"] = g.mpls([Item("00090", extra.start, extra.end)])
    return g.write_bd(root, pls, {k: g.clpi(c.start, c.end) for k, c in clips.items()},
                      streams={k: c.data for k, c in clips.items()})


@pytest.fixture
def short_episodes(monkeypatch):
    """A synthetic episode lasts seconds; the rule wants ten minutes. Set
    between series_volume's extra (1.1 s) and its episodes (1.5 s)."""
    from app.services.disc import analyze as analyze_mod

    monkeypatch.setattr(analyze_mod, "EPISODE_MIN", 1.3)


@pytest.fixture
def api(settings_file, monkeypatch):
    from app.main import app
    from tests.conftest import local_client

    # the synthetic titles last seconds, not minutes: nothing is "too short"
    settings_file(disc__min_title_seconds=0)
    fresh = QueueManager()
    fresh.store.load()
    import app.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "queue_manager", fresh)
    return local_client(app), fresh


# ------------------------------------------------------------------- scan

def test_scanning_a_disc_lists_every_title_with_a_verdict(api, tmp_path):
    client, _ = api
    root, _ = film_disc(tmp_path)
    r = client.get("/api/disc/scan", params={"path": str(root)})
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["kind"] == "bd" and report["name"] == "Film (2001)"
    # beside the disc folder by default: Film (2001)/ → Film (2001).mkv
    assert report["output_mode"] == "beside" and report["output_dir"] == str(root.parent)
    by_id = {t["id"]: t for t in report["titles"]}
    assert by_id["00800.mpls"]["category"] == "main" and by_id["00800.mpls"]["selected"]
    assert by_id["00800.mpls"]["output"] == "Film (2001).mkv"
    assert by_id["00001.mpls"]["category"] == "duplicate"
    assert by_id["00003.mpls"]["label"] == "花絮 01" and by_id["00003.mpls"]["selected"]
    assert by_id["00003.mpls"]["output"] == "Film (2001).花絮01.mkv"
    # listed in the order they would be written
    assert [t["category"] for t in report["titles"]][:2] == ["main", "extra"]
    audio = [s for s in by_id["00800.mpls"]["streams"] if s["kind"] == "audio"]
    assert audio[0]["language"] == "jpn" and audio[0]["language_name"] == "日语"


def test_the_extras_setting_decides_their_default_tick(api, tmp_path, settings_file):
    client, _ = api
    root, _ = film_disc(tmp_path)
    settings_file(disc__export_extras=False, disc__min_title_seconds=0)
    report = client.get("/api/disc/scan", params={"path": str(root)}).json()
    extra = next(t for t in report["titles"] if t["category"] == "extra")
    assert not extra["selected"]


def test_scanning_something_that_is_not_a_disc_says_why(api, tmp_path):
    client, _ = api
    r = client.get("/api/disc/scan", params={"path": str(tmp_path)})
    assert r.status_code == 400 and "BDMV" in r.json()["detail"]


# ------------------------------------------------------------------ queue

def test_an_empty_selection_is_frozen_into_the_defaults(api, tmp_path):
    client, manager = api
    root, _ = film_disc(tmp_path)
    r = client.post("/api/queue/disc", json={"path": str(root)})
    assert r.status_code == 200, r.text
    entry = manager.store.entries[0]
    assert entry.kind == "disc" and entry.request is None
    assert entry.disc.titles == ["00800.mpls", "00003.mpls"]
    # everything the list showed is settled into the entry, not left to the run
    assert (entry.disc.name, entry.disc.output_mode, entry.disc.episode_start,
            entry.disc.extra_start) == ("Film (2001)", "beside", 1, 1)
    assert r.json()["entry"]["summary"].startswith("原盘封装 · 2 个标题")


def test_titles_that_are_not_on_the_disc_or_cannot_be_written_are_refused(api, tmp_path):
    client, manager = api
    root, _ = film_disc(tmp_path)
    r = client.post("/api/queue/disc", json={"path": str(root), "titles": ["00999.mpls"]})
    assert r.status_code == 400 and "00999.mpls" in r.json()["detail"]
    meta = g.simple_bd(tmp_path / "Meta", {"00800": [("00001", 0, 5400)]})
    r = client.post("/api/queue/disc", json={"path": str(meta)})
    assert r.status_code == 400 and "元数据" in r.json()["detail"]
    assert manager.store.entries == []


def test_an_encrypted_disc_is_refused_before_it_is_queued(api, tmp_path):
    client, manager = api
    root, _ = film_disc(tmp_path, encrypted=True)
    report = client.get("/api/disc/scan", params={"path": str(root)}).json()
    assert report["encrypted"] and not any(t["selectable"] for t in report["titles"])
    r = client.post("/api/queue/disc", json={"path": str(root)})
    assert r.status_code == 400 and "加密" in r.json()["detail"]
    assert manager.store.entries == []


def _run(manager, timeout=120.0):
    """Drive the queue until its current entry is finished."""
    from app.services.pipeline import manager as jobs

    manager.step()
    entry = next(e for e in manager.store.entries if e.status == "running")
    job = jobs.get(entry.job_id)
    deadline = time.monotonic() + timeout
    while job.status.stage not in ("done", "failed", "cancelled"):
        assert time.monotonic() < deadline, job.status
        time.sleep(0.05)
    manager.step()
    return entry, job


def test_a_queued_disc_is_written_next_to_itself(api, tmp_path):
    client, manager = api
    root, clips = film_disc(tmp_path)
    client.post("/api/queue/disc", json={"path": str(root)})
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    film, extra = root.parent / "Film (2001).mkv", root.parent / "Film (2001).花絮01.mkv"
    assert entry.result_files == [str(film), str(extra)]
    assert _frames(film) == clips["00001"].frames + clips["00002"].frames
    assert _frames(extra) == clips["00003"].frames
    with av.open(str(film)) as c:
        assert len(c.chapters()) == 2
    view = client.get("/api/queue").json()["entries"][0]
    assert view["kind"] == "disc" and view["result_files"] == entry.result_files


def test_the_job_log_ticks_what_this_job_writes(api, tmp_path):
    """The log's disc listing is read to find out what a job did; its ticks
    must be the job's selection, not the analysis's defaults (found on a
    real box set: one episode was queued, the log ticked all six)."""
    client, manager = api
    root, _ = film_disc(tmp_path)
    client.post("/api/queue/disc", json={"path": str(root), "titles": ["00003.mpls"]})
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    log = job.logfile.path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("☑ 00003.mpls") for line in log)
    assert any(line.startswith("☐ 00800.mpls") for line in log)
    assert not any(line.startswith("☑ 00800.mpls") for line in log)


def test_a_retry_skips_what_was_already_written(api, tmp_path):
    """A disc that failed half way must not be written again from the
    start: the film done in the first attempt stays, and no 片名.2.mkv
    appears beside it."""
    client, manager = api
    root, _ = film_disc(tmp_path)
    client.post("/api/queue/disc", json={"path": str(root)})
    first, _ = _run(manager)
    assert first.status == "done"
    retried = manager.retry(first.id)
    assert retried.kind == "disc" and retried.disc == first.disc
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    assert entry.result_files == first.result_files
    assert sorted(p.name for p in root.parent.glob("*.mkv")) == [
        "Film (2001).mkv", "Film (2001).花絮01.mkv"]
    assert any("上次已经封装好了" in e.log for e in job.events)


def test_the_output_never_overwrites_an_existing_file(api, tmp_path):
    client, manager = api
    root, _ = film_disc(tmp_path, extras=False)
    (root.parent / "Film (2001).mkv").write_bytes(b"somebody else's file")
    client.post("/api/queue/disc", json={"path": str(root)})
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    assert (root.parent / "Film (2001).mkv").read_bytes() == b"somebody else's file"
    assert entry.result_files == [str(root.parent / "Film (2001).2.mkv")]


def test_the_mkvs_go_where_the_page_says(api, tmp_path, settings_file):
    client, manager = api
    root, _ = film_disc(tmp_path, extras=False)
    client.post("/api/queue/disc", json={"path": str(root), "output_mode": "inside"})
    entry, job = _run(manager)
    assert entry.result_files == [str(root / "Film (2001).mkv")], job.status.error
    out = tmp_path / "out"
    client.post("/api/queue/disc", json={"path": str(root), "output_mode": "custom",
                                         "output_dir": str(out)})
    entry, job = _run(manager)
    assert entry.result_files == [str(out / "Film (2001).mkv")], job.status.error
    r = client.post("/api/queue/disc", json={"path": str(root), "output_mode": "custom"})
    assert r.status_code == 400 and "文件夹" in r.json()["detail"]
    # the settings say where, unless the page does
    settings_file(disc__min_title_seconds=0, disc__output_mode="custom",
                  disc__output_dir=str(out))
    report = client.get("/api/disc/scan", params={"path": str(root)}).json()
    assert report["output_mode"] == "custom" and report["output_dir"] == str(out)
    report = client.get("/api/disc/scan", params={"path": str(root),
                                                  "output_mode": "beside"}).json()
    assert report["output_dir"] == str(tmp_path)


# ------------------------------------------------------------------ batch

def test_the_batch_scan_lists_every_disc_and_why_one_cannot_be_read(api, tmp_path):
    client, _ = api
    lib = tmp_path / "lib"
    film_disc(lib)
    (lib / "Broken.iso").write_bytes(bytes(g.SECTOR * 400))
    (lib / "docs").mkdir()
    (lib / "docs" / "readme.txt").write_text("x")
    r = client.post("/api/disc/batch-scan", json={"path": str(lib)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [Path(d["root"]).name for d in body["discs"]] == ["Film (2001)"]
    assert body["discs"][0]["output_dir"] == str(lib)
    assert [Path(s["path"]).name for s in body["skipped"]] == ["Broken.iso"]
    assert "UDF" in body["skipped"][0]["reason"]
    r = client.post("/api/disc/batch-scan", json={"path": str(lib / "docs")})
    assert r.status_code == 400 and "没有找到原盘" in r.json()["detail"]


def test_a_box_set_is_queued_as_one_group_and_numbered_on(api, tmp_path, short_episodes):
    client, manager = api
    box = tmp_path / "Show [BD]"
    series_volume(box / "SHOW_VOL01", tmp_path)
    series_volume(box / "SHOW_VOL02", tmp_path)
    scan = client.post("/api/disc/batch-scan", json={"path": str(box)}).json()
    second = scan["discs"][1]
    assert (second["name"], second["mode"], second["episode_start"],
            second["extra_start"]) == ("SHOW", "series", 4, 2)
    assert second["volume"]["index"] == 2 and "从第 4 集编起" in second["volume"]["note"]
    r = client.post("/api/queue/disc-batch", json={
        "path": str(box), "discs": [{"path": d["path"]} for d in scan["discs"]]})
    assert r.status_code == 200, r.text
    first_entry, second_entry = manager.store.entries
    assert first_entry.group_id and first_entry.group_id == second_entry.group_id
    assert first_entry.group_title == str(box)
    # the numbering is settled now: the run must not need VOL01 any more
    frozen = second_entry.disc
    assert (frozen.name, frozen.episode_start, frozen.extra_start, frozen.own_name) == \
        ("SHOW", 4, 2, "SHOW.VOL02")
    for _ in range(2):
        entry, job = _run(manager)
        assert entry.status == "done", job.status.error
    assert sorted(p.name for p in box.glob("*.mkv")) == sorted(
        [f"SHOW.E0{i}.mkv" for i in range(1, 7)] + ["SHOW.花絮01.mkv", "SHOW.花絮02.mkv"])
    assert manager.retry(first_entry.id).group_id == first_entry.group_id


def test_a_batch_with_a_disc_that_cannot_be_queued_queues_nothing(api, tmp_path):
    client, manager = api
    lib = tmp_path / "lib"
    root, _ = film_disc(lib)
    meta = g.simple_bd(lib / "Meta Only", {"00800": [("00001", 0, 5400)]})
    discs = [{"path": str(root)}, {"path": str(meta)}]
    r = client.post("/api/queue/disc-batch", json={"path": str(lib), "discs": discs})
    assert r.status_code == 400
    assert "Meta Only" in r.json()["detail"] and "元数据" in r.json()["detail"]
    assert manager.store.entries == []
    discs[1]["include"] = False           # left out, the rest goes ahead
    r = client.post("/api/queue/disc-batch", json={"path": str(lib), "discs": discs})
    assert r.status_code == 200 and r.json()["count"] == 1


# ------------------------------------------------- the rest of the program

def test_a_batch_scan_does_not_translate_a_disc_file_by_file(tmp_path):
    root, _ = film_disc(tmp_path / "lib")
    (tmp_path / "lib" / "other.mkv").write_bytes(b"x")
    (tmp_path / "lib" / "image.iso").write_bytes(b"x")
    scan = scan_media(tmp_path / "lib")
    assert [p.name for p in scan.to_translate] == ["other.mkv"]
    assert sorted(p.name for p in scan.discs) == ["Film (2001)", "image.iso"]


def test_the_disc_picker_lists_discs_and_images(api, tmp_path):
    client, _ = api
    film_disc(tmp_path)
    (tmp_path / "plain").mkdir()
    (tmp_path / "image.iso").write_bytes(b"x")
    (tmp_path / "film.mkv").write_bytes(b"x")
    r = client.get("/api/fs/browse", params={"path": str(tmp_path), "mode": "disc"}).json()
    assert r["disc_dirs"] == ["Film (2001)"]
    assert [f["name"] for f in r["files"]] == ["image.iso"] and r["files"][0]["kind"] == "disc"
    assert not r["is_disc"]
    inside = client.get("/api/fs/browse", params={"path": str(tmp_path / "Film (2001)"),
                                                  "mode": "disc"}).json()
    assert inside["is_disc"]
    # the ordinary picker is unchanged
    plain = client.get("/api/fs/browse", params={"path": str(tmp_path)}).json()
    assert "disc_dirs" not in plain and [f["name"] for f in plain["files"]] == ["film.mkv"]

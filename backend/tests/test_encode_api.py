"""The 压制 page's endpoints, and the chains that run through an encode.

Two ways in: the 压制 page (one file or a folder, optionally with subtitles
after) and the 原盘 page's third button (remux → encode → subtitles). The
last tests here run the second one end to end on a synthetic Blu-ray: the
lossless MKV comes out, its encode goes in right behind the disc, replaces
it under the same name, and queues the subtitles at the end.
"""

from __future__ import annotations

from pathlib import Path

import av

from app.services import encode
from tests.test_disc_api import _run, api, film_disc  # noqa: F401 — api is a fixture
from tests.mediagen import make_source

FAST = {"video_codec": "libx264", "preset": "ultrafast"}


# ------------------------------------------------------------------ reading


def test_the_encoders_endpoint_adds_what_the_encode_form_needs(api):
    client, _ = api
    body = client.get("/api/media/encoders").json()
    caps = {c["id"]: c for c in body["capabilities"]}
    assert caps["libx265"]["tunes"] == ["animation", "grain"] and caps["libx265"]["ten_bit"]
    assert caps["libx264"]["quality"]["default"] == 20
    assert body["deinterlace"] is True
    # what the 翻译任务 page reads is unchanged
    assert body["video"][0]["id"] == "copy" and body["presets"][0] == "ultrafast"


def test_probing_a_file_reads_its_streams_and_a_few_frames(api, tmp_path):
    client, _ = api
    source = make_source(tmp_path / "i.mkv", interlaced=True, audio=[
        {"codec": "flac", "format": "s16", "language": "jpn"}])
    r = client.get("/api/encode/probe", params={"path": str(source)})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["video"]["codec"] == "h264" and info["video"]["interlaced"] == 1.0
    assert info["audio"][0]["lossless"] and info["audio"][0]["language"] == "jpn"
    assert not info["encoded"]
    missing = client.get("/api/encode/probe", params={"path": str(tmp_path / "no.mkv")})
    assert missing.status_code == 400


def test_a_folder_scan_lists_videos_but_never_the_pieces_of_a_disc(api, tmp_path):
    client, _ = api
    make_source(tmp_path / "a.mkv")
    (tmp_path / "sub").mkdir()
    make_source(tmp_path / "sub" / "b.mkv")
    (tmp_path / "a.mkv.part").write_bytes(b"half")
    (tmp_path / "song.flac").write_bytes(b"not a video")
    root, _ = film_disc(tmp_path / "discs")
    body = client.post("/api/encode/scan", json={"path": str(tmp_path)}).json()
    assert [Path(f["relative"]).as_posix() for f in body["files"]] == ["a.mkv", "sub/b.mkv"]
    assert body["discs"] == [str(root)]
    flat = client.post("/api/encode/scan",
                       json={"path": str(tmp_path), "recursive": False}).json()
    assert [f["relative"] for f in flat["files"]] == ["a.mkv"]


# ------------------------------------------------------------------ queueing


def test_what_cannot_be_encoded_is_refused_when_queued(api, tmp_path):
    client, manager = api
    source = make_source(tmp_path / "f.mkv")

    def post(**body):
        return client.post("/api/queue/encode", json={"source": str(source), **body})

    assert post(options={"video_codec": "h264_nvenc"}).status_code == 400
    assert "10bit" in post(options={"video_codec": "libvpx-vp9", "bit_depth": "10"}).json()["detail"]
    assert "film" in post(options={"video_codec": "libx265", "tune": "film"}).json()["detail"]
    assert "哪个文件夹" in post(output_mode="custom").json()["detail"]
    same = post(output_mode="custom", output_dir=str(tmp_path)).json()["detail"]
    assert "原文件所在的文件夹" in same
    assert "目标语言" in post(subtitles={"target_language": " "}).json()["detail"]
    dropped = post(options={"subtitles": "none"}, subtitles={"text_source": "subtitle"})
    assert "字幕轨" in dropped.json()["detail"]
    assert client.post("/api/queue/encode", json={"source": str(tmp_path / "x.mkv")}) \
        .status_code == 400
    assert manager.store.entries == []
    # the page never replaces anything: a field saying so is simply not taken
    ok = post(replace_source=True, options=FAST)
    assert ok.status_code == 200, ok.text
    assert manager.store.entries[0].encode.replace_source is False


def test_a_batch_is_all_or_nothing_and_keeps_its_sub_folders(api, tmp_path):
    client, manager = api
    (tmp_path / "S1").mkdir()
    (tmp_path / "S2").mkdir()
    a = make_source(tmp_path / "S1" / "01.mkv")
    b = make_source(tmp_path / "S2" / "01.mkv")
    body = {"path": str(tmp_path), "files": [str(a), str(b), str(tmp_path / "gone.mkv")],
            "options": FAST}
    r = client.post("/api/queue/encode-batch", json=body)
    assert r.status_code == 400 and "gone.mkv" in r.json()["detail"]
    assert manager.store.entries == []
    out = tmp_path / "out"
    body.update(files=[str(a), str(b)], output_mode="custom", output_dir=str(out),
                subtitles={"target_language": "English", "series_mode": True})
    r = client.post("/api/queue/encode-batch", json=body)
    assert r.status_code == 200, r.text
    entries = manager.store.entries
    assert [Path(e.encode.output_dir) for e in entries] == [out / "S1", out / "S2"]
    assert len({e.group_id for e in entries}) == 1
    assert entries[0].then.series_id == f"enc-{entries[0].group_id}"


def test_an_encode_runs_writes_beside_its_source_and_says_what_it_saved(api, tmp_path):
    client, manager = api
    source = make_source(tmp_path / "Film.mkv", frames=24, audio=[
        {"codec": "flac", "format": "s16", "language": "jpn"}])
    r = client.post("/api/queue/encode",
                    json={"source": str(source), "options": FAST,
                          "subtitles": {"target_language": "English"}})
    assert r.status_code == 200, r.text
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    out = tmp_path / "Film.H264.mkv"
    assert entry.result_files == [str(out)] and source.exists()
    assert entry.note.startswith("已压制：Film.H264.mkv")
    assert "已自动加入 1 条字幕任务" in entry.note
    follow = manager.store.entries[-1]
    assert (follow.request.video_path, follow.origin_kind) == (str(out), "encode")
    with av.open(str(out)) as c:
        assert [s.codec_context.name for s in c.streams] == ["h264", "eac3"]
    # the same again finds its file and does not write Film.H264.2.mkv
    # (the translation it queued is not what this test runs)
    manager.cancel(follow.id)
    again = manager.retry(entry.id)
    entry2, _ = _run(manager)
    assert entry2.id == again.id and entry2.result_files == [str(out)]
    assert not (tmp_path / "Film.H264.2.mkv").exists()


def test_the_cpu_switch_slows_a_running_encode_and_says_so(api, tmp_path, monkeypatch):
    """压制让出 CPU end to end: switched on from the 列队 page, picked up by the
    encode as it runs. The other programs are faked at 60% of the machine, so
    the encode is held to 20% (half of the 40% left) whatever this machine is
    really doing."""
    import os
    import time

    from app.services import cpuyield

    client, manager = api
    monkeypatch.setattr(cpuyield, "_enabled", False)
    monkeypatch.setattr(cpuyield, "SAMPLE", 0.02)
    monkeypatch.setattr(cpuyield, "WINDOW", 0.06)
    cpus = os.cpu_count() or 1
    monkeypatch.setattr(cpuyield, "read_system", lambda: (
        time.process_time() + 0.6 * time.monotonic() * cpus, time.monotonic() * cpus))
    source = make_source(tmp_path / "Film.mkv", frames=480, size=(320, 240))
    assert client.post("/api/queue/cpu-yield", json={"enabled": True}).status_code == 200
    client.post("/api/queue/encode", json={"source": str(source), "options": FAST})
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    logged = [e.log for e in job.events if e.log]
    assert any(line.startswith("压制让路已开启") for line in logged)
    assert any("其他程序的 CPU 占用 60%" in line and "压制开始限速" in line for line in logged)
    assert any(line.startswith("压制让路：限速 1 次") for line in logged)
    assert cpuyield.status()["active"] is False   # the page stops showing it


def test_an_encode_that_outgrows_the_memory_limit_stops_and_leaves_nothing(
        api, tmp_path, monkeypatch):
    """The limit, end to end: the encode is stopped, the job says why, and
    neither the half-written file nor a finished one is left behind."""
    import time

    from app.services import memguard

    client, manager = api
    monkeypatch.setattr(memguard, "_limit_gb", 0.005)     # 5 MB: any encode is over
    monkeypatch.setattr(memguard, "CHECK_SECONDS", 0.0)
    from app.services.pipeline import manager as jobs

    stages = []     # the job's stage each time memory is handed back
    real_trim = memguard.trim

    def trim():
        stages.append(jobs.get(manager.store.entries[0].job_id).status.stage)
        real_trim()

    monkeypatch.setattr(memguard, "trim", trim)
    source = make_source(tmp_path / "Film.mkv", frames=240, size=(640, 480))
    client.post("/api/queue/encode", json={"source": str(source), "options": FAST})
    entry, job = _run(manager)
    assert entry.status == "failed"
    assert "超过上限 0.0 GB，已停止" in entry.error
    assert source.exists() and not list(tmp_path.glob("Film.H264*"))   # no .mkv, no .part
    assert memguard.status()["active"] is False
    # before measuring, and once the job is over (the runner's last act, so
    # it may land a moment after the queue sees the failure)
    deadline = time.monotonic() + 10
    while "failed" not in stages and time.monotonic() < deadline:
        time.sleep(0.05)
    assert stages == ["encoding", "failed"]


def test_an_encode_logs_how_much_memory_it_took(api, tmp_path, monkeypatch):
    from app.services import memguard

    client, manager = api
    monkeypatch.setattr(memguard, "_limit_gb", 0.0)
    monkeypatch.setattr(memguard, "CHECK_SECONDS", 0.0)
    source = make_source(tmp_path / "Film.mkv", frames=48, size=(320, 240))
    client.post("/api/queue/encode", json={"source": str(source), "options": FAST})
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    logged = [e.log for e in job.events if e.log]
    assert any(line.startswith("内存保护：压制最多用") for line in logged)
    assert any(line.startswith("压制占用内存峰值") for line in logged)


# -------------------------------------------------- the 原盘 page's third button


def _pipeline(client, root, **extra):
    return client.post("/api/queue/disc", json={
        "path": str(root), "encode": {"options": FAST},
        "subtitles": {"target_language": "English"}, **extra})


def test_the_full_pipeline_is_mkv_only_and_keeps_what_the_subtitles_need(api, tmp_path):
    client, manager = api
    root, _ = film_disc(tmp_path)
    r = client.post("/api/queue/disc", json={
        "path": str(root), "encode": {"options": {**FAST, "container": "mp4"}}})
    assert r.status_code == 400 and "MKV" in r.json()["detail"]
    r = client.post("/api/queue/disc", json={
        "path": str(root), "encode": {"options": {**FAST, "audio_languages": ["eng"]}},
        "subtitles": {"target_language": "简体中文", "audio_language": "jpn"}})
    assert r.status_code == 400 and "音轨" in r.json()["detail"]
    assert manager.store.entries == []


def test_remux_then_encode_then_subtitles_end_to_end(api, tmp_path):
    client, manager = api
    root, _ = film_disc(tmp_path)
    other = make_source(tmp_path / "other.mkv")
    r = _pipeline(client, root)
    assert r.status_code == 200, r.text
    client.post("/api/queue/encode", json={"source": str(other), "options": FAST})
    disc, _ = _run(manager)
    assert disc.status == "done"
    film = tmp_path / "Film (2001).mkv"
    extra = tmp_path / "Film (2001).花絮01.mkv"
    assert disc.result_files == [str(film), str(extra)]
    lossless = film.stat().st_size
    # the two encodes run next, before the entry queued after the disc
    kinds = [(e.kind, e.encode.source if e.encode else "") for e in manager.store.entries[1:]]
    assert kinds == [("encode", str(film)), ("encode", str(extra)), ("encode", str(other))]
    first, job = _run(manager)
    assert first.status == "done", job.status.error
    # replaced under its own name: no second file, and the file says what it is
    assert first.result_files == [str(film)]
    tags = encode.read_tags(film)
    assert tags["encode"] and tags["remux"]["title"] == "00800.mpls"
    assert not list(tmp_path.glob("*.H264.mkv")) and film.stat().st_size != lossless
    with av.open(str(film)) as c:
        # the picture re-encoded; the disc's AC-3 is lossy, so copied as it is
        assert [s.codec_context.name for s in c.streams if s.type != "subtitle"][:2] \
            == ["h264", "ac3"]
    _run(manager)                                   # the extra's encode
    subtitles = [e for e in manager.store.entries if e.kind == "job"]
    assert [e.request.video_path for e in subtitles] == [str(film), str(extra)]
    assert manager.store.entries[-1].request.video_path == str(extra)
    assert all(e.origin_kind == "encode" and e.request.series_id.startswith("disc-")
               for e in subtitles)

    # a retry of the disc finds its titles by their tags — lossless or
    # encoded by now — and writes nothing again, not even as 片名.2.mkv
    for e in subtitles:
        manager.cancel(e.id)            # translating is not what this test runs
    retried = manager.retry(disc.id)
    while manager.store.get(retried.id).status == "queued":
        entry, job = _run(manager)
    assert manager.store.get(retried.id).status == "done"
    assert not list(tmp_path.glob("*.2.mkv"))
    assert "已压制过或已在列队里" in manager.store.get(retried.id).note


def test_keeping_the_lossless_file_names_the_encode_apart(api, tmp_path):
    client, manager = api
    root, _ = film_disc(tmp_path, extras=False)
    r = client.post("/api/queue/disc", json={
        "path": str(root), "encode": {"options": FAST, "keep_lossless": True}})
    assert r.status_code == 200, r.text
    from app.services.jobqueue import describe_entry
    assert "保留无损版" in describe_entry(manager.store.entries[0])
    _run(manager)
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    assert entry.result_files == [str(tmp_path / "Film (2001).H264.mkv")]
    assert encode.read_tags(tmp_path / "Film (2001).mkv")["encode"] is None


def test_a_fresh_retry_of_a_remux_does_not_write_the_film_twice(api, tmp_path):
    """用当前设置重试 changes the settings fingerprint, and with it the
    checkpoint the remux kept its record in; the tag in the file is what
    still knows the title is done."""
    client, manager = api
    root, _ = film_disc(tmp_path, extras=False)
    client.post("/api/queue/disc", json={"path": str(root)})
    first, _ = _run(manager)
    assert first.status == "done"
    manager.retry(first.id, fresh=True)
    second, job = _run(manager)
    assert second.status == "done" and second.result_files == first.result_files
    assert not (tmp_path / "Film (2001).2.mkv").exists()
    assert any("上次已经封装好了" in e.log for e in job.events)

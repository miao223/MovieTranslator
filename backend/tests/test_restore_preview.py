"""试看 (services/restore_preview.py): a few moments of a film through
Lanczos and every model, measured once on the whole film, on the heavy-job
slot. The models are the fake engine (tests/fake_engine.py): nearest ×2."""

import sys
import threading
import time
from pathlib import Path

import av
import pytest

from app.models.schemas import EncodeOptions, RestoreSettings
from app.services import pipeline, restore, restore_preview as rp
from tests.conftest import local_client
from tests.mediagen import make_fields

ENGINE = RestoreSettings(engine_python=sys.executable)


@pytest.fixture(autouse=True)
def fake_models(monkeypatch, settings_file):
    made = []

    def make(opts, log, engine=None):
        if not opts.ai_model:
            return None
        client = restore.EngineClient(sys.executable, {"model": "double"}, log, module="tests.fake_engine")
        made.append(client)
        return restore.EngineStage(client, "假模型")

    monkeypatch.setattr(restore, "make_stage", make)
    monkeypatch.setattr(rp, "SEGMENT_SECONDS", 1.0)
    rp._current = None
    yield made
    p = rp.current()
    if p is not None:
        p.cancel.set()
        if p.thread is not None:
            p.thread.join(timeout=30)
    # straight, not through monkeypatch: its undo would put this one back
    rp._current = None


def wait(p, timeout=120.0, until=("done", "failed", "cancelled")):
    end = time.monotonic() + timeout
    while p.status not in until and time.monotonic() < end:
        time.sleep(0.05)
    return p.status


def frames(path: Path) -> int:
    with av.open(str(path)) as c:
        return sum(1 for packet in c.demux(c.streams.video[0]) if packet.size)


def test_a_preview_puts_every_method_side_by_side(tmp_path):
    source = make_fields(tmp_path / "tape.mkv", frames=360, size=(128, 96))
    p = rp.start(source, EncodeOptions(deinterlace="detect", crop_auto=True, upscale=720), ENGINE)
    assert wait(p) == "done", p.error
    view = p.view()
    assert len(view["segments"]) == 3 and all(s == 1.0 for _, s in view["segments"])
    assert set(view["methods"]) == {"lanczos", "realbasicvsr", "realviformer", "liveaction_span"}
    # bob: two frames a source frame, a second of 29.97 → ~60 frames a segment
    for method, figures in view["methods"].items():
        assert 150 <= figures["frames"] <= 180, (method, figures)
        assert figures["flicker"] == figures["flicker"]          # measured, not NaN
    for method in ("realbasicvsr", "realviformer", "liveaction_span"):
        assert view["methods"][method]["model_fps"] == 100.0     # the engine's own figure
        assert view["methods"][method]["film_model_hours"] is not None
    assert "model_fps" not in view["methods"]["lanczos"]
    video = rp.file_path(view["files"]["video"])
    with av.open(str(video)) as c:
        cc = c.streams.video[0].codec_context
        # four 960×720 panels (128×96 4:3 → 720p, square pixels)
        assert (cc.width, cc.height) == (1920, 1440)
    lanczos = sum(frames(p.folder / f"seg{i}.lanczos.mkv") for i in range(3))
    assert frames(video) == lanczos
    assert rp.file_path(view["files"]["sheet"]).stat().st_size > 0
    assert rp.file_path("seg0.mkv") is None                      # only what the view lists


def test_the_film_is_measured_once_not_each_clip(tmp_path):
    # a one-second clip can't be classified (classify wants 60 frames of
    # motion) nor its field order voted on reliably: every segment must run
    # with what the whole film measured, and none may measure again
    source = make_fields(tmp_path / "tape.mkv", frames=360, size=(128, 96), bottom_first=True,
                         flagged=False)
    p = rp.start(source, EncodeOptions(deinterlace="detect", upscale=720), ENGINE)
    assert wait(p) == "done", p.error
    assert "反交错：还原 60 帧" in p.resolved and "场序：下场优先" in p.resolved
    # the cadence is measured once, on the film; a clip measuring again would
    # log it again (and, at 60 frames, say it cannot tell)
    assert sum("片源节奏" in line for line in p.log) == 1, p.log
    assert not [line for line in p.log if "测不出" in line]
    assert not [line for line in p.log if line.startswith("场序：从画面测得")]   # encode_file's own vote


def test_it_waits_for_the_heavy_job_slot(tmp_path):
    source = make_fields(tmp_path / "tape.mkv", frames=360, size=(128, 96))
    assert pipeline._run_slot.acquire(timeout=5)
    try:
        p = rp.start(source, EncodeOptions(deinterlace="bob", upscale=720), ENGINE)
        time.sleep(1.0)
        assert p.status == "waiting" and "等待" in p.stage
    finally:
        pipeline._run_slot.release()
    assert wait(p) == "done", p.error


def test_cancelling_while_waiting_gives_back_nothing_it_never_took(tmp_path):
    source = make_fields(tmp_path / "tape.mkv", frames=360, size=(128, 96))
    assert pipeline._run_slot.acquire(timeout=5)
    try:
        p = rp.start(source, EncodeOptions(deinterlace="bob", upscale=720), ENGINE)
        time.sleep(0.5)
        assert p.status == "waiting"
        rp.cancel()
        assert wait(p, timeout=2) == "cancelled"
        # still ours: a release on the way out would hand the running job's slot away
        assert not pipeline._run_slot.acquire(timeout=0)
    finally:
        pipeline._run_slot.release()


def test_cancelling_stops_it_and_leaves_no_engine_running(tmp_path, fake_models):
    source = make_fields(tmp_path / "tape.mkv", frames=360, size=(128, 96))
    p = rp.start(source, EncodeOptions(deinterlace="bob", upscale=720), ENGINE)
    end = time.monotonic() + 60
    while not fake_models and time.monotonic() < end:     # into the first model
        time.sleep(0.02)
    rp.cancel()
    assert wait(p) == "cancelled"
    for client in fake_models:
        assert client.proc.poll() is not None
    assert pipeline._run_slot.acquire(timeout=5)       # given back
    pipeline._run_slot.release()


def test_a_new_preview_replaces_the_running_one(tmp_path):
    first = make_fields(tmp_path / "a.mkv", frames=360, size=(128, 96))
    second = make_fields(tmp_path / "b.mkv", frames=360, size=(128, 96))
    old = rp.start(first, EncodeOptions(deinterlace="bob", upscale=720), ENGINE)
    new = rp.start(second, EncodeOptions(deinterlace="bob", upscale=720), ENGINE)
    assert old.status == "cancelled" and rp.current() is new
    assert not old.folder.exists()                     # its files went with it
    assert wait(new) == "done", new.error


@pytest.mark.parametrize("options,engine,said", [
    (EncodeOptions(upscale=0), ENGINE, "放大到"),
    (EncodeOptions(upscale=720), RestoreSettings(), "修复引擎"),
    (EncodeOptions(upscale=720, video_codec="copy"), ENGINE, "画面保持原样"),
])
def test_what_cannot_be_previewed_is_refused_at_once(tmp_path, options, engine, said):
    source = make_fields(tmp_path / "tape.mkv", frames=60, size=(128, 96))
    with pytest.raises(rp.PreviewError, match=said):
        rp.start(source, options, engine)


def test_a_short_film_is_looked_at_once_in_its_middle():
    assert rp.moments(2.0) == [(0.5, 1.0)]
    assert len(rp.moments(3.5)) == 1
    assert [round(s, 2) for s, _ in rp.moments(100.0)] == [24.5, 49.5, 74.5]


def test_the_preview_endpoints(tmp_path, settings_file):
    source = make_fields(tmp_path / "tape.mkv", frames=360, size=(128, 96))
    settings_file(restore__engine_python=sys.executable)
    from app.main import app
    client = local_client(app)
    assert client.get("/api/restore/preview").json() == {"status": "idle"}
    r = client.post("/api/restore/preview", json={"source": str(source), "options": {"upscale": 0}})
    assert r.status_code == 400 and "放大到" in r.json()["detail"]
    r = client.post("/api/restore/preview", json={"source": str(source),
                                                  "options": {"upscale": 720, "deinterlace": "bob"}})
    assert r.status_code == 200, r.text
    assert wait(rp.current()) == "done", rp.current().error
    view = client.get("/api/restore/preview").json()
    assert view["status"] == "done" and view["files"]["video"] == "compare.mp4"
    r = client.get("/api/restore/preview/files/compare.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and r.headers["content-type"] == "video/mp4"
    assert "attachment" not in r.headers.get("content-disposition", "")
    assert client.get("/api/restore/preview/files/sheet.jpg").headers["content-type"] == "image/jpeg"
    assert client.get("/api/restore/preview/files/seg0.mkv").status_code == 404
    assert client.get("/api/restore/preview/files/..%2F..%2Fsettings.json").status_code == 404
    assert client.delete("/api/restore/preview").json()["status"] == "done"   # nothing to cancel

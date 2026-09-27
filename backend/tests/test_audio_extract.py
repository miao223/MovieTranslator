"""The 音频 page: one track of a video, as 16 kHz mono FLAC (lossless,
the default) or Opus (small).

The sources are synthesised (mediagen.make_source) with a different tone
level per track, so the output tells which track was taken. Everything
runs through the queue, the only way in, like the 压制 page.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from app.models.schemas import AudioOptions, AudioRequest
from app.services import audio, audioextract
from tests.mediagen import make_source
from tests.test_disc_api import _run, api  # noqa: F401 — api is a fixture

TRACKS = [{"codec": "aac", "language": "jpn", "title": "日本語", "amp": 0.05},
          {"codec": "flac", "format": "s16", "language": "eng", "amp": 0.6}]


def film(path: Path, frames: int = 96) -> Path:
    """Four seconds of picture and two audio tracks."""
    return make_source(path, frames=frames, audio=TRACKS)


def samples(path: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.concatenate([f.to_ndarray().reshape(-1) for f in c.decode(audio=0)])


def read(path: Path):
    """(container, channels, seconds, peak level)."""
    with av.open(str(path)) as c:
        st = c.streams.audio[0]
        codec, channels = c.format.name, st.codec_context.layout.nb_channels
        rate = st.codec_context.sample_rate
    x = samples(path)
    peak = float(np.abs(x.astype(np.float64)).max()) / (32768 if x.dtype.kind == "i" else 1)
    return codec, channels, len(x) / rate, peak


def test_the_default_is_a_flac_the_pipeline_would_hear_sample_for_sample(api, tmp_path):
    client, manager = api
    (tmp_path / "lib").mkdir()
    source = film(tmp_path / "lib" / "Film.mkv")
    r = client.post("/api/queue/audio", json={"source": str(source)})
    assert r.status_code == 200, r.text
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    out = tmp_path / "lib" / "Film.flac"
    assert entry.result_files == [str(out)]
    assert entry.note.startswith("已提取：Film.flac")
    codec, channels, seconds, _ = read(out)
    assert (codec, channels) == ("flac", 1)
    assert seconds == pytest.approx(4.0, abs=0.1)
    assert audioextract.read_tag(out)["format"] == "flac"
    # lossless: exactly the WAV that translating the video itself transcribes
    # (stream 1: the default track, the first audio stream after the picture)
    wav = audio.extract_audio(source, tmp_path / "direct.wav", track_index=1)
    assert np.array_equal(samples(out), samples(wav))
    # …and translating the FLAC transcribes that same WAV again
    again = audio.extract_audio(out, tmp_path / "again.wav")
    assert again.read_bytes() == wav.read_bytes()
    # nothing half-written is left behind
    assert sorted(p.name for p in out.parent.iterdir()) == ["Film.flac", "Film.mkv"]


def test_a_chosen_track_as_a_small_opus(api, tmp_path):
    client, manager = api
    source = film(tmp_path / "Film.mkv")
    info = client.get("/api/audio/probe", params={"path": str(source)}).json()
    assert [t["language"] for t in info["tracks"]] == ["jpn", "eng"]
    assert info["default_track"] == info["tracks"][0]["index"] and not info["extracted"]
    loud = info["tracks"][1]["index"]
    r = client.post("/api/queue/audio", json={
        "source": str(source), "track": loud, "options": {"format": "opus"}})
    assert r.status_code == 200, r.text
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    out = tmp_path / "Film.opus"
    codec, channels, seconds, peak = read(out)
    assert (codec, channels) == ("ogg", 1)
    with av.open(str(out)) as c:
        assert c.streams.audio[0].codec.name == "opus"
    assert seconds == pytest.approx(4.0, abs=0.1)
    assert peak > 0.3          # the loud English track, not the quiet Japanese one
    # 24 kbps × 4 s is 12 kB of audio; the Ogg pages add a little
    assert out.stat().st_size < 20_000
    # the page now says it has been done
    again = client.get("/api/audio/probe", params={"path": str(source)}).json()
    assert again["extracted"] == str(out)


def test_what_cannot_be_extracted_is_refused_when_queued(api, tmp_path):
    client, manager = api
    source = film(tmp_path / "Film.mkv")
    assert client.post("/api/queue/audio", json={
        "source": str(source), "track": 9}).status_code == 400
    assert client.post("/api/queue/audio", json={
        "source": str(tmp_path / "gone.mkv")}).status_code == 400
    assert "哪个文件夹" in client.post("/api/queue/audio", json={
        "source": str(source), "output_mode": "custom"}).json()["detail"]
    silent = make_source(tmp_path / "silent.mkv")
    assert client.post("/api/queue/audio", json={"source": str(silent)}).status_code == 400
    assert manager.store.entries == []


def test_never_over_an_existing_file_and_a_retry_finds_its_own(api, tmp_path):
    client, manager = api
    source = film(tmp_path / "Film.mkv")
    theirs = tmp_path / "Film.flac"
    theirs.write_bytes(b"someone else's")
    client.post("/api/queue/audio", json={"source": str(source)})
    entry, job = _run(manager)
    assert entry.status == "done", job.status.error
    ours = tmp_path / "Film.2.flac"
    assert entry.result_files == [str(ours)] and theirs.read_bytes() == b"someone else's"
    # the same again finds Film.2.flac by its tag instead of writing Film.3.flac
    retried = manager.retry(entry.id)
    assert retried.kind == "audio" and retried.audio == entry.audio
    entry2, job2 = _run(manager)
    assert entry2.status == "done" and entry2.result_files == [str(ours)]
    assert "已经提取过了" in job2.status.message
    assert not (tmp_path / "Film.3.flac").exists()


def test_a_batch_chooses_by_language_and_keeps_its_sub_folders(api, tmp_path):
    client, manager = api
    (tmp_path / "S1").mkdir()
    a = film(tmp_path / "S1" / "01.mkv")
    b = make_source(tmp_path / "02.mkv", frames=48, audio=TRACKS[:1])   # no English
    make_source(tmp_path / "silent.mkv")
    (tmp_path / "old.flac").write_bytes(b"not listed: not a video")
    scan = client.post("/api/audio/scan", json={"path": str(tmp_path)}).json()
    assert sorted(Path(f["relative"]).as_posix() for f in scan["files"]) == ["02.mkv", "S1/01.mkv"]
    assert [Path(s["path"]).name for s in scan["skipped"]] == ["silent.mkv"]

    out = tmp_path / "out"
    r = client.post("/api/queue/audio-batch", json={
        "path": str(tmp_path), "files": [str(a), str(b)], "language": "eng",
        "output_mode": "custom", "output_dir": str(out)})
    assert r.status_code == 200, r.text
    entries = manager.store.entries
    assert [Path(e.audio.output_dir) for e in entries] == [out / "S1", out]
    assert len({e.group_id for e in entries}) == 1
    first, _ = _run(manager)
    second, job = _run(manager)
    assert (first.status, second.status) == ("done", "done")
    assert read(out / "S1" / "01.flac")[3] > 0.3                 # English
    assert read(out / "02.flac")[2] == pytest.approx(2.0, abs=0.1)
    assert any("没有英语音轨" in e.log for e in job.events)      # fell back, and said so


def test_cancelling_stops_the_extraction_and_leaves_no_part_file(api, tmp_path, monkeypatch):
    client, manager = api
    (tmp_path / "lib").mkdir()
    source = film(tmp_path / "lib" / "Film.mkv")
    request = AudioRequest(source=str(source), options=AudioOptions())
    with pytest.raises(InterruptedError):
        audioextract.extract(source, tmp_path / "x.opus", request, 1,
                             should_cancel=lambda: True)
    # through the queue: cancelled once the file has been started (PyAV
    # creates it with the first packet), and nothing is left behind
    real = audioextract.extract

    def cancelled_halfway(src, out, req, track, **kw):
        real(src, out, req, track, **{**kw, "should_cancel": lambda: out.exists()})

    monkeypatch.setattr(audioextract, "extract", cancelled_halfway)
    client.post("/api/queue/audio", json={"source": str(source)})
    entry, job = _run(manager)
    assert entry.status == "cancelled", job.status.error
    assert sorted(p.name for p in source.parent.iterdir()) == ["Film.mkv"]


def test_the_pipeline_wav_is_unchanged_by_the_shared_loop(tmp_path):
    """write_track's defaults are the WAV extract_audio always wrote."""
    from app.services import audio

    source = film(tmp_path / "Film.mkv")
    wav = audio.extract_audio(source, tmp_path / "a.wav")
    with av.open(str(wav)) as c:
        st = c.streams.audio[0]
        assert (c.format.name, st.codec_context.name, st.codec_context.sample_rate,
                st.codec_context.layout.nb_channels) == ("wav", "pcm_s16le", 16000, 1)
    assert audioextract.read_tag(wav) is None

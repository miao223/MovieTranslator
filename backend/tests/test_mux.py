"""Embedding the subtitle as a soft track in a new video.

The promise is narrow and checkable: the picture and sound come through
untouched, the subtitle plays without a sidecar file, and nothing about
the source file changes. The tests exist to hold that line — above all
that no re-encode ever creeps in.
"""

from pathlib import Path

import av
import pytest

from app.models.schemas import SubtitleLine, SubtitleSettings
from app.services import mux, subtitle
from tests.test_audio_tracks import make_multitrack_video

LINES = [
    SubtitleLine(index=1, start=0.5, end=1.2, text="Hello there", translation="你好"),
    SubtitleLine(index=2, start=1.4, end=1.9, text="Goodbye", translation="再见"),
]


@pytest.fixture
def video(tmp_path):
    return make_multitrack_video(tmp_path / "film.mkv")


def _write_subs(tmp_path, styled=False, mode="bilingual") -> Path:
    settings = SubtitleSettings(style_enabled=styled)
    if styled:
        text = subtitle.build_ass(LINES, settings, mode=mode)
        path = tmp_path / "subs.ass"
    else:
        text = subtitle.build_srt(LINES, settings, mode=mode)
        path = tmp_path / "subs.srt"
    path.write_text(text, encoding="utf-8")
    return path


def _streams(path: Path):
    with av.open(str(path)) as c:
        return [(s.type, s.codec_context.name, dict(s.metadata)) for s in c.streams]


def _frames(path: Path) -> int:
    """Frames a player would actually get. Counting packets is not enough —
    a dropped keyframe leaves the packets that follow it undecodable."""
    with av.open(str(path)) as c:
        return sum(len(p.decode()) for p in c.demux(c.streams.video[0]))


def make_bframe_video(path: Path, frames: int = 48) -> Path:
    """A source with B-frames — i.e. what real films are.

    Without them a video has no reordering, every packet carries a DTS, and
    the whole class of timestamp bugs this module had is invisible. Verified
    against real releases (h264/mp4, matroska, webm) and `ffmpeg -c copy`;
    this clip reproduces the same behaviour without shipping a fixture.
    """
    import numpy as np

    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=24)
        video.width, video.height = 128, 96
        video.pix_fmt = "yuv420p"
        video.options = {"bf": "3", "g": "24"}
        for i in range(frames):
            img = np.zeros((96, 128, 3), dtype=np.uint8)
            img[:, (i * 4) % 128:((i * 4) % 128) + 16] = 255  # motion to encode
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
    return path


# ------------------------------------------------------------------ naming


def test_the_new_file_sits_beside_the_old_one_under_a_language_suffix(tmp_path):
    out = mux.output_path(tmp_path / "film.mkv", "简体中文")
    assert out == tmp_path / "film.zh.mkv"
    assert mux.output_path(tmp_path / "film.mp4", "English").name == "film.en.mkv"
    # an unknown target language still produces something distinguishable
    assert mux.output_path(tmp_path / "film.mkv", "Esperanto").name == "film.sub.mkv"


def test_an_existing_file_is_never_overwritten(tmp_path):
    """The output lands in the user's video library; a name collision must
    not cost them a file."""
    (tmp_path / "film.zh.mkv").write_bytes(b"someone's video")
    assert mux.output_path(tmp_path / "film.mkv", "简体中文").name == "film.zh.2.mkv"


# -------------------------------------------------------------- the remux


def test_the_picture_and_sound_are_copied_not_re_encoded(video, tmp_path):
    """The whole point of soft subs: same streams, new container. If this
    ever starts transcoding, a two-hour film goes from a file copy to an
    hour of CPU — and the quality is gone for good."""
    subs = _write_subs(tmp_path)
    out = mux.embed(video, subs, tmp_path / "out.mkv", "简体中文")

    before = _streams(video)
    after = _streams(out)
    assert [(t, c) for t, c, _ in after[:3]] == [(t, c) for t, c, _ in before]
    assert [t for t, _, _ in after] == ["video", "audio", "audio", "subtitle"]


def test_every_audio_track_keeps_its_language_tag(video, tmp_path):
    """add_stream_from_template copies codec parameters but not metadata,
    and a multi-track release is chosen by exactly those tags."""
    out = mux.embed(video, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文")
    langs = [m.get("language") for t, _, m in _streams(out) if t == "audio"]
    assert langs == ["jpn", "eng"]
    titles = [m.get("title") for t, _, m in _streams(out) if t == "audio"]
    assert titles == ["日本語", "English dub"]


def test_the_subtitle_track_announces_itself(video, tmp_path):
    """A track nobody can find is the same as no track: it carries the
    language, a readable name, and the default flag."""
    out = mux.embed(
        video, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文",
        track_title="简体中文字幕",
    )
    with av.open(str(out)) as c:
        track = c.streams.subtitles[0]
        assert track.metadata["language"] == "chi"
        assert track.metadata["title"] == "简体中文字幕"
        assert track.disposition & av.stream.Disposition.default


def test_the_embedded_cues_are_the_ones_we_wrote(video, tmp_path):
    """Same text, same timings as the sidecar file would have had — the
    embedded track is that file, remuxed."""
    out = mux.embed(video, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文")
    with av.open(str(out)) as c:
        track = c.streams.subtitles[0]
        cues = [
            (float(p.pts * p.time_base), bytes(p).decode("utf-8"))
            for p in c.demux(track) if p.dts is not None
        ]
    assert [round(t, 2) for t, _ in cues] == [0.5, 1.4]
    assert cues[0][1] == "Hello there\n你好"
    assert cues[1][1] == "Goodbye\n再见"


def test_styling_survives_because_the_ass_header_rides_along(video, tmp_path):
    """ASS keeps its sizes and colours in codecpar.extradata, which PyAV
    offers no way to set — copying the stream from the .ass file is what
    makes styled output embeddable at all."""
    out = mux.embed(video, _write_subs(tmp_path, styled=True), tmp_path / "o.mkv", "")
    with av.open(str(out)) as c:
        track = c.streams.subtitles[0]
        assert track.codec_context.name in ("ass", "ssa")
        header = bytes(track.codec_context.extradata or b"").decode("utf-8", "replace")
    assert "[V4+ Styles]" in header and "Arial" in header


def test_attached_fonts_come_through(video, tmp_path):
    """A release keeps the fonts its subtitles are styled with as container
    attachments. Drop them and the film still plays, but its own subtitles
    render in whatever the player falls back to."""
    with_font = tmp_path / "withfont.mkv"
    with av.open(str(video)) as vin, \
         av.open(str(with_font), "w", format="matroska") as out:
        mapping = {}
        for stream in vin.streams:
            copy = out.add_stream_from_template(stream)
            copy.metadata.update(dict(stream.metadata))
            mapping[stream.index] = copy
        out.add_attachment("MyFont.ttf", "application/x-truetype-font", b"FONTDATA" * 8)
        for packet in vin.demux():
            if not packet.size:
                continue
            packet.stream = mapping[packet.stream.index]
            out.mux(packet)

    muxed = mux.embed(with_font, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文")
    with av.open(str(muxed)) as c:
        fonts = [s for s in c.streams if s.type == "attachment"]
        assert [s.metadata["filename"] for s in fonts] == ["MyFont.ttf"]
        assert bytes(fonts[0].data) == b"FONTDATA" * 8


def test_a_subtitle_track_already_in_the_video_is_kept(tmp_path):
    """Ours is an addition, not a replacement — a release's own subtitles
    and its attached fonts have to come through."""
    source = tmp_path / "withsubs.mkv"
    make_multitrack_video(source)
    with_extra = tmp_path / "extra.mkv"
    existing = _write_subs(tmp_path)
    mux.embed(source, existing, with_extra, "English")  # first pass adds one

    out = mux.embed(with_extra, existing, tmp_path / "twice.mkv", "简体中文")
    kinds = [t for t, _, _ in _streams(out)]
    assert kinds.count("subtitle") == 2


# ------------------------------------------------------- failure handling


def test_a_cancelled_mux_leaves_nothing_behind(video, tmp_path):
    """A half-copied film in the library looks exactly like a real one
    until the moment it is played."""
    out = tmp_path / "out.mkv"
    with pytest.raises(InterruptedError):
        mux.embed(
            video, _write_subs(tmp_path), out, "简体中文",
            should_cancel=lambda: True,
        )
    assert not out.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_a_broken_source_leaves_nothing_behind(tmp_path):
    broken = tmp_path / "broken.mkv"
    broken.write_bytes(b"not a video")
    out = tmp_path / "out.mkv"
    with pytest.raises(Exception):
        mux.embed(broken, _write_subs(tmp_path), out, "简体中文")
    assert not out.exists() and list(tmp_path.glob("*.part")) == []


def test_the_source_video_is_never_touched(video, tmp_path):
    before = video.read_bytes()
    mux.embed(video, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文")
    assert video.read_bytes() == before


# --------------------------------------------------- the pipeline's stage


@pytest.fixture
def job_factory(tmp_path, monkeypatch):
    """Build a real Job without touching the user's log folder."""
    from app.core import joblog
    from app.models.schemas import JobRequest
    from app.services.pipeline import Job

    monkeypatch.setattr(joblog, "_base_dir", lambda: tmp_path / "logs")

    def make(video: Path, **kwargs):
        return Job(JobRequest(video_path=str(video), embed_subtitle=True, **kwargs))

    return make


def test_the_stage_embeds_and_keeps_the_subtitle_out_of_the_video_folder(
    video, tmp_path, job_factory
):
    """Embed mode's promise: one new video, no sidecar file left behind —
    but the subtitle still exists in the work dir, because the download
    button and the debug artifacts both want it."""
    from app.services.pipeline import manager

    job = job_factory(video, target_language="简体中文")
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = manager._embed_subtitle(
        job, job.request, workdir, video, ".srt",
        subtitle.build_srt(LINES, SubtitleSettings()),
    )

    assert out == video.parent / "film.zh.mkv" and out.is_file()
    assert job.status.video_filename == str(out)
    assert not (video.parent / "film.srt").exists()  # nothing beside the video
    assert (workdir / "film.srt").is_file() and job.srt_path == workdir / "film.srt"


def test_a_failed_mux_falls_back_to_writing_the_subtitle(
    video, tmp_path, job_factory, monkeypatch
):
    """An hour of transcription and translation sits behind this step. If
    the copy cannot be made, the job still has to hand back its subtitle."""
    from app.services import pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("磁盘已满")

    monkeypatch.setattr(pipeline.mux, "embed", boom)
    job = job_factory(video)
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = pipeline.manager._embed_subtitle(
        job, job.request, workdir, video, ".srt",
        subtitle.build_srt(LINES, SubtitleSettings()),
    )

    assert out is None  # the caller writes the subtitle file itself
    assert job.status.video_filename == ""
    assert any("磁盘已满" in e.log for e in job.events)


def test_a_cancelled_mux_still_cancels_the_job(video, tmp_path, job_factory):
    """Cancellation must not be swallowed by the fallback — a user who
    pressed cancel does not want a subtitle file appearing instead."""
    from app.services.pipeline import manager

    job = job_factory(video)
    job.cancel_event.set()
    workdir = tmp_path / "work"
    workdir.mkdir()
    with pytest.raises(InterruptedError):
        manager._embed_subtitle(
            job, job.request, workdir, video, ".srt",
            subtitle.build_srt(LINES, SubtitleSettings()),
        )
    assert not list(video.parent.glob("*.zh.mkv"))


# ------------------------------------------------------------ time origin


def _video_starting_at(path: Path, offset: float) -> Path:
    """A source whose timeline starts late, as TS captures and edit lists do."""
    import numpy as np

    with av.open(str(path), "w") as c:
        v = c.add_stream("libx264", rate=8)
        v.width, v.height = 64, 48
        v.pix_fmt = "yuv420p"
        packets = []
        for i in range(16):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts = i
            packets += list(v.encode(frame))
        packets += list(v.encode(None))
        for p in packets:
            shift = round(offset / float(p.time_base))
            p.pts += shift
            p.dts += shift
            c.mux(p)
    return path


def test_a_source_that_starts_late_does_not_drift_the_subtitles(tmp_path):
    """The transcript is timed from audio decoded from the first frame, so
    it always starts at zero. A container whose own clock starts at 10s
    would put every cue ten seconds early if copied as-is."""
    source = _video_starting_at(tmp_path / "late.mkv", 10.0)
    with av.open(str(source)) as c:
        assert c.start_time / av.time_base == pytest.approx(10.0)

    out = mux.embed(source, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文")

    with av.open(str(out)) as c:
        first_video = min(
            float(p.pts * p.time_base)
            for p in c.demux(c.streams.video[0]) if p.dts is not None
        )
    assert first_video < 0.5  # pulled back to the subtitles' origin
    with av.open(str(out)) as c:
        cues = [
            float(p.pts * p.time_base)
            for p in c.demux(c.streams.subtitles[0]) if p.dts is not None
        ]
    assert [round(t, 2) for t in cues] == [0.5, 1.4]  # unchanged


# ------------------------------------------------ what real films look like


def test_not_a_single_frame_is_lost_from_a_source_with_b_frames(tmp_path):
    """The bug this test exists for: the remux idiom `if packet.dts is None:
    continue` throws away real content. Matroska stores no DTS at all, so its
    demuxer returns packets with dts unset — and the first of them is the
    opening keyframe. Measured on a real release: 24 of 1104 frames gone and
    the film opening on an undecodable GOP. Every source without B-frames
    hides this, which is why the earlier tests all passed."""
    source = make_bframe_video(tmp_path / "bframes.mkv")
    expected = _frames(source)
    assert expected == 48

    out = mux.embed(source, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文")
    assert _frames(out) == expected


@pytest.mark.parametrize("styled", [False, True])
def test_both_subtitle_formats_ride_along_intact(tmp_path, styled):
    source = make_bframe_video(tmp_path / "bframes.mkv")
    out = mux.embed(
        source, _write_subs(tmp_path, styled=styled),
        tmp_path / f"out_{styled}.mkv", "简体中文",
    )
    assert _frames(out) == 48
    with av.open(str(out)) as c:
        track = c.streams.subtitles[0]
        assert track.codec_context.name in (("ass", "ssa") if styled else ("srt",))
        cues = [bytes(p).decode("utf-8") for p in c.demux(track) if p.size]
    assert len(cues) == len(LINES)
    assert "你好" in cues[0] and "再见" in cues[1]


def test_a_codec_mkv_cannot_hold_fails_loudly(tmp_path, monkeypatch):
    """PyAV asks libavformat at normal compliance, which turns down a few
    old codecs (msmpeg4v2, wmv, vc1, adpcm) that MKV can technically carry.
    Skipping the stream produced a file with no picture at all — the job
    must fail here instead, so the caller writes a subtitle file."""
    source = make_bframe_video(tmp_path / "bframes.mkv")
    real = av.container.OutputContainer.add_stream_from_template

    def refuse_video(self, template, *args, **kwargs):
        if template.type == "video":
            raise ValueError("'matroska' format does not support 'msmpeg4v2' codec")
        return real(self, template, *args, **kwargs)

    monkeypatch.setattr(
        av.container.OutputContainer, "add_stream_from_template", refuse_video
    )
    out = tmp_path / "out.mkv"
    with pytest.raises(ValueError, match="无法容纳"):
        mux.embed(source, _write_subs(tmp_path), out, "简体中文")
    assert not out.exists() and list(tmp_path.glob("*.part")) == []


def test_only_our_subtitle_track_is_the_default_one(video, tmp_path):
    """A release often flags its own subtitle default. Leaving that in place
    means the player may show theirs instead of the translation just made."""
    first = mux.embed(video, _write_subs(tmp_path), tmp_path / "one.mkv", "English")
    second = mux.embed(first, _write_subs(tmp_path), tmp_path / "two.mkv", "简体中文")

    with av.open(str(second)) as c:
        defaults = [
            s.metadata.get("language")
            for s in c.streams
            if s.type == "subtitle" and s.disposition & av.stream.Disposition.default
        ]
    assert defaults == ["chi"]

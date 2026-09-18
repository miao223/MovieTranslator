"""Embedding the subtitle as a soft track in a new video.

The promise is narrow and checkable: the picture and sound come through
untouched, the subtitle plays without a sidecar file, and nothing about
the source file changes. The tests exist to hold that line — above all
that no re-encode ever creeps in.
"""

from fractions import Fraction
from pathlib import Path

import av
import pytest

from app.models.schemas import EmbedSettings, SubtitleLine, SubtitleSettings
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


def test_a_target_language_typed_in_by_hand_is_still_understood(tmp_path):
    """界面上的目标语言可以手输，所以名单外的值不等于「不知道」。

    一个直接写下的语言代码既是他要的语言、又正好是这套后缀的形状，认下来
    比退回 sub/und 有用得多；认不出的写法照旧退回，绝不猜。
    """
    assert mux.language_of("Português") == ("pt", "por")
    assert mux.language_of("pt") == ("pt", "por")
    assert mux.language_of("  NL  ") == ("nl", "dut")
    assert mux.language_of("Esperanto") == mux.FALLBACK
    assert mux.language_of("") == mux.FALLBACK
    assert mux.output_path(tmp_path / "film.mkv", "Italiano").name == "film.it.mkv"


def test_the_source_language_names_the_track_and_the_file():
    """纯原文的产物是片子自己的语言，那不在 LANGUAGES 那七个目标里。"""
    assert mux.source_language_of("ja") == ("ja", "jpn", "日语字幕")
    assert mux.source_language_of(" JA ") == ("ja", "jpn", "日语字幕")
    # 判不出来就说判不出来：标着 und 的轨道比标着 eng 的日语轨道有用得多
    assert mux.source_language_of("") == ("orig", "und", "原文字幕")
    # 认得的语言给中文名，哪怕它不在「可选目标语言」那张表里
    assert mux.source_language_of("sv") == ("sv", "swe", "瑞典语字幕")
    # 彻底表外的语言仍然说真话，而不是退回一句「原文」
    assert mux.source_language_of("cy") == ("cy", "cy", "cy字幕")


def test_a_given_suffix_overrides_the_target_language(tmp_path):
    """纯原文按源语言命名，与目标语言无关。"""
    out = mux.output_path(tmp_path / "film.mkv", "简体中文", "mkv", suffix="ja")
    assert out == tmp_path / "film.ja.mkv"


def test_the_overriding_suffix_still_refuses_to_overwrite(tmp_path):
    """防覆盖的循环对新参数同样生效——它防的是用户的片库。"""
    (tmp_path / "film.ja.mkv").write_bytes(b"someone's video")
    assert mux.output_path(
        tmp_path / "film.mkv", "简体中文", "mkv", suffix="ja"
    ).name == "film.ja.2.mkv"


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


def test_the_track_is_tagged_with_the_language_it_is_actually_in(video, tmp_path):
    """纯原文的轨道装的是原文，标签必须跟着原文走而不是翻译目标——
    一条内容是日语、标签写 chi 的轨道，播放器和媒体库都会选错。"""
    out = mux.embed(
        video, _write_subs(tmp_path, mode="original_only"), tmp_path / "out.mkv",
        "简体中文", track_title="日语字幕", track_language="jpn",
    )
    with av.open(str(out)) as c:
        track = c.streams.subtitles[0]
        assert track.metadata["language"] == "jpn"
        assert track.metadata["title"] == "日语字幕"


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
    and its attached fonts have to come through.

    Counting streams is not enough: the first track has to still say what it
    said, which is exactly what services/subsource.py can now check.
    """
    from app.services import subsource

    source = tmp_path / "withsubs.mkv"
    make_multitrack_video(source)
    with_extra = tmp_path / "extra.mkv"
    existing = _write_subs(tmp_path)
    mux.embed(source, existing, with_extra, "English")  # first pass adds one

    out = mux.embed(with_extra, existing, tmp_path / "twice.mkv", "简体中文")
    kinds = [t for t, _, _ in _streams(out)]
    assert kinds.count("subtitle") == 2

    tracks = subsource.list_tracks(out)
    assert [t["language"] for t in tracks] == ["eng", "chi"]
    first = subsource.read_cues(out, tracks[0])
    # the two display lines of a bilingual cue come back as one line, which
    # is what the LLM protocol wants; the words and times are untouched
    assert [l.text for l in first] == ["Hello there 你好", "Goodbye 再见"]
    assert [round(l.start, 2) for l in first] == [0.5, 1.4]


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


def test_the_pure_original_stage_names_everything_after_the_source_language(
    video, tmp_path, job_factory
):
    """detected 一个值，三个落点：文件名后缀、轨道标签、轨道标题。

    片名.zh.mkv 装着日语原文是彻头彻尾的谎；而工作目录里那份字幕也要带后缀，
    因为它就是「下载字幕」按钮吐给用户的文件名。
    """
    from app.services.pipeline import manager

    job = job_factory(video, target_language="简体中文", output_mode="original_only")
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = manager._embed_subtitle(
        job, job.request, workdir, video, ".srt",
        subtitle.build_srt(LINES, SubtitleSettings(), mode="original_only"),
        detected="ja",
    )

    assert out == video.parent / "film.ja.mkv" and out.is_file()
    assert job.srt_path == workdir / "film.ja.srt"
    with av.open(str(out)) as c:
        track = c.streams.subtitles[0]
        assert track.metadata["language"] == "jpn"
        assert track.metadata["title"] == "日语字幕"


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


# --------------------------------------------------------------------------
# Re-encoding and the second container. Everything above this line describes
# the default path, which must keep behaving exactly as it did; everything
# below only happens when EmbedSettings asks for it.
# --------------------------------------------------------------------------


def _vfr_video(path: Path, pts_ms=(0, 10, 12, 40, 41, 100, 101, 102, 200, 300, 400)) -> Path:
    """A source whose frames are not evenly spaced.

    Constant-rate material cannot detect a wrong encoder time base, because
    quantising to 1/fps maps such timestamps onto themselves. Only irregular
    spacing shows the damage.
    """
    import numpy as np

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height = 128, 96
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.options = {"bf": "2", "g": "8"}
        for i, when in enumerate(pts_ms):
            img = np.zeros((96, 128, 3), dtype=np.uint8)
            img[:, (i * 7) % 128:((i * 7) % 128) + 16] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = when, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def _frame_pts(path: Path) -> list:
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        return sorted(
            round(float(f.pts * stream.time_base) * 1000)
            for p in c.demux(stream) for f in p.decode()
        )


def _cue_texts(path: Path) -> list:
    """The cues as a player would read them, for either subtitle carrier."""
    with av.open(str(path)) as c:
        track = c.streams.subtitles[0]
        tx3g = track.codec_context.name == "mov_text"
        out = []
        for p in c.demux(track):
            if not p.size:
                continue
            raw = bytes(p)
            # tx3g: a big-endian uint16 length, then the text. The MP4 muxer
            # inserts empty samples of its own to fill the gaps between cues.
            body = raw[2:].decode("utf-8") if tx3g else raw.decode("utf-8")
            if body:
                out.append((round(float(p.pts * p.time_base), 2), body))
    return out


def test_re_encoding_loses_not_a_single_frame(tmp_path):
    """The trap this guards is that the copy path's `not packet.size` filter
    looks exactly like the right thing to reuse. It is not: that empty packet
    is what makes the decoder release the frames it is holding for B-frame
    reorder, and skipping it silently shortened the film."""
    source = make_bframe_video(tmp_path / "bframes.mkv")
    assert _frames(source) == 48
    out = mux.embed(
        source, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文",
        opts=EmbedSettings(video_codec="libx264", preset="ultrafast"),
    )
    assert _frames(out) == 48


def test_a_variable_rate_source_keeps_every_timestamp(tmp_path):
    """Guards the encoder's time base.

    Setting it on the stream rather than the codec context reads back as if
    it worked, but the encoder never looks there and derives 1/fps instead.
    Measured before the fix: 0,10,12,40,41,100… came back 0,0,0,42,42,83…,
    which slides the audio and every cue with it.
    """
    source = _vfr_video(tmp_path / "vfr.mkv")
    before = _frame_pts(source)
    out = mux.embed(
        source, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文",
        opts=EmbedSettings(video_codec="libx264", preset="ultrafast"),
    )
    assert _frame_pts(out) == before


def test_a_ten_bit_source_is_not_quietly_flattened_to_eight(tmp_path):
    """A new stream defaults to yuv420p, so saying nothing costs the depth."""
    import numpy as np

    source = tmp_path / "deep.mkv"
    with av.open(str(source), "w") as c:
        stream = c.add_stream("libx264", rate=24)
        stream.width, stream.height = 128, 96
        stream.pix_fmt = "yuv420p10le"
        for i in range(12):
            img = np.zeros((96, 128, 3), dtype=np.uint8)
            img[:, i * 8:i * 8 + 16] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame = frame.reformat(format="yuv420p10le")
            frame.pts = i
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode(None):
            c.mux(packet)

    out = mux.embed(
        source, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文",
        opts=EmbedSettings(video_codec="libx265", preset="ultrafast"),
    )
    with av.open(str(out)) as c:
        assert c.streams.video[0].codec_context.pix_fmt == "yuv420p10le"


def test_re_encoding_the_picture_leaves_the_sound_alone(video, tmp_path):
    """Only the video was asked for, so the audio must still be a copy —
    language tags and all, which is what multi-track releases are picked by."""
    out = mux.embed(
        video, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文",
        opts=EmbedSettings(video_codec="libx264", preset="ultrafast"),
    )
    kinds = [(t, c) for t, c, _ in _streams(out)]
    assert kinds[0] == ("video", "h264")  # re-encoded, but still h264
    assert [c for t, c in kinds if t == "audio"] == ["aac", "aac"]
    langs = [m.get("language") for t, _, m in _streams(out) if t == "audio"]
    assert langs == ["jpn", "eng"]


def test_a_cover_image_is_copied_rather_than_re_encoded(tmp_path):
    """Releases carry a poster as a second video stream flagged attached_pic.
    Re-encoding that as if it were the film is both pointless and wrong."""
    import numpy as np

    source = tmp_path / "withcover.mkv"
    with av.open(str(source), "w") as c:
        main = c.add_stream("libx264", rate=24)
        main.width, main.height = 128, 96
        main.pix_fmt = "yuv420p"
        cover = c.add_stream("mjpeg", rate=1)
        cover.width, cover.height = 32, 32
        cover.pix_fmt = "yuvj420p"
        cover.disposition = av.stream.Disposition.attached_pic
        for i in range(12):
            img = np.zeros((96, 128, 3), dtype=np.uint8)
            img[:, i * 8:i * 8 + 16] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in main.encode(frame):
                c.mux(packet)
        art = av.VideoFrame.from_ndarray(
            np.full((32, 32, 3), 200, dtype=np.uint8), format="rgb24")
        art.pts = 0
        for packet in cover.encode(art):
            c.mux(packet)
        for packet in main.encode(None):
            c.mux(packet)
        for packet in cover.encode(None):
            c.mux(packet)

    out = mux.embed(
        source, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文",
        opts=EmbedSettings(video_codec="libx265", preset="ultrafast"),
    )
    codecs = [c for t, c, _ in _streams(out) if t == "video"]
    assert codecs == ["hevc", "mjpeg"]  # the film re-encoded, the poster not


def test_mp4_carries_the_subtitle_as_a_mov_text_track(video, tmp_path):
    """MP4 accepts neither SRT nor ASS, and PyAV cannot encode subtitles at
    all, so the track is tx3g built by hand. What matters is that a player
    reads back the same text at the same times."""
    out = mux.embed(
        video, _write_subs(tmp_path), tmp_path / "out.mp4", "简体中文",
        opts=EmbedSettings(container="mp4"),
    )
    with av.open(str(out)) as c:
        assert c.streams.subtitles[0].codec_context.name == "mov_text"
    assert _cue_texts(out) == [(0.5, "Hello there\n你好"), (1.4, "Goodbye\n再见")]


def test_mp4_keeps_the_two_lines_of_a_styled_cue_apart(video, tmp_path):
    """The ASS the styled path emits carries override tags and \\N breaks.
    Stripping those must not also join the original and its translation —
    that is exactly what subsource's own cleaner does for the LLM."""
    out = mux.embed(
        video, _write_subs(tmp_path, styled=True), tmp_path / "out.mp4", "简体中文",
        opts=EmbedSettings(container="mp4"),
    )
    assert _cue_texts(out) == [(0.5, "Hello there\n你好"), (1.4, "Goodbye\n再见")]


def test_mp4_keeps_the_lyric_marks(video, tmp_path):
    """♪ is ordinary UTF-8 and must survive into the tx3g payload."""
    lines = [SubtitleLine(index=1, start=0.5, end=1.2,
                          text="♪ Crawl out ♪", translation="♪ 爬出来 ♪")]
    path = tmp_path / "lyric.srt"
    path.write_text(subtitle.build_srt(lines, SubtitleSettings()), encoding="utf-8")
    out = mux.embed(video, path, tmp_path / "out.mp4", "简体中文",
                    opts=EmbedSettings(container="mp4"))
    assert _cue_texts(out) == [(0.5, "♪ Crawl out ♪\n♪ 爬出来 ♪")]


def test_mp4_loses_no_frames_either(tmp_path):
    """B-frames mean the first DTS is negative; MP4 stores that in an edit
    list where matroska could not, but the packets still have to all arrive."""
    source = make_bframe_video(tmp_path / "bframes.mkv")
    out = mux.embed(source, _write_subs(tmp_path), tmp_path / "out.mp4", "简体中文",
                    opts=EmbedSettings(container="mp4"))
    assert _frames(out) == 48


def test_mp4_says_what_it_had_to_leave_behind(video, tmp_path):
    """A source subtitle track has nowhere to go in an MP4. Dropping it is
    the only option; dropping it silently is not."""
    first = mux.embed(video, _write_subs(tmp_path), tmp_path / "one.mkv", "English")
    notes = []
    out = mux.embed(first, _write_subs(tmp_path), tmp_path / "two.mp4", "简体中文",
                    opts=EmbedSettings(container="mp4"), log=notes.append)
    assert any("无法容纳" in n for n in notes)
    # theirs is gone, ours is the only subtitle left
    subs = [(c, m.get("language")) for t, c, m in _streams(out) if t == "subtitle"]
    assert subs == [("mov_text", "chi")]


@pytest.mark.parametrize("codec,expected", [
    ("libx264", "h264"),
    ("libx265", "hevc"),
    # AV1 reads back under the decoder's name, not the encoder's
    ("libsvtav1", "libdav1d"),
])
def test_every_software_encoder_produces_a_playable_file(tmp_path, codec, expected):
    source = make_bframe_video(tmp_path / "bframes.mkv")
    out = mux.embed(
        source, _write_subs(tmp_path), tmp_path / f"out_{codec}.mkv", "简体中文",
        opts=EmbedSettings(video_codec=codec, preset="ultrafast", quality=40),
    )
    assert _frames(out) == 48
    with av.open(str(out)) as c:
        assert c.streams.video[0].codec_context.name == expected


@pytest.mark.skipif(
    any(e["id"] == "h264_nvenc" for e in mux.available_encoders()),
    reason="this machine really has NVENC",
)
def test_an_encoder_this_machine_lacks_fails_before_writing_anything(video, tmp_path):
    """A hardware encoder builds a stream quite happily on a machine with no
    such card — only avcodec_open2 fails. Opening it up front is what keeps
    the failure from landing after the header and some audio are written."""
    out = tmp_path / "out.mkv"
    with pytest.raises(ValueError, match="无法使用编码器"):
        mux.embed(video, _write_subs(tmp_path), out, "简体中文",
                  opts=EmbedSettings(video_codec="h264_nvenc"))
    assert not out.exists() and list(tmp_path.glob("*.part")) == []


def test_the_probe_finds_the_encoders_the_wheel_ships(tmp_path):
    ids = {e["id"] for e in mux.available_encoders()}
    assert {"libx264", "libx265", "libsvtav1"} <= ids
    # hardware entries are machine-dependent: assert only that any that do
    # show up are labelled as such, never that they are present or absent
    assert all(e["hardware"] == bool("nvenc" in e["id"] or "qsv" in e["id"]
                                     or "amf" in e["id"])
               for e in mux.available_encoders())


def test_the_output_is_named_after_the_container(tmp_path):
    video = tmp_path / "film.mkv"
    video.touch()
    assert mux.output_path(video, "简体中文").name == "film.zh.mkv"
    assert mux.output_path(video, "简体中文", "mp4").name == "film.zh.mp4"
    # and it still refuses to overwrite something already there
    (tmp_path / "film.zh.mp4").touch()
    assert mux.output_path(video, "简体中文", "mp4").name == "film.zh.2.mp4"


def test_the_audio_can_be_re_encoded_on_its_own(video, tmp_path):
    """Asking for an audio codec must not drag the picture into an encode."""
    out = mux.embed(
        video, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文",
        opts=EmbedSettings(audio_codec="flac"),
    )
    kinds = [(t, c) for t, c, _ in _streams(out)]
    assert kinds[0] == ("video", "h264")
    assert [c for t, c in kinds if t == "audio"] == ["flac", "flac"]
    langs = [m.get("language") for t, _, m in _streams(out) if t == "audio"]
    assert langs == ["jpn", "eng"]


def test_the_pipeline_stage_honours_the_container_choice(
    video, tmp_path, job_factory
):
    """The stage decides the output name, so the container has to reach it —
    otherwise the mp4 would be written under a .mkv name."""
    from app.services.pipeline import manager

    job = job_factory(video, target_language="简体中文",
                      embed={"container": "mp4"})
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = manager._embed_subtitle(
        job, job.request, workdir, video, ".srt",
        subtitle.build_srt(LINES, SubtitleSettings()),
    )
    assert out == video.parent / "film.zh.mp4" and out.is_file()
    assert _cue_texts(out) == [(0.5, "Hello there\n你好"), (1.4, "Goodbye\n再见")]


def test_the_encoder_list_is_served_to_the_page(tmp_path):
    from app.main import app
    from tests.conftest import local_client

    with local_client(app) as client:
        body = client.get("/api/media/encoders").json()
    assert body["video"][0]["id"] == "copy"  # always first, always available
    assert {"libx264", "libx265"} <= {e["id"] for e in body["video"]}
    assert [c["value"] for c in body["containers"]] == ["mkv", "mp4"]
    assert "aac" in {a["id"] for a in body["audio"]}


@pytest.mark.skipif(
    any(e["id"] == "h264_nvenc" for e in mux.available_encoders()),
    reason="this machine really has NVENC",
)
def test_a_job_asking_for_a_missing_encoder_is_refused_at_once(video, tmp_path):
    """Not at mux time: that runs after an hour of transcription and
    translation, and 'this machine has no such encoder' is knowable now."""
    from app.main import app
    from tests.conftest import local_client

    with local_client(app) as client:
        reply = client.post("/api/jobs", json={
            "video_path": str(video),
            "embed_subtitle": True,
            "embed": {"video_codec": "h264_nvenc"},
        })
    assert reply.status_code == 400
    assert "无法使用编码器" in reply.json()["detail"]


def test_an_audio_codec_the_container_cannot_hold_is_re_encoded_not_fatal(
    video, tmp_path, monkeypatch
):
    """MP4 has no room for DTS or TrueHD, and the bundled ffmpeg has no
    encoder for either, so copying such a track was never going to work.
    An hour of transcription and translation stands behind this step, so one
    audio track gets re-encoded rather than the whole job being lost.

    Faked, because a source with a codec this build cannot even encode
    cannot be built here — which is the same reason the fallback exists.
    """
    real = av.container.OutputContainer.add_stream_from_template

    def picky(self, template, *args, **kwargs):
        if template.type == "audio":
            raise ValueError("mp4 does not support 'dts' codec")
        return real(self, template, *args, **kwargs)

    monkeypatch.setattr(
        av.container.OutputContainer, "add_stream_from_template", picky)
    notes = []
    out = mux.embed(video, _write_subs(tmp_path), tmp_path / "out.mkv", "简体中文",
                    log=notes.append)

    assert [c for t, c, _ in _streams(out) if t == "audio"] == ["aac", "aac"]
    assert [m.get("language") for t, _, m in _streams(out) if t == "audio"] \
        == ["jpn", "eng"]
    assert sum("已转为 AAC" in n for n in notes) == 2

"""压制 (re-encoding a video): the engine in services/encode.py.

What it promises, and what these tests hold it to: every frame arrives,
timestamps are the source's own, the picture keeps its shape, audio is
re-encoded only where that buys space (and fitted to what the encoder
takes), everything else — chapters, fonts, subtitles, languages — comes
along, and the lossless file a pipeline remuxed is only ever replaced by
an encode that checked out.
"""

import os
import struct
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from app.models.schemas import EncodeOptions, EncodeRequest
from app.services import encode, mux
from tests.mediagen import make_avi, make_source


def run(source: Path, **options) -> tuple[Path, encode.Result, list]:
    """Encode *source* the way the pipeline does, and return the file."""
    request = EncodeRequest(source=str(source), options=EncodeOptions(
        **{"preset": "ultrafast", **options}))
    target = encode.output_target(source, request)
    part = target.with_name(target.name + ".part")
    lines: list = []
    result = encode.encode_file(source, part, request, log=lines.append)
    problems = encode.verify(part, result)
    assert problems == [], problems
    final, why = encode.finalize(part, source, target, request, None)
    assert why == ""
    return final, result, lines


def frames_of(path: Path) -> int:
    with av.open(str(path)) as c:
        return sum(len(p.decode()) for p in c.demux(c.streams.video[0]))


def pts_of(path: Path) -> list:
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        return sorted(round(float(p.pts * p.time_base), 3)
                      for p in c.demux(v) if p.size and p.pts is not None)


# ------------------------------------------------------------------ picture


def test_the_default_is_hevc_in_10bit_and_every_frame_arrives(tmp_path):
    source = make_source(tmp_path / "film.mkv", frames=30,
                         audio=[{"codec": "aac", "language": "jpn", "title": "日本語"}])
    out, result, lines = run(source)
    assert out.name == "film.HEVC.mkv"
    with av.open(str(out)) as c:
        video = c.streams.video[0]
        assert video.codec_context.name == "hevc"
        assert video.codec_context.pix_fmt == "yuv420p10le"
        # the AAC track is lossy: copied as it is, language and title intact
        a = c.streams.audio[0]
        assert a.codec_context.name == "aac"
        assert (a.metadata["language"], a.metadata["title"]) == ("jpn", "日本語")
    assert frames_of(out) == frames_of(source) == 30
    assert result.stats.frames_out == result.stats.video_packets == 30
    assert encode.read_tags(out)["encode"]["source"] == "film.mkv"
    assert any("原样复制" in line for line in lines)


def test_variable_frame_timing_survives_the_filter_graph(tmp_path):
    """The deinterlace/scale graph hands frames back in another time base;
    the encoder must still land every frame on its own instant."""
    tb = Fraction(1, 1000)
    when = [0, 10, 12, 40, 41, 100, 101, 102, 200, 300, 400]
    source = make_source(tmp_path / "vfr.mkv", frames=len(when), pts=when, time_base=tb)
    out, _, _ = run(source, video_codec="libx264")   # auto: bwdif is in the graph
    assert pts_of(out) == pts_of(source)
    big = make_source(tmp_path / "vfr720.mkv", frames=len(when), pts=when, time_base=tb,
                      size=(1280, 720))
    scaled, _, _ = run(big, video_codec="libx264", max_height=480)
    assert pts_of(scaled) == pts_of(big)
    with av.open(str(scaled)) as c:
        assert (c.streams.video[0].codec_context.width,
                c.streams.video[0].codec_context.height) == (854, 480)


def _decoded(path: Path) -> list:
    with av.open(str(path)) as c:
        return list(c.decode(c.streams.video[0]))


def test_auto_deinterlace_leaves_progressive_frames_alone_and_mends_flagged_ones(tmp_path):
    progressive = _decoded(make_source(tmp_path / "p.mkv", frames=12))
    interlaced = _decoded(make_source(tmp_path / "i.mkv", frames=12, interlaced=True))
    assert not any(f.interlaced_frame for f in progressive)
    assert all(f.interlaced_frame for f in interlaced)
    chain = [("bwdif", "mode=send_frame:parity=auto:deint=interlaced")]

    def through(frames, path):
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            filt = encode.VideoFilter(stream, list(chain), Fraction(1), Fraction(24))
            out = []
            for f in frames:
                out += filt.push(f)
            return out + filt.flush()

    kept = through(progressive, tmp_path / "p.mkv")
    mended = through(interlaced, tmp_path / "i.mkv")
    assert len(kept) == len(progressive) and len(mended) == len(interlaced)
    assert all(np.array_equal(a.to_ndarray(), b.to_ndarray()) for a, b in zip(progressive, kept))
    assert any(not np.array_equal(a.to_ndarray(), b.to_ndarray())
               for a, b in zip(interlaced, mended))


def test_ivtc_turns_thirty_frames_into_twenty_four_from_the_same_start(tmp_path):
    tb = Fraction(1001, 30000)
    source = make_source(tmp_path / "ntsc.mkv", frames=60, rate=Fraction(30000, 1001),
                         interlaced=True, time_base=tb, pts=list(range(60)))
    out, result, lines = run(source, video_codec="libx264", deinterlace="ivtc")
    assert any("反胶片过带" in line for line in lines)
    assert result.stats.frames_in == 60
    assert result.stats.frames_out == 48          # 5 frames in, 4 out
    assert pts_of(out)[0] == pts_of(source)[0]     # decimate's lag taken back
    with av.open(str(out)) as c:
        assert abs(float(c.streams.video[0].average_rate) - 24000 / 1001) < 0.01


def test_ivtc_is_not_run_where_it_would_delete_real_frames():
    """Soft telecine decodes as progressive frames with uneven steps; a PAL
    source is not pulldown at all. Both fall back to plain deinterlacing."""
    class Stream:
        average_rate = Fraction(30000, 1001)
        guessed_rate = None
    notes: list = []
    chain, per_frame = encode._deinterlace_chain(
        EncodeOptions(deinterlace="ivtc"), Stream(),
        encode.VideoSample(frames=90, interlaced=0, regular=False), notes.append)
    assert per_frame == 1 and chain[0][0] == "bwdif" and "步长不均" in notes[0]
    Stream.average_rate = Fraction(25)
    chain, per_frame = encode._deinterlace_chain(
        EncodeOptions(deinterlace="ivtc"), Stream(), encode.VideoSample(), notes.append)
    assert per_frame == 1 and "29.97" in notes[1]


def test_bit_depth_follows_the_format_and_what_the_encoder_can_do():
    assert encode.pixel_format("libx265", EncodeOptions(video_codec="libx265"), False)[0] \
        == "yuv420p10le"
    assert encode.pixel_format("libx264", EncodeOptions(video_codec="libx264"), False)[0] \
        == "yuv420p"   # auto: H.264 High 10 barely plays on hardware
    fmt, note = encode.pixel_format(
        "libvpx-vp9", EncodeOptions(video_codec="libvpx-vp9", bit_depth="10"), False)
    assert fmt == "yuv420p" and "10bit" in note   # this build's libvpx has none
    with pytest.raises(ValueError, match="10bit"):
        encode.check_options(EncodeOptions(video_codec="libvpx-vp9", bit_depth="10"))
    with pytest.raises(encode.EncodeError, match="HDR"):
        encode.pixel_format("libvpx-vp9", EncodeOptions(video_codec="libvpx-vp9"), True)


def test_x265_refuses_the_film_tuning_so_the_form_never_offers_it():
    caps = {c["id"]: c for c in encode.capabilities()}
    assert caps["libx265"]["tunes"] == ["animation", "grain"]
    assert "film" in caps["libx264"]["tunes"]
    assert caps["libx265"]["ten_bit"] and not caps["libvpx-vp9"]["ten_bit"]
    assert caps["libsvtav1"]["quality"]["max"] == 63
    with pytest.raises(ValueError, match="film"):
        encode.check_options(EncodeOptions(video_codec="libx265", tune="film"))


def test_geometry_keeps_the_picture_s_shape():
    g = encode.geometry
    assert g(1920, 1080, 1, EncodeOptions(max_height=720)) == (1280, 720, 1)
    assert g(1280, 720, 1, EncodeOptions(max_height=1080)) == (1280, 720, 1)   # never up
    assert g(1080, 1920, 1, EncodeOptions(max_height=720)) == (720, 1280, 1)   # portrait
    # anamorphic DVD: same shape, same sample aspect ratio
    assert g(720, 576, Fraction(16, 15), EncodeOptions(max_height=480)) \
        == (600, 480, Fraction(16, 15))
    # AV1 in MKV cannot keep the ratio: square pixels instead
    assert g(720, 480, Fraction(32, 27), EncodeOptions(video_codec="libsvtav1")) \
        == (854, 480, 1)
    assert g(721, 481, 1, EncodeOptions())[:2] == (720, 480)                    # even sides


def test_an_anamorphic_picture_is_shown_the_same_after_the_encode(tmp_path):
    source = make_source(tmp_path / "dvd.mkv", size=(144, 96), sar=Fraction(32, 27), frames=8)
    hevc, _, _ = run(source, video_codec="libx265")
    with av.open(str(hevc)) as c:
        v = c.streams.video[0]
        assert (v.codec_context.width, v.sample_aspect_ratio) == (144, Fraction(32, 27))
    av1, _, _ = run(source, video_codec="libsvtav1")
    with av.open(str(av1)) as c:
        assert c.streams.video[0].codec_context.width == 170   # 144 * 32/27, even


# -------------------------------------------------------------------- audio


def _audio(path: Path) -> list:
    with av.open(str(path)) as c:
        return [(s.codec_context.name, s.codec_context.layout.name,
                 s.codec_context.sample_rate, dict(s.metadata).get("language"),
                 dict(s.metadata).get("title"), int(s.disposition))
                for s in c.streams.audio]


def test_the_picture_can_stay_as_it_is_while_the_sound_shrinks(tmp_path):
    """画面原样: a remux with smaller audio — the picture bit for bit."""
    source = make_source(tmp_path / "keep.mkv", audio=[
        {"codec": "flac", "layout": "5.1(side)", "format": "s16"}])
    out, result, _ = run(source, video_codec="copy")
    assert out.name == "keep.remux.mkv"
    with av.open(str(source)) as a, av.open(str(out)) as b:
        src = [bytes(p) for p in a.demux(a.streams.video[0]) if p.size]
        dst = [bytes(p) for p in b.demux(b.streams.video[0]) if p.size]
        assert b.streams.audio[0].codec_context.name == "eac3"
    assert src == dst and not result.video_encoded


def test_a_cut_that_opens_mid_gop_can_keep_its_picture(tmp_path):
    """画面原样 on a file cut with a plain stream copy: it used to fail on the
    fifth packet (EINVAL). Now the two undecodable leading pictures are left
    out, said so, and verify() still passes — the copied count is what went
    into the file, and the missing frames are allowed for in the span."""
    from tests.mediagen import make_open_gop_cut

    source = make_open_gop_cut(tmp_path / "cut.mkv")
    out, result, lines = run(source, video_codec="copy")
    assert any("丢掉 2 个解不出来的前导帧" in line for line in lines)
    with av.open(str(source)) as a, av.open(str(out)) as b:
        src = [bytes(p) for p in a.demux(a.streams.video[0]) if p.size]
        dst = [bytes(p) for p in b.demux(b.streams.video[0]) if p.size]
    assert dst == [src[0]] + src[3:], "the keyframe, then everything after the two"
    assert result.stats.dropped == {0: 2}


def test_only_the_lossless_tracks_are_re_encoded(tmp_path):
    source = make_source(tmp_path / "bd.mkv", audio=[
        {"codec": "flac", "layout": "5.1(side)", "format": "s16", "language": "jpn",
         "title": "Japanese FLAC 5.1"},
        {"codec": "ac3", "layout": "5.1(side)", "language": "eng", "title": "Commentary"},
    ])
    out, result, _ = run(source)
    tracks = _audio(out)
    assert tracks[0][:3] == ("eac3", "5.1(side)", 48000)
    assert tracks[0][4] == "E-AC-3 5.1"         # the old title named the old codec
    assert tracks[1][0] == "ac3" and tracks[1][4] == "Commentary"
    # the lossy track is copied bit for bit
    with av.open(str(source)) as a, av.open(str(out)) as b:
        src = [bytes(p) for p in a.demux(a.streams.audio[1]) if p.size]
        dst = [bytes(p) for p in b.demux(b.streams.audio[1]) if p.size]
    assert src == dst
    assert result.stats.copied[2] == len(src)


def test_audio_is_fitted_to_what_each_encoder_takes(tmp_path):
    """Opus has no 44.1k and no (side) layouts; AC-3 / E-AC-3 stop at 5.1.
    Before, each of these failed the whole write at its first packet."""
    source = make_source(tmp_path / "a.mkv", audio=[
        {"codec": "flac", "layout": "5.1(side)", "rate": 44100, "format": "s16"},
        {"codec": "flac", "layout": "7.1", "format": "s16"},
    ])
    opus, _, _ = run(source, audio_codec="libopus")
    assert [t[:3] for t in _audio(opus)] == [("opus", "5.1", 48000), ("opus", "7.1", 48000)]
    eac3, _, lines = run(source, audio_codec="eac3")
    assert [t[:3] for t in _audio(eac3)] == [("eac3", "5.1(side)", 44100),
                                             ("eac3", "5.1(side)", 48000)]
    assert any("7.1 降为 5.1" in line for line in lines)


def test_a_24_bit_track_stays_24_bit_in_flac(tmp_path):
    source = make_source(tmp_path / "pcm.mkv", audio=[
        {"codec": "pcm_s24le", "layout": "stereo", "format": "s32"}])
    out, _, _ = run(source, audio_codec="flac")
    with av.open(str(out)) as c:
        cc = c.streams.audio[0].codec_context
        assert cc.name == "flac" and cc.format.name.startswith("s32")


def test_a_stereo_downmix_does_not_clip(tmp_path):
    source = make_source(tmp_path / "loud.mkv", audio=[
        {"codec": "flac", "layout": "5.1(side)", "format": "s16", "amp": 0.9}])
    out, _, _ = run(source, audio_codec="flac", audio_mixdown="stereo")
    with av.open(str(out)) as c:
        a = c.streams.audio[0]
        assert a.codec_context.layout.name == "stereo"
        peak = max(np.abs(f.to_ndarray()).max() for f in c.decode(a))
    assert peak <= 32767 * 0.95


def test_language_filters_keep_untagged_tracks_and_never_the_last_audio(tmp_path):
    source = make_source(tmp_path / "langs.mkv", audio=[
        {"codec": "aac", "language": "jpn"}, {"codec": "aac", "language": "eng"},
        {"codec": "aac", "language": "und"}])
    with av.open(str(source)) as c:
        plans, _ = encode.plan_streams(c, EncodeOptions(audio_languages=["ja"]))
        assert [p.action for p in plans if p.kind == "audio"] == ["copy", "drop", "copy"]
        plans, notes = encode.plan_streams(c, EncodeOptions(audio_languages=["fre"]))
        # jpn and eng go, but the untagged one is kept — not "nothing left"
        assert [p.action for p in plans if p.kind == "audio"] == ["drop", "drop", "copy"]
    only_tagged = make_source(tmp_path / "tagged.mkv", audio=[{"codec": "aac", "language": "jpn"}])
    with av.open(str(only_tagged)) as c:
        plans, notes = encode.plan_streams(c, EncodeOptions(audio_languages=["fre"]))
        assert [p.action for p in plans if p.kind == "audio"] == ["copy"]
        assert "一条音轨都不剩" in notes[0]


# ------------------------------------------------------------- the container


def test_chapters_move_with_the_timeline(tmp_path):
    source = make_source(tmp_path / "late.mkv", frames=48, offset=10.0,
                         chapters=[(10.0, 11.0), (11.0, 12.0)])
    out, result, _ = run(source)
    assert result.chapters == 2
    with av.open(str(out)) as c:
        starts = [round(float(ch["start"] * ch["time_base"]), 3) for ch in c.chapters()]
    assert starts == [0.0, 1.0]


def test_mkv_keeps_subtitles_and_fonts_and_mp4_says_it_cannot(tmp_path):
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,100 --> 00:00:00,500\nHello\n", encoding="utf-8")
    source = make_source(tmp_path / "rel.mkv", subtitle=srt, font=True,
                         audio=[{"codec": "aac"}])
    out, _, _ = run(source)
    with av.open(str(out)) as c:
        kinds = [s.type for s in c.streams]
    assert kinds.count("subtitle") == 1 and kinds.count("attachment") == 1
    mp4, _, lines = run(source, container="mp4")
    with av.open(str(mp4)) as c:
        assert [s.type for s in c.streams if s.type != "data"] == ["video", "audio"]
        assert c.streams.video[0].codec_context.codec_tag == "hvc1"
    assert any("MP4 装不下" in line for line in lines)
    assert encode.read_tags(mp4)["encode"] is not None


def test_a_cover_image_travels_as_a_copy(tmp_path):
    source = tmp_path / "cover.mkv"
    with av.open(str(source), "w") as c:
        main = c.add_stream("libx264", rate=24)
        main.width, main.height = 64, 48
        cover = c.add_stream("mjpeg", rate=1)
        cover.width, cover.height = 32, 32
        cover.pix_fmt = "yuvj420p"
        cover.disposition = av.stream.Disposition.attached_pic
        for i in range(8):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
            frame.pts = i
            for p in main.encode(frame):
                c.mux(p)
        art = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 200, np.uint8), format="rgb24")
        art.pts = 0
        for p in list(cover.encode(art)) + list(cover.encode(None)) + list(main.encode(None)):
            c.mux(p)
    out, _, _ = run(source)
    with av.open(str(out)) as c:
        assert [s.codec_context.name for s in c.streams.video] == ["hevc", "mjpeg"]


# ------------------------------------------------------------ failure modes


def test_cancelling_leaves_nothing_behind(tmp_path):
    source = make_source(tmp_path / "c.mkv", frames=48)
    request = EncodeRequest(source=str(source), options=EncodeOptions(preset="ultrafast"))
    target = encode.output_target(source, request)
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"stale")
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 5

    lines: list = []
    with pytest.raises(InterruptedError):
        encode.encode_file(source, part, request, log=lines.append, should_cancel=cancel)
    assert not part.exists() and not target.exists()
    assert "残留文件" in lines[0]


def test_one_undecodable_packet_is_counted_not_fatal():
    class Bad:
        def decode(self):
            raise av.error.InvalidDataError(1094995529, "Invalid data found")

    stats = encode.Stats()
    pipe = encode.AudioPipe(None, 0, None, stats, 3)
    assert pipe.feed(Bad()) == []
    assert stats.bad == {3: 1}
    stats.seen[3] = 100
    stats.bad[3] = encode.BAD_PACKET_FLOOR + 1
    with pytest.raises(encode.EncodeError, match="解码失败"):
        encode._check_bad(stats)


def test_verify_notices_a_missing_frame(tmp_path):
    source = make_source(tmp_path / "v.mkv")
    request = EncodeRequest(source=str(source), options=EncodeOptions(preset="ultrafast"))
    target = encode.output_target(source, request)
    part = target.with_name(target.name + ".part")
    result = encode.encode_file(source, part, request)
    assert encode.verify(part, result) == []
    result.stats.video_packets += 1
    result.stats.frames_out += 1
    assert any("画面应有" in p for p in encode.verify(part, result))


def test_an_encoder_this_machine_lacks_fails_before_anything_is_written(tmp_path):
    source = make_source(tmp_path / "n.mkv")
    with pytest.raises(ValueError, match="h264_nvenc"):
        encode.check_options(EncodeOptions(video_codec="h264_nvenc"))
    request = EncodeRequest(source=str(source), options=EncodeOptions(video_codec="h264_nvenc"))
    part = tmp_path / "n.out.part"
    with pytest.raises(ValueError):
        encode.encode_file(source, part, request)
    assert not part.exists()


def test_hdr10_metadata_is_spelled_the_way_each_encoder_takes_it():
    # R, G, B, white point as AVRationals, then min/max luminance, then flags
    prim = [(34000, 50000), (16000, 50000), (13250, 50000), (34500, 50000),
            (7500, 50000), (3000, 50000)]
    raw = struct.pack("<22i", *[x for pair in prim for x in pair],
                      15635, 50000, 16450, 50000, 1, 10000, 10000000, 10000, 1, 1)
    sample = encode.VideoSample(mastering=raw, light=struct.pack("<2I", 1000, 400))
    x265 = encode.hdr_params("libx265", sample, encode.PQ)
    assert x265 == ("hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)"
                    "WP(15635,16450)L(10000000,1):max-cll=1000,400")
    svt = encode.hdr_params("libsvtav1", sample, encode.PQ)
    assert svt.startswith("enable-hdr=1:mastering-display=G(0.2650,0.6900)")
    assert svt.endswith("content-light=1000,400")
    assert encode.hdr_params("libx265", sample, 1) == ""          # not PQ
    assert encode.hdr_params("hevc_nvenc", sample, encode.PQ) == ""


def test_dolby_vision_without_an_hdr10_base_is_refused(tmp_path, monkeypatch):
    source = make_source(tmp_path / "dv.mkv")
    monkeypatch.setattr(encode, "sample_video",
                        lambda *a, **k: encode.VideoSample(frames=1, dovi=True))
    request = EncodeRequest(source=str(source), options=EncodeOptions(preset="ultrafast"))
    part = tmp_path / "dv.out.part"
    with pytest.raises(encode.EncodeError, match="profile 5"):
        encode.encode_file(source, part, request)
    assert not part.exists()


# ------------------------------------------------------------ replacing


def _lossless(tmp_path: Path, name: str = "Film.E01.mkv", tagged: bool = True) -> Path:
    tags = {encode.REMUX_TAG: encode.remux_tag("/disc/Film", "00001.mpls")} if tagged else None
    return make_source(tmp_path / name, frames=12, tags=tags,
                       audio=[{"codec": "flac", "format": "s16", "language": "jpn"}])


def _replace(source: Path, **opts):
    request = EncodeRequest(source=str(source), replace_source=True,
                            options=EncodeOptions(preset="ultrafast", **opts))
    target = encode.output_target(source, request)
    part = target.with_name(target.name + ".part")
    before = source.stat()
    result = encode.encode_file(source, part, request)
    assert encode.verify(part, result) == []
    return request, target, part, before


def test_the_lossless_file_is_swapped_for_its_encode_under_the_same_name(tmp_path):
    source = _lossless(tmp_path)
    request, target, part, before = _replace(source)
    final, why = encode.finalize(part, source, target, request, before)
    assert (final, why) == (source, "")
    tags = encode.read_tags(source)
    assert tags["encode"] and tags["remux"]["title"] == "00001.mpls"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Film.E01.mkv"]
    # a re-run (the queue died before recording it) finds the work done
    assert encode.existing_encode(source, request) == source


def test_a_file_this_program_did_not_remux_is_never_replaced(tmp_path):
    source = _lossless(tmp_path, tagged=False)
    request, target, part, before = _replace(source)
    final, why = encode.finalize(part, source, target, request, before)
    assert final == target and "不是本程序封装" in why
    assert encode.read_tags(source)["encode"] is None
    assert target.exists()


def test_a_source_changed_during_the_encode_is_not_replaced(tmp_path):
    source = _lossless(tmp_path)
    request, target, part, before = _replace(source)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))
    final, why = encode.finalize(part, source, target, request, before)
    assert final == target and "改动" in why


def test_a_locked_lossless_file_keeps_both(tmp_path, monkeypatch):
    source = _lossless(tmp_path)
    request, target, part, before = _replace(source)
    real = os.replace

    def locked(src, dst):
        if Path(dst) == source:
            raise PermissionError(13, "in use")
        return real(src, dst)

    monkeypatch.setattr(encode.os, "replace", locked)
    final, why = encode.finalize(part, source, target, request, before, retries=1, wait=0)
    assert final == target and "占用" in why
    assert encode.read_tags(source)["encode"] is None


def test_a_rerun_of_a_kept_encode_finds_it_instead_of_writing_a_second(tmp_path):
    source = make_source(tmp_path / "k.mkv")
    out, _, _ = run(source)
    request = EncodeRequest(source=str(source), options=EncodeOptions(preset="ultrafast"))
    assert encode.existing_encode(source, request) == out
    other = EncodeRequest(source=str(source), options=EncodeOptions(preset="ultrafast", quality=30))
    assert encode.existing_encode(source, other) is None


def test_the_summary_line_says_what_will_happen():
    assert encode.describe_options(EncodeOptions()) == \
        "H.265 10bit（x265） · CRF 22 · medium · 无损音轨→E-AC-3"
    line = encode.describe_options(EncodeOptions(
        video_codec="libx264", rate_control="bitrate", bitrate_kbps=8000, max_height=720,
        deinterlace="ivtc", audio_codec="copy", container="mp4"))
    assert line == "H.264 8bit（x264） · 8000 kbps · medium · ≤720p · 反胶片过带 · 音频原样 · MP4"


def test_an_avi_with_b_frames_keeps_its_frame_rate_and_every_frame(tmp_path):
    """AVI stores no presentation times; the decoder hands H.264 B-frames
    back labelled in decode order (1,3,4,2,6,7,5…). Read as they are, the
    sampled rate came out as 6.56 fps on a real VHS capture and the encoder
    was fed timestamps that run backwards."""
    source = make_avi(tmp_path / "capture.avi", frames=30, rate=25)
    with av.open(str(source)) as c:
        labels = [f.pts for f in c.decode(c.streams.video[0])]
    assert labels != sorted(labels)                     # the defect is really there
    with av.open(str(source)) as c:
        clock = mux.FrameClock.of(c.streams.video[0])
        fixed = [clock(f) for f in c.decode(c.streams.video[0])]
    # every one in order, the decoder's flushed last frames (no dts) included
    assert fixed == sorted(fixed) and len(set(fixed)) == 30
    sample = encode.sample_video(source, 0)
    assert abs(1 / sample.step - 25) < 0.01 and sample.regular
    out, result, _lines = run(source, video_codec="libx264")
    assert result.stats.frames_out == frames_of(out) == 30
    times = pts_of(out)
    steps = {round(b - a, 3) for a, b in zip(times, times[1:])}
    assert steps == {0.04}
    with av.open(str(out)) as c:
        assert abs(float(c.streams.video[0].average_rate) - 25) < 0.01

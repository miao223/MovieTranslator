"""Writing disc titles as MKV: pieces stitched onto one timeline, nothing lost.

The streams are real — H.264 with B-frames, AC-3, LPCM, PGS, MPEG-2, VobSub
— encoded by tests/discgen.py with PyAV's own m2ts and DVD muxers, then
wrapped in hand-written MPLS/CLPI/IFO tables. Correctness is judged the way
test_mux.py judges it: by *decoded* frames and samples, not packet counts.
"""

from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np
import pytest

from app.services.disc import open_disc
from app.services.disc.remux import remux_title, vobsub_private
from tests import discgen as g
from tests.discgen import C, Item, P, S
from tests.test_mux import _frames

PID_STREAMS = (
    S("video", 0x1011, 0x1B, fmt=6, rate=1),
    S("audio", 0x1100, 0x81, "jpn", fmt=3, rate=1),
    S("pg", 0x1200, 0x90, "chi"),
)


def bd_disc(tmp_path, clips, playlist, *, streams=PID_STREAMS, name="Film", marks=None, ep=False):
    """clips: id -> discgen.Clip; playlist: [(clip id, in 45k, out 45k, connection)]."""
    items = [Item(c, a, b, connection=conn, streams=streams) for c, a, b, conn in playlist]
    pls = {"00800": g.mpls(items, marks=marks or [(1, i, a) for i, (_, a, _, _) in enumerate(playlist)])}
    info = {cid: g.clpi(c.start, c.end, streams=streams,
                        ep=c.keyframes if ep else (), packets=len(c.data) // 192)
            for cid, c in clips.items()}
    root = g.write_bd(tmp_path / name, pls, info, streams={k: c.data for k, c in clips.items()})
    return open_disc(str(root))


def decoded_samples(path: Path, index: int = 0) -> np.ndarray:
    with av.open(str(path)) as c:
        chunks = [f.to_ndarray() for f in c.decode(c.streams.audio[index])]
    return np.concatenate(chunks, axis=1)


def stream_langs(path: Path):
    with av.open(str(path)) as c:
        return [(s.type, s.codec_context.name, s.metadata.get("language", "")) for s in c.streams]


# ------------------------------------------------------------------ Blu-ray

def test_three_clips_become_one_continuous_film(tmp_path):
    clips = {cid: g.make_m2ts(2.0, start=start, cues=[(0.5, 1.5, "字幕")], tmp=tmp_path)
             for cid, start in (("00001", 0.0), ("00002", 600.0), ("00003", 30.0))}
    playlist = [(cid, c.start, c.end, 1 if n == 0 else 5)
                for n, (cid, c) in enumerate(clips.items())]
    disc = bd_disc(tmp_path, clips, playlist)
    title = disc.titles[0]
    logs = []
    result = remux_title(disc, title, tmp_path / "Film.mkv", log=logs.append)

    assert result.path.is_file() and not (tmp_path / "Film.mkv.part").exists()
    # every picture of every clip, decodable, in one file
    assert _frames(result.path) == sum(c.frames for c in clips.values())
    with av.open(str(result.path)) as out:
        assert abs(out.duration / av.time_base - title.duration) < 0.2
        chapters = out.chapters()
        video_pts = sorted(p.pts * p.time_base for p in out.demux(out.streams.video[0])
                           if p.pts is not None)
        out.seek(0)
        sub_pts = [float(p.pts * p.time_base) for p in out.demux(out.streams.subtitles[0])
                   if p.pts is not None]
    # the clip timelines (0 s, 600 s, 30 s) are gone: one smooth run
    gaps = [float(b - a) for a, b in zip(video_pts, video_pts[1:])]
    assert max(gaps) < 0.05
    # chapters sit where each clip starts on the new timeline
    starts = [c["start"] * c["time_base"] for c in chapters]
    assert [round(float(s)) for s in starts] == [0, 2, 4]
    # each clip's subtitle at 0.5 s into that clip, whatever its own clock said
    assert all(any(abs(t - (k * 2 + 0.5)) < 0.1 for t in sub_pts) for k in range(3))
    # languages come from the disc's stream table, not from the m2ts ("und"
    # is not stored by matroska at all, so the picture reads back empty)
    assert stream_langs(result.path) == [
        ("video", "h264", ""), ("audio", "ac3", "jpn"), ("subtitle", "pgssub", "chi")]
    assert result.dropped == 0


def test_a_play_item_that_takes_only_part_of_its_clip(tmp_path):
    """Seamless branching: a play item that starts mid-clip, found through
    the clip's EP map, and ends before the clip does."""
    clip = g.make_m2ts(4.0, tmp=tmp_path)
    # an in-point more than a second into the clip, so the EP map is used
    key = next(pts for pts, _spn in clip.keyframes if pts > clip.start + 45000 + 45000 // 4)
    playlist = [("00001", key, key + 45000 * 2 - 45000 // 24, 1)]
    disc = bd_disc(tmp_path, {"00001": clip}, playlist, ep=True)
    result = remux_title(disc, disc.titles[0], tmp_path / "part.mkv")
    frames = _frames(result.path)
    assert 44 <= frames <= 48          # two seconds of a 24 fps clip
    with av.open(str(result.path)) as out:
        assert abs(out.duration / av.time_base - 2.0) < 0.2


def test_lpcm_becomes_flac_without_losing_a_bit(tmp_path):
    clip = g.make_m2ts(2.0, audio="pcm_bluray", wide=True, tmp=tmp_path)
    streams = (S("video", 0x1011, 0x1B, fmt=6, rate=1), S("audio", 0x1100, 0x80, "eng", fmt=3, rate=1))
    disc = bd_disc(tmp_path, {"00001": clip}, [("00001", clip.start, clip.end, 1)], streams=streams)
    result = remux_title(disc, disc.titles[0], tmp_path / "lpcm.mkv")
    assert stream_langs(result.path)[1] == ("audio", "flac", "eng")
    with av.open(io.BytesIO(clip.data), format="mpegts") as src:
        original = np.concatenate([f.to_ndarray() for f in src.decode(src.streams.audio[0])], axis=1)
    converted = decoded_samples(result.path)
    n = min(original.shape[1], converted.shape[1])
    assert n > 48000 * 1.9
    assert np.array_equal(original[:, :n] >> 8, converted[:, :n] >> 8)

    pcm = remux_title(disc, disc.titles[0], tmp_path / "pcm.mkv", lpcm="pcm")
    assert stream_langs(pcm.path)[1][1] == "pcm_s24le"


def test_cancelling_leaves_nothing_behind(tmp_path):
    clip = g.make_m2ts(2.0, tmp=tmp_path)
    disc = bd_disc(tmp_path, {"00001": clip}, [("00001", clip.start, clip.end, 1)])
    calls = iter(range(10 ** 6))
    with pytest.raises(InterruptedError):
        remux_title(disc, disc.titles[0], tmp_path / "x.mkv",
                    should_cancel=lambda: next(calls) > 20)
    assert not (tmp_path / "x.mkv").exists() and not (tmp_path / "x.mkv.part").exists()


def test_a_leftover_part_file_is_removed_and_said(tmp_path):
    clip = g.make_m2ts(1.0, tmp=tmp_path)
    disc = bd_disc(tmp_path, {"00001": clip}, [("00001", clip.start, clip.end, 1)])
    (tmp_path / "x.mkv.part").write_bytes(b"stale")
    logs = []
    remux_title(disc, disc.titles[0], tmp_path / "x.mkv", log=logs.append)
    assert any("残留" in line for line in logs)


# ---------------------------------------------------------------------- DVD

def dvd_disc(tmp_path, vobs, cells, *, audio=((0, 2, "ja", 1),), subp=(("zh", 1),),
             subp_control=(0x80000000,), audio_control=(0x8000,), name="Film"):
    data = b"".join(v.data for v in vobs)
    pgc = P(cells, programs=(1,), audio=audio_control, subp=subp_control,
            palette=[0x108080, 0xEB8080] + [0x108080] * 14)
    root = g.write_dvd(tmp_path / name, g.vmg_ifo([(1, 1, 1, 1)]),
                       {1: g.vts_ifo([pgc], [[(1, 1)]], audio=audio, subp=subp,
                                     video=(1 << 14) | (1 << 12))},
                       vobs={1: [data]})
    return open_disc(str(root))


def test_a_dvd_title_across_a_timestamp_discontinuity(tmp_path):
    """Two VOB ids, each starting its clock again — the second cell's first
    picture carries a timestamp from before the first cell's last one."""
    a = g.make_vob(2.0, subtitles=[0.5])
    b = g.make_vob(2.0, subtitles=[1.0])
    sa, sb = len(a.data) // g.SECTOR, len(b.data) // g.SECTOR
    cells = [C(0, sa - 1, 2.0, vob_id=1), C(sa, sa + sb - 1, 2.0, vob_id=2, cell_id=1)]
    disc = dvd_disc(tmp_path, [a, b], cells, subp=(("zh", 9),))
    result = remux_title(disc, disc.titles[0], tmp_path / "dvd.mkv")
    assert _frames(result.path) == a.frames + b.frames
    with av.open(str(result.path)) as out:
        assert abs(out.duration / av.time_base - 4.0) < 0.3
        sub = out.streams.subtitles[0]
        assert b"palette: 000000, ffffff" in (sub.codec_context.extradata or b"")
        assert sub.disposition & av.stream.Disposition.forced
        times = [float(p.pts * p.time_base) for p in out.demux(sub) if p.pts is not None]
    assert len(times) == 2 and abs(times[1] - times[0] - 2.5) < 0.2
    assert stream_langs(result.path) == [
        ("video", "mpeg2video", ""), ("audio", "ac3", "jpn"), ("subtitle", "dvdsub", "chi")]


def test_a_vobsub_track_that_starts_late_is_still_written(tmp_path):
    """The first cell has no subtitle at all, so a demuxer opened on it does
    not know the stream exists; the IFO does, and the track is built anyway."""
    a = g.make_vob(2.0)
    b = g.make_vob(2.0, subtitles=[0.5])
    sa, sb = len(a.data) // g.SECTOR, len(b.data) // g.SECTOR
    cells = [C(0, sa - 1, 2.0, vob_id=1), C(sa, sa + sb - 1, 2.0, vob_id=2)]
    disc = dvd_disc(tmp_path, [a, b], cells)
    result = remux_title(disc, disc.titles[0], tmp_path / "late.mkv")
    with av.open(str(result.path)) as out:
        sub = out.streams.subtitles[0]
        assert b"palette:" in (sub.codec_context.extradata or b"")
        times = [float(p.pts * p.time_base) for p in out.demux(sub) if p.pts is not None]
    assert len(times) == 1 and abs(times[0] - 2.5) < 0.2


def test_dvd_lpcm_becomes_flac(tmp_path):
    a = g.make_vob(2.0, audio="pcm_dvd")
    sa = len(a.data) // g.SECTOR
    disc = dvd_disc(tmp_path, [a], [C(0, sa - 1, 2.0)], audio=((4, 2, "en", 1),),
                    subp=(), subp_control=())
    result = remux_title(disc, disc.titles[0], tmp_path / "lpcm.mkv")
    assert stream_langs(result.path)[1] == ("audio", "flac", "eng")
    with av.open(io.BytesIO(a.data), format="mpeg") as src:
        original = np.concatenate([f.to_ndarray() for f in src.decode(src.streams.audio[0])], axis=1)
    converted = decoded_samples(result.path)
    n = min(original.shape[1], converted.shape[1])
    assert n > 48000 * 1.9 and np.array_equal(original[:, :n], converted[:, :n])


def test_the_palette_is_converted_to_rgb():
    private = vobsub_private((720, 480), [0x108080, 0xEB8080, 0x51F05A])
    assert private.startswith(b"size: 720x480\npalette: 000000, ffffff, ")
    red = private.split(b", ")[2]      # BT.601 studio-range red
    assert int(red[:2], 16) >= 0xF0 and int(red[2:4], 16) < 0x10 and int(red[4:6], 16) < 0x10


def test_a_vobsub_track_first_met_mid_file_is_still_written(tmp_path, monkeypatch):
    """PyAV hands out packets only for the streams that existed when the
    file was opened, and a VOB announces nothing up front: the demuxer only
    knows the streams it met in its probe of the start and in the stretch of
    the end it reads to estimate the duration. A subtitle stream whose lines
    all lie between the two never appears at all. Measured on a real DVD
    (a 7 GB film): both tracks came out empty, 0 of 1034 and 0 of 1010,
    until each piece was opened with subtitle_primer.

    A small VOB hides this — its end-read covers most of the file — so the
    one subtitle sits mid-file, more than the ~250 KB end-read from the end,
    and the start probe is shrunk to match a film's proportions."""
    from app.services.disc import remux

    monkeypatch.setattr(remux, "PROBE", {"probesize": "200000", "analyzeduration": "1000000"})
    a = g.make_vob(40.0, subtitles=[15.0])
    n = len(a.data) // g.SECTOR
    disc = dvd_disc(tmp_path, [a], [C(0, n - 1, 40.0)])

    def sub_times(path):
        with av.open(str(path)) as out:
            sub = out.streams.subtitles[0]
            return [float(p.pts * p.time_base) for p in out.demux(sub) if p.pts is not None]

    result = remux_title(disc, disc.titles[0], tmp_path / "late.mkv")
    times = sub_times(result.path)
    assert len(times) == 1 and abs(times[0] - 15.0) < 0.3, times
    assert _frames(result.path) == a.frames

    # the reason the primer exists: without it, not one packet
    monkeypatch.setattr(remux, "subtitle_primer", lambda sids: b"")
    bare = remux_title(disc, disc.titles[0], tmp_path / "bare.mkv")
    assert sub_times(bare.path) == []

"""修复 (services/restore.py and its steps in services/encode.py).

What it promises: an interlaced recording comes back with a frame per field,
in the order the fields were shot (measured from the picture when the file
does not say), on a clock fine enough to hold them; crop, denoise and
enlarging happen in that order, to the size worked out up front; a model
plugged in between the filter graph and the encoder gets RGB frames and
hands back frames whose timestamps carry straight through; and none of it
changes what an encode without these options does or how it is named.
"""

from fractions import Fraction

import av
import numpy as np
import pytest

from app.models.schemas import EncodeOptions, EncodeRequest, RestoreSettings
from app.services import encode, restore
from tests.mediagen import bar_positions, make_cadence, make_fields, make_source
from tests.test_encode import frames_of, pts_of, run


def steps(positions: list) -> list:
    """Frame-to-frame moves of the bar, leaving out its wrap-around."""
    return [b - a for a, b in zip(positions, positions[1:]) if abs(b - a) < 40]


# ------------------------------------------------------------------- bob


def test_bob_gives_a_frame_per_field_in_the_order_they_were_shot(tmp_path):
    source = make_fields(tmp_path / "tape.mkv", frames=30)
    out, result, lines = run(source, video_codec="libx264", deinterlace="bob")
    assert out.name == "tape.60fps.H264.mkv"
    assert result.stats.frames_in == 30
    assert result.stats.frames_out == frames_of(out) == 60
    assert any("还原 60 帧" in line for line in lines)
    moves = steps(bar_positions(out))
    assert moves and all(m > 0 for m in moves), moves       # 0, 4, 8, 12…
    times = pts_of(out)
    assert len(set(times)) == 60 and times[0] == pts_of(source)[0]
    with av.open(str(out)) as c:
        assert abs(float(c.streams.video[0].average_rate) - 60000 / 1001) < 0.05


def test_bottom_field_first_comes_out_in_order_too(tmp_path):
    source = make_fields(tmp_path / "dv.mkv", frames=30, bottom_first=True)
    out, _result, lines = run(source, video_codec="libx264", deinterlace="bob")
    assert any("下场优先" in line for line in lines)
    assert all(m > 0 for m in steps(bar_positions(out)))


def test_an_unflagged_capture_has_its_field_order_measured(tmp_path):
    """No interlace flags: bwdif alone takes every frame as top-first, and
    a bottom-first capture in a codec with no field flags at all (the
    lossless AVI kind: Huffyuv, Lagarith) would play each pair of fields
    backwards. The picture says bottom-first."""
    source = make_fields(tmp_path / "capture.mkv", frames=30, flagged=False,
                         bottom_first=True)
    with av.open(str(source)) as c:
        assert not any(f.interlaced_frame for f in c.decode(c.streams.video[0]))
    out, _result, lines = run(source, video_codec="libx264", deinterlace="bob")
    assert any("从画面测得下场优先" in line for line in lines)
    assert all(m > 0 for m in steps(bar_positions(out)))


def test_a_field_order_given_by_hand_wins_and_a_wrong_one_is_called_out(tmp_path):
    source = make_fields(tmp_path / "tape.mkv", frames=30)
    out, _result, lines = run(source, video_codec="libx264", deinterlace="bob",
                              field_order="bff")
    assert any("请改选另一种" in line for line in lines)
    assert any(m < 0 for m in steps(bar_positions(out)))     # it jitters, as asked


def test_bob_fits_a_second_field_into_an_avi_capture_clock(tmp_path):
    """AVI counts time in frames (1001/30000): the second field of each
    frame has no tick of its own unless the encoder's clock is finer."""
    source = make_fields(tmp_path / "capture.avi", frames=30, fmt="avi")
    with av.open(str(source)) as c:
        assert c.streams.video[0].time_base == Fraction(1001, 30000)
    out, result, _lines = run(source, video_codec="libx264", deinterlace="bob")
    assert result.stats.frames_out == frames_of(out) == 60
    assert len(set(pts_of(out))) == 60


def test_bob_is_left_alone_where_every_field_already_has_a_frame():
    class Stream:
        average_rate = Fraction(50)
        guessed_rate = None
    notes: list = []
    chain, per_frame = encode._deinterlace_chain(
        EncodeOptions(deinterlace="bob"), Stream(), encode.VideoSample(), notes.append)
    assert per_frame == 1 and "send_frame" in chain[0][1] and "不需要还原" in notes[0]


def test_field_votes_read_which_field_is_nearer_in_time():
    # a scene panning 3 px per field: shots a field apart differ less than
    # shots two fields apart, which is all the vote looks at
    x = np.arange(200)
    scene = np.tile(128 + 100 * np.sin(x / 9.0), (48, 1))
    shots = [scene[:, 3 * k:3 * k + 64].astype(np.uint8) for k in range(21)]

    def weave(first, second, top_first):
        frame = np.empty((96, 64), np.uint8)
        frame[0::2], frame[1::2] = (first, second) if top_first else (second, first)
        return frame

    tff = [weave(shots[i], shots[i + 1], True) for i in range(0, 20, 2)]
    bff = [weave(shots[i], shots[i + 1], False) for i in range(0, 20, 2)]
    # a still picture: both fields the same moment, no vote either way
    still = [weave(shots[0], shots[0], True)] * 10
    assert restore.field_order(restore.field_votes(tff)) == "tff"
    assert restore.field_order(restore.field_votes(bff)) == "bff"
    assert restore.field_order(restore.field_votes(still)) == ""


# ------------------------------------------------------- crop and enlarge


@pytest.mark.parametrize("size,sar,upscale,want", [
    ((720, 480), Fraction(32, 27), 1080, (1920, 1080)),   # NTSC 16:9
    ((720, 480), Fraction(8, 9), 1080, (1440, 1080)),     # NTSC 4:3
    ((720, 576), Fraction(16, 15), 1080, (1440, 1080)),   # PAL 4:3
    ((720, 472), Fraction(8, 9), 1080, (1464, 1080)),     # 8 lines of tape noise cut
    ((1920, 1080), Fraction(1), 1080, (1920, 1080)),      # already there: untouched
    ((1920, 1080), Fraction(1), 720, (1920, 1080)),       # never shrinks
])
def test_enlarging_makes_square_pixels_first(size, sar, upscale, want):
    w, h, out_sar = encode.geometry(*size, sar, EncodeOptions(upscale=upscale))
    assert (w, h) == want
    if upscale > min(size):
        assert out_sar == 1


def test_crop_then_denoise_then_enlarge(tmp_path):
    source = make_source(tmp_path / "tape.mkv", frames=12, size=(128, 96))
    out, _result, lines = run(source, video_codec="libx264", crop_bottom=8, crop_left=4,
                              crop_right=4, denoise="light", upscale=720)
    # 120x88 after the crop, enlarged until the short side is 720
    with av.open(str(out)) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.width, cc.height) == (982, 720)
    assert frames_of(out) == 12
    assert out.name == "tape.720p.H264.mkv"
    assert any("Lanczos 放大" in line and "裁边后 120x88" in line for line in lines)


def test_too_much_crop_is_refused(tmp_path):
    source = make_source(tmp_path / "tape.mkv", frames=4, size=(64, 48))
    request = EncodeRequest(source=str(source), options=EncodeOptions(
        video_codec="libx264", preset="ultrafast", crop_top=24, crop_bottom=20))
    with pytest.raises(encode.EncodeError, match="裁边太多"):
        encode.encode_file(source, tmp_path / "x.mkv.part", request)
    assert not (tmp_path / "x.mkv.part").exists()


# --------------------------------------------------------- a model's place


class DoubleStage:
    """Stands in for an enlarging model: nearest-neighbour x2, in batches
    of three so frames are held back and flushed like a GPU batch would."""

    name = "测试用放大"

    def __init__(self):
        self.held, self.formats, self.seen, self.sizes = [], set(), 0, set()

    def _run(self) -> list:
        out = []
        for frame in self.held:
            self.formats.add(frame.format.name)
            self.sizes.add((frame.width, frame.height))
            rgb = frame.to_ndarray()
            big = av.VideoFrame.from_ndarray(rgb.repeat(2, 0).repeat(2, 1),
                                             format=frame.format.name)
            big.pts, big.time_base = frame.pts, frame.time_base
            out.append(big)
        self.seen += len(self.held)
        self.held = []
        return out

    def push(self, frame) -> list:
        self.held.append(frame)
        return self._run() if len(self.held) == 3 else []

    def flush(self) -> list:
        return self._run()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def stage(monkeypatch):
    made = DoubleStage()
    monkeypatch.setattr(restore, "make_stage", lambda opts, log, engine=None: made)
    return made


def test_a_model_gets_rgb_and_its_frames_reach_the_encoder(tmp_path, stage):
    source = make_source(tmp_path / "tape.mkv", frames=13, size=(128, 96))
    out, result, lines = run(source, video_codec="libx264", upscale=720)
    assert stage.formats == {"rgb24"} and stage.seen == 13
    with av.open(str(out)) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.width, cc.height) == (960, 720)
    assert frames_of(out) == 13
    assert pts_of(out) == pts_of(source)
    assert any("用 测试用放大 放大" in line for line in lines)


def test_a_ten_bit_encode_hands_the_model_sixteen_bit_rgb(tmp_path, stage):
    source = make_source(tmp_path / "tape.mkv", frames=6, size=(128, 96))
    run(source, video_codec="libx265", upscale=720)          # H.265: 10bit by default
    assert stage.formats == {"rgb48le"}


def test_bob_and_a_model_together_keep_every_field(tmp_path, stage):
    """After bwdif the frames are in half the time base; the second filter
    graph must take that from the frames, not from the stream."""
    source = make_fields(tmp_path / "tape.mkv", frames=20)
    out, result, _lines = run(source, video_codec="libx264", deinterlace="bob", upscale=720)
    assert stage.seen == 40 and frames_of(out) == 40
    assert len(set(pts_of(out))) == 40
    assert out.name == "tape.720p.60fps.H264.mkv"


def test_no_model_is_used_where_nothing_is_enlarged(tmp_path, stage):
    source = make_source(tmp_path / "film.mkv", frames=4, size=(1280, 720))
    run(source, video_codec="libx264", upscale=720)
    assert stage.seen == 0


# --------------------------------------------------------- what stays put


def test_a_plain_encode_hashes_and_is_named_as_before(tmp_path):
    # the hash every encode tagged before 修复 existed carries
    assert encode.options_hash(EncodeOptions()) == "e6bd2cda2219"
    assert encode.options_hash(EncodeOptions(deinterlace="bob")) != "e6bd2cda2219"
    assert encode.options_hash(EncodeOptions(crop_bottom=8)) != "e6bd2cda2219"
    source = make_source(tmp_path / "film.mkv", frames=2)
    target = encode.output_target(source, EncodeRequest(source=str(source)))
    assert target.name == "film.HEVC.mkv"
    assert not restore.restoring(EncodeOptions())


def test_what_cannot_be_combined_is_refused_when_queued():
    with pytest.raises(ValueError, match="重编码"):
        encode.check_options(EncodeOptions(video_codec="copy", deinterlace="bob"))
    with pytest.raises(ValueError, match="不能同时"):
        encode.check_options(EncodeOptions(upscale=1080, max_height=720))
    with pytest.raises(ValueError):
        EncodeOptions(crop_bottom=7)                         # odd: chroma would not line up


def test_the_queue_line_says_what_the_restore_does():
    text = encode.describe_options(EncodeOptions(
        deinterlace="bob", field_order="bff", crop_bottom=8, denoise="light", upscale=1080))
    for part in ("还原 60 帧", "下场优先", "裁边 上0 下8", "轻度降噪", "放大到 1080p"):
        assert part in text


@pytest.mark.parametrize("level", ["light", "strong"])
def test_each_denoise_level_runs(tmp_path, level):
    source = make_source(tmp_path / "tape.mkv", frames=8, size=(128, 96))
    out, _result, lines = run(source, video_codec="libx264", denoise=level)
    assert frames_of(out) == 8
    assert any(restore.DENOISE_NAMES[level] in line for line in lines)


def test_a_still_first_look_is_followed_by_a_longer_later_one(tmp_path):
    """The first sample is taken a third of the way in. Where that stretch
    does not move, the order must still come from the picture — falling
    back to the flags would play this unflagged bottom-first capture
    backwards for the whole film."""
    source = make_fields(tmp_path / "capture.mkv", frames=2100, flagged=False,
                         bottom_first=True, still=1050, size=(64, 48))
    out, _result, lines = run(source, video_codec="libx264", deinterlace="bob")
    assert any("从画面测得下场优先" in line for line in lines), lines
    moving = bar_positions(out)[2 * 1050 + 2:]
    assert moving and all(m > 0 for m in steps(moving))


def test_a_picture_that_never_moves_says_what_the_fallback_assumes(tmp_path):
    source = make_fields(tmp_path / "still.mkv", frames=30, still=30)
    _out, _result, lines = run(source, video_codec="libx264", deinterlace="bob")
    assert any("测不出" in line and "按上场优先处理" in line for line in lines)



# ------------------------------------------------------ what the disc is


@pytest.mark.parametrize("kind,mode", [
    ("interlaced", "bob"), ("telecine", "ivtc"), ("shifted", "match"), ("progressive", "off"),
])
def test_the_cadence_is_read_from_the_picture_not_the_flags(tmp_path, kind, mode):
    """Every one of these is flagged interlaced, as every DVD is; measured on
    the user's discs, the four need four different modes."""
    source = make_cadence(tmp_path / f"{kind}.mkv", kind)
    cadence = restore.analyze_cadence(source, 0)
    assert (cadence.kind, cadence.mode) == (kind, mode), cadence.as_dict()


def test_a_cycle_that_field_matching_cannot_undo_is_called_blended():
    # measured on a 1990 film disc: combed 4 frames in 5, field matching in
    # either order still left 41% of 49%
    assert restore.classify(0.49, 0.41, True, 600) == "blended"
    assert restore.classify(0.64, 0.44, False, 600) == "interlaced"
    assert restore.classify(0.38, 0.0, True, 600) == "telecine"
    assert restore.classify(0.9, 0.9, False, 30) == ""          # too little to go on
    assert restore.classify(0.0, 0.0, False, 600, motion=0.3) == ""   # nothing moved: no answer
    # a 2010 TV omnibus: 30p drama with interlaced segments between
    assert restore.classify(0.30, 0.13, False, 600) == "mixed"
    assert restore.CADENCE_MODES["mixed"] == "match"


def test_combing_is_alternating_lines_not_noise():
    rng = np.random.default_rng(3)
    noisy = np.clip(rng.normal(120, 20, (120, 160)), 0, 255).astype(np.uint8)
    assert not restore.combed(noisy)
    a, b = np.zeros((120, 160), np.uint8), np.zeros((120, 160), np.uint8)
    a[:, 40:80], b[:, 52:92] = 220, 220
    woven = a.copy()
    woven[1::2] = b[1::2]
    assert restore.combed(woven)


def test_match_mode_brings_30p_back_whole(tmp_path):
    source = make_cadence(tmp_path / "drama.mkv", "shifted", frames=120)
    out, result, lines = run(source, video_codec="libx264", deinterlace="match")
    assert result.stats.frames_out == frames_of(out) == 120
    with av.open(str(out)) as c:
        frames = [f.to_ndarray(format="gray") for f in c.decode(c.streams.video[0])]
        assert abs(float(c.streams.video[0].average_rate) - 30000 / 1001) < 0.05
    assert sum(restore.combed(f) for f in frames[5:]) == 0
    assert any("只做场匹配" in line for line in lines)


def test_detect_turns_into_the_measured_mode(tmp_path):
    notes: list = []
    film = make_cadence(tmp_path / "film.mkv", "telecine")
    opts = restore.resolve(film, EncodeOptions(deinterlace="detect"), notes.append)
    assert opts.deinterlace == "ivtc" and "24p 硬过带" in notes[0]
    camera = make_cadence(tmp_path / "camera.mkv", "interlaced")
    assert restore.resolve(camera, EncodeOptions(deinterlace="detect"), notes.append).deinterlace == "bob"


def test_detect_runs_through_a_whole_encode(tmp_path):
    source = make_cadence(tmp_path / "film.mkv", "telecine", frames=150)
    out, result, lines = run(source, video_codec="libx264", deinterlace="detect")
    assert result.stats.frames_in == 150 and result.stats.frames_out == 120
    assert any("反胶片过带" in line for line in lines)


def _letterboxed(path, size=(160, 120), bars=(16, 16, 8, 0), frames=24):
    w, h = size
    top, bottom, left, right = bars
    with av.open(str(path), "w", format="matroska") as c:
        v = c.add_stream("libx264", rate=24)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        for i in range(frames):
            img = np.full((h, w, 3), 16, np.uint8)
            img[top:h - bottom, left:w - right] = 90 + (i * 5) % 100
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in v.encode(frame):
                c.mux(packet)
        for packet in v.encode(None):
            c.mux(packet)
    return path


def test_black_bars_are_found_and_cut_on_top_of_a_manual_crop(tmp_path):
    source = _letterboxed(tmp_path / "box.mkv")
    assert restore.detect_borders(source, 0) == (16, 16, 8, 0)
    notes: list = []
    opts = restore.resolve(source, EncodeOptions(crop_auto=True, crop_bottom=4), notes.append)
    assert (opts.crop_top, opts.crop_bottom, opts.crop_left, opts.crop_right) == (16, 20, 8, 0)
    assert not opts.crop_auto and "黑边" in notes[0]


def test_a_full_frame_has_no_bars(tmp_path):
    source = _letterboxed(tmp_path / "night.mkv", bars=(0, 0, 0, 0))
    with av.open(str(source)) as c:
        assert c.streams.video[0].codec_context.height == 120
    assert restore.detect_borders(source, 0) == (0, 0, 0, 0)


def test_an_unflagged_pal_capture_gets_its_shape_back(tmp_path):
    source = make_source(tmp_path / "pal.mkv", frames=6, size=(720, 576))
    assert restore.missing_aspect(720, 576, None)
    out, _result, _lines = run(source, video_codec="libx264", aspect="4:3", upscale=1080)
    with av.open(str(out)) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.width, cc.height) == (1440, 1080)
    out, _result, _lines = run(source, video_codec="libx264", aspect="16:9")
    with av.open(str(out)) as c:
        assert c.streams.video[0].sample_aspect_ratio == Fraction(64, 45)



def test_a_model_never_sees_stretched_pixels(tmp_path, stage):
    """Anamorphic DVD pixels are squared before the model — by stretching,
    never shrinking: 16:9 widens to 854, 4:3 heightens to 540."""
    wide = make_source(tmp_path / "wide.mkv", frames=3, size=(720, 480), sar=Fraction(32, 27))
    run(wide, video_codec="libx264", upscale=1080)
    assert stage.sizes == {(854, 480)}
    stage.sizes.clear()
    narrow = make_source(tmp_path / "narrow.mkv", frames=3, size=(720, 480), sar=Fraction(8, 9))
    out, _result, _lines = run(narrow, video_codec="libx264", upscale=1080)
    assert stage.sizes == {(720, 540)}
    with av.open(str(out)) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.width, cc.height) == (1440, 1080)


# --------------------------------------------------------------- the engine


import sys

from restore_engine import protocol


def fake_engine(monkeypatch, model="double"):
    def make(opts, log, engine=None):
        client = restore.EngineClient(sys.executable, {"model": model}, log, module="tests.fake_engine")
        return restore.EngineStage(client, "假模型")
    monkeypatch.setattr(restore, "make_stage", make)


def test_frames_go_through_the_engine_process_and_keep_their_time(tmp_path, monkeypatch):
    fake_engine(monkeypatch)
    source = make_source(tmp_path / "tape.mkv", frames=10, size=(128, 96))
    out, result, lines = run(source, video_codec="libx264", upscale=720, ai_model="realviformer")
    assert frames_of(out) == 10 and pts_of(out) == pts_of(source)
    with av.open(str(out)) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.width, cc.height) == (960, 720)
    assert any("假引擎：加载完成" in line for line in lines)


@pytest.mark.parametrize("model,said", [("fail", "显存不足"), ("die", "中途退出"), ("nan", "NaN")])
def test_an_engine_failure_fails_the_encode_and_leaves_nothing(tmp_path, monkeypatch, model, said):
    fake_engine(monkeypatch, model)
    source = make_source(tmp_path / "tape.mkv", frames=10, size=(128, 96))
    request = EncodeRequest(source=str(source), options=EncodeOptions(
        video_codec="libx264", preset="ultrafast", upscale=720, ai_model="realviformer"))
    part = tmp_path / "out.mkv.part"
    with pytest.raises(encode.EncodeError, match=said):
        encode.encode_file(source, part, request)
    assert not part.exists()


def test_without_a_model_folder_the_engine_picks_its_own(monkeypatch, settings_file):
    # model_cache_dir unset is the default, and Path(None) used to end every
    # AI encode right there (the GPU bench always passed a folder)
    seen = {}

    class Client:
        def __init__(self, python, init, log, *rest, **kw):
            seen.update(init)
            self.info = {"device": "fake", "precision": "fp32"}

    monkeypatch.setattr(restore, "EngineClient", Client)
    stage = restore.make_stage(EncodeOptions(upscale=720, ai_model="realviformer"), print,
                               RestoreSettings(engine_python="python"))
    assert stage is not None and seen["models_dir"] == ""


def test_field_sampling_does_not_pile_up_threads(tmp_path):
    # with the garbage collector off, what a sample leaves running is what a
    # busy process would carry until it comes round: a fixed few per sample,
    # not a converter's threads per sampled frame
    import gc
    import re
    from pathlib import Path
    status = Path("/proc/self/status")
    if not status.exists():
        pytest.skip("counts threads through /proc")

    def threads():
        return int(re.search(r"Threads:\s+(\d+)", status.read_text()).group(1))

    source = make_fields(tmp_path / "tape.mkv", frames=50)
    gc.collect()
    gc.disable()
    try:
        before = threads()
        encode.sample_video(source, 0, count=10, fields=True)
        short = threads() - before
        middle = threads()
        encode.sample_video(source, 0, count=40, fields=True)
        long = threads() - middle
    finally:
        gc.enable()
        gc.collect()
    assert long - short < 15, (short, long)


def test_the_engine_needs_somewhere_to_run():
    with pytest.raises(restore.EngineError, match="修复引擎"):
        restore.EngineClient("", {"model": "x"}, print)
    with pytest.raises(ValueError, match="放大到"):
        encode.check_options(EncodeOptions(ai_model="realviformer"))


def test_frames_cross_the_pipe_unchanged():
    import io
    for dtype in (np.uint8, np.uint16):
        array = (np.arange(6 * 4 * 3) * 997 % 65535).astype(dtype).reshape(6, 4, 3)
        pipe = io.BytesIO()
        protocol.send_frame(pipe, array)
        protocol.send_json(pipe, {"end": True})
        pipe.seek(0)
        kind, back = protocol.receive(pipe)
        assert kind == b"F" and back.dtype == dtype and np.array_equal(back, array)
        assert protocol.receive(pipe) == (b"J", {"end": True})
        assert protocol.receive(pipe) == (None, None)


def test_a_still_picture_says_nothing_about_its_cadence(tmp_path):
    source = make_fields(tmp_path / "still.mkv", frames=150, still=150)
    assert restore.analyze_cadence(source, 0).kind == ""
    notes: list = []
    opts = restore.resolve(source, EncodeOptions(deinterlace="detect"), notes.append)
    assert opts.deinterlace == "bob" and "测不出" in notes[0]      # flagged interlaced: assume so

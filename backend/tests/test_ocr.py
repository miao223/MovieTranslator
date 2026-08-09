"""Reading a graphic subtitle track — Blu-ray PGS and its relatives.

The fixtures are real PGS bitstreams built by tests/pgs.py, so these tests
exercise the actual demux/decode path rather than a stand-in. What they
defend, in order of how much damage getting it wrong does:

* the picture handed to the recogniser is the glyphs and nothing else —
  outline included is a fatter glyph with its counters closed, and that is
  fatal for kanji long before it is noticeable in English;
* cue times are the ones the disc stored;
* a model that loses count of the lines it was given is not believed.

The recognisers themselves are faked. rapidocr is an optional extra that CI
does not install, and the vision engine costs money — neither belongs in a
test suite, and neither is what this module gets wrong.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import JobRequest, LLMSettings, OcrSettings
from app.services import ocr
from tests.pgs import render, write_sup, write_two_region_sup

CUES = [(1.0, 3.0, "Hello there"), (4.0, 6.0, "What do you suppose\nthis is?")]


def track_for(path: Path) -> dict:
    return {"index": 0, "path": str(path), "text": False, "forced": False,
            "default": True, "language": "", "language_name": "未标注语言",
            "title": path.name, "codec": "PGS"}


def sup(tmp_path, cues=CUES, style="outline") -> tuple[Path, dict]:
    path = write_sup(tmp_path / f"{style}.sup", cues, style=style)
    return path, track_for(path)


def ink_ratio(image: Image.Image) -> float:
    return float((np.array(image) < 128).mean())


# ------------------------------------------------------------ binarisation


@pytest.mark.parametrize("style", ["outline", "antialiased", "box"])
def test_only_the_glyphs_survive_whatever_the_palette_looks_like(tmp_path, style):
    """Three ways a disc draws a subtitle, one answer.

    ``antialiased`` is the one that matters: it is what real discs ship, and
    a fixed erosion depth used to land on top of the widest outline index
    and let it through — the ink ratio nearly doubled and every ``e`` filled
    in solid.
    """
    path, track = sup(tmp_path, style=style)
    cues = ocr.bitmap_cues(path, track, upscale=1)
    assert len(cues) == 2
    for cue in cues:
        assert 0.05 < ink_ratio(cue.image) < 0.25


def test_the_glyphs_come_out_the_same_weight_with_and_without_soft_edges(tmp_path):
    hard = ocr.bitmap_cues(*sup(tmp_path, style="outline"), upscale=1)
    soft = ocr.bitmap_cues(*sup(tmp_path, style="antialiased"), upscale=1)
    for a, b in zip(hard, soft):
        # the soft edge legitimately adds a little; a doubling means the
        # outline got in
        assert ink_ratio(b.image) == pytest.approx(ink_ratio(a.image), abs=0.03)


def test_a_plate_with_the_text_knocked_out_is_not_read_as_a_black_rectangle(tmp_path):
    """No transparency to measure depth against, so the plate is whatever
    covers the most ground and the letters are the rest."""
    path, track = sup(tmp_path, style="box")
    cue = ocr.bitmap_cues(path, track, upscale=1)[0]
    assert ink_ratio(cue.image) < 0.3


def test_ink_is_never_the_background():
    """A pathological rect whose 'interior' is a plate inside a transparent
    frame still comes back with the smaller half as the letters."""
    field = np.zeros((40, 40), dtype=np.uint8)
    field[5:35, 5:35] = 1   # an opaque plate
    field[15:25, 15:25] = 2  # a small mark on it
    assert ocr.ink_mask(field).mean() < 0.5


def test_one_cue_split_across_regions_stays_two_lines(tmp_path):
    """Discs often put each line of a subtitle in its own region; pasting
    them back at their own coordinates is what keeps them apart."""
    path = write_two_region_sup(tmp_path / "two.sup", 1.0, 3.0, "Top", "Bottom")
    cue = ocr.bitmap_cues(path, track_for(path), upscale=1)[0]
    rows = (np.array(cue.image) < 128).any(axis=1)
    # two bands of ink with a gap between them
    breaks = np.diff(rows.astype(int))
    assert list(breaks).count(1) == 2


def test_upscaling_multiplies_both_sides(tmp_path):
    path, track = sup(tmp_path)
    one = ocr.bitmap_cues(path, track, upscale=1)[0].image
    three = ocr.bitmap_cues(path, track, upscale=3)[0].image
    assert three.size == (one.width * 3, one.height * 3)


# ------------------------------------------------------------------ timing


def test_cue_times_are_the_ones_the_disc_stored(tmp_path):
    """PGS ends a cue with its own clear event. Trusting packet.duration
    instead gives 0xFFFFFFFF — a subtitle lasting forty-nine days."""
    path, track = sup(tmp_path)
    cues = ocr.bitmap_cues(path, track, upscale=1)
    assert [(round(c.start, 2), round(c.end, 2)) for c in cues] == [(1.0, 3.0), (4.0, 6.0)]


def test_a_clear_event_produces_no_cue_of_its_own(tmp_path):
    path, track = sup(tmp_path, cues=[(1.0, 3.0, "Only one")])
    assert len(ocr.bitmap_cues(path, track, upscale=1)) == 1


def test_cancelling_stops_the_decode(tmp_path):
    path, track = sup(tmp_path)
    with pytest.raises(InterruptedError):
        ocr.bitmap_cues(path, track, should_cancel=lambda: True)


# ------------------------------------------------------------ the pipeline


def fake_engine(texts, scores=None):
    def run(cues, _lang):
        return [(t, (scores or {}).get(t, 0.9)) for t in texts[: len(cues)]]
    return run


def test_recognised_text_becomes_numbered_lines_on_the_disc_timings(tmp_path):
    path, track = sup(tmp_path)
    lines = ocr.read_cues(path, track, OcrSettings(),
                          engine=fake_engine(["Hello there", "What do you suppose"]))
    assert [(l.index, round(l.start, 2), round(l.end, 2), l.text) for l in lines] == [
        (1, 1.0, 3.0, "Hello there"),
        (2, 4.0, 6.0, "What do you suppose"),
    ]


def test_a_cue_the_engine_read_nothing_out_of_is_dropped(tmp_path):
    path, track = sup(tmp_path)
    stats: dict = {}
    lines = ocr.read_cues(path, track, OcrSettings(), stats=stats,
                          engine=fake_engine(["Hello there", "   "]))
    assert [l.text for l in lines] == ["Hello there"]
    assert stats["blank"] == 1


def test_reading_nothing_at_all_is_an_error_not_an_empty_subtitle(tmp_path):
    path, track = sup(tmp_path)
    with pytest.raises(ValueError, match="没有识别出任何文字"):
        ocr.read_cues(path, track, OcrSettings(), engine=fake_engine(["", ""]))


def test_a_track_with_no_pictures_says_so(tmp_path):
    path = tmp_path / "empty.sup"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        ocr.read_cues(path, track_for(path), OcrSettings(),
                      engine=fake_engine(["x"]))


# ----------------------------------------------------------- local engine


def test_the_missing_dependency_says_how_to_install_it(monkeypatch, tmp_path):
    """rapidocr is an optional extra, so "not installed" is a normal state
    and has to be actionable rather than an ImportError traceback."""
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "rapidocr":
            raise ImportError("no module named rapidocr")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setattr(ocr, "_engine", None)
    monkeypatch.setattr(ocr, "_engine_key", None)
    with pytest.raises(RuntimeError) as exc:
        ocr._get_engine("japan", None, None)
    assert 'pip install -e ".[ocr]"' in str(exc.value)
    assert "视觉大模型" in str(exc.value)  # the way out that needs no install


@pytest.mark.parametrize("code,model", [
    ("ja", "japan"), ("en", "en"), ("zh", "ch"), ("ko", "korean"),
    ("", "en"), ("de", "latin"),
])
def test_each_language_maps_to_its_recognition_model(code, model):
    assert ocr.rec_language(code) == model


def fake_rapidocr(monkeypatch, versions=("PP-OCRv4", "PP-OCRv5", "PP-OCRv6"),
                  refuse=(), unknown_globals=()):
    """A stand-in for the library that refuses what the real one refuses.

    RapidOCR validates the parameter dictionary before it looks at a single
    model: the version and the model size have to arrive as enum instances,
    and an unknown ``Global.*`` key is an error. Neither was true of what we
    sent, and every test here passed anyway, because nothing was checking
    the shape of the dictionary — only the real machine found out.
    """
    import enum
    import sys
    import types

    def enum_of(name, values):
        return enum.Enum(name, {v.replace("-", "_").replace(".", "_"): v
                                for v in values})

    # the subset of Global that the shipped config.yaml actually declares
    global_keys = {"text_score", "use_det", "use_cls", "use_rec", "font_path",
                   "log_level", "model_root_dir", "min_side_len", "max_side_len"}
    global_keys -= set(unknown_globals)
    enum_params = {"engine_type", "model_type", "ocr_version", "task_type"}
    attempts: list = []

    class FakeRapidOCR:
        def __init__(self, params=None):
            params = params or {}
            label = tuple(getattr(params.get(k), "value", "")
                          for k in ("Rec.ocr_version", "Rec.model_type"))
            attempts.append((label, params))
            for key, value in params.items():
                section, _, name = key.rpartition(".")
                if section == "Global" and name not in global_keys:
                    raise ValueError(f"{key} is not a valid key.")
                if name in enum_params and not isinstance(value, enum.Enum):
                    raise TypeError(f"The value of {key} must be Enum Type.")
            if " ".join(label) in refuse:
                raise ValueError(f"Unsupported configuration: {label}")

    mod = types.ModuleType("rapidocr")
    mod.RapidOCR = FakeRapidOCR
    mod.OCRVersion = enum_of("OCRVersion", versions)
    mod.ModelType = enum_of("ModelType", ("mobile", "small", "server"))
    mod.LangRec = enum_of("LangRec", ("ch", "en", "japan", "korean", "latin",
                                      "cyrillic"))
    monkeypatch.setitem(sys.modules, "rapidocr", mod)
    monkeypatch.setattr(ocr, "_engine", None)
    monkeypatch.setattr(ocr, "_engine_key", None)
    monkeypatch.setattr(ocr, "_engine_label", "")
    return attempts


def test_the_parameters_are_the_shape_the_library_demands(monkeypatch):
    """The strings this used to send were rejected on sight — *The value of
    Rec.ocr_version must be Enum Type* — before a single cue was read."""
    attempts = fake_rapidocr(monkeypatch)
    assert ocr._get_engine("japan", None, None) is not None
    (_, params), = attempts
    assert params["Rec.ocr_version"].value == "PP-OCRv6"
    assert params["Rec.lang_type"].value == "japan"
    assert params["Global.use_cls"] is False  # a subtitle is never upside down


def test_a_model_this_release_does_not_ship_falls_back_to_the_next(monkeypatch):
    """Which languages each line covers keeps moving — v5 never had Japanese
    at all — and the library only says so when it goes looking for the file."""
    attempts = fake_rapidocr(monkeypatch, refuse=("PP-OCRv6 small",))
    assert ocr._get_engine("japan", None, None) is not None
    assert [label for label, _ in attempts] == [
        ("PP-OCRv6", "small"), ("PP-OCRv4", "mobile"),
    ]
    # which one answered decides how the recognition reads, so it is on record
    assert ocr._engine_label == "PP-OCRv4 mobile"


def test_a_release_that_predates_a_line_never_asks_for_it(monkeypatch):
    attempts = fake_rapidocr(monkeypatch, versions=("PP-OCRv4",))
    ocr._get_engine("japan", None, None)
    assert [label for label, _ in attempts] == [("PP-OCRv4", "mobile")]


def test_a_language_only_the_old_line_reads_goes_straight_there(monkeypatch):
    attempts = fake_rapidocr(monkeypatch)
    ocr._get_engine("korean", None, None)
    assert [label for label, _ in attempts] == [("PP-OCRv4", "mobile")]


def test_a_setting_this_release_has_never_heard_of_is_not_fatal(monkeypatch):
    """An unknown ``Global.*`` key is an error, not something ignored, so one
    renamed setting would otherwise take every candidate down with it."""
    attempts = fake_rapidocr(monkeypatch, unknown_globals=("use_cls",))
    assert ocr._get_engine("korean", None, None) is not None
    assert len(attempts) == 2  # rejected once, then asked for nothing extra
    assert not [k for k in attempts[-1][1] if k.startswith("Global.")]


def test_when_nothing_loads_the_error_names_the_proxy_switch(monkeypatch):
    """The likeliest cause is a blocked download, and that has a setting."""
    fake_rapidocr(monkeypatch, refuse=("PP-OCRv6 small", "PP-OCRv4 mobile"))
    with pytest.raises(RuntimeError, match="模型下载走代理"):
        ocr._get_engine("japan", None, None)


def test_the_models_are_kept_where_startup_does_not_wipe_them(monkeypatch, tmp_path):
    attempts = fake_rapidocr(monkeypatch)
    monkeypatch.setattr("app.services.asr.get_model_cache_dir",
                        lambda: str(tmp_path))
    ocr._get_engine("japan", None, None)
    (_, params), = attempts
    assert params["Global.model_root_dir"] == str(tmp_path / "rapidocr")
    assert "Global.model_dir" not in params  # Global has no such key at all


def test_boxes_are_read_top_to_bottom_then_left_to_right():
    """A two-line cue comes back as several boxes, and the order they were
    detected in is not the order a person reads them."""
    class Result:
        txts = ["second half", "SIGN", "first half"]
        scores = [0.9, 0.8, 0.95]
        boxes = [
            [[10, 60], [200, 60], [200, 90], [10, 90]],
            [[300, 10], [400, 10], [400, 40], [300, 40]],
            [[10, 10], [200, 10], [200, 40], [10, 40]],
        ]

    text, score = ocr._one(lambda _img: Result(), Image.new("L", (8, 8), 255))
    assert text == "first half SIGN second half"
    assert score == pytest.approx(0.883, abs=0.01)


# ---------------------------------------------------------- vision engine


class FakeVision:
    """Records the requests and replays scripted replies.

    The last reply sticks once the script runs out, so "this endpoint is
    simply broken" is one entry rather than a guess at the call count.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature, **kw):
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply

        class Obj:
            pass

        msg, choice, resp = Obj(), Obj(), Obj()
        msg.content = reply
        choice.message = msg
        resp.choices = [choice]
        return resp


def vision_cues(n=3):
    return [ocr.Cue(float(i), i + 1.0, render(f"line {i}"))
            for i in range(1, n + 1)]


def test_the_sheet_carries_the_pictures_and_their_numbers():
    client = FakeVision(["[1] one\n[2] two\n[3] three"])
    out = ocr._read_vision(vision_cues(), LLMSettings(model="m"),
                           OcrSettings(vision_batch=10), client=client)
    assert [t for t, _s in out] == ["one", "two", "three"]
    content = client.calls[0][0]["content"]
    # PNG, never JPEG: ringing around one-bit glyphs is lost accuracy
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "原样抄写" in content[0]["text"]


def test_the_sheet_numbers_are_drawn_big_enough_to_survive_downscaling():
    """The server shrinks the sheet as one picture, so the label shrinks with
    the subtitle. PIL's unscaled default font is 11px tall — on a sheet two
    thousand pixels wide that comes back as a smudge, and a label the model
    cannot read fails the coverage check on every single sheet."""
    sheet = ocr._sheet(vision_cues(2), 1)
    column = np.array(sheet)[:, :ocr.LABEL_WIDTH]
    rows = np.where((column < 128).any(axis=1))[0]
    assert rows.size, "the numbers are not on the sheet at all"
    assert (column < 128).sum() > 300  # 47 per label at the default size


def test_a_sheet_the_model_lost_count_of_is_halved_before_being_abandoned():
    """Half a subtitle is worse than a clear failure — the caller can fall
    back to speech recognition, but it cannot fill in holes it never sees.
    Only a single cue that will not read ends the pass, though: the usual
    reason a self-hosted endpoint refuses a sheet is a limit of its own."""
    client = FakeVision(["[1] one\n[3] three"])  # never covers its numbers
    with pytest.raises(RuntimeError, match="第 1 条上连续失败"):
        ocr._read_vision(vision_cues(), LLMSettings(model="m"),
                         OcrSettings(vision_batch=10), client=client)
    assert len(client.calls) == 4  # three at once, twice; then one, twice


def test_a_sheet_the_endpoint_will_not_take_is_split_and_read():
    """What a local deployment usually objects to is the size of the
    request, and it stops objecting once the sheet is halved."""
    client = FakeVision([
        RuntimeError("payload too large"), RuntimeError("payload too large"),
        "[1] one\n[2] two", "[3] three\n[4] four",
    ])
    out = ocr._read_vision(vision_cues(4), LLMSettings(model="m"),
                           OcrSettings(vision_batch=10), client=client)
    assert [t for t, _s in out] == ["one", "two", "three", "four"]


def test_a_server_that_answers_without_any_choices_says_what_it_said():
    """A local endpoint can answer 200 with `choices: null` and the real
    complaint alongside it; reading choices[0] off that gives a TypeError
    naming neither the server nor the problem."""
    class NoChoices(FakeVision):
        def create(self, model, messages, temperature, **kw):
            self.calls.append(messages)

            class Resp:
                choices = None
                error = {"message": "model not loaded"}

            return Resp()

    client, said = NoChoices([""]), []
    with pytest.raises(RuntimeError, match="连续失败"):
        ocr._read_vision(vision_cues(1), LLMSettings(model="m"),
                         OcrSettings(vision_batch=10), client=client,
                         log=said.append)
    assert any("model not loaded" in line for line in said)
    assert any("图 " in line for line in said)  # and how big the picture was


def test_a_transient_api_error_gets_one_more_try():
    client = FakeVision([RuntimeError("502"), "[1] one\n[2] two\n[3] three"])
    out = ocr._read_vision(vision_cues(), LLMSettings(model="m"),
                           OcrSettings(vision_batch=10), client=client)
    assert [t for t, _s in out] == ["one", "two", "three"]


def test_the_sheets_are_numbered_continuously_across_batches():
    client = FakeVision(["[1] one\n[2] two", "[3] three\n[4] four"])
    out = ocr._read_vision(vision_cues(4), LLMSettings(model="m"),
                           OcrSettings(vision_batch=2), client=client)
    assert [t for t, _s in out] == ["one", "two", "three", "four"]


def test_an_unreadable_cue_is_kept_as_an_empty_line_not_skipped():
    """Skipping it would shift every number after it — the failure mode this
    whole project checks for."""
    client = FakeVision(["[1] one\n[2] [无]\n[3] three"])
    out = ocr._read_vision(vision_cues(), LLMSettings(model="m"),
                           OcrSettings(vision_batch=10), client=client)
    assert [t for t, _s in out] == ["one", "", "three"]


@pytest.mark.parametrize("reply,expected", [
    ("[1] one\n[2] two", {1: "one", 2: "two"}),
    ("```\n[1] one\n```", {1: "one"}),
    ("- [1] one\n• [2] two", {1: "one", 2: "two"}),
    ("1. hello", {1: "hello"}),
])
def test_the_reply_parser_tolerates_the_usual_model_noise(reply, expected):
    assert ocr.parse_sheet(reply) == expected


# -------------------------------------------------------- through the job


def video_with_pgs(tmp_path) -> Path:
    """A real mkv carrying a real PGS track, built by the production muxer."""
    from app.services import mux
    from tests.test_audio_tracks import make_multitrack_video

    src = make_multitrack_video(tmp_path / "film.mkv")
    supfile = write_sup(tmp_path / "p.sup", CUES)
    return mux.embed(src, supfile, tmp_path / "withpgs.mkv", "English",
                     track_title="PGS")


def test_a_graphic_track_in_a_real_video_is_found_and_marked(tmp_path):
    from app.services import subsource

    tracks = subsource.all_tracks(video_with_pgs(tmp_path))
    assert [t["codec"] for t in tracks] == ["PGSSUB"]
    assert tracks[0]["text"] is False
    assert "OCR" in subsource.describe_track(tracks[0])


def test_a_sup_file_beside_the_video_is_offered_too(tmp_path):
    from app.services import subsource
    from tests.test_audio_tracks import make_multitrack_video

    video = make_multitrack_video(tmp_path / "film.mkv")
    write_sup(tmp_path / "film.sup", CUES)
    tracks = subsource.all_tracks(video)
    assert [Path(t["path"]).name for t in tracks] == ["film.sup"]
    assert tracks[0]["text"] is False  # a .sup holds pictures like any PGS


def test_the_job_routes_a_graphic_track_to_ocr_and_then_proofreads_it(
    tmp_path, settings_file, monkeypatch
):
    """The whole point, end to end: no audio decoded, no whisper model, and
    — unlike a subtitle a person typed — the proofreading pass does run,
    because a machine produced these characters."""
    from app.models.schemas import PromptSettings
    from app.services import asr, audio, ocr as ocr_mod, pipeline, refine, translator
    from tests.test_translator import FakeClient

    settings_file(work_dir=str(tmp_path / "work"),
                  prompts=PromptSettings(refine_enabled=True, mark_lyrics=False))
    video = video_with_pgs(tmp_path)
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: pytest.fail("ASR ran"))
    monkeypatch.setattr(audio, "extract_audio", lambda *a, **k: pytest.fail("audio ran"))
    # the recogniser stands in; what is under test is the routing
    monkeypatch.setattr(
        ocr_mod, "_read_rapidocr",
        lambda cues, *a, **k: [("He11o there", 0.9), ("Second l1ne", 0.9)][: len(cues)],
    )
    seen: dict = {}
    real_refine = refine.refine_lines

    def spy(lines, *a, **kw):
        seen["source"] = kw.get("source")
        return real_refine(lines, *a, **kw)

    monkeypatch.setattr(refine, "refine_lines", spy)
    fake = FakeClient([
        "[1] Hello there\n[2] Second line",   # the proofread
        "Tom → 汤姆",                          # the glossary
        "[1] 你好啊\n[2] 第二行",              # the translation
    ])
    # refine imports the factory by name, so it holds its own binding — patch
    # both or the proofread quietly falls back to its input and the test
    # passes while the pass it is checking never ran
    monkeypatch.setattr(translator, "make_openai_client", lambda *a, **k: fake)
    monkeypatch.setattr(refine, "make_openai_client", lambda *a, **k: fake)

    workdir = tmp_path / "work" / "jobs" / "x"
    workdir.mkdir(parents=True)
    job = pipeline.Job(JobRequest(video_path=str(video), text_source="subtitle"))
    pipeline.manager._execute(job, job.request, workdir)

    assert job.status.stage == "done"
    assert seen["source"] == "ocr"  # the look-alike-glyph wording, not homophones
    written = Path(job.status.srt_filename).read_text(encoding="utf-8")
    assert "你好啊" in written
    # the digits the recogniser mistook for letters were put right, which is
    # the entire reason this pass runs on OCR output at all
    assert "Hello there" in written and "He11o" not in written


def test_a_batch_still_falls_back_to_speech_when_ocr_cannot_run(
    tmp_path, settings_file, monkeypatch
):
    from app.core.debuglog import DebugLog
    from app.models.schemas import AppSettings
    from app.services import ocr as ocr_mod, pipeline

    video = video_with_pgs(tmp_path)
    monkeypatch.setattr(
        ocr_mod, "read_cues",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(ocr_mod.MISSING_DEPENDENCY)),
    )
    job = pipeline.Job(JobRequest(
        video_path=str(video), text_source="subtitle", subtitle_fallback_asr=True
    ))
    got = pipeline.manager._import_subtitle(
        job, job.request, AppSettings(), DebugLog(None, enabled=False)
    )
    assert got is None  # the caller falls through to _transcribe
    assert any("改用语音识别" in e.log for e in job.events)


def test_a_single_file_job_fails_instead_of_silently_transcribing(
    tmp_path, monkeypatch
):
    from app.core.debuglog import DebugLog
    from app.models.schemas import AppSettings
    from app.services import ocr as ocr_mod, pipeline

    video = video_with_pgs(tmp_path)
    monkeypatch.setattr(
        ocr_mod, "read_cues",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(ocr_mod.MISSING_DEPENDENCY)),
    )
    job = pipeline.Job(JobRequest(video_path=str(video), text_source="subtitle"))
    with pytest.raises(RuntimeError, match="pip install"):
        pipeline.manager._import_subtitle(
            job, job.request, AppSettings(), DebugLog(None, enabled=False)
        )


# ------------------------------------------------------- the proofread pass


def test_the_speech_prompt_is_untouched_by_the_new_one():
    """The OCR variant must not drift the pass that was tuned on films.

    Byte-identical, because "the ASR path behaves exactly as before" is the
    property that keeps a second source from becoming a regression in the
    first.
    """
    from app.services.refine import build_refine_prompt

    asr = build_refine_prompt("ja", "藤堂 → 藤堂")
    assert asr == build_refine_prompt("ja", "藤堂 → 藤堂", source="asr")
    assert "输入是语音识别产生的字幕行" in asr
    assert "同音/近音词" in asr
    assert "口吃" in asr


def test_a_glyph_fix_on_a_short_line_survives_the_fidelity_check():
    """Measured in words, "He11o there" → "Hello there" keeps one word of
    two — 0.50, under the floor — and two-word subtitle lines are the common
    case. Every correction this pass exists to make would be thrown away."""
    from app.services.refine import _is_faithful

    assert _is_faithful("Hello there", "He11o there", ocr=True)
    assert not _is_faithful("Hello there", "He11o there")  # the word-based one


def test_the_ocr_check_still_refuses_a_rewrite():
    from app.services.refine import _is_faithful

    assert not _is_faithful("Something else entirely", "He11o there", ocr=True)
    assert not _is_faithful("", "He11o there", ocr=True)


def test_the_ocr_prompt_corrects_but_never_merges():
    """The line breaks came from a person; only the characters came from a
    machine. Merging them would undo work done against the picture."""
    from app.services.refine import build_refine_prompt

    prompt = build_refine_prompt("ja", source="ocr")
    assert "禁止合并行" in prompt and "一一对应" in prompt
    assert "字形相近" in prompt
    assert "同音" not in prompt  # that is the other machine's mistake
    assert "口吃" not in prompt  # ...and so is that
    assert "OCR" in prompt

"""Reading a subtitle the release already carries, instead of transcribing.

The claim being defended: what comes out of a video's own subtitle track is
exactly what a person typed into it — same words, same times — with the
formatting that only means something to a renderer taken off, and with
nothing invented, dropped or re-timed on the way.

The fixtures are built by the production muxer: a known list of cues is
written to .srt/.ass, embedded with mux.embed, and read back here. A test
that passes is therefore a genuine round trip through both halves.
"""

from pathlib import Path

import av
import pytest

from app.models.schemas import (
    AppSettings,
    JobRequest,
    SubtitleLine,
    SubtitleSettings,
)
from app.services import mux, subsource, subtitle
from tests.test_audio_tracks import make_multitrack_video

CUES = [
    SubtitleLine(index=1, start=0.5, end=1.2, text="Hello there"),
    SubtitleLine(index=2, start=1.4, end=1.9, text="Goodbye my friend"),
]


def write_subs(path: Path, lines=CUES, styled=False) -> Path:
    settings = SubtitleSettings(style_enabled=styled)
    text = (subtitle.build_ass if styled else subtitle.build_srt)(
        lines, settings, mode="translation_only"
    )
    # translation_only, and translation is empty, so build_* falls back to
    # the original text — one line per cue, which is what a real subtitle is
    path.write_text(text, encoding="utf-8")
    return path


def video_with_subs(tmp_path, lines=CUES, styled=False, language="English") -> Path:
    src = make_multitrack_video(tmp_path / "film.mkv")
    subs = write_subs(tmp_path / ("subs.ass" if styled else "subs.srt"), lines, styled)
    return mux.embed(src, subs, tmp_path / "withsubs.mkv", language,
                     track_title="Full")


def write_ass(path: Path, *events: str) -> Path:
    """An .ass carrying exactly the dialogue lines given."""
    head = (
        "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
        " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
        " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
        " Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,56,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,60,60,40,1\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV,"
        " Effect, Text\n"
    )
    path.write_text(head + "\n".join(events) + "\n", encoding="utf-8")
    return path


def read(video: Path, track=None, **kw):
    tracks = subsource.all_tracks(video)
    return subsource.read_cues(video, track or subsource.pick_track(tracks), **kw)


# ------------------------------------------------------------ enumeration


def test_an_embedded_track_reports_what_the_picker_needs(tmp_path):
    out = video_with_subs(tmp_path)
    tracks = subsource.list_tracks(out)
    assert len(tracks) == 1
    t = tracks[0]
    with av.open(str(out)) as c:  # the index is the container's, not a count
        assert t["index"] == c.streams.subtitles[0].index
    assert t["language"] == "eng"
    assert t["language_name"] == "英语"
    assert t["title"] == "Full"
    assert t["codec"] == "SRT"
    assert t["default"] is True and t["forced"] is False
    assert t["text"] is True
    assert t["path"] == ""


def test_a_video_without_subtitles_is_not_an_error(tmp_path):
    plain = make_multitrack_video(tmp_path / "plain.mkv")
    assert subsource.list_tracks(plain) == []
    assert subsource.all_tracks(plain) == []


def test_sidecar_files_are_offered_alongside_the_embedded_ones(tmp_path):
    out = video_with_subs(tmp_path)
    write_subs(out.with_suffix(".srt"))
    write_subs(out.parent / f"{out.stem}.ja.ass", styled=True)
    write_subs(tmp_path / "someone-elses.srt")  # different stem: not ours

    tracks = subsource.all_tracks(out)
    names = [Path(t["path"]).name for t in tracks if t["path"]]
    assert sorted(names) == ["withsubs.ja.ass", "withsubs.srt"]
    # the .ja. in the middle is a language tag, and it is read as one
    ja = next(t for t in tracks if t["path"].endswith(".ja.ass"))
    assert ja["language"] == "jpn" and ja["index"] == -1


def test_the_description_says_why_a_track_cannot_be_used(tmp_path):
    bitmap = {"index": 4, "path": "", "codec": "HDMV_PGS_SUBTITLE",
              "language": "eng", "language_name": "英语", "title": "",
              "default": False, "forced": False, "text": False}
    assert "图形字幕" in subsource.describe_track(bitmap)
    assert "字幕轨 #4" in subsource.describe_track(bitmap)


# ---------------------------------------------------------------- picking


def test_the_default_readable_track_is_the_one_picked():
    tracks = [
        {"index": 2, "path": "", "text": True, "forced": True, "default": False,
         "language": "eng", "codec": "SRT"},
        {"index": 3, "path": "", "text": True, "forced": False, "default": True,
         "language": "eng", "codec": "SRT"},
    ]
    assert subsource.pick_track(tracks)["index"] == 3


def test_a_forced_track_is_never_chosen_over_a_full_one():
    """A forced track holds only signs and foreign lines — a few dozen cues
    where the full one has thousands."""
    tracks = [
        {"index": 2, "path": "", "text": True, "forced": True, "default": True,
         "language": "eng", "codec": "SRT"},
        {"index": 3, "path": "", "text": True, "forced": False, "default": False,
         "language": "eng", "codec": "SRT"},
    ]
    assert subsource.pick_track(tracks)["index"] == 3
    # ...unless it is all there is
    assert subsource.pick_track(tracks[:1])["index"] == 2


def test_a_language_preference_beats_the_default_flag():
    tracks = [
        {"index": 2, "path": "", "text": True, "forced": False, "default": True,
         "language": "eng", "codec": "SRT"},
        {"index": 3, "path": "", "text": True, "forced": False, "default": False,
         "language": "jpn", "codec": "ASS"},
    ]
    assert subsource.pick_track(tracks, language="ja")["index"] == 3


BITMAP = {"index": 2, "path": "", "text": False, "forced": False,
          "default": True, "language": "eng", "codec": "HDMV_PGS_SUBTITLE",
          "language_name": "英语", "title": ""}
TEXT = {"index": 3, "path": "", "text": True, "forced": False,
        "default": False, "language": "jpn", "codec": "ASS",
        "language_name": "日语", "title": ""}


def test_a_text_track_always_beats_a_graphic_one():
    """Graphic tracks are readable now (services/ocr.py), but OCR costs
    minutes and can misread; a text track is free and exact."""
    assert subsource.pick_track([BITMAP, TEXT])["index"] == 3
    assert subsource.pick_track([TEXT, BITMAP])["index"] == 3
    # ...even when the graphic one is the flagged default, as it is here


def test_a_graphic_track_is_returned_when_it_is_asked_for_or_is_all_there_is():
    assert subsource.pick_track([BITMAP, TEXT], index=2)["index"] == 2
    assert subsource.pick_track([BITMAP])["index"] == 2


def test_no_subtitle_at_all_is_its_own_message():
    with pytest.raises(ValueError, match="没有内建字幕轨"):
        subsource.pick_track([])


def test_a_stale_index_falls_back_instead_of_failing():
    """Same rule as audio.pick_track: the video may have been re-picked
    after the track was chosen, and losing the job over it helps nobody."""
    tracks = [{"index": 3, "path": "", "text": True, "forced": False,
               "default": True, "language": "eng", "codec": "SRT"}]
    assert subsource.pick_track(tracks, index=99)["index"] == 3


# ------------------------------------------------------------ reading back


@pytest.mark.parametrize("styled", [False, True], ids=["srt", "ass"])
def test_the_cues_come_back_exactly_as_they_went_in(tmp_path, styled):
    lines = read(video_with_subs(tmp_path, styled=styled))
    assert [l.text for l in lines] == [c.text for c in CUES]
    assert [round(l.start, 3) for l in lines] == [0.5, 1.4]
    assert [round(l.end, 3) for l in lines] == [1.2, 1.9]
    assert [l.index for l in lines] == [1, 2]


def test_a_sidecar_file_reads_the_same_as_an_embedded_track(tmp_path):
    video = make_multitrack_video(tmp_path / "film.mkv")
    write_subs(tmp_path / "film.srt")
    lines = read(video)
    assert [l.text for l in lines] == [c.text for c in CUES]
    assert [round(l.start, 3) for l in lines] == [0.5, 1.4]


def test_formatting_that_only_a_renderer_understands_is_taken_off(tmp_path):
    video = make_multitrack_video(tmp_path / "film.mkv")
    write_ass(
        tmp_path / "film.ass",
        r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\an8}Sign on the wall",
        r"Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,- Where are you?\N- Right here",
        r"Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,<i>Whispering</i> now",
        r"Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,Wait{\i1} for{\i0} me\hplease",
    )
    assert [l.text for l in read(video)] == [
        "Sign on the wall",
        # a cue is one line for the LLM; the renderer re-wraps it later
        "- Where are you? - Right here",
        "Whispering now",
        "Wait for me please",
    ]


def test_a_vector_drawing_is_not_a_line_of_dialogue(tmp_path):
    """{\\p1} switches the cue body to drawing commands. "m 0 0 l 100 0"
    reads as content to any character test, and would be translated."""
    video = make_multitrack_video(tmp_path / "film.mkv")
    write_ass(
        tmp_path / "film.ass",
        r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\p1}m 0 0 l 100 0 100 100{\p0}",
        r"Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,{\pos(960,540)}Real dialogue",
    )
    stats: dict = {}
    lines = read(video, stats=stats)
    assert [l.text for l in lines] == ["Real dialogue"]  # \pos survives, \p1 does not
    assert stats["blank"] == 1


def test_cues_with_nothing_to_read_are_dropped(tmp_path):
    video = make_multitrack_video(tmp_path / "film.mkv")
    write_ass(
        tmp_path / "film.ass",
        r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,♪♪",
        r"Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,...",
        r"Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,Something said",
    )
    stats: dict = {}
    assert [l.text for l in read(video, stats=stats)] == ["Something said"]
    assert stats["blank"] == 2


def test_a_karaoke_repeat_is_folded_but_a_real_one_is_not(tmp_path):
    """Effect tracks re-emit the same words while they are still on screen.
    A character saying the same thing again later is a separate subtitle."""
    video = make_multitrack_video(tmp_path / "film.mkv")
    write_ass(
        tmp_path / "film.ass",
        r"Dialogue: 0,0:00:20.00,0:00:22.00,Default,,0,0,0,,{\k30}Sing{\k20}ing along",
        r"Dialogue: 0,0:00:21.00,0:00:23.00,Default,,0,0,0,,Singing along",
        r"Dialogue: 0,0:00:30.00,0:00:32.00,Default,,0,0,0,,Singing along",
    )
    stats: dict = {}
    lines = read(video, stats=stats)
    assert [(round(l.start, 1), round(l.end, 1), l.text) for l in lines] == [
        (20.0, 23.0, "Singing along"),
        (30.0, 32.0, "Singing along"),
    ]
    assert stats["folded"] == 1


def test_the_subtitles_own_music_marks_are_believed(tmp_path):
    """The file already says which lines are sung — no need to ask a model,
    and the ♪ have to survive to the output either way."""
    video = make_multitrack_video(tmp_path / "film.mkv")
    write_ass(
        tmp_path / "film.ass",
        r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,♪ We'll meet again ♪",
        r"Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,Turn that off",
    )
    lines = read(video)
    assert [l.is_lyric for l in lines] == [True, False]
    # stripped on the way in so the model is not asked to translate a symbol;
    # lyrics.apply_marks puts them back around the finished text
    assert lines[0].text == "We'll meet again"


def test_an_empty_track_is_reported_rather_than_returned_empty(tmp_path):
    video = make_multitrack_video(tmp_path / "film.mkv")
    write_ass(tmp_path / "film.ass")
    with pytest.raises(ValueError, match="没有可读的字幕内容"):
        read(video)


def test_a_source_that_starts_late_still_reads_from_zero(tmp_path):
    """Mirror image of the muxer: it pulls the copy back to zero, so cues
    read out of one have to be pulled back by the same amount or a
    read-then-embed round trip would drift."""
    from tests.test_mux import _video_starting_at

    late = _video_starting_at(tmp_path / "late.mkv", 10.0)
    with_subs = mux.embed(late, write_subs(tmp_path / "s.srt"),
                          tmp_path / "out.mkv", "English")
    lines = read(with_subs)
    assert [round(l.start, 2) for l in lines] == [0.5, 1.4]


# --------------------------------------------------------------- language


def test_the_tracks_own_tag_is_the_first_answer(tmp_path):
    out = video_with_subs(tmp_path, language="日本語")
    assert subsource.iso2(subsource.list_tracks(out)[0]["language"]) == "ja"


@pytest.mark.parametrize("expected,text", [
    ("ja", ["これは何だと思う?", "森林写真じゃない?"]),
    ("zh", ["你觉得这是什么？", "不是森林的照片吗？"]),
    ("ko", ["이게 뭐라고 생각해?", "숲 사진 아니야?"]),
    ("ru", ["Что это такое?", "Это же фото леса?"]),
    ("en", ["What do you suppose this is?", "Not a forest photo?"]),
    ("fr", ["Que pensez-vous que ce soit ?", "Une photo de la forêt, non ?"]),
    ("de", ["Was denkst du, ist das nicht?", "Ein Waldfoto, oder?"]),
])
def test_an_untagged_subtitle_is_identified_from_its_text(expected, text):
    lines = [SubtitleLine(index=i, start=i, end=i + 1, text=t)
             for i, t in enumerate(text, start=1)]
    assert subsource.detect_language(lines) == expected


def test_a_stray_foreign_character_does_not_decide_the_language():
    lines = [SubtitleLine(index=1, start=0, end=1,
                          text="The Tokyo 東京 office and the rest of that team")]
    assert subsource.detect_language(lines) == "en"


def test_nothing_readable_means_no_guess():
    lines = [SubtitleLine(index=1, start=0, end=1, text="...")]
    assert subsource.detect_language(lines) == ""


# --------------------------------------------------------------- pipeline


def _job(tmp_path, **kw):
    from app.services.pipeline import Job

    return Job(JobRequest(video_path=str(tmp_path / "film.mkv"),
                          text_source="subtitle", **kw))


def test_importing_never_reaches_the_recogniser(tmp_path, monkeypatch):
    """The whole point: no model is loaded, no audio is decoded."""
    from app.core.debuglog import DebugLog
    from app.services import asr, audio, pipeline

    out = video_with_subs(tmp_path)
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: pytest.fail("ASR ran"))
    monkeypatch.setattr(audio, "extract_audio", lambda *a, **k: pytest.fail("audio ran"))

    job = pipeline.Job(JobRequest(video_path=str(out), text_source="subtitle"))
    lines, detected, kind = pipeline.manager._import_subtitle(
        job, job.request, AppSettings(), DebugLog(None, enabled=False)
    )
    assert [l.text for l in lines] == [c.text for c in CUES]
    assert detected == "en"
    assert kind == "subtitle"  # typed by a person, so no proofreading pass
    assert job.status.stage == "importing"


def test_a_whole_job_runs_from_the_subtitle_and_skips_preprocessing(
    tmp_path, settings_file, monkeypatch
):
    """End to end with no audio, no model and no network.

    Also pins the one behaviour the extra source changes downstream: the
    refine pass is for repairing speech recognition, and a subtitle written
    by a person has nothing for it to repair.
    """
    from app.models.schemas import PromptSettings
    from app.services import pipeline, refine, translator
    from tests.test_translator import FakeClient

    # work_dir keeps the cache and the job log inside tmp_path rather than in
    # the developer's own MovieTranslator directory
    settings_file(work_dir=str(tmp_path / "work"),
                  prompts=PromptSettings(refine_enabled=True, mark_lyrics=False))
    out = video_with_subs(tmp_path)
    monkeypatch.setattr(
        refine, "refine_lines", lambda *a, **k: pytest.fail("refine ran")
    )
    fake = FakeClient(["Tom → 汤姆", "[1] 你好啊\n[2] 再见了朋友"])
    monkeypatch.setattr(translator, "make_openai_client", lambda *a, **k: fake)

    workdir = tmp_path / "work" / "jobs" / "x"
    workdir.mkdir(parents=True)
    job = pipeline.Job(JobRequest(video_path=str(out), text_source="subtitle"))
    pipeline.manager._execute(job, job.request, workdir)

    assert job.status.stage == "done"
    written = Path(job.status.srt_filename).read_text(encoding="utf-8")
    assert "你好啊" in written and "Hello there" in written
    assert any("已跳过转写预处理" in e.log for e in job.events)


def test_a_single_file_job_fails_loudly_when_the_subtitle_is_unusable(tmp_path):
    """Someone who picked a track by hand is waiting seconds, not an hour."""
    from app.core.debuglog import DebugLog
    from app.services import pipeline

    plain = make_multitrack_video(tmp_path / "plain.mkv")
    job = pipeline.Job(JobRequest(video_path=str(plain), text_source="subtitle"))
    with pytest.raises(ValueError, match="没有内建字幕轨"):
        pipeline.manager._import_subtitle(
            job, job.request, AppSettings(), DebugLog(None, enabled=False)
        )


def test_a_batch_job_transcribes_that_file_instead(tmp_path):
    """One episode shipped without a subtitle must not stop the season."""
    from app.core.debuglog import DebugLog
    from app.services import pipeline

    plain = make_multitrack_video(tmp_path / "plain.mkv")
    job = pipeline.Job(JobRequest(
        video_path=str(plain), text_source="subtitle", subtitle_fallback_asr=True
    ))
    got = pipeline.manager._import_subtitle(
        job, job.request, AppSettings(), DebugLog(None, enabled=False)
    )
    assert got is None  # the caller falls through to _transcribe
    assert any("改用语音识别" in e.log for e in job.events)


def test_the_batch_sets_that_fallback_and_a_single_job_does_not(tmp_path):
    from app.models.schemas import BatchRequest
    from app.services import batch as batch_mod

    made: list[JobRequest] = []

    class Recorder:
        def create(self, req):
            made.append(req)
            raise RuntimeError("stop here")  # no need to actually run one

    (tmp_path / "ep1.mkv").write_bytes(b"not really a video")
    monkey = batch_mod.job_manager
    batch_mod.job_manager = Recorder()
    try:
        batch_mod.batch_manager.create(BatchRequest(
            directory=str(tmp_path), text_source="subtitle",
            subtitle_language="eng", skip_existing_srt=False,
        ))
    finally:
        batch_mod.job_manager = monkey

    assert made and made[0].subtitle_fallback_asr is True
    assert made[0].text_source == "subtitle"
    assert made[0].subtitle_language == "eng"
    assert JobRequest(video_path="x").subtitle_fallback_asr is False


# ------------------------------------------------------------------- API


def test_the_probe_endpoint_lists_both_kinds(tmp_path):
    from app.main import app
    from tests.conftest import local_client

    out = video_with_subs(tmp_path)
    write_subs(out.with_suffix(".srt"))
    with local_client(app) as client:
        resp = client.get("/api/media/subtitle-tracks", params={"path": str(out)})
    assert resp.status_code == 200
    kinds = [(t["index"], t["text"]) for t in resp.json()]
    assert (-1, True) in kinds  # the sidecar
    assert len(kinds) == 2


def test_the_probe_endpoint_is_quiet_about_having_nothing_to_offer(tmp_path):
    """Most films have no subtitle track; that is an answer, not a failure."""
    from app.main import app
    from tests.conftest import local_client

    plain = make_multitrack_video(tmp_path / "plain.mkv")
    with local_client(app) as client:
        resp = client.get("/api/media/subtitle-tracks", params={"path": str(plain)})
    assert resp.status_code == 200 and resp.json() == []

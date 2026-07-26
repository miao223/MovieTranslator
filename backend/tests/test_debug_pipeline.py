"""The debug log must actually capture every stage, not just exist.

Each stage writes into it from a different module, so a wiring mistake in
one of them is invisible until someone needs the file — which is exactly
when it is too late to re-run a two-hour transcription.
"""

from app.core.debuglog import DebugLog
from app.models.schemas import (
    LLMSettings,
    SubtitleLine,
    SubtitleSettings,
)
from app.services import segmenter, subtitle
from app.services.asr import Segment
from app.services.pipeline import JobManager
from app.services.refine import refine_lines
from app.services.translator import Translator
from app.services.vet import vet_recovered
from tests.test_translator import FakeClient


class Settings:
    subtitle = SubtitleSettings()


def test_every_stage_writes_into_the_debug_log(tmp_path):
    path = tmp_path / "movie.debug.log"
    debug = DebugLog(path, enabled=True)

    segments = vet_recovered(
        [
            Segment(0.0, 2.0, "ちゃんと話そう"),
            Segment(2.1, 2.4, "JR東日本E233系電車", recovered=True),
            Segment(2.5, 4.0, "それでいいよね"),
        ],
        LLMSettings(model="m"),
        client=FakeClient(["[R1] 丢弃 车站广播，与本片无关"]),
        debug=debug,
    )
    assert [s.text for s in segments] == ["ちゃんと話そう", "それでいいよね"]

    lines = segmenter.segment_lines(segments, SubtitleSettings(), debug=debug)
    assert lines

    lines = refine_lines(
        lines,
        LLMSettings(model="m"),
        SubtitleSettings(),
        client=FakeClient(["[1] ちゃんと話そう。\n[2] それでいいよね？"]),
        glossary="藤堂 → 藤堂",
        debug=debug,
    )

    translator = Translator(
        LLMSettings(model="m"),
        target_language="简体中文",
        client=FakeClient(["藤堂 → 藤堂", "[1] 好好谈谈吧\n[2] 这样可以吧"]),
        debug=debug,
    )
    translator.translate(lines)
    JobManager._debug_final(debug, lines, Settings())

    text = path.read_text(encoding="utf-8")
    for section in ("二次识别复核", "分句结果", "转写预处理", "翻译（translator）", "最终字幕"):
        assert section in text, f"missing section: {section}"
    assert "system prompt" in text
    assert "车站广播，与本片无关" in text      # why a recovered line was dropped
    assert "藤堂" in text                     # glossary reached the prompt
    assert "好好谈谈吧" in text                # final translation recorded
    assert "间隔阈值 gap_limit" in text        # the adaptive threshold used


def test_debug_off_leaves_no_file_and_no_cost(tmp_path):
    path = tmp_path / "movie.debug.log"
    debug = DebugLog(path, enabled=False)
    lines = segmenter.segment_lines([(0.0, 2.0, "hello there")], SubtitleSettings(), debug=debug)
    JobManager._debug_final(debug, lines, Settings())
    assert not path.exists()

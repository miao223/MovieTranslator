from app.models.schemas import SubtitleLine, SubtitleSettings
from app.services.subtitle import build_srt, format_timestamp


def make_lines():
    return [
        SubtitleLine(index=1, start=0.0, end=2.5, text="Hello there.", translation="你好。"),
        SubtitleLine(index=2, start=2.5, end=5.0, text="General Kenobi!", translation="肯诺比将军！"),
    ]


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(-1) == "00:00:00,000"
    # rounding must not produce 1000 ms
    assert format_timestamp(1.9996) == "00:00:02,000"


def test_bilingual_srt_translation_bottom():
    srt = build_srt(make_lines(), SubtitleSettings(), mode="bilingual")
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].splitlines() == [
        "1",
        "00:00:00,000 --> 00:00:02,500",
        "Hello there.",
        "你好。",
    ]


def test_bilingual_translation_top():
    settings = SubtitleSettings(bilingual_layout="translation_top")
    srt = build_srt(make_lines(), settings, mode="bilingual")
    assert srt.splitlines()[2] == "你好。"


def test_translation_only_and_fallback():
    lines = make_lines()
    lines[1].translation = ""  # missing translation falls back to original
    srt = build_srt(lines, SubtitleSettings(), mode="translation_only")
    blocks = srt.strip().split("\n\n")
    assert blocks[0].splitlines()[2] == "你好。"
    assert blocks[1].splitlines()[2] == "General Kenobi!"

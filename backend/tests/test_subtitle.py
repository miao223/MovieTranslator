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


# ------------------------------------------------------- display wrapping


def test_wrap_returns_short_text_untouched():
    from app.services.subtitle import wrap_display_text

    assert wrap_display_text("Hello there.", 42) == ["Hello there."]


def test_wrap_breaks_near_the_middle_at_a_word_boundary():
    from app.services.subtitle import wrap_display_text

    parts = wrap_display_text("Yeah, like that thought never entered my mind.", 30)
    assert len(parts) == 2
    assert all(len(p) <= 30 for p in parts)
    assert " ".join(parts) == "Yeah, like that thought never entered my mind."


def test_wrap_never_hard_cuts_an_unsplittable_string():
    from app.services.subtitle import wrap_display_text

    assert wrap_display_text("a" * 60, 20) == ["a" * 60]


def test_wrap_at_most_two_lines():
    from app.services.subtitle import wrap_display_text

    long_text = " ".join(["word"] * 40)
    assert len(wrap_display_text(long_text, 20)) <= 2


def test_srt_wraps_both_languages():
    from app.models.schemas import SubtitleLine, SubtitleSettings
    from app.services.subtitle import build_srt

    line = SubtitleLine(
        index=1, start=0.0, end=3.0,
        text="Yeah, like that thought never entered my mind.",
        translation="是啊，好像我压根就没往那儿想过这件事情。",
    )
    out = build_srt([line], SubtitleSettings(max_chars_per_line=12))
    body = out.split("\n", 2)[2]
    assert len(body.strip().splitlines()) == 4  # 2 original + 2 translation


def test_ass_wraps_with_backslash_n():
    from app.models.schemas import SubtitleLine, SubtitleSettings
    from app.services.subtitle import build_ass

    line = SubtitleLine(
        index=1, start=0.0, end=3.0,
        text="Yeah, like that thought never entered my mind.",
        translation="是啊，我压根没往那儿想。",
    )
    out = build_ass([line], SubtitleSettings(max_chars_per_line=24, style_enabled=True))
    dialogue = [l for l in out.splitlines() if l.startswith("Dialogue:")][0]
    # the original wraps into two balanced halves joined by an ASS line break
    assert "thought\\Nnever" in dialogue
    assert "{" not in line.text  # sanity: escaping untouched by the wrap

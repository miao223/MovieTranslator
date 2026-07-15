from app.models.schemas import SubtitleLine, SubtitleSettings
from app.services.subtitle import _ass_color, build_ass, format_ass_timestamp


def make_lines():
    return [
        SubtitleLine(index=1, start=0.0, end=2.5, text="Hello there.", translation="你好。"),
        SubtitleLine(index=2, start=2.5, end=5.0, text="Bye {brace}", translation="再见"),
    ]


def test_ass_color_conversion():
    assert _ass_color("#FFFFFF") == "&H00FFFFFF"
    assert _ass_color("#FF0000") == "&H000000FF"  # red → BGR order
    assert _ass_color("#12AB34") == "&H0034AB12"
    assert _ass_color("bad") == "&H00FFFFFF"  # fallback


def test_ass_timestamp():
    assert format_ass_timestamp(0) == "0:00:00.00"
    assert format_ass_timestamp(3661.55) == "1:01:01.55"


def test_ass_structure_and_styles():
    s = SubtitleSettings(
        style_enabled=True, font_size=60, original_font_size=36,
        translation_color="#FFEE00", original_color="#B4B4B4",
    )
    ass = build_ass(make_lines(), s, mode="bilingual")
    assert "PlayResX: 1920" in ass
    assert "Style: Default,Arial,60,&H0000EEFF," in ass  # translation style
    # bilingual default (translation_bottom): original styled inline first
    assert "{\\fs36\\c&H00B4B4B4&}Hello there.{\\r}\\N你好。" in ass
    assert ass.count("Dialogue:") == 2
    # braces in text are neutralised, not parsed as tags
    assert "{brace}" not in ass and "（brace）" in ass


def test_ass_translation_top_and_only():
    s = SubtitleSettings(style_enabled=True)
    top = build_ass(make_lines(), s.model_copy(update={"bilingual_layout": "translation_top"}))
    assert "你好。\\N{\\fs" in top
    only = build_ass(make_lines(), s, mode="translation_only")
    assert "Hello there." not in only.split("[Events]")[1]
    assert "你好。" in only

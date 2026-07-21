from app.models.schemas import PromptSettings
from app.services.translator import build_system_prompt


def test_default_prompt_contains_all_asr_rules():
    p = build_system_prompt(PromptSettings(), "简体中文", synopsis="太空歌剧")
    assert "简体中文" in p
    assert "[行号] 译文" in p
    assert "同音" in p           # fix_asr_errors
    assert "同一句话被拆成" in p  # link_fragments
    assert "片假名" in p          # normalize_loanwords
    assert "42 个字符" in p       # limit_length uses max_line_chars default
    assert "太空歌剧" in p


def test_switches_remove_rules():
    p = build_system_prompt(
        PromptSettings(
            fix_asr_errors=False,
            link_fragments=False,
            normalize_loanwords=False,
            limit_length=False,
        ),
        "简体中文",
    )
    assert "同音" not in p
    assert "片假名" not in p
    assert "个字符" not in p


def test_glossary_and_extra_included():
    p = build_system_prompt(
        PromptSettings(glossary="Gandalf → 甘道夫", extra="歌词保留原文"),
        "简体中文",
    )
    assert "Gandalf → 甘道夫" in p
    assert "歌词保留原文" in p


def test_custom_override_wins():
    p = build_system_prompt(
        PromptSettings(custom_system_prompt="翻译成{target_language}。简介：{synopsis}"),
        "日本語",
        synopsis="剧情X",
    )
    assert p == "翻译成日本語。简介：剧情X"


def test_max_line_chars_propagates():
    p = build_system_prompt(PromptSettings(), "简体中文", max_line_chars=20)
    assert "20 个字符" in p


# --------------------------------- no-borrowing rule (the line-shift bug)


def test_prompt_forbids_borrowing_content_across_lines():
    p = build_system_prompt(PromptSettings(), "简体中文")
    assert "严禁把下一行的内容提前翻译到本行" in p
    assert "宁可单行译文读起来不完整" in p


def test_prompt_no_longer_invites_reflowing():
    """The old wording asked for lines to 'read as one fluent sentence',
    which is exactly what made the model move content between lines."""
    p = build_system_prompt(PromptSettings(), "简体中文")
    assert "使连起来读是通顺的一句话" not in p


def test_no_borrowing_rule_survives_every_switch_off():
    off = PromptSettings(
        fix_asr_errors=False, link_fragments=False,
        normalize_loanwords=False, limit_length=False, tone="",
    )
    assert "严禁把下一行的内容提前翻译到本行" in build_system_prompt(off, "简体中文")

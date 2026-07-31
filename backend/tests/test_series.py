"""Series mode: one glossary shared by every episode of a batch.

The whole feature is a promise about consistency, so the tests are about
what happens when two episodes disagree — not about parsing for its own
sake.
"""

import pytest

from app.services import series

REPLY = """
以下是主要人名/术语对照表：

- 佐藤健一 → 佐藤健一
- タカシ → 隆
1. 白金台 → 白金台
| 呪術廻戦 → 咒术回战 |
"""


# ----------------------------------------------------------------- parsing


def test_terms_survive_the_shapes_models_actually_use():
    """Bullets, numbering, table pipes — the prompt asks for none of them
    and models produce all of them."""
    assert series.parse_terms(REPLY) == [
        ("佐藤健一", "佐藤健一"),
        ("タカシ", "隆"),
        ("白金台", "白金台"),
        ("呪術廻戦", "咒术回战"),
    ]


def test_only_an_arrow_makes_a_term():
    """Accepting ':' as well would turn the instruction the model is
    echoing back into a glossary entry."""
    prose = (
        "```\n"
        "注意：右侧必须是简体中文译名\n"
        "好的，我明白了。\n"
        "| 原文 | 译名 |\n"
        "```\n"
    )
    assert series.parse_terms(prose) == []


def test_a_header_row_is_not_a_term():
    assert series.parse_terms("原文 → 译名\n佐藤 → 佐藤") == [("佐藤", "佐藤")]


def test_a_sentence_is_not_a_term():
    """A model that starts narrating must not push real terms out."""
    long_side = "这句话很长" * 20
    assert series.parse_terms(f"{long_side} → 译文\n佐藤 → 佐藤") == [("佐藤", "佐藤")]


# --------------------------------------------------------------- accumulate


def test_the_first_episode_to_name_something_decides_it():
    g = series.SeriesGlossary(id="s1")
    assert g.learn("タカシ → 隆")[0] == 1
    added, clashes = g.learn("タカシ → 高志\nユキ → 雪")

    assert added == 1  # only ユキ is new
    assert g.terms["タカシ"] == "隆"  # episode 1 holds
    assert len(clashes) == 1 and "隆" in clashes[0] and "高志" in clashes[0]
    assert g.conflicts == clashes  # kept for the human to look at


def test_an_agreeing_episode_reports_no_conflict():
    g = series.SeriesGlossary(id="s2")
    g.learn("タカシ → 隆")
    added, clashes = g.learn("タカシ → 隆")
    assert (added, clashes) == (0, [])


def test_the_users_own_table_outranks_the_model():
    """Terms written by hand in settings are the authority: never
    overwritten, and never absorbed into the accumulated table either —
    they are already in the prompt, ahead of it."""
    user = "タカシ → 塔卡西"
    g = series.SeriesGlossary(id="s3")
    added, clashes = g.learn("タカシ → 隆\nユキ → 雪", user_glossary=user)

    assert added == 1 and clashes == []
    assert "タカシ" not in g.terms
    merged = g.merged(user)
    assert merged.splitlines() == ["タカシ → 塔卡西", "ユキ → 雪"]


def test_the_cap_keeps_the_earliest_terms():
    g = series.SeriesGlossary(id="s4")
    g.learn("\n".join(f"名{i} → 名{i}" for i in range(series.MAX_TERMS + 50)))
    assert len(g) == series.MAX_TERMS
    assert "名0" in g.terms and f"名{series.MAX_TERMS + 10}" not in g.terms


# ------------------------------------------------------------------ merged


def test_merged_puts_the_users_text_first_and_verbatim():
    """It may hold notes and formatting of their own; it is not ours to
    rewrite."""
    user = "# 我自己的表\nタカシ → 塔卡西"
    g = series.SeriesGlossary(id="s5")
    g.learn("ユキ → 雪")
    assert g.merged(user) == "# 我自己的表\nタカシ → 塔卡西\nユキ → 雪"


def test_merged_degrades_to_each_side_alone():
    g = series.SeriesGlossary(id="s6")
    assert g.merged("タカシ → 塔卡西") == "タカシ → 塔卡西"
    assert g.merged("") == ""
    g.learn("ユキ → 雪")
    assert g.merged("") == "ユキ → 雪"


# ------------------------------------------------------------------- store


def test_the_store_hands_back_the_same_table_and_nothing_for_a_stray_id():
    g = series.create("batch-x")
    g.learn("ユキ → 雪")
    assert series.get("batch-x") is g
    assert len(series.get("batch-x")) == 1
    # a single-file job carries no series id and must never find a table
    assert series.get("") is None
    assert series.get("no-such-batch") is None


# ---------------------------------------------------------------- pipeline


def test_a_job_without_a_series_translates_with_the_users_glossary_only():
    """Series mode off — the whole of a normal job's behaviour must be
    unchanged, which for this feature means: exactly what was configured."""
    assert series.for_job("", "タカシ → 塔卡西") == (None, "タカシ → 塔卡西")
    assert series.for_job("no-such-batch", "") == (None, "")


def test_a_job_in_a_series_gets_the_batch_table_behind_the_users():
    shared = series.create("batch-for-job")
    shared.learn("ユキ → 雪")
    found, glossary = series.for_job("batch-for-job", "タカシ → 塔卡西")
    assert found is shared
    assert glossary == "タカシ → 塔卡西\nユキ → 雪"


def test_the_cached_settings_object_is_never_written_to():
    """load_settings() hands back the instance config caches, so a job that
    edited prompts.glossary in place would rewrite every later job's prompt
    — including jobs of other batches. The pipeline copies instead; this is
    what that copy has to guarantee."""
    from app.models.schemas import AppSettings

    settings = AppSettings()
    settings.prompts.glossary = "タカシ → 塔卡西"
    shared = series.create("batch-copy")
    shared.learn("ユキ → 雪")

    _, glossary = series.for_job("batch-copy", settings.prompts.glossary)
    prompts = settings.prompts.model_copy(update={"glossary": glossary})

    assert "ユキ" in prompts.glossary
    assert settings.prompts.glossary == "タカシ → 塔卡西"


@pytest.mark.parametrize(
    "mode,count,marker",
    [("global", 5, "全局上下文模式"), ("chunked", 40, "滑动窗口分块")],
)
def test_the_translator_hands_back_the_glossary_it_settled_on(mode, count, marker):
    """Series mode reads exactly this, and both modes build a glossary by a
    different route — one in-conversation, one from a sample of the film."""
    from tests.test_translator import FakeClient  # mock LLM

    from app.models.schemas import LLMSettings, SubtitleLine
    from app.services.translator import Translator

    # CJK counts a token per character, which is what pushes the chunked
    # case over its context budget
    lines = [
        SubtitleLine(index=i, start=i, end=i + 1, text="这是一句台词" * 4)
        for i in range(1, count + 1)
    ]
    replies = ["タカシ → 隆"] + [
        "\n".join(f"[{line.index}] 译文{line.index}" for line in lines)
    ] * 4
    logs = []
    tr = Translator(
        LLMSettings(context_limit=100_000 if mode == "global" else 1_000),
        target_language="简体中文",
        client=FakeClient(replies),
        log=logs.append,
    )
    tr.translate(lines)

    assert any(marker in line for line in logs), logs  # the mode under test
    assert series.parse_terms(tr.glossary_text) == [("タカシ", "隆")]
    assert all(line.translation for line in lines)

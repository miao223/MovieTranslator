"""Translator protocol tests with a scripted fake OpenAI client."""

import pytest

from app.models.schemas import LLMSettings, SubtitleLine
from app.services.translator import (
    TranslationError,
    Translator,
    estimate_tokens,
    parse_translations,
)


class FakeClient:
    """Returns queued responses; records every request's messages."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature, **kw):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("no scripted response left")
        content = self.responses.pop(0)

        class Msg:
            pass

        msg = Msg()
        msg.content = content
        choice = Msg()
        choice.message = msg
        resp = Msg()
        resp.choices = [choice]
        return resp


def make_lines(n):
    return [
        SubtitleLine(index=i, start=float(i), end=float(i + 1), text=f"line {i}")
        for i in range(1, n + 1)
    ]


def settings(**kw):
    defaults = dict(base_url="http://x", api_key="k", model="m", batch_size=3)
    defaults.update(kw)
    return LLMSettings(**defaults)


# ------------------------------------------------------------- parsing


def test_parse_translations_basic():
    text = "[1] 你好\n[2] 世界"
    assert parse_translations(text) == {1: "你好", 2: "世界"}


def test_parse_translations_tolerates_fences_and_variants():
    text = "```\n[1] 你好\n2. 世界\n[3]：再见\n继续的一行\n```"
    parsed = parse_translations(text)
    assert parsed[1] == "你好"
    assert parsed[2] == "世界"
    assert parsed[3].startswith("再见")
    assert "继续的一行" in parsed[3]


def test_estimate_tokens_cjk_heavier_than_latin():
    assert estimate_tokens("你好世界") > estimate_tokens("abcd")


# --------------------------------------------------------- global mode


def test_global_mode_translates_all_lines():
    lines = make_lines(5)  # batch_size=3 → 2 batches
    fake = FakeClient([
        "术语表：\nfoo → 富",  # glossary
        "[1] 一\n[2] 二\n[3] 三",
        "[4] 四\n[5] 五",
    ])
    tr = Translator(settings(), "简体中文", client=fake)
    tr.translate(lines)
    assert [l.translation for l in lines] == ["一", "二", "三", "四", "五"]
    # first request carries the whole numbered transcript
    first_user = fake.calls[0][1]["content"]
    for i in range(1, 6):
        assert f"[{i}] line {i}" in first_user
    # timestamps never sent
    assert "start" not in first_user and "-->" not in first_user


def test_missing_lines_are_rerequested():
    lines = make_lines(3)
    fake = FakeClient([
        "glossary",
        "[1] 一\n[3] 三",      # line 2 missing
        "[2] 二",               # repair round
    ])
    tr = Translator(settings(), "简体中文", client=fake)
    tr.translate(lines)
    assert lines[1].translation == "二"
    # repair request mentions the missing line's original text
    assert "line 2" in fake.calls[2][-1]["content"]


def test_persistent_missing_line_raises():
    lines = make_lines(2)
    fake = FakeClient(["glossary", "[1] 一", "nope", "nope"])
    tr = Translator(settings(batch_size=5), "简体中文", client=fake)
    with pytest.raises(TranslationError):
        tr.translate(lines)


def test_synopsis_included_in_system_prompt():
    lines = make_lines(1)
    fake = FakeClient(["glossary", "[1] 一"])
    tr = Translator(settings(), "简体中文", synopsis="一部太空歌剧", client=fake)
    tr.translate(lines)
    assert "一部太空歌剧" in fake.calls[0][0]["content"]


# -------------------------------------------------------- chunked mode


def test_chunked_mode_when_context_too_small():
    lines = make_lines(6)
    # force chunking: tiny context limit
    fake = FakeClient([
        "glossary",                     # sampled glossary pass
        "[1] 一\n[2] 二\n[3] 三",
        "[4] 四\n[5] 五\n[6] 六",
    ])
    tr = Translator(settings(context_limit=1000, batch_size=3), "简体中文", client=fake)
    # est*3 > 1000 requires lots of text; fake it by padding lines
    for l in lines:
        l.text = "word " * 120 + f"line {l.index}"
    tr.translate(lines)
    assert all(l.translation for l in lines)
    # second chunk carries tail context of the first (original → translation)
    second_chunk_prompt = fake.calls[2][1]["content"]
    assert "→" in second_chunk_prompt
    assert "一" in second_chunk_prompt

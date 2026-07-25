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


def test_glossary_request_demands_target_language_names():
    lines = make_lines(1)
    fake = FakeClient(["glossary", "[1] 一"])
    tr = Translator(settings(), "简体中文", client=fake)
    tr.translate(lines)
    glossary_request = fake.calls[0][1]["content"]
    assert "简体中文译名" in glossary_request
    assert "罗马音" in glossary_request  # explicit prohibition present
    system = fake.calls[0][0]["content"]
    assert "不得用罗马音" in system


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


# ------------------------------------------------- batch alignment (the shift)
#
# Every line came back with the right number and every check passed, yet an
# English film went out 82% misaligned: the model had lost its place in the
# transcript and numbered its output correctly anyway. Line length is the
# one property that survives translation, so the correlation across a batch
# collapses the moment the rows slide against each other.


def aligned_lines(n=40):
    """Alternating short and long lines, so length carries information."""
    out = []
    for i in range(1, n + 1):
        text = "Yes." if i % 2 else f"This is a considerably longer line number {i} here."
        out.append(SubtitleLine(index=i, start=float(i), end=float(i) + 0.9, text=text))
    return out


def zh_for(text):
    return "是。" if text == "Yes." else "这是一行相当长的字幕内容，用来测试长度相关性。"


def reply_for(lines, shift=0):
    """Numbered reply; with shift>0 each line gets its neighbour's text."""
    parts = []
    for i, line in enumerate(lines):
        src = lines[(i + shift) % len(lines)]
        parts.append(f"[{line.index}] {zh_for(src.text)}")
    return "\n".join(parts)


def test_length_correlation_separates_aligned_from_shifted():
    from app.services.translator import _length_correlation

    lines = aligned_lines()
    for line in lines:
        line.translation = zh_for(line.text)
    assert _length_correlation(lines) > 0.9

    shifted = aligned_lines()
    for i, line in enumerate(shifted):
        line.translation = zh_for(shifted[(i + 1) % len(shifted)].text)
    assert _length_correlation(shifted) < 0.45


def test_a_shifted_batch_is_detected_and_re_requested():
    lines = aligned_lines()
    client = FakeClient([
        "glossary",
        reply_for(lines, shift=1),   # the model lost its place
        reply_for(lines, shift=0),   # ...and gets it right when told
    ])
    logs = []
    Translator(LLMSettings(model="m", batch_size=200), target_language="简体中文",
               client=client, log=logs.append).translate(lines)
    assert any("疑似整批错位" in m for m in logs)
    assert all(l.translation == zh_for(l.text) for l in lines)


def test_a_batch_that_stays_shifted_keeps_the_first_answer_and_warns():
    lines = aligned_lines()
    client = FakeClient([
        "glossary",
        reply_for(lines, shift=1),
        reply_for(lines, shift=1),   # retry is no better
    ])
    logs = []
    Translator(LLMSettings(model="m", batch_size=200), target_language="简体中文",
               client=client, log=logs.append).translate(lines)
    assert any("建议人工核对" in m for m in logs)
    assert all(l.translation for l in lines)  # never left empty


def test_an_aligned_batch_costs_no_extra_request():
    lines = aligned_lines()
    client = FakeClient(["glossary", reply_for(lines)])
    Translator(LLMSettings(model="m", batch_size=200), target_language="简体中文",
               client=client).translate(lines)
    assert len(client.calls) == 2  # glossary + the one batch, no re-request


def test_the_batch_request_restates_its_own_source_lines():
    """Locating line 601 by counting through the transcript is what slipped."""
    lines = aligned_lines()
    client = FakeClient(["glossary", reply_for(lines)])
    Translator(LLMSettings(model="m", batch_size=200), target_language="简体中文",
               client=client).translate(lines)
    batch_request = next(
        m["content"] for m in client.calls[-1]
        if m["role"] == "user" and "请输出第" in m["content"]
    )
    assert "[1] Yes." in batch_request
    assert f"[{lines[-1].index}]" in batch_request

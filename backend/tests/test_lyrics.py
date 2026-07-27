"""Marking what is sung rather than spoken.

The invariant every test here defends: this pass only annotates. It never
adds, drops or rewrites a line, so every failure degrades to "nothing
marked" — which is exactly the behaviour with the switch off.
"""

import pytest

from app.models.schemas import LLMSettings, SubtitleLine
from app.services.lyrics import (
    apply_marks,
    build_lyrics_prompt,
    mark_lyrics,
    parse_ranges,
    strip_marks,
    whisper_agreement,
)
from tests.test_translator import FakeClient

LLM = LLMSettings(model="test-model")


def L(i, text):
    return SubtitleLine(index=i, start=float(i), end=i + 1.0, text=text)


# The real case: four lines of a song sitting in a stretch of silence
# between two pieces of dialogue, which the vetting pass had misfiled as
# "unrelated to this film" for want of a lyrics category.
SAMPLE = [
    L(1, "We send a search party to the surface to find my dad."),
    L(2, "There's a brighter side to every dark night."),
    L(3, "And there's a smiling face in every crowd."),
    L(4, "All your troubles will soon fade away."),
    L(5, "There's a brighter, brighter side."),
    L(6, "To the surface?"),
]


def run(responses, lines=None, **kw):
    client = FakeClient(responses)
    ls = lines if lines is not None else [l.model_copy() for l in SAMPLE]
    out = mark_lyrics(ls, LLM, client=client, **kw)
    return out, client


def flags(lines):
    return [l.is_lyric for l in lines]


GOOD = "[2-5] There's a brighter side | brighter, brighter side."


# --------------------------------------------------------------- happy path


def test_a_verified_range_marks_exactly_its_lines():
    out, client = run([GOOD])
    assert flags(out) == [False, True, True, True, True, False]
    assert len(out) == len(SAMPLE)  # never adds or drops
    assert len(client.calls) == 1


def test_a_single_line_range_works():
    out, _ = run(["[3] And there's a smiling face | in every crowd."])
    assert flags(out) == [False, False, True, False, False, False]


def test_no_lyrics_at_all():
    out, _ = run(["无"])
    assert not any(flags(out))


# ------------------------------------------------------- verification fails
#
# Each of these leaves the transcript exactly as it arrived.


@pytest.mark.parametrize("reply, why", [
    ("[2-99] There's a brighter side | brighter side.", "行号越界"),
    ("[5-2] There's a brighter side | brighter side.", "区间倒置"),
    ("[2-5] Completely different words here | nothing like it", "首尾对不上"),
    ("[2-5] There's a brighter side | nothing like this at all", "尾片段对不上"),
    ("", "空响应"),
    ("I think lines 2 to 5 are a song.", "没有可解析的区间"),
])
def test_a_range_that_does_not_verify_marks_nothing(reply, why):
    out, _ = run([reply])
    assert not any(flags(out)), why
    assert len(out) == len(SAMPLE)


def test_overlapping_ranges_keep_the_first_and_drop_the_second():
    out, _ = run([
        "[2-3] There's a brighter side | in every crowd.\n"
        "[3-5] And there's a smiling face | brighter, brighter side."
    ])
    assert flags(out) == [False, True, True, False, False, False]


def test_a_range_claiming_most_of_the_film_is_refused():
    """Guards against "[1-500] the whole film is a song" — but the limit
    never bites below MAX_RANGE_LINES, so a long musical number survives."""
    lines = [L(i, f"ordinary spoken line number {i}") for i in range(1, 501)]
    out, _ = run(["[1-500] ordinary spoken line | line number 500"], lines=lines)
    assert not any(flags(out))


def test_a_long_musical_number_is_not_refused_for_its_length():
    lines = [L(i, f"ordinary spoken line number {i}") for i in range(1, 501)]
    out, _ = run(["[10-60] ordinary spoken line | line number 60"], lines=lines)
    assert sum(flags(out)) == 51


def test_a_failing_endpoint_never_raises_and_marks_nothing():
    class Boom:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            raise RuntimeError("connection refused")

    out = mark_lyrics([l.model_copy() for l in SAMPLE], LLM, client=Boom())
    assert not any(flags(out))
    assert len(out) == len(SAMPLE)


def test_cancellation_still_propagates():
    with pytest.raises(InterruptedError):
        mark_lyrics(
            [l.model_copy() for l in SAMPLE], LLM,
            client=FakeClient([]), should_cancel=lambda: True,
        )


# ------------------------------------------------------------------ prompt


def test_the_request_carries_no_timestamps():
    _out, client = run([GOOD])
    sent = "\n".join(m["content"] for m in client.calls[0])
    assert "-->" not in sent
    for line in SAMPLE:
        assert f"{line.start}" not in sent


def test_singing_by_characters_counts_as_lyrics():
    """A single criterion the model can actually apply: it cannot see the
    screen, so "is someone on camera singing" would be guesswork."""
    prompt = build_lyrics_prompt("en")
    assert "生日歌" in prompt
    assert "拿不准时不要标" in prompt


# ------------------------------------------------------------- ♪ handling


def test_marks_are_applied_to_both_languages_and_only_to_lyrics():
    lines = [l.model_copy() for l in SAMPLE]
    lines[1].is_lyric = True
    lines[1].translation = "每个黑夜都有光明的一面。"
    lines[0].translation = "我们派搜索队去地表找我爸。"
    apply_marks(lines)
    assert lines[1].text == "♪ There's a brighter side to every dark night. ♪"
    assert lines[1].translation == "♪ 每个黑夜都有光明的一面。 ♪"
    assert "♪" not in lines[0].text
    assert "♪" not in lines[0].translation


def test_marks_are_renormalised_after_the_model_mangles_them():
    """The translator is asked to keep the ♪ and mostly does. A marker it
    dropped, doubled or moved onto the neighbour would be worse than none,
    so the flag has the last word."""
    lines = [L(1, "♪ a song ♪"), L(2, "♪ spoken line"), L(3, "another song")]
    lines[0].is_lyric = True
    lines[2].is_lyric = True
    lines[0].translation = "一首歌"            # model dropped both marks
    lines[1].translation = "♪ 对白 ♪"          # model invented marks
    lines[2].translation = "♪ 另一首歌"        # model kept only one
    apply_marks(lines)
    assert lines[0].text == "♪ a song ♪"
    assert lines[0].translation == "♪ 一首歌 ♪"
    assert lines[1].text == "spoken line"
    assert lines[1].translation == "对白"
    assert lines[2].translation == "♪ 另一首歌 ♪"


def test_strip_marks_leaves_no_double_spaces():
    assert strip_marks("♪  Crawl out through the fallout  ♪") == \
        "Crawl out through the fallout"


def test_marking_is_idempotent():
    lines = [L(1, "a song")]
    lines[0].is_lyric = True
    apply_marks(lines)
    apply_marks(lines)
    assert lines[0].text == "♪ a song ♪"


def test_the_mark_is_not_counted_as_an_invented_word():
    """refine's fidelity check spends a small budget on words the model
    added; a ♪ eating that budget would cost real corrections."""
    from app.services.refine import _is_faithful

    assert _is_faithful("♪ Crawl out through the fallout ♪",
                        "Crawl out through the fallout")


# --------------------------------------------------------- whisper vs ours


def test_agreement_with_whisper_separates_the_two_disagreements():
    lines = [L(1, "a song"), L(2, "spoken"), L(3, "another song")]
    lines[0].is_lyric = True
    lines[2].is_lyric = True
    both, ours, theirs, ours_alone, theirs_alone = whisper_agreement(
        lines, [(0.5, 1.5), (1.5, 2.5)]   # whisper marked lines 1 and 2
    )
    assert (both, ours, theirs) == (1, 1, 1)
    assert [l.index for l in ours_alone] == [3]
    assert [l.index for l in theirs_alone] == [2]


# ------------------------------------------------------------ parse_ranges


def test_parse_tolerates_fences_and_bullets():
    assert parse_ranges(
        "```\n- [2-5] head words | tail words\n[9] one | one\n```"
    ) == [(2, 5, "head words", "tail words"), (9, 9, "one", "one")]

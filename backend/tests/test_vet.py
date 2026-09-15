"""Vetting of the second pass's output.

The invariant every test here defends: vetting may only ever remove
``recovered`` segments. Whatever the model says, however badly the request
fails, the first pass comes out whole — that is what makes it safe to let
an LLM delete subtitles at all.
"""

import pytest

from app.models.schemas import LLMSettings
from app.services.asr import Segment
from app.services.vet import (
    parse_verdicts,
    render_transcript,
    vet_recovered,
)
from tests.test_translator import FakeClient

LLM = LLMSettings(model="test-model")


def S(start, end, text, recovered=False):
    return Segment(start, end, text, recovered=recovered)


# The shape of the real problem: two fabrications from whisper's training
# data and one stretch of genuine dialogue, both inside a first pass that
# is about a haunted apartment.
SAMPLE = [
    S(10.0, 12.0, "部屋の隅に何かいるような気がして"),
    S(12.5, 14.0, "JR東日本E233系電車", recovered=True),
    S(15.0, 17.0, "振り返っても誰もいないんです"),
    S(20.0, 22.0, "今どこ?会社?", recovered=True),
    S(22.0, 23.0, "今下なの", recovered=True),
    S(30.0, 32.0, "もしもし"),
]


def run(responses, segments=None, **kw):
    client = FakeClient(responses)
    segs = segments if segments is not None else list(SAMPLE)
    out = vet_recovered(segs, LLM, client=client, **kw)
    return out, client


def texts(segments):
    return [s.text for s in segments]


FIRST_PASS = [s.text for s in SAMPLE if not s.recovered]


# ------------------------------------------------------------- no-op path


def test_a_transcript_with_nothing_recovered_costs_nothing():
    client = FakeClient([])
    segs = [S(0.0, 1.0, "a"), S(1.0, 2.0, "b")]
    assert vet_recovered(segs, LLM, client=client) == segs
    assert client.calls == []


# --------------------------------------------------------------- verdicts


def test_verdicts_are_applied_and_the_first_pass_is_untouched():
    out, client = run(["[R1] 丢弃 车站广播，与本片无关\n[R2] 保留\n[R3] 保留"])
    assert texts(out) == [
        "部屋の隅に何かいるような気がして",
        "振り返っても誰もいないんです",
        "今どこ?会社?",
        "今下なの",
        "もしもし",
    ]
    assert len(client.calls) == 1


def test_dropping_everything_leaves_exactly_the_first_pass():
    out, _ = run(["[R1] 丢弃 无关\n[R2] 丢弃 无关\n[R3] 丢弃 无关"])
    assert texts(out) == FIRST_PASS


def test_keeping_everything_changes_nothing():
    out, _ = run(["[R1] 保留\n[R2] 保留\n[R3] 保留"])
    assert texts(out) == texts(SAMPLE)


# ------------------------------------------------- verification and failure
#
# Every failure mode drops the recovered lines rather than keeping them:
# that is the direction which preserves the floor (the film as it was
# before the second pass existed).


def test_a_missing_verdict_voids_the_whole_chunk():
    out, _ = run(["[R1] 丢弃 无关\n[R2] 保留"])  # nothing said about R3
    assert texts(out) == FIRST_PASS


def test_an_unknown_number_voids_the_whole_chunk():
    out, _ = run(["[R1] 保留\n[R2] 保留\n[R3] 保留\n[R9] 保留"])
    assert texts(out) == FIRST_PASS


def test_an_empty_reply_voids_the_whole_chunk():
    out, _ = run([""])
    assert texts(out) == FIRST_PASS


def test_a_failing_endpoint_never_raises():
    class Boom:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kw):
            self.calls += 1
            raise RuntimeError("connection refused")

    client = Boom()
    out = vet_recovered(list(SAMPLE), LLM, client=client)
    assert texts(out) == FIRST_PASS
    assert client.calls == 2  # one retry, because a blip must not eat dialogue


def test_a_retry_that_succeeds_keeps_the_dialogue():
    out, client = run(["garbage", "[R1] 丢弃 无关\n[R2] 保留\n[R3] 保留"])
    assert "今どこ?会社?" in texts(out)
    assert "JR東日本E233系電車" not in texts(out)
    assert len(client.calls) == 2
    # at temperature 0, repeating the request verbatim would repeat the
    # answer — the retry has to say what was wrong with it
    retry = client.calls[1][-1]["content"]
    assert "R1 R2 R3" in retry


def test_cancellation_still_propagates():
    with pytest.raises(InterruptedError):
        vet_recovered(
            list(SAMPLE), LLM, client=FakeClient([]), should_cancel=lambda: True
        )


# ----------------------------------------------------------------- prompt


def test_reviewed_lines_are_marked_and_interleaved_in_time_order():
    """Position in the transcript is what makes the judgement easy — asked
    in isolation "JR東日本E233系電車" is just a noun phrase."""
    ids = {id(s): n for n, s in enumerate([s for s in SAMPLE if s.recovered], 1)}
    body = render_transcript(SAMPLE, ids)
    assert body.splitlines() == [
        "[1] 部屋の隅に何かいるような気がして",
        "*[R1] JR東日本E233系電車",
        "[2] 振り返っても誰もいないんです",
        "*[R2] 今どこ?会社?",
        "*[R3] 今下なの",
        "[3] もしもし",
    ]


def test_the_request_carries_no_timestamps():
    """Project-wide rule; here it costs nothing because interleaving already
    encodes the timing."""
    _out, client = run(["[R1] 保留\n[R2] 保留\n[R3] 保留"])
    sent = "\n".join(m["content"] for m in client.calls[0])
    assert "-->" not in sent
    for seg in SAMPLE:
        assert f"{seg.start}" not in sent


def test_a_transcript_that_is_all_recovered_says_so_instead_of_pointing_at_nothing():
    """When the first pass yields nothing, the prompt's premise is gone.

    It normally tells the model that the unmarked lines are confirmed and
    to judge against them; with every line recovered there are no unmarked
    lines, and one film reached exactly that state — its first pass came
    back as 577 dashes, all of which this pipeline now drops. Claiming a
    context that is not in the request is worse than admitting there is
    none, and the user is told the pass is running blind.
    """
    segments = [
        S(10.0, 12.0, "今どこ?会社?", recovered=True),
        S(12.5, 14.0, "今下なの", recovered=True),
    ]
    logged: list[str] = []
    out, client = run(
        ["[R1] 保留\n[R2] 保留"], segments=segments, log=logged.append
    )
    system = client.calls[0][0]["content"]
    assert "没有已确认的识别结果" in system
    assert "请把它们当作判断依据" not in system
    assert any("⚠" in line and "第一遍识别" in line for line in logged)
    assert texts(out) == ["今どこ?会社?", "今下なの"]


def test_the_usual_prompt_still_anchors_on_the_confirmed_lines():
    _out, client = run(["[R1] 保留\n[R2] 保留\n[R3] 保留"])
    system = client.calls[0][0]["content"]
    assert "请把它们当作判断依据" in system
    assert "没有已确认的识别结果" not in system


def test_the_synopsis_reaches_the_model_when_given():
    _out, client = run(
        ["[R1] 保留\n[R2] 保留\n[R3] 保留"], synopsis="心霊ドキュメンタリー"
    )
    assert "心霊ドキュメンタリー" in client.calls[0][0]["content"]


# ---------------------------------------------------------------- chunking


def test_chunks_without_anything_to_review_send_no_request():
    segments = [S(float(i), i + 1.0, f"平凡な台詞 {i}") for i in range(400)]
    segments.append(S(500.0, 501.0, "パン粉を少しずつ入れて混ぜます", recovered=True))
    out, client = run(["[R1] 丢弃 料理番組の字幕"], segments=segments)
    assert len(out) == 400
    assert len(client.calls) == 1  # one chunk had the line; the rest cost nothing


def test_a_long_transcript_splits_and_each_chunk_is_judged():
    segments = []
    for i in range(900):
        segments.append(S(float(i), i + 0.5, f"これは何番目かの台詞です {i}"))
        if i in (10, 890):
            segments.append(S(i + 0.5, i + 0.9, f"recovered {i}", recovered=True))
    out, client = run(["[R1] 保留", "[R2] 丢弃 無関係"], segments=segments)
    assert len(client.calls) == 2
    assert "recovered 10" in texts(out)
    assert "recovered 890" not in texts(out)
    # the unmarked lines keep counting up across chunks: a second chunk that
    # restarts at [1] reads like a second transcript
    second_chunk = client.calls[1][1]["content"]
    assert "\n[1] " not in second_chunk


# ------------------------------------------------------------ parse_verdicts


def test_parse_tolerates_fences_reasons_and_stray_bullets():
    parsed = parse_verdicts(
        "```\n"
        "[R1] 保留\n"
        "- [R2] 丢弃 车站广播，与本片无关\n"
        "R3: 保留\n"
        "总共 3 条\n"
        "```"
    )
    assert parsed == {
        1: (True, ""),
        2: (False, "车站广播，与本片无关"),
        3: (True, ""),
    }


def test_parse_accepts_english_verdicts():
    assert parse_verdicts("[R1] keep\n[R2] drop off-topic") == {
        1: (True, ""),
        2: (False, "off-topic"),
    }


# --------------------------------------------------- a slip in a long chunk
#
# The run that motivated the tolerance: 547 recovered lines went out in one
# request, the model answered every one of them, and two of its answers read
# `[R163] 保保留`. The verdict regex could not see those two lines, coverage
# was declared broken, and 985 seconds of real dialogue were discarded —
# twice, because the retry fumbled a different line. What follows pins both
# halves: a slip of a few lines in five hundred costs those lines only, and
# a reply that is actually short still voids the chunk.


def big(n_recovered=547, anchors_every=10):
    """A transcript shaped like that run: mostly recovered, anchors between."""
    segments = []
    for i in range(1, n_recovered + 1):
        if i % anchors_every == 0:
            segments.append(S(i * 2.0, i * 2.0 + 1.0, f"確認済みの台詞 {i}"))
        segments.append(S(i * 2.0 + 1.0, i * 2.0 + 1.8, f"補足の台詞 {i}", recovered=True))
    return segments


def reply_for(numbers, drop=(), garble=(), omit=()):
    """The model's answer, with the named numbers mistyped or missing."""
    lines = []
    for n in numbers:
        if n in omit:
            continue
        if n in garble:
            lines.append(f"[R{n}] 保保留")  # the measured typo
        elif n in drop:
            lines.append(f"[R{n}] 丢弃 与本片无关")
        else:
            lines.append(f"[R{n}] 保留")
    return "\n".join(lines)


def recovered_texts(segments):
    return [s.text for s in segments if s.text.startswith("補足")]


def test_exact_coverage_is_what_it_was_before_the_tolerance():
    segments = big()
    numbers = range(1, 548)
    out, client = run([reply_for(numbers)], segments=segments)
    assert len(recovered_texts(out)) == 547
    assert len(client.calls) == 1


def test_a_slip_of_two_in_five_hundred_drops_only_those_two():
    segments = big()
    logged = []
    out, client = run([reply_for(range(1, 548), omit={163, 259})],
                      segments=segments, log=logged.append)
    assert len(client.calls) == 1, "a two-line slip must not cost a retry"
    assert len(recovered_texts(out)) == 545
    assert "補足の台詞 163" not in recovered_texts(out)
    assert "補足の台詞 259" not in recovered_texts(out)
    assert any("R163 R259" in line for line in logged)
    assert any("未收到可识别的判定" in line for line in logged)


def test_a_garbled_verdict_word_is_dropped_not_kept():
    """`保保留` is unreadable, and unreadable is never read as 保留."""
    segments = big()
    out, client = run([reply_for(range(1, 548), garble={163, 259})],
                      segments=segments)
    assert len(client.calls) == 1
    assert "補足の台詞 163" not in recovered_texts(out)
    assert len(recovered_texts(out)) == 545


def test_the_reason_says_which_kind_of_failure_it_was():
    from app.services.vet import UNJUDGED_REASON, UNREADABLE_REASON, parse_reply

    verdicts, unreadable = parse_reply("[R1] 保留\n[R2] 保保留\n")
    assert set(verdicts) == {1}
    assert unreadable == {2}
    assert UNJUDGED_REASON != UNREADABLE_REASON


def test_a_truncated_reply_still_fails_and_retries():
    """400 of 547 answered is a cut-off response, not a slip."""
    segments = big()
    short = reply_for(range(1, 401))
    out, client = run([short, short], segments=segments)
    assert len(client.calls) == 2
    assert recovered_texts(out) == []


def test_the_budget_is_zero_below_fifty_items():
    """Under 50 lines the tolerance rounds to nothing — today's behaviour."""
    small = big(n_recovered=49, anchors_every=1000)
    out, client = run([reply_for(range(1, 50), omit={7})] * 2, segments=small)
    assert recovered_texts(out) == []
    assert len(client.calls) == 2

    fifty = big(n_recovered=50, anchors_every=1000)
    out, client = run([reply_for(range(1, 51), omit={7})], segments=fifty)
    assert len(recovered_texts(out)) == 49
    assert len(client.calls) == 1


def test_an_unknown_number_still_voids_the_chunk_even_with_full_coverage():
    """A number nobody sent means the model lost its place — not a slip."""
    segments = big(n_recovered=100, anchors_every=1000)
    reply = reply_for(range(1, 101)) + "\n[R101] 保留"
    out, client = run([reply, reply], segments=segments)
    assert recovered_texts(out) == []
    assert len(client.calls) == 2


def test_unjudged_lines_are_never_kept():
    import random

    rng = random.Random(7)
    segments = big()
    missing = set(rng.sample(range(1, 548), 10))
    out, _ = run([reply_for(range(1, 548), omit=missing)], segments=segments)
    survived = set(recovered_texts(out))
    assert all(f"補足の台詞 {n}" not in survived for n in missing)
    assert len(survived) == 547 - len(missing)


def test_two_consecutive_failures_stop_the_calls():
    """GIVE_UP_AFTER: an unreachable endpoint costs two calls, not one a chunk."""
    from app.services import vet

    long_transcript = []
    for i in range(1, 1201):
        long_transcript.append(S(i * 2.0, i * 2.0 + 1.5,
                                 f"これは十分に長い確認済みの台詞です {i}"))
        long_transcript.append(S(i * 2.0 + 1.5, i * 2.0 + 1.9,
                                 f"補足の台詞 {i}", recovered=True))
    chunks = vet._chunks(long_transcript, LLM.context_limit)
    assert len(chunks) >= 3, "the transcript must actually split"

    out, client = run(["garbage"] * 10, segments=long_transcript)
    assert len(client.calls) == vet.GIVE_UP_AFTER * 2
    assert recovered_texts(out) == []

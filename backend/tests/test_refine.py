"""Transcript preprocessing: verification and fallback behaviour.

The point of this layer is that its output is checkable — same language in,
same language out. These tests pin down what gets rejected, because a bad
merge silently accepted here is exactly what shifted a whole film.
"""

import pytest

from app.models.schemas import LLMSettings, SubtitleLine, SubtitleSettings
from app.services.refine import (
    _covers_exactly,
    _is_faithful,
    parse_units,
    refine_lines,
)
from tests.test_translator import FakeClient


def L(index, start, end, text):
    return SubtitleLine(index=index, start=start, end=end, text=text)


# the real cues that triggered the bug
SAMPLE = [
    L(1, 327.1, 328.8, "Yeah, like that thought never entered my"),
    L(2, 328.8, 329.3, "mind."),
    L(3, 333.16, 334.06, "Okay, come on."),
]

LLM = LLMSettings(model="test-model")
SUB = SubtitleSettings()


def run(responses, lines=None, subtitle=None):
    client = FakeClient(responses)
    out = refine_lines(
        lines or [l.model_copy() for l in SAMPLE],
        LLM,
        subtitle or SUB,
        client=client,
    )
    return out, client


# ------------------------------------------------------------- parse_units


def test_parse_units_handles_ranges_and_singles():
    units = parse_units("[7-8] Merged sentence.\n[9] Alone.")
    assert units == [(7, 8, "Merged sentence."), (9, 9, "Alone.")]


def test_parse_units_ignores_fences_and_folds_continuations():
    units = parse_units("```\n[1] first\ncontinued\n```")
    assert units == [(1, 1, "first continued")]


# ---------------------------------------------------------- coverage check


def test_coverage_accepts_contiguous_ranges():
    assert _covers_exactly([(1, 2, "x"), (3, 3, "y")], SAMPLE)


@pytest.mark.parametrize("units", [
    [(1, 1, "x"), (3, 3, "y")],           # line 2 dropped
    [(1, 2, "x"), (2, 3, "y")],           # overlapping
    [(3, 3, "y"), (1, 2, "x")],           # out of order
    [(1, 2, "x")],                        # truncated
    [(1, 2, "x"), (3, 4, "y")],           # invented a line
])
def test_coverage_rejects_broken_mappings(units):
    assert not _covers_exactly(units, SAMPLE)


def test_bad_coverage_falls_back_to_input():
    out, _ = run(["[1] Yeah, like that thought never entered my\n[3] Okay, come on."])
    assert [l.text for l in out] == [l.text for l in SAMPLE]
    assert [l.index for l in out] == [1, 2, 3]


# --------------------------------------------------------------- fidelity


def test_fidelity_accepts_identity_and_small_corrections():
    assert _is_faithful("a b c", "a b c")
    assert _is_faithful("Okay, come on!", "Okay, come on.")


def test_fidelity_rejects_a_rewrite():
    assert not _is_faithful("totally different words here", "a b c d")


def test_fidelity_rejects_hallucinated_additions():
    """Keeping every source word is not enough — invented content keeps
    them all and would pass a retention-only check."""
    assert not _is_faithful("Okay, and then he killed the man.", "Okay.")


def test_hallucinated_line_falls_back_to_the_source():
    lines = [L(1, 0.0, 2.0, "Okay.")]
    out, _ = run(["[1] Okay, and then he killed the man."], lines=lines)
    assert out[0].text == "Okay."


def test_rewritten_unit_is_rejected_but_neighbours_survive():
    out, _ = run([
        "[1-2] Something completely unrelated invented by the model.\n"
        "[3] Okay, come on."
    ])
    # the bogus merge is dropped, its source lines kept verbatim
    assert [l.text for l in out] == [l.text for l in SAMPLE]


# ------------------------------------------------------------ happy path


def test_split_sentence_is_merged_with_real_timestamps():
    out, _ = run([
        "[1-2] Yeah, like that thought never entered my mind.\n[3] Okay, come on."
    ])
    assert len(out) == 2
    assert out[0].text == "Yeah, like that thought never entered my mind."
    assert (out[0].start, out[0].end) == (327.1, 329.3)
    assert [l.index for l in out] == [1, 2]


def test_asr_word_correction_is_kept():
    out, _ = run([
        "[1-2] Yeah, like that thought never entered my mind.\n[3] Okay, come on!"
    ])
    assert out[1].text == "Okay, come on!"


def test_timestamps_are_never_sent_to_the_model():
    _, client = run([
        "[1-2] Yeah, like that thought never entered my mind.\n[3] Okay, come on."
    ])
    sent = "\n".join(m["content"] for m in client.calls[0])
    assert "327" not in sent and "-->" not in sent


# --------------------------------------- local constraints the model can't see


def test_merge_across_a_long_pause_is_rejected():
    lines = [
        L(1, 0.0, 2.0, "Something happens and"),
        L(2, 6.0, 7.0, "then more happens."),
    ]
    out, _ = run(["[1-2] Something happens and then more happens."], lines=lines)
    assert len(out) == 2  # 4s pause: two utterances, not one sentence


def test_merge_beyond_the_char_budget_is_rejected():
    lines = [L(1, 0.0, 2.0, "a " * 40), L(2, 2.0, 4.0, "b " * 40)]
    out, _ = run([f"[1-2] {'a ' * 40}{'b ' * 40}"], lines=lines,
                 subtitle=SubtitleSettings(max_chars_per_line=42))
    assert len(out) == 2


def test_merge_beyond_the_duration_cap_is_rejected():
    lines = [
        L(1, 0.0, 5.0, "This sentence starts here and"),
        L(2, 5.5, 12.0, "keeps going for a very long time."),
    ]
    out, _ = run(
        ["[1-2] This sentence starts here and keeps going for a very long time."],
        lines=lines,
    )
    assert len(out) == 2  # 12s cue would outstay its welcome


# ------------------------------------------------------ failure containment


def test_model_error_returns_the_input_unchanged():
    class Boom:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            raise RuntimeError("connection reset")

    out = refine_lines([l.model_copy() for l in SAMPLE], LLM, SUB, client=Boom())
    assert [l.text for l in out] == [l.text for l in SAMPLE]


def test_empty_reply_falls_back():
    out, _ = run([""])
    assert [l.text for l in out] == [l.text for l in SAMPLE]


def test_cancellation_propagates():
    with pytest.raises(InterruptedError):
        refine_lines(
            [l.model_copy() for l in SAMPLE], LLM, SUB,
            client=FakeClient([]), should_cancel=lambda: True,
        )


# ------------------------------------------------------------- chunking


def test_long_transcript_is_chunked_and_each_chunk_validated():
    lines = [L(i, i * 2.0, i * 2.0 + 1.5, f"Line number {i} says something.")
             for i in range(1, 201)]
    # tiny context budget forces several chunks
    llm = LLMSettings(model="m", context_limit=1_000)
    client = FakeClient([])

    def create(model, messages, temperature, **kw):
        client.calls.append(messages)
        body = messages[-1]["content"]
        echoed = [l for l in body.splitlines() if l.startswith("[")]

        class M:
            pass

        msg, choice, resp = M(), M(), M()
        msg.content = "\n".join(echoed)
        choice.message = msg
        resp.choices = [choice]
        return resp

    client.create = create
    out = refine_lines(lines, llm, SUB, client=client)
    assert len(client.calls) > 1
    assert [l.text for l in out] == [l.text for l in lines]
    assert [l.index for l in out] == list(range(1, 201))


# --------------------------------------------- segmenter → refine → translate


def test_full_chain_keeps_every_line_paired_with_its_own_translation():
    """The bug end-to-end: fragmented cues in, one cue per sentence out,
    each translation sitting on the line it belongs to."""
    from app.models.schemas import SubtitleSettings
    from app.services.segmenter import segment_lines
    from app.services.subtitle import build_srt
    from app.services.translator import Translator

    raw = [  # as whisper emitted them
        (327.100, 328.800, "Yeah, like that thought never entered my"),
        (328.800, 329.300, "mind."),
        (333.160, 334.060, "Okay, come on."),
        (334.140, 334.760, "Somebody, somebody."),
    ]
    lines = segment_lines(raw, SubtitleSettings())
    assert [l.text for l in lines] == [
        "Yeah, like that thought never entered my mind.",
        "Okay, come on.",
        "Somebody, somebody.",
    ]

    # preprocessing has nothing left to merge here; it must not disturb them
    echo = FakeClient(["\n".join(f"[{l.index}] {l.text}" for l in lines)])
    lines = refine_lines(lines, LLM, SUB, client=echo)

    reply = "\n".join(f"[{l.index}] 译文{l.index}" for l in lines)
    translator = Translator(
        LLM, target_language="简体中文", client=FakeClient(["术语表", reply]),
    )
    translator.translate(lines)

    assert [(l.index, l.translation) for l in lines] == [
        (1, "译文1"), (2, "译文2"), (3, "译文3"),
    ]
    srt = build_srt(lines, SubtitleSettings())
    # 45 chars > 42: one cue, wrapped onto two display lines
    assert "Yeah, like that thought\nnever entered my mind.\n译文1" in srt
    assert "Okay, come on.\n译文2" in srt
    assert "00:05:27,100 --> 00:05:29,300" in srt  # merged cue keeps real times


def test_a_full_film_is_split_to_fit_the_output_limit():
    """One request per film would exceed every model's max output tokens,
    get truncated, fail coverage, and silently disable the whole feature."""
    from app.services.refine import REFINE_CHUNK_TOKENS, _chunks
    from app.services.translator import estimate_tokens

    film = [L(i, i * 3.0, i * 3.0 + 2.0,
              "This is a fairly typical line of movie dialogue.")
            for i in range(1, 1201)]
    chunks = _chunks(film, LLMSettings().context_limit)
    assert len(chunks) > 10
    assert sum(len(c) for c in chunks) == 1200
    for c in chunks:
        assert sum(estimate_tokens(l.text) + 6 for l in c) <= REFINE_CHUNK_TOKENS + 20


def test_a_dead_endpoint_gives_up_instead_of_timing_out_every_chunk():
    lines = [L(i, i * 3.0, i * 3.0 + 2.0,
               "This is a fairly typical line of movie dialogue.")
             for i in range(1, 601)]
    attempts = []

    class Dead:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            attempts.append(1)
            raise RuntimeError("connection refused")

    out = refine_lines(lines, LLMSettings(model="m"), SUB, client=Dead())
    assert len(attempts) == 3  # GIVE_UP_AFTER, not one per chunk
    assert [l.text for l in out] == [l.text for l in lines]
    assert [l.index for l in out] == list(range(1, 601))

import pytest

from app.models.schemas import SubtitleSettings
from app.services.asr import Segment, Word
from app.services.segmenter import _split_text, segment_lines


def test_short_segment_untouched():
    lines = segment_lines([(0.0, 2.0, "Hello world.")], SubtitleSettings())
    assert len(lines) == 1
    assert lines[0].index == 1
    assert lines[0].text == "Hello world."
    assert lines[0].start == 0.0
    assert lines[0].end == 2.0


def test_long_segment_split_at_sentences():
    text = "This is the first sentence. And here comes the second one! Finally a third."
    lines = segment_lines([(0.0, 9.0, text)], SubtitleSettings(max_chars_per_line=40))
    assert len(lines) >= 2
    assert all(len(l.text) <= 40 for l in lines)
    assert lines[0].start == 0.0
    for a, b in zip(lines, lines[1:]):
        assert b.start >= a.start


def test_indexes_are_sequential():
    segs = [(0.0, 2.0, "One."), (2.0, 4.0, "Two."), (4.0, 6.0, "Three.")]
    lines = segment_lines(segs, SubtitleSettings())
    assert [l.index for l in lines] == [1, 2, 3]


def test_split_text_no_punctuation_uses_spaces():
    chunks = _split_text("aaa bbb ccc ddd eee fff", 10)
    assert all(len(c) <= 10 for c in chunks)


def test_split_text_hard_cut():
    chunks = _split_text("a" * 25, 10)
    assert chunks == ["a" * 10, "a" * 10, "a" * 5]


# --------------------------- stretched-segment pathology (the よかった bug)


def test_short_text_with_bogus_duration_stays_one_line():
    # 5 chars over 12.4s: whisper attached trailing silence. Must NOT be
    # character-cut into fragments, and the duration must be capped.
    lines = segment_lines([(499.06, 511.49, "よかったー")], SubtitleSettings())
    assert len(lines) == 1
    assert lines[0].text == "よかったー"
    assert lines[0].end - lines[0].start < 4.0


def test_medium_text_with_bogus_duration_no_char_fragments():
    text = "私好きですよそういう話ワクワクするし"
    lines = segment_lines([(528.0, 570.35, text)], SubtitleSettings())
    # may split at clause boundaries but never into 1-3 char shards
    assert all(len(l.text) >= 4 for l in lines)
    assert lines[-1].end - lines[0].start < 15.0


# --------------------------------------------------- word-timestamp path


def w(start, end, text):
    return Word(start, end, text)


def test_words_break_at_sentence_end_with_real_times():
    seg = Segment(0.0, 20.0, "irrelevant", words=[
        w(0.0, 0.4, "よかった"), w(0.4, 0.6, "ー。"),
        w(10.0, 10.5, "そういう"), w(10.5, 11.0, "話。"),
    ])
    lines = segment_lines([seg], SubtitleSettings())
    assert len(lines) == 2
    assert lines[0].text == "よかったー。"
    assert lines[0].end <= 1.0  # real word end, not the stretched 20s
    assert lines[1].start == 10.0  # real start after the silence gap


def test_words_break_on_silence_gap():
    seg = Segment(0.0, 30.0, "x", words=[
        w(0.0, 0.5, "Hello"), w(0.6, 1.0, " there"),
        w(9.0, 9.5, " again"),  # 8s gap → new line
    ])
    lines = segment_lines([seg], SubtitleSettings())
    assert len(lines) == 2
    assert lines[0].text == "Hello there"
    assert lines[1].text == "again"
    assert lines[1].start == 9.0


def test_words_respect_the_two_display_line_budget():
    # max_chars_per_line is per DISPLAY line; a cue may fill two of them
    words = [w(i * 0.5, i * 0.5 + 0.4, f" word{i}") for i in range(20)]
    lines = segment_lines([Segment(0, 10, "x", words=words)], SubtitleSettings(max_chars_per_line=30))
    assert all(len(l.text) <= 60 for l in lines)
    assert any(len(l.text) > 30 for l in lines)  # the old per-cue limit is gone


def test_tiny_fragment_merged_into_previous():
    segs = [(0.0, 2.0, "なんかいいね"), (2.1, 2.4, "し")]
    lines = segment_lines(segs, SubtitleSettings())
    assert len(lines) == 1
    assert lines[0].text == "なんかいいねし"
    assert lines[0].end == 2.4


# ------------------------------------------- sentence merging (the shift bug)
#
# Real cues from the user's Friends S01 run. Sentences cut mid-phrase left
# lines like "mind." with no content of their own, and the translator filled
# them with the NEXT line's meaning — shifting the whole file by one.


def test_sentence_split_across_cues_is_rejoined():
    segs = [
        (327.100, 328.800, "Yeah, like that thought never entered my"),
        (328.800, 329.300, "mind."),
    ]
    lines = segment_lines(segs, SubtitleSettings())
    assert len(lines) == 1
    assert lines[0].text == "Yeah, like that thought never entered my mind."
    assert (lines[0].start, lines[0].end) == (327.100, 329.300)


def test_complete_sentence_is_not_glued_to_the_next_one():
    segs = [
        (336.420, 336.920, "Okay."),
        (336.920, 339.080, "Okay, um, senior year of college"),
        (339.080, 340.660, "on a pool table."),
    ]
    lines = segment_lines(segs, SubtitleSettings())
    assert [l.text for l in lines] == [
        "Okay.",
        "Okay, um, senior year of college on a pool table.",
    ]


def test_merge_stops_at_a_real_pause():
    segs = [(0.0, 2.0, "Something happens and"), (5.0, 6.0, "then more")]
    lines = segment_lines(segs, SubtitleSettings())
    assert len(lines) == 2  # 3s of silence: not one sentence, whatever grammar says


def test_merge_respects_the_char_budget():
    segs = [(0.0, 2.0, "a" * 60), (2.0, 4.0, "b" * 40)]
    lines = segment_lines(segs, SubtitleSettings(max_chars_per_line=42))
    assert len(lines) == 2  # 60 + 40 > 84


def test_cjk_merge_keeps_no_space():
    segs = [(0.0, 2.0, "私好きですよ"), (2.2, 3.0, "そういう話")]
    lines = segment_lines(segs, SubtitleSettings())
    assert lines[0].text == "私好きですよそういう話"


def test_word_path_breaks_at_a_clause_boundary():
    words = [w(i * 0.3, i * 0.3 + 0.25, t) for i, t in enumerate(
        ["This", " is", " a", " long", " clause,", " and", " another", " part", " here"]
    )]
    lines = segment_lines([Segment(0, 3, "x", words=words)], SubtitleSettings(max_chars_per_line=20))
    assert lines[0].text == "This is a long clause,"  # not cut after "another"


def test_open_ended_ratio_measures_fragmentation():
    from app.services.segmenter import open_ended_ratio

    lines = segment_lines(
        [(0.0, 1.0, "Done."), (2.0, 3.0, "Also done!"), (6.0, 7.0, "but not this")],
        SubtitleSettings(),
    )
    assert open_ended_ratio(lines) == pytest.approx(1 / 3)

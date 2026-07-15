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


def test_words_respect_max_chars():
    words = [w(i * 0.5, i * 0.5 + 0.4, f" word{i}") for i in range(20)]
    lines = segment_lines([Segment(0, 10, "x", words=words)], SubtitleSettings(max_chars_per_line=30))
    assert all(len(l.text) <= 30 for l in lines)


def test_tiny_fragment_merged_into_previous():
    segs = [(0.0, 2.0, "なんかいいね"), (2.1, 2.4, "し")]
    lines = segment_lines(segs, SubtitleSettings())
    assert len(lines) == 1
    assert lines[0].text == "なんかいいねし"
    assert lines[0].end == 2.4

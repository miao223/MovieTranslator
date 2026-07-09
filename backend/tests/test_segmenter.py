from app.models.schemas import SubtitleSettings
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
    # time is contiguous and monotonic
    assert lines[0].start == 0.0
    for a, b in zip(lines, lines[1:]):
        assert b.start >= a.start
    assert abs(lines[-1].end - 9.0) < 0.01


def test_indexes_are_sequential():
    segs = [(0.0, 2.0, "One."), (2.0, 4.0, "Two."), (4.0, 6.0, "Three.")]
    lines = segment_lines(segs, SubtitleSettings())
    assert [l.index for l in lines] == [1, 2, 3]


def test_split_text_no_punctuation_uses_spaces():
    chunks = _split_text("aaa bbb ccc ddd eee fff", 10)
    assert all(len(c) <= 10 for c in chunks)
    assert " ".join(chunks).replace("  ", " ") == "aaa bbb ccc ddd eee fff"


def test_split_text_hard_cut():
    chunks = _split_text("a" * 25, 10)
    assert chunks == ["a" * 10, "a" * 10, "a" * 5]

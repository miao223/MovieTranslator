"""Turn whisper segments into subtitle lines.

Preferred path: word-level timestamps. Lines are assembled word by word and
broken at sentence punctuation, silence gaps, or when the length/duration
limits are hit — every cue keeps REAL start/end times, so stretched or
fabricated timings cannot occur.

Fallback path (word timestamps disabled): the classic text splitter, with
two guards against whisper's stretched-segment pathology: a duration far
beyond what the text could occupy is capped to a speaking-rate estimate,
and short texts are never hard-cut into character fragments.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

from app.models.schemas import SubtitleLine, SubtitleSettings
from app.services.asr import Segment, Word

# split preference: sentence enders > clause punctuation > spaces
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？…])\s*")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:，；：、])\s*")
_SENTENCE_END = re.compile(r"[.!?。！？…]$")
# same punctuation as the two regexes above, as sets for last-char tests
_SENTENCE_CHARS = ".!?。！？…"
_CLAUSE_CHARS = ",;:，；：、"

MIN_LINE_DURATION = 0.5  # seconds
GAP_BREAK = 1.5  # silence between words that forces a new line (s)
NO_HARD_SPLIT_BELOW = 12  # never character-cut texts shorter than this
CHARS_PER_SECOND = 3.0  # conservative speaking-rate floor for bogus-duration cap
# How short is "too short to mean anything"? The number depends on the
# script, because scripts delimit differently — see text_units(). Counting
# characters everywhere is what let English fragments through every rule
# here: "around" is six characters but one word, and left on its own it has
# no counterpart in the target language, so the translator folds it into the
# previous line and shifts the rest of the film.
FRAGMENT_CJK_CHARS = 2
FRAGMENT_LATIN_WORDS = 2
# ...and a latin fragment has to be short in characters too. Word count
# alone would call a single 60-character token a fragment; whatever that
# is (a URL, a run-on transcription artefact), it is not the tail of a
# sentence. Real ones sit far below: "around" 6, "holding on" 10,
# "legal counsel" 13.
FRAGMENT_LATIN_MAX_CHARS = 20
# ...and a fragment gets merged back regardless of the measured gap: a cue
# that short is precisely the cue whose word timestamps cannot be trusted,
# so a large gap next to it is evidence the timing is wrong, not that the
# speech is apart. Trusting the gap produced 「て」/「かこれ…」 → "什" / "么…".
MERGE_FRAGMENT_GAP = GAP_BREAK
# a time-driven break must never strand a piece this short: whisper
# occasionally reports a multi-second gap in the middle of a single word
# (「伊」…25s…「勢が6票」), and honouring it splits the word in half
MIN_PIECE_CJK_CHARS = 3
MIN_PIECE_LATIN_WORDS = 1

# A cue may occupy two display lines (the 42-chars figure of subtitle style
# guides is per DISPLAY line, not per cue), so accumulation gets twice the
# budget and max_chars_per_line only governs wrapping in subtitle.py.
# Enforcing it per cue is what shredded normal English sentences into
# "…never entered my" + "mind." — which in turn let the translator reflow
# content across lines and shift the whole file.
CUE_CHAR_BUDGET_RATIO = 2
MERGE_GAP = 0.8  # max silence between lines of the same sentence (s)
MERGE_MAX_DURATION = 7.0  # a merged cue never lasts longer than this (s)
# a clause-boundary break must leave at least this share of the buffer behind,
# otherwise breaking at the overflowing word is the lesser evil
MIN_CLAUSE_BREAK_RATIO = 0.4


def is_cjk_text(text: str) -> bool:
    """Is this written in a script that does not separate words with spaces?"""
    counted = [ch for ch in text if not ch.isspace()]
    if not counted:
        return False
    cjk = sum(
        1
        for ch in counted
        if "぀" <= ch <= "ヿ" or "㐀" <= ch <= "鿿" or "가" <= ch <= "힯"
    )
    return cjk >= len(counted) * 0.5


def text_units(text: str) -> int:
    """Length of *text* in the units its script actually delimits.

    Chinese and Japanese write without spaces, so a character is the
    meaningful unit. Latin scripts delimit words, and there a character
    count says nothing about whether a line can stand on its own.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped) if is_cjk_text(stripped) else len(stripped.split())


def is_short_text(text: str, cjk_chars: int, latin_words: int) -> bool:
    """Is *text* at or below the given limit, in its own script's units?"""
    size = text_units(text)
    if size == 0:
        return False
    return size <= (cjk_chars if is_cjk_text(text.strip()) else latin_words)


def is_fragment_text(text: str) -> bool:
    """Too short to carry meaning on its own, whatever the script.

    Sentence punctuation is the veto: "Okay." and "Yeah." are one word
    each and are complete lines, while "around" and "years" are the tail
    of a sentence whose head is in the previous cue. Ending on a full stop
    is the ASR telling us which of the two this is.
    """
    stripped = text.strip()
    if _SENTENCE_END.search(stripped):
        return False
    if not is_cjk_text(stripped) and len(stripped) > FRAGMENT_LATIN_MAX_CHARS:
        return False
    return is_short_text(stripped, FRAGMENT_CJK_CHARS, FRAGMENT_LATIN_WORDS)


def _cap_bogus_duration(text: str, start: float, end: float) -> float:
    """Return a sane end time: whisper sometimes attaches long silence."""
    duration = end - start
    plausible = max(len(text) / CHARS_PER_SECOND, MIN_LINE_DURATION) * 2
    if duration > plausible:
        return start + max(len(text) / CHARS_PER_SECOND, MIN_LINE_DURATION)
    return end


# ------------------------------------------------------------- word path


def _clause_break_index(buf: Sequence[Word]) -> Optional[int]:
    """Index to split *buf* at so the first part ends on a natural boundary.

    Prefers the latest sentence ender, then the latest clause punctuation.
    Returns None when no boundary leaves a substantial enough first part —
    breaking mid-phrase is then the lesser evil.
    """
    total = len("".join(w.text for w in buf).strip())
    if total == 0:
        return None
    for chars in (_SENTENCE_CHARS, _CLAUSE_CHARS):
        for i in range(len(buf) - 1, 0, -1):
            head = "".join(w.text for w in buf[:i]).strip()
            if len(head) < total * MIN_CLAUSE_BREAK_RATIO:
                break
            if head[-1] in chars:
                return i
    return None


def _trustworthy_start(take: Sequence[Word]) -> float:
    """When does this cue's speech really begin?

    A cue can only contain a gap longer than ``GAP_BREAK`` because the
    break there was suppressed to avoid stranding a fragment — i.e. because
    whisper claimed several seconds of silence in the middle of one word.
    One of the two timestamps around that gap is wrong, and it is the
    fragment's: a word transcribed as a single character is what whisper
    misplaces. So step over such gaps while the text before them is still
    too short to be a line, and start the cue where the rest is spoken.
    Otherwise 「伊勢が6票」 appears on screen 25 seconds early.
    """
    for i in range(1, len(take)):
        head = "".join(w.text for w in take[:i]).strip()
        if not is_short_text(head, MIN_PIECE_CJK_CHARS, MIN_PIECE_LATIN_WORDS):
            break
        if take[i].start - take[i - 1].end > GAP_BREAK:
            return take[i].start
    return take[0].start


def _lines_from_words(words: Sequence[Word], settings: SubtitleSettings) -> List[SubtitleLine]:
    lines: List[SubtitleLine] = []
    buf: List[Word] = []
    budget = max(settings.max_chars_per_line * CUE_CHAR_BUDGET_RATIO, 1)

    def flush(split_at: Optional[int] = None):
        """Emit ``buf[:split_at]`` as a line; whatever is left stays buffered."""
        if not buf:
            return
        take = buf[:split_at] if split_at is not None else list(buf)
        rest = buf[split_at:] if split_at is not None else []
        text = "".join(w.text for w in take).strip()
        if text:
            end = max(take[-1].end, take[0].start + MIN_LINE_DURATION)
            start = _trustworthy_start(take)
            # backstop for anything the scan above did not catch: a span far
            # beyond max_duration is a bad timestamp, not a long line, and
            # the end is the trustworthy side of it (that is where the audio
            # just was)
            start = max(start, end - settings.max_duration)
            lines.append(
                SubtitleLine(
                    index=0, start=round(start, 3), end=round(end, 3), text=text
                )
            )
        else:
            rest = list(buf) if split_at is not None else []
        buf.clear()
        buf.extend(rest)

    for word in words:
        if buf:
            head = "".join(w.text for w in buf).strip()
            candidate_len = len(head + word.text.rstrip())
            too_long = candidate_len > budget
            # measure the cue's length from where its speech really starts,
            # not from a timestamp already known to be wrong: with a bogus
            # leading word 577s early, every following word looks "too slow"
            # and the buffer gets cut after four kana — 「マッショ」 + 「ンは…」
            too_slow = (
                word.end - _trustworthy_start(buf) > settings.max_duration
            )
            gap = word.start - buf[-1].end > GAP_BREAK
            # a break here would leave `head` alone as a cue; below a few
            # characters that is not a line but half a word, and the
            # translator has no choice but to render it as half a word
            strands_fragment = is_short_text(
                head, MIN_PIECE_CJK_CHARS, MIN_PIECE_LATIN_WORDS
            )
            if (gap or too_slow) and not strands_fragment:
                flush()  # time-driven breaks always cut the whole buffer
            elif too_long:
                flush(_clause_break_index(buf))
        buf.append(word)
        if _SENTENCE_END.search(word.text.strip()):
            flush()
    flush()
    return lines


# ------------------------------------------------------------- text path


def _split_text(text: str, max_chars: int) -> List[str]:
    """Split *text* into chunks of at most *max_chars*, at natural boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    for pattern in (_SENTENCE_BREAK, _CLAUSE_BREAK):
        parts = [p for p in pattern.split(text) if p.strip()]
        if len(parts) > 1:
            chunks: List[str] = []
            current = ""
            for part in parts:
                candidate = (current + " " + part).strip() if current else part.strip()
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part.strip()
            if current:
                chunks.append(current)
            out: List[str] = []
            for c in chunks:
                out.extend(_split_text(c, max_chars) if len(c) > max_chars else [c])
            return out

    words = text.split(" ")
    if len(words) > 1:
        chunks, current = [], ""
        for w in words:
            candidate = (current + " " + w).strip() if current else w
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = w
        if current:
            chunks.append(current)
        out = []
        for c in chunks:
            out.extend(_split_text(c, max_chars) if len(c) > max_chars else [c])
        return out

    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _lines_from_text(
    start: float, end: float, text: str, settings: SubtitleSettings
) -> List[SubtitleLine]:
    end = _cap_bogus_duration(text, start, end)
    duration = max(end - start, MIN_LINE_DURATION)
    chunks = _split_text(text, settings.max_chars_per_line)
    # honour max_duration only when the text is long enough to survive the
    # extra splitting — short texts with inflated durations were the source
    # of the 2-chars-per-5-seconds fragment bug
    if chunks and duration / len(chunks) > settings.max_duration:
        import math

        need = math.ceil(duration / settings.max_duration)
        while len(chunks) < need:
            longest = max(range(len(chunks)), key=lambda i: len(chunks[i]))
            if len(chunks[longest]) < NO_HARD_SPLIT_BELOW:
                break
            sub = _split_text(chunks[longest], max(len(chunks[longest]) // 2, 1))
            if len(sub) <= 1:
                break
            chunks[longest : longest + 1] = sub

    total_chars = sum(len(c) for c in chunks) or 1
    lines: List[SubtitleLine] = []
    t = start
    for chunk in chunks:
        share = duration * len(chunk) / total_chars
        lines.append(
            SubtitleLine(
                index=0,
                start=round(t, 3),
                end=round(min(t + share, end), 3),
                text=chunk,
            )
        )
        t += share
    return lines


# ------------------------------------------------------- sentence merging


def _needs_space(ch: str) -> bool:
    """CJK scripts write without spaces between words; latin scripts don't."""
    return not ("　" <= ch <= "鿿" or "＀" <= ch <= "￯")


def _join_text(a: str, b: str) -> str:
    if not a or not b:
        return a or b
    if _needs_space(a[-1]) and _needs_space(b[0]):
        return f"{a} {b}"
    return a + b


def _can_merge(
    prev: SubtitleLine,
    line: SubtitleLine,
    budget: int,
    max_duration: float,
    punctuated: bool = True,
) -> bool:
    gap = line.start - prev.end
    # a fragment on EITHER side makes this pair a fragment pair: whichever
    # cue is too short to stand alone is the one whose timing is suspect
    if is_fragment_text(prev.text) or is_fragment_text(line.text):
        if is_fragment_text(line.text) and _SENTENCE_END.search(prev.text.strip()):
            # the fragment opens the NEXT utterance, so gluing it backwards
            # splits a word across cues just as surely: "I'm gonna call
            # Odessa PD. I" + "understand it now…"
            return False
        if gap > MERGE_FRAGMENT_GAP:  # a stray fragment, not a separate line
            return False
    else:
        if not punctuated:
            # Rejoining two substantial cues rests entirely on "the previous
            # one does not end a sentence". Without punctuation that is not
            # a finding, it is the absence of one — and acting on it glued
            # four sentences from three speakers into one 6.6s cue. Leave
            # the pair alone; refine restores the punctuation and then
            # decides, with the words in front of it.
            return False
        if _SENTENCE_END.search(prev.text):
            return False  # previous cue is a complete sentence, leave it alone
        if gap > MERGE_GAP:
            return False  # a real pause: same sentence or not, don't glue cues
    if len(_join_text(prev.text, line.text)) > budget:
        return False
    return line.end - prev.start <= max_duration


def _merge_sentence_units(
    lines: List[SubtitleLine], settings: SubtitleSettings, punctuated: bool = True
) -> List[SubtitleLine]:
    """Glue cues that whisper cut mid-sentence back into whole sentences.

    A cue ending without sentence punctuation is the tell: whatever follows
    it after a short gap is the rest of that sentence. Leaving those halves
    apart is what pushes the translator into re-flowing content across lines.

    That tell only exists in a punctuated transcript. When *punctuated* is
    False only the fragment repair runs — see ``_can_merge``.
    """
    budget = max(settings.max_chars_per_line * CUE_CHAR_BUDGET_RATIO, 1)
    max_duration = max(settings.max_duration, MERGE_MAX_DURATION)
    merged: List[SubtitleLine] = []
    for line in lines:
        if merged and _can_merge(merged[-1], line, budget, max_duration, punctuated):
            merged[-1].text = _join_text(merged[-1].text, line.text)
            merged[-1].end = line.end
        else:
            merged.append(line)
    return merged


def _fix_overlaps(lines: List[SubtitleLine]) -> int:
    """Keep cues strictly ordered after a start time has been pulled back.

    Clamping a bogus start (see ``flush``) can move a cue behind its
    predecessor; two cues on screen at once is the one artefact a player
    cannot recover from. Returns how many cues had to be nudged.
    """
    fixed = 0
    for prev, line in zip(lines, lines[1:]):
        if line.start < prev.end:
            line.start = round(min(prev.end, line.end - 0.05), 3)
            fixed += 1
    return fixed


# Punctuation counts as a usable signal only when the ASR actually supplies
# it. One transcript came back with 6% of its cues punctuated — enough for
# has_sentence_punctuation() to answer yes, nowhere near enough to tell
# where a sentence ends. The cut sits at "absent more often than present"
# rather than at any measured value, so it encodes no single film's numbers.
UNPUNCTUATED_ABOVE = 0.5


def is_effectively_unpunctuated(lines: Sequence[SubtitleLine]) -> bool:
    """Should punctuation-based rules be trusted on this transcript?"""
    return open_ended_ratio(lines) > UNPUNCTUATED_ABOVE


def has_sentence_punctuation(lines: Sequence[SubtitleLine]) -> bool:
    """Did the ASR punctuate at all?

    Almost every heuristic downstream — where to break a line, whether a
    cue is a complete sentence, where to wrap the display text — reads
    sentence punctuation. Whisper sometimes returns a whole film without a
    single one, and then all of them silently become no-ops instead of
    misbehaving visibly. Callers use this to say so in the log.
    """
    return any(_SENTENCE_END.search(l.text.strip()) for l in lines)


def open_ended_ratio(lines: Sequence[SubtitleLine]) -> float:
    """Share of cues not ending on sentence punctuation — the fragmentation
    metric reported in the job log (a healthy film sits below ~0.1).

    Meaningless when the ASR emitted no punctuation at all; check
    ``has_sentence_punctuation`` before reading anything into it.
    """
    if not lines:
        return 0.0
    open_ended = sum(1 for l in lines if not _SENTENCE_END.search(l.text.strip()))
    return open_ended / len(lines)


def fragment_count(lines: Sequence[SubtitleLine]) -> int:
    """Cues too short to carry meaning on their own — the metric that would
    have exposed the "half a word per cue" bug (17% of one film's cues)."""
    return sum(
        1
        for l in lines
        if is_fragment_text(l.text)
        and not _SENTENCE_END.search(l.text.strip())
    )


# ---------------------------------------------------------------- public


def segment_lines(
    segments: Sequence[Union[Segment, tuple]],
    settings: SubtitleSettings,
    debug=None,
) -> List[SubtitleLine]:
    lines: List[SubtitleLine] = []
    for seg in segments:
        if isinstance(seg, tuple):  # legacy (start, end, text) form
            seg = Segment(seg[0], seg[1], seg[2])
        if not seg.text.strip():
            continue
        if seg.words:
            lines.extend(_lines_from_words(seg.words, settings))
        else:
            lines.extend(_lines_from_text(seg.start, seg.end, seg.text, settings))

    # A cue's real start is only known once flush() has discounted whisper's
    # bogus ones (see _trustworthy_start), and that can move a cue minutes
    # past where its segment claimed to begin — behind cues that genuinely
    # come first. Everything below reads this list as chronological: merging
    # joins neighbours, _fix_overlaps compares each cue against the previous
    # one, and the translator takes context from line order. So restore time
    # order here, where the starts are final. A film whose timestamps behave
    # is already sorted, and this is then a no-op.
    lines.sort(key=lambda l: (l.start, l.end))

    raw = [l.model_copy() for l in lines] if debug is not None and debug.enabled else []
    punctuated = not is_effectively_unpunctuated(lines)
    merged = _merge_sentence_units(lines, settings, punctuated)
    nudged = _fix_overlaps(merged)

    for i, line in enumerate(merged, start=1):
        line.index = i

    if debug is not None and debug.enabled:
        _report(debug, raw, merged, nudged, punctuated)
    return merged


def _report(
    debug,
    raw: List[SubtitleLine],
    merged: List[SubtitleLine],
    nudged: int,
    punctuated: bool,
):
    from app.core.debuglog import fmt_cue, percentiles

    debug.section("分句结果（segmenter）")
    debug.kv("断行后条数", len(raw))
    debug.kv("句子合并后条数", len(merged))
    debug.kv("因起点夹紧而顺延的 cue", nudged)
    debug.kv("ASR 是否输出句末标点", "是" if has_sentence_punctuation(merged) else "否 ⚠")
    debug.kv(
        "句子合并策略",
        "完整（转写有标点可依据）" if punctuated
        else "仅修复碎片，整句合并交给预处理（转写标点不可用 ⚠）",
    )
    debug.kv("未完句结尾占比", f"{open_ended_ratio(merged):.0%}")
    debug.kv(
        f"碎片 cue（中日文≤{FRAGMENT_CJK_CHARS}字／西文≤{FRAGMENT_LATIN_WORDS}词）",
        f"{fragment_count(merged)} / {len(merged)}",
    )
    gaps = [b.start - a.end for a, b in zip(merged, merged[1:])]
    debug.kv("相邻 cue 间隔(s)", percentiles(gaps))
    debug.kv("cue 时长(s)", percentiles([l.end - l.start for l in merged]))
    debug.kv(
        "语音总时长/覆盖",
        f"{sum(l.end - l.start for l in merged):.0f}s"
        + (f"，末尾 {merged[-1].end:.0f}s" if merged else ""),
    )
    debug.line("\n合并前（断行直出）：")
    debug.lines(fmt_cue(i, l.start, l.end, l.text) for i, l in enumerate(raw, 1))
    debug.line("\n合并后（送入预处理的内容）：")
    debug.lines(fmt_cue(l.index, l.start, l.end, l.text) for l in merged)

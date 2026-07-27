"""Find what is sung rather than spoken, and mark it.

Why this exists
---------------
On one English film a single song met three different fates depending only
on which stage of the pipeline produced it:

* picked up by the first pass (15 lines, the title song among them) — kept,
  translated as dialogue, unmarked;
* picked up by the second pass with whisper's own ``♪`` prefix (90 lines) —
  discarded by the vetting pass;
* picked up by the second pass without it (6 lines) — judged inconsistently.

Two things caused that. Whisper's ``♪`` is not a judgement about singing at
all: it appeared 0 times in 434 first-pass segments and 90 times in 216
second-pass ones, i.e. it tracks whether the VAD was on, not whether anyone
was singing. And the vetting prompt had no category for lyrics — its labels
are dialogue / off-topic / filler / annotation, where "annotation" means the
``【…】``/``♪`` *format*. A song playing in the film fits none of them, so
the model had to force it into the nearest one.

The model was not the limit. Given the same transcript it identified an
unmarked lyric with no surrounding context at all, and misfiled four lines
of an obvious song that happened to sit in a silent stretch — the hard case
right, the easy one wrong. What it lacked was a label, so this pass gives it
one and asks nothing else of it.

Safety
------
This pass only ever *annotates*: it sets ``is_lyric`` and never adds,
removes or rewrites a line. So every failure mode — a malformed reply, a
range that does not verify, an unreachable endpoint — degrades to "nothing
marked", which is exactly the behaviour with the feature switched off.

Timestamps are never sent (project-wide rule); the model sees numbered
source lines and answers with line ranges.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from app.models.schemas import LLMSettings, NetworkSettings, SubtitleLine
from app.services.refine import _overlap, _words
from app.services.translator import chat_completion, estimate_tokens, make_openai_client

LogFn = Callable[[str], None]

MARK = "♪"

# `[120-135] head text | tail text`
_RANGE_RE = re.compile(
    r"^\s*[*\-•]?\s*\[\s*(\d+)\s*(?:[-–~]\s*(\d+)\s*)?\]\s*(.*?)\s*$"
)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$")

# The reply is one short line per stretch of song, so the binding limit is
# the transcript going in. A 90-minute film measured under 10k tokens.
LYRICS_CHUNK_TOKENS = 12_000
MARKER_TOKENS = 3
# share of the first/last line's words the quoted snippet must account for.
# This is the "did you lose your place" check — the failure this project has
# been bitten by more than any other.
MIN_SNIPPET_OVERLAP = 0.6
# No single stretch of song may claim more of the film than this — the check
# is there to catch a runaway "[1-500] the whole film is a song", not to
# second-guess a long musical number, so it never bites below MAX_RANGE_LINES.
MAX_RANGE_SHARE = 0.20
MAX_RANGE_LINES = 60
GIVE_UP_AFTER = 2


def build_lyrics_prompt(language_hint: str = "") -> str:
    lang = f"（原文语言：{language_hint}）" if language_hint else ""
    return "\n".join([
        f"你是字幕校对员{lang}。下面是一部影片的完整字幕原文，格式 `[行号] 原文`。",
        "请找出其中**唱出来**的内容，而不是说出来的内容。",
        "",
        "算作歌词：",
        "- 背景插曲、片头曲、片尾曲的歌词；",
        "- 片中人物自己唱的歌（生日歌、婚礼上的演奏与合唱、角色哼的小调）；",
        "- 收音机、电视、留声机里放出来的歌。",
        "不算歌词：",
        "- 正常对白、旁白、独白；",
        "- 朗读、念稿、背诵诗文、喊口号；",
        "- 只是措辞文艺的台词。",
        "",
        "可用的判断线索：歌词往往押韵、有重复的段落或副歌、句式工整、"
        "相邻几行连起来是完整的一段，而且与前后的对白不构成问答关系。",
        "**拿不准时不要标**——漏标只是少一个记号，错标会给正常对白套上歌词标记。",
        "",
        "输出格式：每一段连续的歌词输出一行",
        "`[起始行号-结束行号] 起始行开头几个词 | 结束行结尾几个词`",
        "竖线两侧照抄原文中的片段，用来核对行号没有数错。单行的歌词写 `[行号] …… | ……`。",
        "区间必须升序、互不重叠。全片没有歌词时输出 `无`。",
        "不要输出解释、代码块标记或任何多余内容。",
        "示例：",
        "[120-123] There's a brighter side | brighter, brighter side.",
        "[210] Happy birthday dear boy | happy birthday to you.",
    ])


def parse_ranges(reply: str) -> List[Tuple[int, int, str, str]]:
    """Parse `[a-b] head | tail` into (first, last, head, tail) tuples."""
    out: List[Tuple[int, int, str, str]] = []
    for raw in reply.splitlines():
        if _FENCE_RE.match(raw):
            continue
        m = _RANGE_RE.match(raw)
        if not m:
            continue
        first = int(m.group(1))
        last = int(m.group(2)) if m.group(2) else first
        body = m.group(3)
        head, _, tail = body.partition("|")
        out.append((first, last, head.strip(), tail.strip()))
    return out


def _snippet_matches(snippet: str, line_text: str) -> bool:
    """Is *snippet* really a piece of *line_text*?

    Compared by word overlap rather than literally: the model retypes the
    quote and may normalise punctuation or casing, but it cannot invent the
    words of a line it never looked at.
    """
    words = _words(snippet)
    if not words:
        return False
    kept, _total, _added = _overlap(snippet, line_text)
    return kept >= max(1, round(len(words) * MIN_SNIPPET_OVERLAP))


def _verify(
    rng: Tuple[int, int, str, str],
    by_index: dict[int, SubtitleLine],
    total: int,
) -> str:
    """Empty string when the range is usable, otherwise why it is not."""
    first, last, head, tail = rng
    if last < first:
        return "区间倒置"
    if first not in by_index or last not in by_index:
        return f"行号越界（全片 1-{total}）"
    limit = max(MAX_RANGE_LINES, int(total * MAX_RANGE_SHARE))
    if last - first + 1 > limit:
        return f"区间过长（{last - first + 1} 行 > 上限 {limit} 行）"
    if not _snippet_matches(head, by_index[first].text):
        return f"起始片段对不上第 {first} 行"
    if not _snippet_matches(tail, by_index[last].text):
        return f"结束片段对不上第 {last} 行"
    return ""


def _chunks(lines: Sequence[SubtitleLine], context_limit: int) -> List[List[SubtitleLine]]:
    budget = min(max(context_limit // 3, 2_000), LYRICS_CHUNK_TOKENS)
    out: List[List[SubtitleLine]] = []
    current: List[SubtitleLine] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line.text) + MARKER_TOKENS
        if current and used + cost > budget:
            out.append(current)
            current, used = [], 0
        current.append(line)
        used += cost
    if current:
        out.append(current)
    return out


def _numbered(lines: Sequence[SubtitleLine]) -> str:
    return "\n".join(f"[{l.index}] {l.text}" for l in lines)


def _tally(usage: Optional[dict], resp, dbg) -> None:
    reported = getattr(resp, "usage", None)
    if usage is None or reported is None:
        return
    cached = getattr(reported, "prompt_tokens_details", None)
    usage["calls"] += 1
    usage["prompt"] += getattr(reported, "prompt_tokens", 0) or 0
    usage["completion"] += getattr(reported, "completion_tokens", 0) or 0
    usage["cached"] += getattr(cached, "cached_tokens", 0) or 0
    if dbg:
        dbg.line(
            f"    tokens: 输入 {getattr(reported, 'prompt_tokens', 0)}"
            f"（缓存命中 {getattr(cached, 'cached_tokens', 0) or 0}）"
            f" 输出 {getattr(reported, 'completion_tokens', 0)}"
        )


def strip_marks(text: str) -> str:
    """Remove any ♪ the model put there, and the space it leaves behind."""
    return re.sub(r"\s+", " ", text.replace(MARK, " ")).strip()


def apply_marks(lines: Sequence[SubtitleLine]) -> None:
    """Make the ♪ on every line agree with its ``is_lyric`` flag.

    Run after translation as well as before it: the translator is asked to
    keep the marks, and it mostly does, but a marker that silently went
    missing or landed on the neighbouring line would be worse than none. The
    flag is the truth; the text is rewritten to match it.
    """
    for line in lines:
        line.text = strip_marks(line.text)
        line.translation = strip_marks(line.translation)
        if line.is_lyric:
            if line.text:
                line.text = f"{MARK} {line.text} {MARK}"
            if line.translation:
                line.translation = f"{MARK} {line.translation} {MARK}"


def mark_lyrics(
    lines: List[SubtitleLine],
    llm: LLMSettings,
    language_hint: str = "",
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    client=None,  # injectable for tests
    network: Optional[NetworkSettings] = None,
    debug=None,
    usage: Optional[dict] = None,
) -> List[SubtitleLine]:
    """Flag the sung lines in *lines*. Never adds, drops or rewrites a line."""
    log = log or (lambda _m: None)
    if not lines:
        return lines

    client = client if client is not None else make_openai_client(llm, network)
    system = build_lyrics_prompt(language_hint)
    chunks = _chunks(lines, llm.context_limit)
    by_index = {l.index: l for l in lines}
    dbg = debug if debug is not None and debug.enabled else None

    if dbg:
        dbg.section("歌词识别（LLM 判断哪些行是唱出来的）")
        dbg.kv("送审", f"{len(lines)} 行")
        dbg.kv("分块数", len(chunks))
        dbg.block("system prompt", system)

    marked: List[SubtitleLine] = []
    rejected: List[str] = []
    consecutive_failures = 0
    no_thinking = llm.disable_thinking

    for n, chunk in enumerate(chunks, start=1):
        if should_cancel and should_cancel():
            raise InterruptedError("cancelled")
        if consecutive_failures >= GIVE_UP_AFTER:
            continue
        user = "请找出以下字幕里唱出来的内容：\n" + _numbered(chunk)
        if dbg:
            dbg.block(f"第 {n} 块 请求", user)
        try:
            resp, no_thinking = chat_completion(
                client,
                model=llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,  # mechanical judgement: no creativity wanted
                no_thinking=no_thinking,
            )
            reply = resp.choices[0].message.content or ""
            _tally(usage, resp, dbg)
            if dbg:
                dbg.block(f"第 {n} 块 响应", reply)
            consecutive_failures = 0
        except (InterruptedError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 — annotation is best-effort
            consecutive_failures += 1
            log(f"⚠ 第 {chunk[0].index}-{chunk[-1].index} 行歌词识别未生效（{exc}）")
            if dbg:
                dbg.line(f"\n⚠ 第 {n} 块失败：{exc}")
            continue

        previous_last = 0
        for rng in parse_ranges(reply):
            first, last, _head, _tail = rng
            reason = _verify(rng, by_index, len(lines))
            if not reason and first <= previous_last:
                reason = f"与上一个区间重叠（上一个到第 {previous_last} 行）"
            if reason:
                rejected.append(f"[{first}-{last}] {reason}")
                continue
            previous_last = last
            for i in range(first, last + 1):
                by_index[i].is_lyric = True
                marked.append(by_index[i])

    if dbg and rejected:
        dbg.line(f"\n被校验否决的区间（{len(rejected)} 个）：")
        dbg.lines("  " + r for r in rejected)

    log(
        f"歌词识别完成：{len(marked)} / {len(lines)} 行判定为歌词"
        f"（{len(marked) / len(lines):.0%}）"
        + (f"，{len(rejected)} 个区间未通过校验" if rejected else "")
    )
    if dbg:
        from app.core.debuglog import fmt_time

        dbg.line(f"\n识别结果：{len(marked)} / {len(lines)} 行")
        dbg.lines(
            f"  [{l.index}] {fmt_time(l.start)} {MARK} {l.text}" for l in marked
        )
    return lines


def whisper_agreement(
    lines: Sequence[SubtitleLine], whisper_marked: Sequence[tuple[float, float]]
) -> tuple[int, int, int, List[SubtitleLine], List[SubtitleLine]]:
    """Cross-tabulate our judgement against whisper's own ``♪`` prefixes.

    Whisper's marking cannot be trusted on its own — it tracks the decoding
    mode, not the singing — but it is an *independent* second opinion, and
    that is enough to make this pass measurable without a human subtitle:
    where the two agree, confidence is high; where they disagree is the list
    worth listening to.
    """
    def touched(line: SubtitleLine) -> bool:
        return any(
            min(line.end, e) - max(line.start, s) > 0.2 for s, e in whisper_marked
        )

    both = only_ours = only_whisper = 0
    ours_alone: List[SubtitleLine] = []
    whisper_alone: List[SubtitleLine] = []
    for line in lines:
        w = touched(line)
        if line.is_lyric and w:
            both += 1
        elif line.is_lyric:
            only_ours += 1
            ours_alone.append(line)
        elif w:
            only_whisper += 1
            whisper_alone.append(line)
    return both, only_ours, only_whisper, ours_alone, whisper_alone

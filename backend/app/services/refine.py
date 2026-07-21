"""Transcript preprocessing: rejoin sentences the ASR cut apart, fix its
word-level mistakes — all in the SOURCE language, before translation.

Why this exists
---------------
Whisper cuts sentences mid-phrase ("…never entered my" / "mind."). A line
that carries no meaning of its own has no counterpart in the target
language, so the translator puts the whole sentence on the first line and
then fills the second with the NEXT line's content — shifting every line
after it, and the shift compounds because the model sees its own shifted
output in the conversation history.

The decisive property of doing this here rather than compensating during
translation: **input and output are the same language**, so the model's
answer can be verified mechanically. Every source line must be covered by
exactly one output unit, and each unit's text must still be recognisably
the concatenation of its source lines. A translation can never be checked
that way, which is why the shift went unnoticed for a whole film.

The model never sees timestamps (project-wide rule): merged cues take the
real start/end of the lines they came from, and constraints that need
timing (pauses, cue duration) are enforced locally afterwards.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

from app.models.schemas import LLMSettings, NetworkSettings, SubtitleLine, SubtitleSettings
from app.services.segmenter import (
    CUE_CHAR_BUDGET_RATIO,
    MERGE_MAX_DURATION,
    _join_text,
    open_ended_ratio,
)
from app.services.translator import estimate_tokens, make_openai_client

LogFn = Callable[[str], None]
ProgressFn = Callable[[float], None]  # 0..1

# `[110] text` or `[110-111] text`
_UNIT_RE = re.compile(r"^\s*\[(\d+)(?:\s*[-–~]\s*(\d+))?\]\s*(.*\S)\s*$")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$")

# a unit must still look like its source lines (see _is_faithful)
MIN_KEPT = 0.6  # share of source words that must survive
MAX_ADDED_RATIO = 0.25  # share of the unit's words that may be new
MAX_ADDED_FLOOR = 2  # ...but always allow a couple, so short lines can be fixed
# one chunk is restated in full by the reply, so it must fit the model's
# OUTPUT limit (commonly 4k-16k tokens), not just its context window
REFINE_CHUNK_TOKENS = 1_500
# lines further apart than this are separate utterances no matter what the
# grammar suggests — the model cannot know, it never sees timestamps
MAX_INTERNAL_GAP = 1.2
# how many lines of the previous chunk are shown as read-only context
CONTEXT_LINES = 2
# consecutive chunk failures after which the pass gives up entirely — an
# unreachable endpoint must not cost one timeout per chunk before translation
GIVE_UP_AFTER = 3

_WORD_RE = re.compile(r"[0-9A-Za-z']+|[^\sA-Za-z0-9']")


def build_refine_prompt(language_hint: str = "") -> str:
    lang = f"（原文语言：{language_hint}）" if language_hint else ""
    return "\n".join([
        f"你是字幕转写整理员{lang}。输入是语音识别产生的字幕行，格式 `[行号] 原文`。",
        "你的任务只有三件事：",
        "1. 把被错误切断的同一句话合并成一行（这是最重要的任务）；",
        "2. 修正明显的语音识别错误：同音/近音词、漏词、错词；",
        "3. 删除口吃、重复词、语气噪音等明显的识别杂音。",
        "严格禁止：",
        "- 禁止翻译，输出必须与输入是同一种语言；",
        "- 禁止改写措辞、润色文风、增加或删减实质内容；",
        "- 禁止把一行拆成多行；",
        "- 人名、地名、专有名词保持原样，不要改动。",
        "输出格式：每行 `[行号] 整理后的原文`；合并多行时写成 `[起始行号-结束行号] 整理后的原文`。",
        "必须覆盖输入的全部行号，顺序递增，不得跳过、重复或新增行号。",
        "不要输出解释、注释、代码块标记或任何多余内容。",
        "示例：",
        "输入: [7] Yeah, like that thought never entered my / [8] mind. / [9] Okay, come on.",
        "输出: [7-8] Yeah, like that thought never entered my mind. / [9] Okay, come on.",
    ])


def _numbered(lines: Sequence[SubtitleLine]) -> str:
    return "\n".join(f"[{line.index}] {line.text}" for line in lines)


def parse_units(text: str) -> List[tuple[int, int, str]]:
    """Parse `[n] text` / `[n-m] text` into (first, last, text) triples."""
    units: List[tuple[int, int, str]] = []
    for raw in text.splitlines():
        if _FENCE_RE.match(raw):
            continue
        m = _UNIT_RE.match(raw)
        if not m:
            if units and raw.strip():  # continuation of the previous unit
                first, last, body = units[-1]
                units[-1] = (first, last, f"{body} {raw.strip()}")
            continue
        first = int(m.group(1))
        last = int(m.group(2)) if m.group(2) else first
        units.append((first, last, m.group(3).strip()))
    return units


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _overlap(candidate: str, source: str) -> tuple[int, int, int]:
    """(source words kept, source word count, words the model added)."""
    src = _words(source)
    pool: dict[str, int] = {}
    for word in src:
        pool[word] = pool.get(word, 0) + 1
    kept = added = 0
    for word in _words(candidate):
        if pool.get(word, 0) > 0:
            pool[word] -= 1
            kept += 1
        else:
            added += 1
    return kept, len(src), added


def _is_faithful(candidate: str, source: str) -> bool:
    """Is the model's version still the source text, not a rewrite?

    Two independent guards, because the two ways of losing content look
    nothing alike: dropping most of the line (deletion) and stuffing in
    text that was never spoken (hallucination — which keeps every source
    word and would sail through a "words retained" check alone).
    """
    kept, total, added = _overlap(candidate, source)
    if total == 0:
        return True
    if kept < total * MIN_KEPT:
        return False
    # a few new words are legitimate ASR corrections; a flood is invention
    return added <= max(MAX_ADDED_FLOOR, len(_words(candidate)) * MAX_ADDED_RATIO)


def _covers_exactly(units: Sequence[tuple[int, int, str]], lines: Sequence[SubtitleLine]) -> bool:
    """Ranges must be ascending, contiguous and cover every line once."""
    expected = lines[0].index
    for first, last, _text in units:
        if first != expected or last < first:
            return False
        expected = last + 1
    return expected == lines[-1].index + 1


def _apply_units(
    units: Sequence[tuple[int, int, str]],
    lines: Sequence[SubtitleLine],
    settings: SubtitleSettings,
) -> tuple[List[SubtitleLine], int, int]:
    """Turn validated units into cues. Returns (lines, merged, corrected)."""
    by_index = {line.index: line for line in lines}
    budget = max(settings.max_chars_per_line * CUE_CHAR_BUDGET_RATIO, 1)
    max_duration = max(settings.max_duration, MERGE_MAX_DURATION)

    out: List[SubtitleLine] = []
    merged = corrected = 0
    for first, last, text in units:
        sources = [by_index[i] for i in range(first, last + 1)]
        joined = ""
        for src in sources:
            joined = _join_text(joined, src.text)

        reject = not _is_faithful(text, joined)
        if not reject and len(sources) > 1:
            # constraints the model could not check: it never sees timestamps
            spans_pause = any(
                b.start - a.end > MAX_INTERNAL_GAP for a, b in zip(sources, sources[1:])
            )
            reject = (
                spans_pause
                or len(text) > budget
                or sources[-1].end - sources[0].start > max_duration
            )
        if reject:
            out.extend(sources)  # keep the local result for this range
            continue

        if len(sources) > 1:
            merged += len(sources) - 1
        if text != joined:
            corrected += 1
        out.append(
            SubtitleLine(
                index=0,
                start=sources[0].start,
                end=sources[-1].end,
                text=text,
            )
        )
    return out, merged, corrected


def _chunks(lines: List[SubtitleLine], context_limit: int) -> List[List[SubtitleLine]]:
    """Split the transcript so one request fits both context AND output limits.

    The reply repeats the chunk in full, so the binding constraint is the
    model's max output tokens — a whole film in one request gets truncated,
    fails the coverage check, and silently falls back to doing nothing.
    """
    budget = min(max(context_limit // 3, 500), REFINE_CHUNK_TOKENS)
    out: List[List[SubtitleLine]] = []
    current: List[SubtitleLine] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line.text) + 6  # + the "[n] " marker
        if current and used + cost > budget:
            out.append(current)
            current, used = [], 0
        current.append(line)
        used += cost
    if current:
        out.append(current)
    return out


def refine_lines(
    lines: List[SubtitleLine],
    llm: LLMSettings,
    subtitle: SubtitleSettings,
    language_hint: str = "",
    log: Optional[LogFn] = None,
    progress: Optional[ProgressFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    client=None,  # injectable for tests
    network: Optional[NetworkSettings] = None,
) -> List[SubtitleLine]:
    """Rejoin and clean up *lines*; returns renumbered cues.

    Never raises for model/network problems: any chunk that fails validation
    (or fails outright) falls back to its input lines, so a broken
    preprocessing pass can only leave the transcript as good as it was.
    """
    log = log or (lambda _m: None)
    progress = progress or (lambda _p: None)
    if not lines:
        return lines

    client = client if client is not None else make_openai_client(llm, network)
    system = build_refine_prompt(language_hint)
    chunks = _chunks(lines, llm.context_limit)
    before = open_ended_ratio(lines)

    result: List[SubtitleLine] = []
    merged = corrected = failed = 0
    consecutive_failures = 0
    for n, chunk in enumerate(chunks, start=1):
        if should_cancel and should_cancel():
            raise InterruptedError("cancelled")
        if consecutive_failures >= GIVE_UP_AFTER:
            # endpoint is down or the model cannot follow the format: stop
            # burning time on the remaining chunks, translation still runs
            result.extend(chunk)
            failed += 1
            continue
        if len(chunks) > 1:
            log(f"预处理第 {n}/{len(chunks)} 块（第 {chunk[0].index}-{chunk[-1].index} 行）…")
        context = result[-CONTEXT_LINES:] if result else []
        user = ""
        if context:
            user += (
                "上文（已整理，仅供参考，不要输出）：\n"
                + "\n".join(l.text for l in context)
                + "\n\n"
            )
        user += "请整理以下字幕行：\n" + _numbered(chunk)

        try:
            resp = client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,  # mechanical task: no creativity wanted
            )
            reply = resp.choices[0].message.content or ""
            units = parse_units(reply)
            if not units or not _covers_exactly(units, chunk):
                raise ValueError("行号覆盖校验未通过")
            applied, m, c = _apply_units(units, chunk, subtitle)
            consecutive_failures = 0
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 — preprocessing is best-effort
            failed += 1
            consecutive_failures += 1
            log(f"⚠ 第 {chunk[0].index}-{chunk[-1].index} 行预处理未生效（{exc}），沿用原始分行")
            if consecutive_failures >= GIVE_UP_AFTER:
                log(f"⚠ 连续 {GIVE_UP_AFTER} 块预处理失败，跳过剩余部分，直接进入翻译")
            applied, m, c = list(chunk), 0, 0

        result.extend(applied)
        merged += m
        corrected += c
        progress(n / len(chunks))

    for i, line in enumerate(result, start=1):
        line.index = i

    log(
        f"预处理完成：{len(lines)} 行 → {len(result)} 条"
        f"（合并 {merged} 处、纠错 {corrected} 处"
        + (f"、{failed} 块回退" if failed else "")
        + f"），未完句结尾占比 {before:.0%} → {open_ended_ratio(result):.0%}"
    )
    return result

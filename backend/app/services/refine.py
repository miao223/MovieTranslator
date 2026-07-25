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
    has_sentence_punctuation,
    is_fragment_text,
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
# grammar suggests — the model cannot know, it never sees timestamps.
# A floor, not the value actually used: see _gap_limit().
MAX_INTERNAL_GAP = 1.2
# ...and however inflated a film's timings are, a pause this long between
# two SUBSTANTIAL lines is a real one
MAX_INTERNAL_GAP_CEILING = 2.5
# share of the film's own inter-cue gaps the limit is placed at, so a
# transcript whose timings are uniformly padded still gets merged. With a
# fixed 1.2s, one film had 70% of its cue pairs vetoed before the model's
# proposal was even looked at, and the whole pass merged 6 lines out of 764.
GAP_PERCENTILE = 0.40
# How much a fragment merge may overflow the per-cue character budget.
# The budget protects readability; a fragment left alone breaks
# correctness — it has no counterpart in the target language, so the
# translator folds it into the neighbouring line and every line after it
# shifts. A third display line is the price, and it is the cheaper one:
# 32 of one English film's 70 budget rejections stranded a fragment, and
# the file came out 82% misaligned.
FRAGMENT_BUDGET_RATIO = 1.5
# how many lines of the previous chunk are shown as read-only context
CONTEXT_LINES = 2
# consecutive chunk failures after which the pass gives up entirely — an
# unreachable endpoint must not cost one timeout per chunk before translation
GIVE_UP_AFTER = 3

_WORD_RE = re.compile(r"[0-9A-Za-z']+|[^\sA-Za-z0-9']")
_PUNCT = set("。、，,.!?！？…‥：:；;「」『』（）()[]【】\"“”'‘’-—―~～·・/\\|*&#")
_SENTENCE_CHARS = ".!?。！？…"
_SENTENCE_END_RE = re.compile(rf"[{re.escape(_SENTENCE_CHARS)}]$")
_SENTENCE_CHARS_RE = re.compile(rf"[{re.escape(_SENTENCE_CHARS)}]")


def build_refine_prompt(language_hint: str = "", glossary: str = "") -> str:
    lang = f"（原文语言：{language_hint}）" if language_hint else ""
    parts = [
        f"你是字幕转写整理员{lang}。输入是语音识别产生的字幕行，格式 `[行号] 原文`。",
        "你的任务只有四件事：",
        "1. 把被错误切断的同一句话合并成一行（这是最重要的任务）。"
        "只有一两个字的行几乎一定是上一行或下一行的一部分，务必合并；",
        "2. 修正明显的语音识别错误：同音/近音词、漏词、错词；",
        "3. 删除口吃、重复词、语气噪音等明显的识别杂音；",
        "4. 补回缺失的句末标点（。？！等）和必要的停顿标点。"
        "语音识别常常整片不输出标点，而断句、换行、合并都依赖标点。"
        "只增删标点，一个字都不要改。",
        "严格禁止：",
        "- 禁止翻译，输出必须与输入是同一种语言；",
        "- 禁止改写措辞、润色文风、增加或删减实质内容；",
        "- 禁止把一行拆成多行；",
    ]
    if glossary.strip():
        parts += [
            "- 专有名词按下面的对照表纠正（识别结果与表中某个词读音相近时，"
            "改成表中的原文写法）；表中没有的人名、地名保持原样。",
            "专有名词对照表：\n" + glossary.strip(),
        ]
    else:
        parts.append("- 人名、地名、专有名词保持原样，不要改动。")
    parts += [
        "输出格式：每行 `[行号] 整理后的原文`；合并多行时写成 `[起始行号-结束行号] 整理后的原文`。",
        "必须覆盖输入的全部行号，顺序递增，不得跳过、重复或新增行号。",
        "不要输出解释、注释、代码块标记或任何多余内容。",
        "示例：",
        "输入: [7] Yeah, like that thought never entered my / [8] mind / [9] Okay, come on",
        "输出: [7-8] Yeah, like that thought never entered my mind. / [9] Okay, come on.",
    ]
    return "\n".join(parts)


def _tally(usage: Optional[dict], resp, dbg) -> None:
    """Add this call's reported tokens to the shared tally.

    Preprocessing restates every chunk in full, which made it 65% of one
    film's runtime — leaving it out of the accounting hid the bulk of both
    the time and the cost.
    """
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
    """Content tokens only.

    Punctuation is excluded on purpose: the pass is now asked to restore
    the sentence punctuation the ASR omitted, and counting a added 「。」as
    an invented word would spend the hallucination budget on exactly the
    edit we requested — while a rewrite that only reshuffles punctuation
    was never something the check needed to catch.
    """
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _PUNCT]


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


def _is_fragment(line: SubtitleLine) -> bool:
    """Too short to be a line of its own — so its timing means nothing.

    Sized in the units the script delimits (see segmenter.text_units): a
    character count called "around" a substantial line and let it through.
    """
    return is_fragment_text(line.text)


def _is_single_sentence(text: str) -> bool:
    """Does the model's version of a unit read as one sentence?

    This is the difference between the two things a merge can mean. One
    sentence is the model *asserting* that these lines are a single
    utterance — a statement about language, which a timestamp is in no
    position to refute. Several sentences is the model merely grouping
    neighbouring lines for convenience, and there a measured pause really
    is evidence about whether they belong together.
    """
    stripped = text.strip().rstrip(_SENTENCE_CHARS + "\"'」』）)")
    return bool(text.strip()) and not _SENTENCE_CHARS_RE.search(stripped)


def _gap_limit(lines: Sequence[SubtitleLine]) -> float:
    """The pause length that counts as a real pause *in this transcript*.

    Cue boundaries come from whisper, and how much silence it leaves around
    speech varies wildly with the VAD settings and the film. A constant
    tuned on one transcript vetoes almost every merge on another, and the
    veto happens before the model's judgement is even consulted. Reading
    the limit off the file's own gap distribution keeps the rule's meaning
    ("shorter than most pauses here") stable across films.

    Caveat this limit cannot escape: it is measured on the cues *our own*
    splitting produced, so finer input lowers it — which is backwards, as
    finer input is what needs merging most. Splitting one film's lines from
    725 to 1172 dropped it from 1.80s to 1.38s and cost 28 correct merges.
    ``_is_single_sentence`` is what keeps that from mattering: the merges
    it protects are exactly the ones a pause should not be deciding.
    """
    gaps = sorted(
        b.start - a.end for a, b in zip(lines, lines[1:]) if b.start > a.end
    )
    if not gaps:
        return MAX_INTERNAL_GAP
    nth = gaps[min(int(len(gaps) * GAP_PERCENTILE), len(gaps) - 1)]
    return min(max(nth, MAX_INTERNAL_GAP), MAX_INTERNAL_GAP_CEILING)


def _apply_units(
    units: Sequence[tuple[int, int, str]],
    lines: Sequence[SubtitleLine],
    settings: SubtitleSettings,
    gap_limit: float = MAX_INTERNAL_GAP,
    rejections: Optional[List[str]] = None,
) -> tuple[List[SubtitleLine], int, int]:
    """Turn validated units into cues. Returns (lines, merged, corrected).

    *rejections* collects a human-readable reason per discarded unit; the
    debug log prints them, because "the pass merged 6 lines out of 764" is
    only actionable once you can see which rule threw the rest away.
    """
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

        reason = "" if _is_faithful(text, joined) else "改写幅度过大（保真校验未通过）"
        start, end = sources[0].start, sources[-1].end
        if not reason and len(sources) > 1:
            # constraints the model could not check: it never sees timestamps.
            # A unit the model wrote as ONE sentence is a claim about the
            # language, so the pause only has to stay physically plausible;
            # a pause longer than a whole cue may be displayed for is not an
            # intra-sentence pause under any reading.
            limit = (
                max(gap_limit, settings.max_duration)
                if _is_single_sentence(text)
                else gap_limit
            )
            for a, b in zip(sources, sources[1:]):
                gap = b.start - a.end
                if gap <= limit:
                    continue
                # a pause measured against a one-or-two-character cue is not
                # a pause, it is a bad timestamp — the model reading the words
                # is the better judge there, so let its merge stand
                if _is_fragment(a) or _is_fragment(b):
                    continue
                reason = f"[{a.index}]-[{b.index}] 间隔 {gap:.1f}s > {limit:.1f}s"
                break
            # A merge that rescues a fragment may run past the width budget:
            # a wide cue is a readability cost, a stranded fragment is a
            # correctness one. The allowance is bounded so the cue still
            # fits on screen.
            width = budget
            if any(_is_fragment(s) for s in sources):
                width = int(budget * FRAGMENT_BUDGET_RATIO)
            if not reason and len(text) > width:
                reason = f"合并后 {len(text)} 字 > 每条上限 {width} 字"
            if not reason and end - start > max_duration:
                # An over-long span with a fragment on one edge is that
                # fragment's bogus timestamp, not a genuinely long cue:
                # drop the untrustworthy edge instead of the whole merge.
                if _is_fragment(sources[0]):
                    start = end - max_duration
                elif _is_fragment(sources[-1]):
                    end = start + max_duration
                else:
                    reason = f"跨度 {end - start:.1f}s > 单条上限 {max_duration:.1f}s"
        if reason:
            if rejections is not None:
                rejections.append(f"[{first}-{last}] {reason} | {text}")
            out.extend(sources)  # keep the local result for this range
            continue

        if len(sources) > 1:
            merged += len(sources) - 1
        if text != joined:
            corrected += 1
        out.append(
            SubtitleLine(
                index=0,
                start=round(start, 3),
                end=round(end, 3),
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
    glossary: str = "",
    debug=None,
    usage: Optional[dict] = None,
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
    system = build_refine_prompt(language_hint, glossary)
    chunks = _chunks(lines, llm.context_limit)
    before = open_ended_ratio(lines)
    gap_limit = _gap_limit(lines)
    dbg = debug if debug is not None and debug.enabled else None

    if dbg:
        dbg.section("转写预处理（refine）")
        dbg.kv("分块数", len(chunks))
        dbg.kv("间隔阈值 gap_limit", f"{gap_limit:.2f}s（本片自适应，下限 {MAX_INTERNAL_GAP}s）")
        dbg.kv("输入是否有句末标点", "是" if has_sentence_punctuation(lines) else "否 ⚠")
        dbg.block("system prompt", system)

    result: List[SubtitleLine] = []
    merged = corrected = failed = 0
    consecutive_failures = 0
    all_rejections: List[str] = []
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
            _tally(usage, resp, dbg)
            if dbg:
                dbg.block(f"第 {n} 块 请求", user)
                dbg.block(f"第 {n} 块 响应", reply)
            units = parse_units(reply)
            if not units or not _covers_exactly(units, chunk):
                raise ValueError("行号覆盖校验未通过")
            rejections: List[str] = []
            applied, m, c = _apply_units(units, chunk, subtitle, gap_limit, rejections)
            if dbg and rejections:
                dbg.line(f"\n第 {n} 块 被本地约束否决的合并（{len(rejections)} 条）：")
                dbg.lines("  " + r for r in rejections)
            all_rejections.extend(rejections)
            consecutive_failures = 0
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 — preprocessing is best-effort
            failed += 1
            consecutive_failures += 1
            log(f"⚠ 第 {chunk[0].index}-{chunk[-1].index} 行预处理未生效（{exc}），沿用原始分行")
            if consecutive_failures >= GIVE_UP_AFTER:
                log(f"⚠ 连续 {GIVE_UP_AFTER} 块预处理失败，跳过剩余部分，直接进入翻译")
            if dbg:
                dbg.line(f"\n⚠ 第 {n} 块失败并回退：{exc}")
            applied, m, c = list(chunk), 0, 0

        result.extend(applied)
        merged += m
        corrected += c
        progress(n / len(chunks))

    for i, line in enumerate(result, start=1):
        line.index = i

    punctuated = has_sentence_punctuation(result)
    log(
        f"预处理完成：{len(lines)} 行 → {len(result)} 条"
        f"（合并 {merged} 处、纠错 {corrected} 处"
        + (f"、{failed} 块回退" if failed else "")
        + (f"、{len(all_rejections)} 处合并被时间约束否决" if all_rejections else "")
        + f"），未完句结尾占比 {before:.0%} → {open_ended_ratio(result):.0%}"
        + ("" if punctuated else "（⚠ 全片无句末标点，该比例仅供参考）")
    )
    if dbg:
        from app.core.debuglog import fmt_cue

        dbg.line(f"\n预处理结果：{len(lines)} → {len(result)} 条，"
                 f"合并 {merged}、纠错 {corrected}、否决 {len(all_rejections)}、回退 {failed} 块")
        dbg.kv("输出是否有句末标点", "是" if punctuated else "否 ⚠")
        dbg.line("\n预处理后（送入翻译的内容）：")
        dbg.lines(fmt_cue(l.index, l.start, l.end, l.text) for l in result)
    return result

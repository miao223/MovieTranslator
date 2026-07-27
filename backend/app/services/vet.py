"""Vetting of what the second ASR pass recovered.

Why this exists
---------------
The second pass (``asr.second_pass``) re-transcribes with the VAD switched
off precisely where the VAD said there was no speech. That is where whisper
is at its worst: it was trained on subtitle files, so over non-speech it
emits the things subtitle files contain — closing lines ("thanks for
watching", "see you next time"), station announcements, encyclopedia
entries, cooking instructions. They are grammatical, they compress
normally, and no confidence gate sees anything wrong with them.

One film's 154 recovered segments broke down as: 36 fabricated (135s), 47
non-lexical (94s), 71 genuine lines of dialogue (157s) the first pass had
missed entirely. A phrase blacklist caught 11 of the 36 with zero false
positives, but the rest were open-ended content no list can enumerate.

So the question has to change from "does this look like boilerplate" to
"does this belong to THIS film" — and only a model holding the film's own
transcript can answer that.

The property that makes it safe to hand this to an LLM: **the worst case
has a floor**. Vetting can only delete what the second pass added; the
first pass is never touched. If every judgement were wrong the result is
merely the film as it was before the second pass existed. Nothing else in
this pipeline has that guarantee, which is why a failed request here drops
the recovered lines rather than keeping them — the fallback direction is
the one that preserves the floor.

The model never sees timestamps (project-wide rule). It does not need to:
the lines under review are rendered *interleaved into the transcript in
time order*, so their position carries the timing for free — and that
context is what makes the judgement easy. Asked in isolation, "JR東日本
E233系電車" is just a noun phrase; sitting between two lines about a
haunted apartment, it is obviously not from this film.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.models.schemas import LLMSettings, NetworkSettings
from app.services.asr import Segment
from app.services.translator import chat_completion, estimate_tokens, make_openai_client

LogFn = Callable[[str], None]

# `[R12] 保留` / `[R12] 丢弃 电车报站` — tolerant of the usual model noise
_VERDICT_RE = re.compile(
    r"^\s*[*\-•]?\s*\[?\s*R\s*(\d+)\s*\]?\s*[:：]?\s*(保留|丢弃|keep|drop)\s*[:：,，]?\s*(.*?)\s*$",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$")
_DROP_WORDS = {"丢弃", "drop"}

# One request restates the transcript but answers with one short verdict per
# reviewed line, so the binding constraint is the input side only — chunks
# can be far larger than the refine pass's. A two-hour film's whole
# transcript measured 4,900 tokens, i.e. a single call.
VET_CHUNK_TOKENS = 12_000
# cost of the "*[R123] " / "[123] " prefix each line is rendered with.
# Guessing high here is not harmless: it splits a film that would have fit
# in one request, and the second chunk then judges its lines with only the
# tail of the transcript for context — which is the whole point of the pass.
MARKER_TOKENS = 3
# consecutive chunk failures after which we stop calling: an unreachable
# endpoint must not cost one timeout per chunk
GIVE_UP_AFTER = 2


def build_vet_prompt(
    language_hint: str = "", synopsis: str = "", mark_lyrics: bool = False
) -> str:
    lang = f"（原文语言：{language_hint}）" if language_hint else ""
    parts = [
        f"你是字幕校对员{lang}。下面是一部影片的语音识别转写，按时间顺序排列。",
        "以 `*[R编号]` 开头的行是【补充识别】结果：它们来自语音活动检测判定为"
        "「无语音」的片段，是关闭检测后重新识别得到的，可靠性明显低于其他行。"
        "没有标记的 `[编号]` 行是已确认属于本片的识别结果，请把它们当作判断依据。",
        "补充识别有两类已知问题：",
        "1. 语音识别模型是用字幕文件训练的，在没有人声的段落上会吐出与本片毫无关系、"
        "但语句完全通顺的内容——视频片尾语（感谢观看、记得订阅、下期再见）、"
        "车站广播、百科条目、烹饪教程等等。它们无法靠语法或通顺程度识别，"
        "只能靠「与这部影片无关」识别；",
        "2. 把环境音、音乐、呼吸声当成人声，产出没有意义的填充音。",
        "",
        "请逐条判断每个 `*[R编号]` 行是否应当留在这部影片的字幕里：",
        "- 保留：与上下文连贯的对白；有明确含义的应答、呼唤、惊叫"
        "（例如「はい」「うん」「ただいま」、喊人名、尖叫）；",
        "- 丢弃：与本片题材、剧情、上下文都对不上的内容，哪怕它读起来很正常；",
        "- 丢弃：没有意义的填充音（孤立的「ん」「あ」「うー」「えー」之类）；",
    ]
    if mark_lyrics:
        # The lyrics pass (services/lyrics.py) marks songs later, over the
        # whole film. Dropping them here would take that decision away from
        # it — and it would take it inconsistently, since only the second
        # pass's lines ever reach this prompt.
        parts += [
            "- 丢弃：**不含任何词句的**音效与音乐标记（`♪♪`、`【ドアの音】`、`(laughs)` 等）；",
            "- 保留：有实际词句的歌词（例如 `♪ You're the only one I'll ever love`）——"
            "影片里放出来的歌属于本片，后续会另行标注，此处不要丢弃。",
        ]
    else:
        parts.append(
            "- 丢弃：字幕文件式的音效与音乐标注（【…】、♪、(laughs) 等），以及"
            "影片配乐的歌词——它们不是台词。"
        )
    parts += [
        "",
        "判断标准是「这句话是否属于这部影片」，不是「这句话是否通顺」。",
        "拿不准时：读起来像本片人物之间的对话就保留；"
        "像是从别处（视频平台、广播、教程、说明书）搬过来的成句内容就丢弃。",
    ]
    if synopsis.strip():
        parts += ["", "本片剧情简介（供判断题材用）：\n" + synopsis.strip()]
    parts += [
        "",
        "输出格式：每个 `*[R编号]` 输出一行，写 `[R编号] 保留` 或 `[R编号] 丢弃 简短理由`。",
        "必须且只能覆盖全部 `*[R编号]`，不得新增、遗漏或改写编号。",
        "不要输出没有标记的行，不要输出解释、代码块标记或任何多余内容。",
        "示例：",
        "[R7] 保留",
        "[R8] 丢弃 车站广播，与本片无关",
    ]
    return "\n".join(parts)


def render_transcript(
    segments: Sequence[Segment], ids: Dict[int, int], first_plain: int = 1
) -> str:
    """The transcript in time order, with the lines under review marked.

    *ids* maps ``id(segment)`` to its review number. Position in this text
    is the only timing information the model gets, and the only timing
    information it needs. *first_plain* keeps the unmarked lines' numbering
    continuous across chunks.
    """
    out: List[str] = []
    plain = first_plain - 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        review = ids.get(id(seg))
        if review is None:
            plain += 1
            out.append(f"[{plain}] {text}")
        else:
            out.append(f"*[R{review}] {text}")
    return "\n".join(out)


def parse_verdicts(reply: str) -> Dict[int, Tuple[bool, str]]:
    """Parse ``[R12] 保留`` / ``[R12] 丢弃 reason`` into {n: (keep, reason)}."""
    result: Dict[int, Tuple[bool, str]] = {}
    for raw in reply.splitlines():
        if _FENCE_RE.match(raw):
            continue
        m = _VERDICT_RE.match(raw)
        if not m:
            continue
        keep = m.group(2).lower() not in _DROP_WORDS
        result[int(m.group(1))] = (keep, m.group(3).strip())
    return result


def _covers_exactly(verdicts: Dict[int, Tuple[bool, str]], expected: Sequence[int]) -> bool:
    """A verdict for every reviewed line, and for nothing else."""
    return bool(expected) and set(verdicts) == set(expected)


def _chunks(segments: Sequence[Segment], context_limit: int) -> List[List[Segment]]:
    """Split the timeline so one request fits the model's context.

    Chunks stay contiguous in time, so every reviewed line keeps the
    neighbours that make it judgeable.
    """
    budget = min(max(context_limit // 3, 2_000), VET_CHUNK_TOKENS)
    out: List[List[Segment]] = []
    current: List[Segment] = []
    used = 0
    for seg in segments:
        cost = estimate_tokens(seg.text) + MARKER_TOKENS
        if current and used + cost > budget:
            out.append(current)
            current, used = [], 0
        current.append(seg)
        used += cost
    if current:
        out.append(current)
    return out


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


def _duration(segments: Sequence[Segment]) -> float:
    return sum(s.end - s.start for s in segments)


def vet_recovered(
    segments: List[Segment],
    llm: LLMSettings,
    language_hint: str = "",
    synopsis: str = "",
    mark_lyrics: bool = False,
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    client=None,  # injectable for tests
    network: Optional[NetworkSettings] = None,
    debug=None,
    usage: Optional[dict] = None,
) -> List[Segment]:
    """Drop the second pass's output that does not belong to this film.

    Segments from the first pass are returned untouched, always. Only
    ``recovered`` ones can be removed, and a chunk whose verdicts cannot be
    obtained or verified loses all of its recovered segments — see the
    module docstring for why that is the safe direction.
    """
    log = log or (lambda _m: None)
    review = [s for s in segments if s.recovered]
    if not review:
        return segments

    ids = {id(s): n for n, s in enumerate(review, start=1)}
    client = client if client is not None else make_openai_client(llm, network)
    system = build_vet_prompt(language_hint, synopsis, mark_lyrics)
    chunks = _chunks(segments, llm.context_limit)
    dbg = debug if debug is not None and debug.enabled else None

    if dbg:
        dbg.section("二次识别复核（LLM 判断是否属于本片）")
        dbg.kv("送审", f"{len(review)} 段 / {_duration(review):.0f}s")
        dbg.kv("分块数", len(chunks))
        dbg.block("system prompt", system)

    dropped: List[Tuple[Segment, str]] = []
    kept: List[Segment] = []
    failed_chunks = 0
    consecutive_failures = 0
    no_thinking = llm.disable_thinking
    plain_seen = 0

    for n, chunk in enumerate(chunks, start=1):
        if should_cancel and should_cancel():
            raise InterruptedError("cancelled")
        pending = [s for s in chunk if s.recovered]
        first_plain = plain_seen + 1
        plain_seen += sum(1 for s in chunk if not s.recovered and s.text.strip())
        if not pending:
            continue  # nothing to ask about: no request, no tokens
        expected = [ids[id(s)] for s in pending]

        if consecutive_failures >= GIVE_UP_AFTER:
            dropped += [(s, "复核未能进行") for s in pending]
            continue

        user = (
            "以下是影片的转写，请判断其中带 `*[R编号]` 标记的行：\n"
            + render_transcript(chunk, ids, first_plain)
        )
        if dbg:
            dbg.block(f"第 {n} 块 请求", user)

        verdicts: Optional[Dict[int, Tuple[bool, str]]] = None
        last_error: Optional[Exception] = None
        # One retry, because a network blip must not silently eat real
        # dialogue. At temperature 0 a repeat of the same request would just
        # repeat the same bad answer, so a format failure gets told what
        # was wrong with it.
        for attempt in (1, 2):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            if attempt == 2 and isinstance(last_error, ValueError):
                messages.append({
                    "role": "user",
                    "content": "上一次的回答没有对每个编号各给恰好一行判定。"
                               "请重新输出，必须且只能覆盖以下编号，每个一行：\n"
                               + " ".join(f"R{i}" for i in expected),
                })
            try:
                resp, no_thinking = chat_completion(
                    client,
                    model=llm.model,
                    messages=messages,
                    temperature=0,  # mechanical judgement: no creativity wanted
                    no_thinking=no_thinking,
                )
                reply = resp.choices[0].message.content or ""
                _tally(usage, resp, dbg)
                if dbg:
                    dbg.block(f"第 {n} 块 响应（第 {attempt} 次）", reply)
                parsed = parse_verdicts(reply)
                if not _covers_exactly(parsed, expected):
                    raise ValueError(
                        f"判定覆盖校验未通过（送审 {len(expected)} 条，"
                        f"收到 {len(parsed)} 条有效判定）"
                    )
                verdicts = parsed
                break
            except (InterruptedError, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001 — vetting must never fail a job
                last_error = exc
                if dbg:
                    dbg.line(f"\n⚠ 第 {n} 块第 {attempt} 次失败：{exc}")

        if verdicts is None:
            failed_chunks += 1
            consecutive_failures += 1
            dropped += [(s, "复核失败") for s in pending]
            log(
                f"⚠ 第 {n} 块复核失败（{last_error}），"
                f"该块二次识别的 {len(pending)} 段 / {_duration(pending):.0f}s 已丢弃"
            )
            if consecutive_failures >= GIVE_UP_AFTER:
                log(f"⚠ 连续 {GIVE_UP_AFTER} 块复核失败，剩余二次识别内容一律丢弃")
            continue

        consecutive_failures = 0
        for seg in pending:
            keep, reason = verdicts[ids[id(seg)]]
            if keep:
                kept.append(seg)
            else:
                dropped.append((seg, reason or "与本片无关"))

    drop_ids = {id(s) for s, _ in dropped}
    result = [s for s in segments if id(s) not in drop_ids]

    log(
        f"二次识别复核：送审 {len(review)} 段 / {_duration(review):.0f}s，"
        f"保留 {len(kept)} 段 / {_duration(kept):.0f}s，"
        f"丢弃 {len(dropped)} 段 / {_duration([s for s, _ in dropped]):.0f}s"
        + (f"（其中 {failed_chunks} 块因复核失败整块丢弃）" if failed_chunks else "")
    )
    if dbg:
        from app.core.debuglog import fmt_time

        dbg.line(
            f"\n复核结果：保留 {len(kept)} 段 / {_duration(kept):.0f}s，"
            f"丢弃 {len(dropped)} 段 / {_duration([s for s, _ in dropped]):.0f}s"
        )
        dbg.line("\n逐条判定：")
        verdict_by_id = {id(s): f"丢弃 {r}" for s, r in dropped}
        dbg.lines(
            f"  [R{ids[id(s)]}] {fmt_time(s.start)} "
            f"{verdict_by_id.get(id(s), '保留')} | {s.text}"
            for s in review
        )
    return result

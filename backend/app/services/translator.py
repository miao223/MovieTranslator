"""Global-context subtitle translation over an OpenAI-compatible API.

Protocol (the core idea of this project):

1. The full film transcript is sent ONCE as numbered lines ``[n] text`` —
   no timestamps, to save tokens. The model first produces a glossary of
   names/terms to keep the whole translation self-consistent.
2. Translations are then requested batch by batch ("output lines i..j")
   inside the SAME conversation, so the model always holds the full film
   context while sidestepping per-response output-token limits.
3. Every batch is parsed and validated by line number; missing lines are
   re-requested, with a per-line retry as last resort.
4. If the transcript would overflow the model's context window, we fall
   back to sliding-window chunking: each chunk carries the glossary plus
   the tail of the previous chunk (original + translation) for continuity.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from app.models.schemas import (
    LLMSettings,
    NetworkSettings,
    PromptSettings,
    SubtitleLine,
)

LogFn = Callable[[str], None]
ProgressFn = Callable[[float], None]  # 0..1


def make_openai_client(settings: LLMSettings, network: Optional[NetworkSettings] = None):
    """OpenAI-compatible client, routed through the proxy when enabled."""
    import httpx
    from openai import OpenAI

    http_client = None
    if network and network.llm_via_proxy and network.proxy_url.strip():
        http_client = httpx.Client(proxy=network.proxy_url.strip())
    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key or "EMPTY",
        http_client=http_client,
    )


def build_system_prompt(
    prompts: PromptSettings,
    target_language: str,
    synopsis: str = "",
    max_line_chars: int = 42,
) -> str:
    """Assemble the system prompt from prompt settings.

    A non-empty custom_system_prompt overrides everything; it may contain
    {target_language} / {synopsis} placeholders.
    """
    if prompts.custom_system_prompt.strip():
        return (
            prompts.custom_system_prompt
            .replace("{target_language}", target_language)
            .replace("{synopsis}", synopsis.strip())
        )

    rules = [
        "严格保持行号对应，每行输出格式为 `[行号] 译文`，一行原文对应一行译文，"
        "不得合并、拆分或跳过任何行。",
        # the shift bug: a line holding only a sentence tail has nothing to
        # say on its own, and the model "solves" it by pulling the next
        # line's content forward — offsetting every line after it
        "每行译文只能翻译该行原文本身的内容。若某行原文只是上一句话的残尾"
        "（例如只有一两个词），译文也只能对应这几个词；"
        "严禁把下一行的内容提前翻译到本行，也严禁把本行内容推迟到下一行。"
        "宁可单行译文读起来不完整，也不得跨行搬运内容。\n"
        "   正确：`[7] 我没想到` / `[8] 这一点`；"
        "错误：`[7] 我没想到这一点` / `[8]（挪用了第 9 行的内容）`",
        "全片人名、地名、术语的译法必须前后一致。",
    ]
    if prompts.tone.strip():
        rules.append(prompts.tone.strip())
    if prompts.fix_asr_errors:
        rules.append(
            "字幕文本来自语音识别，可能存在同音/近音词、断词错误。"
            "若某处明显不通顺，请结合上下文推断说话人的本意后翻译，"
            "不要逐字硬译识别错误的词。"
        )
    if prompts.link_fragments:
        rules.append(
            "相邻行可能是同一句话被拆成的两半。理解时要结合上下文，"
            "但输出时不得因此调整内容的归属：该行有多少内容就译多少内容，"
            "跨行的语序差异宁可保留，也不得把内容挪到别的行去凑通顺。"
        )
    if prompts.normalize_loanwords:
        rules.append(
            "源语言中的音译外来词（如日语片假名的人名、地名、品牌、术语），"
            "同一个词全片必须采用统一的译法，并收入译名对照表；"
            "语音识别可能把同一个词写得不一致，请按发音判断是否为同一词并统一处理。"
            f"译法必须是{target_language}（人名地名用通行译名，无通行译名的按发音翻译），"
            "不得用罗马音、拼音或原文字母拼写充当译文；"
            f"仅当该词在{target_language}中习惯直接使用原文时（如品牌名、缩写）才可保留。"
        )
    if prompts.limit_length:
        rules.append(
            f"译文是屏幕字幕，长度尽量接近原文，单行不超过 {max_line_chars} 个字符，"
            "宁可精炼不可冗长。"
        )
    rules.append("不要输出任何解释、注释或多余内容。")

    parts = [
        f"你是一名资深电影字幕翻译。请把用户提供的字幕逐行翻译成{target_language}。",
        "要求：",
    ]
    parts.extend(f"{i}. {rule}" for i, rule in enumerate(rules, start=1))
    if prompts.glossary.strip():
        parts.append("已知译名对照表（必须优先遵守）：\n" + prompts.glossary.strip())
    if synopsis.strip():
        parts.append(f"剧情简介（供理解上下文）：{synopsis.strip()}")
    if prompts.extra.strip():
        parts.append(prompts.extra.strip())
    return "\n".join(parts)

_LINE_RE = re.compile(r"^\s*\[?(\d+)\]?[.、:：]?\s*(.*\S)\s*$")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$")

MAX_BATCH_RETRIES = 2


class TranslationError(Exception):
    pass


def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK ≈ 1 token/char, other text ≈ 1 token/4 chars."""
    cjk = sum(1 for c in text if "⺀" <= c <= "鿿" or "　" <= c <= "ヿ")
    other = len(text) - cjk
    return cjk + other // 4 + 1


def _numbered(lines: List[SubtitleLine]) -> str:
    return "\n".join(f"[{line.index}] {line.text}" for line in lines)


def parse_translations(text: str) -> dict[int, str]:
    """Parse ``[n] translation`` lines from a model response.

    Tolerates code fences and continuation lines (appended to the previous
    numbered line).
    """
    result: dict[int, str] = {}
    current: Optional[int] = None
    for raw in text.splitlines():
        if _FENCE_RE.match(raw):
            continue
        m = _LINE_RE.match(raw)
        if m:
            current = int(m.group(1))
            result[current] = m.group(2).strip()
        elif current is not None and raw.strip():
            result[current] += " " + raw.strip()
    return result


class Translator:
    def __init__(
        self,
        settings: LLMSettings,
        target_language: str,
        synopsis: str = "",
        log: Optional[LogFn] = None,
        progress: Optional[ProgressFn] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        client=None,  # injectable for tests
        prompts: Optional[PromptSettings] = None,
        max_line_chars: int = 42,
        network: Optional[NetworkSettings] = None,
    ):
        self.settings = settings
        self.target_language = target_language
        self.synopsis = synopsis
        self.prompts = prompts or PromptSettings()
        self.max_line_chars = max_line_chars
        self.log = log or (lambda _msg: None)
        self.progress = progress or (lambda _p: None)
        self.should_cancel = should_cancel or (lambda: False)
        self.client = client if client is not None else make_openai_client(settings, network)

    # ------------------------------------------------------------ helpers

    def _chat(self, messages: List[dict]) -> str:
        if self.should_cancel():
            raise InterruptedError("cancelled")
        resp = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
        )
        content = resp.choices[0].message.content or ""
        if not content.strip():
            raise TranslationError("LLM 返回了空响应")
        return content

    def _system_prompt(self) -> str:
        return build_system_prompt(
            self.prompts, self.target_language, self.synopsis, self.max_line_chars
        )

    # ------------------------------------------------------------ public

    def translate(self, lines: List[SubtitleLine]) -> None:
        """Fill ``line.translation`` for every line, in place."""
        if not lines:
            return
        full_text = _numbered(lines)
        est = estimate_tokens(full_text)
        # conversation will hold: input once + all translations + overhead ≈ 3x
        if est * 3 <= self.settings.context_limit:
            self.log(f"全局上下文模式（约 {est} tokens）")
            self._translate_global(lines)
        else:
            self.log(
                f"文本约 {est} tokens，超出模型上下文预算，"
                "切换为滑动窗口分块模式"
            )
            self._translate_chunked(lines)

    # ------------------------------------------------------- global mode

    def _translate_global(self, lines: List[SubtitleLine]) -> None:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    "以下是全片字幕（行号+原文）。请先通读全文，"
                    "输出一份主要人名/术语的译名对照表"
                    f"（每行一条，格式：原文 → {self.target_language}译名）。"
                    f"注意：右侧必须是{self.target_language}译名"
                    "（人名地名用通行译名，无通行译名的按发音翻译成"
                    f"{self.target_language}），不得填写罗马音、拼音或原文字母；"
                    "之后我会分批向你索取各行译文。\n\n" + _numbered(lines)
                ),
            },
        ]
        self.log("正在通读全文并生成术语/译名对照表…（长片首个响应可能需要 1-2 分钟）")
        glossary = self._chat(messages)
        messages.append({"role": "assistant", "content": glossary})
        self.log("术语表已生成：\n" + glossary.strip()[:800])

        batch = self.settings.batch_size
        done = 0
        for i in range(0, len(lines), batch):
            chunk = lines[i : i + batch]
            first, last = chunk[0].index, chunk[-1].index
            self.log(f"请求第 {first}-{last} 行译文…")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"请输出第 {first} 行到第 {last} 行的译文，"
                        "每行格式 `[行号] 译文`，逐行对应，勿遗漏。"
                    ),
                }
            )
            reply = self._request_batch(messages, chunk)
            messages.append({"role": "assistant", "content": reply})
            done += len(chunk)
            self.progress(done / len(lines))
            self.log(f"已翻译 {done}/{len(lines)} 行")

    def _request_batch(self, messages: List[dict], chunk: List[SubtitleLine]) -> str:
        """Ask for one batch, validate line coverage, re-request what's missing.

        Returns the assistant text to keep in the conversation history.
        """
        reply = self._chat(messages)
        parsed = parse_translations(reply)
        by_index = {line.index: line for line in chunk}
        for idx, line in by_index.items():
            if idx in parsed:
                line.translation = parsed[idx]

        missing = [idx for idx in by_index if not by_index[idx].translation]
        retries = 0
        while missing and retries < MAX_BATCH_RETRIES:
            retries += 1
            self.log(f"缺失 {len(missing)} 行译文，补翻中（第 {retries} 次）…")
            ask = (
                "以下行的译文缺失，请补充输出，每行格式 `[行号] 译文`：\n"
                + "\n".join(f"[{i}] {by_index[i].text}" for i in missing)
            )
            fix = self._chat(
                messages + [{"role": "assistant", "content": reply},
                            {"role": "user", "content": ask}]
            )
            parsed = parse_translations(fix)
            for idx in list(missing):
                if idx in parsed and parsed[idx]:
                    by_index[idx].translation = parsed[idx]
            missing = [idx for idx in by_index if not by_index[idx].translation]

        if missing:
            raise TranslationError(
                f"多次重试后仍缺失第 {missing[:10]} 等 {len(missing)} 行译文"
            )
        return reply

    # ------------------------------------------------------ chunked mode

    def _translate_chunked(self, lines: List[SubtitleLine]) -> None:
        glossary = self._build_glossary_from_sample(lines)
        batch = self.settings.batch_size
        tail_context = 10  # lines of previous chunk carried for continuity
        done = 0
        prev: List[SubtitleLine] = []
        for i in range(0, len(lines), batch):
            chunk = lines[i : i + batch]
            self.log(f"请求第 {chunk[0].index}-{chunk[-1].index} 行译文（分块模式）…")
            context_block = ""
            if prev:
                tail = prev[-tail_context:]
                context_block = "上一段结尾（原文 → 已定稿译文，保持衔接与一致）：\n" + "\n".join(
                    f"[{line.index}] {line.text} → {line.translation}" for line in tail
                ) + "\n\n"
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"译名对照表（必须遵守）：\n{glossary}\n\n"
                        + context_block
                        + "请翻译以下字幕，每行格式 `[行号] 译文`，逐行对应，勿遗漏：\n"
                        + _numbered(chunk)
                    ),
                },
            ]
            self._request_batch(messages, chunk)
            prev = chunk
            done += len(chunk)
            self.progress(done / len(lines))
            self.log(f"已翻译 {done}/{len(lines)} 行")

    def _build_glossary_from_sample(self, lines: List[SubtitleLine]) -> str:
        """Sample lines evenly across the film to build a glossary that fits."""
        budget = max(self.settings.context_limit // 3, 2_000)
        sample: List[str] = []
        used = 0
        step = max(len(lines) * estimate_tokens(_numbered(lines[:50])) // (50 * budget), 1)
        for line in lines[::step]:
            t = estimate_tokens(line.text)
            if used + t > budget:
                break
            sample.append(line.text)
            used += t
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    "以下是全片字幕的抽样内容。请输出主要人名/地名/术语的译名对照表"
                    f"（每行一条，格式：原文 → {self.target_language}译名）。"
                    f"注意：右侧必须是{self.target_language}译名"
                    "（人名地名用通行译名，无通行译名的按发音翻译成"
                    f"{self.target_language}），不得填写罗马音、拼音或原文字母。"
                    "不要输出其它内容。\n\n"
                    + "\n".join(sample)
                ),
            },
        ]
        glossary = self._chat(messages)
        self.log("术语表已生成（分块模式）：\n" + glossary.strip()[:800])
        return glossary

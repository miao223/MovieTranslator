"""Series mode: one glossary shared by every video of a batch.

A batch of episodes is translated as N independent films, so each one asks
the model for its own glossary and throws it away (`translator`); episode 1
settles on 「佐藤健一」 and episode 5 on 「佐藤贤一」. Consistent inside a
file, inconsistent across the season.

What this module holds is the season's accumulated 原文 → 译名 table. The
pipeline merges it into `PromptSettings.glossary`, which already reaches
both LLM stages that can act on it — the translator's system prompt
(「已知译名对照表（必须优先遵守）」) and the transcript preprocessing pass.
No new prompt plumbing is needed, and an episode still produces its own
glossary as before: series mode only adds what it must agree with.

Two rules decide every conflict:

- **first one in wins.** Stability is the entire point, so a later episode
  may not rewrite a rendering that earlier episodes already shipped. What
  it wanted instead is recorded — that list is exactly what a human should
  check.
- **the user's own glossary outranks the model's**, always. Terms the user
  wrote in settings are never overwritten and never enter this table.

The table lives in memory for the life of the batch. Nothing is written to
disk unless the user presses 「保存到设置的术语表」.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# An episode contributes 10-40 terms. The cap keeps a 26-episode season from
# crowding out the prompt; past it the earliest — most established — win.
MAX_TERMS = 300
# a term is a name or a phrase; anything longer is the model narrating
MAX_TERM_CHARS = 40

# Only ``→`` (and its ascii spelling) separates a term from its rendering.
# The prompt asks for exactly that, and accepting ``：`` as well would turn
# an instruction like 「注意：右侧必须是中文译名」 into a glossary entry.
_ARROW = re.compile(r"\s*(?:→|⇒|->|=>)\s*")
_BULLET = re.compile(r"^(?:[-*+•·]|\d+[.)、])\s*")
_EDGE = " \t*`_“”\"'（）()"

# header rows of a table the model drew, not terms
_TEMPLATE_SOURCES = {"原文", "原词", "术语", "原文/术语", "source", "term"}
_TEMPLATE_TARGETS = {"译名", "译文", "翻译", "translation"}


def parse_terms(text: str) -> List[Tuple[str, str]]:
    """Pull ``原文 → 译名`` pairs out of whatever the model replied with.

    Tolerates the shapes models actually produce — bullets, numbering,
    markdown tables, fenced blocks — and ignores every line that does not
    carry an arrow, which is what keeps prose out of the table.
    """
    terms: List[Tuple[str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        line = _BULLET.sub("", line.strip("|").strip())
        parts = _ARROW.split(line, maxsplit=1)
        if len(parts) != 2:
            continue
        source = parts[0].strip(_EDGE)
        target = parts[1].strip("|").strip(_EDGE)
        if not source or not target:
            continue
        if len(source) > MAX_TERM_CHARS or len(target) > MAX_TERM_CHARS:
            continue
        if source.lower() in _TEMPLATE_SOURCES and target.lower() in _TEMPLATE_TARGETS:
            continue
        terms.append((source, target))
    return terms


def render_terms(terms) -> str:
    return "\n".join(f"{source} → {target}" for source, target in terms)


@dataclass
class SeriesGlossary:
    """The accumulated table of one batch. Safe to touch from any thread."""

    id: str
    terms: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    # rewrites that were refused, newest last — the human review list
    conflicts: List[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __len__(self) -> int:
        with self.lock:
            return len(self.terms)

    def learn(self, model_reply: str, user_glossary: str = "") -> Tuple[int, List[str]]:
        """Take in one episode's glossary. Returns (newly added, conflicts)."""
        protected = {source for source, _ in parse_terms(user_glossary)}
        added = 0
        clashes: List[str] = []
        with self.lock:
            for source, target in parse_terms(model_reply):
                if source in protected:
                    continue  # the user said so; the model does not get a vote
                settled = self.terms.get(source)
                if settled is None:
                    if len(self.terms) >= MAX_TERMS:
                        continue
                    self.terms[source] = target
                    added += 1
                elif settled != target:
                    clashes.append(f"{source}: 沿用「{settled}」，本集模型给出「{target}」")
            self.conflicts.extend(clashes)
        return added, clashes

    def render(self) -> str:
        """The accumulated terms alone, one ``原文 → 译名`` per line."""
        with self.lock:
            return render_terms(self.terms.items())

    def merged(self, user_glossary: str = "") -> str:
        """What the job should use as its glossary: the user's table first.

        The user's text is carried over verbatim rather than reformatted —
        it is theirs, it may hold notes, and it is the authority the model
        is told to obey first.
        """
        user_text = (user_glossary or "").strip()
        protected = {source for source, _ in parse_terms(user_text)}
        with self.lock:
            learned = render_terms(
                (source, target)
                for source, target in self.terms.items()
                if source not in protected
            )
        if not learned:
            return user_text
        return f"{user_text}\n{learned}" if user_text else learned


# ------------------------------------------------------------------ store

_store: Dict[str, SeriesGlossary] = {}
_store_lock = threading.Lock()


def create(series_id: str) -> SeriesGlossary:
    glossary = SeriesGlossary(id=series_id)
    with _store_lock:
        _store[series_id] = glossary
    return glossary


def get(series_id: str) -> Optional[SeriesGlossary]:
    if not series_id:
        return None
    with _store_lock:
        return _store.get(series_id)


def for_job(series_id: str, user_glossary: str) -> Tuple[Optional[SeriesGlossary], str]:
    """What one job should translate with: (shared table or None, glossary).

    The whole of series mode's effect on a job is this pair, so it lives in
    one place the tests can reach — the pipeline only wires it up.
    """
    shared = get(series_id)
    return shared, shared.merged(user_glossary) if shared else user_glossary

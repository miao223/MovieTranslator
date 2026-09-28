"""A persistent queue whose entries remember the settings they were added under.

The problem this solves is not "run jobs one after another" — that already
happened, because every job thread blocks on pipeline._run_slot. It is that
a job read the settings when it *started*, not when it was submitted, so a
job still waiting its turn silently picked up whatever you changed in the
meantime. Editing settings halfway through a season changed the parameters
from that episode on, while series mode's glossary assumes they all match.

So an entry freezes the settings at the moment it is enqueued, and keeps
them for its whole run. Enqueue A, change anything, enqueue B: A runs the
old way, B the new way.

Two deliberate holes in "freezes everything":

* **API keys are read live** (pipeline.LIVE_KEY_FIELDS). A key that expired
  or was rotated should repair the queue, not break it. They are blanked
  before the snapshot is written, so queue.json cannot leak a credential.
* **work_dir, model_cache_dir, server and mcp ride along but do nothing.**
  cache._base_dir() and asr.get_model_cache_dir() call load_settings()
  themselves and never see the snapshot, and nothing reads server/mcp
  during a job. They are stored anyway so joblog._settings_lines() can
  render a snapshot the same way it renders live settings. Do not "fix"
  this by removing them.

The file lives beside settings.json rather than in the cache directory,
because the cache is wiped on every startup (main.clear_cache) and
work_dir may point at a removable disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import ValidationError

from app.core import config
from app.services import cpuyield, memguard
from app.models.schemas import (
    AppSettings, AudioRequest, DiscRequest, EncodeRequest, JobRequest, QueueEntry,
)

# Terminal states, the same three the rest of the app uses (batch.TERMINAL).
TERMINAL = {"done", "failed", "cancelled"}

# Finished entries are history; they are kept, but not forever. Mirrors
# joblog.KEEP_LOGS — the newest N by finish time, queued and running
# entries never pruned.
KEEP_FINISHED = 50
# A cap so that a client in a loop cannot grow the file without bound.
MAX_QUEUED = 500

INTERRUPTED_NOTE = "上次运行被中断（程序关闭或崩溃），已重新排队，将从头开始"


def queue_path() -> Path:
    """Where the queue lives: next to settings.json.

    Derived from config.settings_path() rather than calling user_config_dir
    again, so that tests/conftest.py's settings_file fixture — which
    redirects that one function — carries the queue into tmp_path with it.
    Otherwise every test run would edit the developer's real queue.
    """
    return config.settings_path().parent / "queue.json"


# ----------------------------------------------------------------- snapshot

def snapshot() -> AppSettings:
    """The current settings, frozen, with every credential removed."""
    # load_settings returns a process-wide shared instance; copy before
    # touching it (CLAUDE.md), and copy deeply because the keys live in a
    # nested model.
    snap = config.load_settings().model_copy(deep=True)
    snap.llm.api_key = ""
    snap.llm.vision_api_key = ""
    snap.llm.audio_api_key = ""
    snap.server.access_token = ""
    return snap


# Parts of AppSettings that do not change what comes out of a job. Excluded
# from the fingerprint so that toggling LAN access or moving the cache does
# not relabel every later entry as a different generation of settings.
# `encode` is only what the 压制 forms start from: every encode entry carries
# its own options, so changing the defaults changes no queued work.
_NOT_OUTPUT_AFFECTING = ("server", "mcp", "work_dir", "model_cache_dir", "encode")


def settings_hash(settings: Optional[AppSettings]) -> str:
    """A short fingerprint of the output-affecting settings."""
    if settings is None:
        return ""
    data = settings.model_dump()
    for key in _NOT_OUTPUT_AFFECTING:
        data.pop(key, None)
    for key in ("api_key", "vision_api_key", "audio_api_key"):
        data.get("llm", {}).pop(key, None)
    # price and latency, not output: switching Flex must not turn every
    # queued entry into "another generation of settings" or orphan a
    # checkpoint (pipeline's checkpoint key is this same hash)
    data.get("asr", {}).pop("api_flex", None)
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def describe_entry(entry: QueueEntry) -> str:
    """The summary line for any kind of entry."""
    if entry.kind == "disc":
        line = describe_disc(entry.disc)
        if entry.then_encode is not None:
            line += f" · 完成后压制：{describe_template(entry.then_encode)}"
            if entry.then is not None:
                line += f" · 再做字幕：{describe(entry.then)}"
        elif entry.then is not None:
            line += f" · 完成后做字幕：{describe(entry.then)}"
        return line
    if entry.kind == "encode":
        line = describe_encode(entry.encode)
        if entry.then is not None:
            line += f" · 完成后做字幕：{describe(entry.then)}"
        if entry.origin:
            line += " · 原盘封装后自动加入"
        return line
    if entry.kind == "audio":
        return describe_audio(entry.audio)
    line = describe(entry.request)
    if entry.origin:
        line += (" · 压制后自动加入" if entry.origin_kind == "encode"
                 else " · 原盘封装后自动加入")
    return line


def describe_encode(request: Optional[EncodeRequest]) -> str:
    """"压制 · H.265 10bit（x265） · CRF 22 · medium · 无损音轨→E-AC-3 · …"."""
    if request is None:
        return ""
    from app.services.encode import describe_options

    parts = ["压制", describe_options(request.options)]
    if request.replace_source:
        parts.append("完成后替换无损版")
    elif request.output_mode == "custom":
        parts.append(f"输出到 {request.output_dir}")
    else:
        parts.append("放在原文件旁边")
    return " · ".join(parts)


def describe_audio(request: Optional[AudioRequest]) -> str:
    """"提取音频 · Opus 24 kbps · 16 kHz 单声道 · 音轨 #2 · 放在原文件旁边"."""
    if request is None:
        return ""
    from app.services.audioextract import describe as describe_options

    parts = ["提取音频", describe_options(request.options)]
    if request.track is not None:
        parts.append(f"音轨 #{request.track}")
    elif request.language:
        parts.append(f"{request.language} 音轨")
    parts.append(f"输出到 {request.output_dir}" if request.output_mode == "custom"
                 else "放在原文件旁边")
    return " · ".join(parts)


def describe_template(request: EncodeRequest) -> str:
    """A disc entry's then_encode, as its summary shows it."""
    from app.services.encode import describe_options

    kept = "删除无损版" if request.replace_source else "保留无损版"
    return f"{describe_options(request.options)} · {kept}"


def describe_disc(request: Optional[DiscRequest]) -> str:
    """One scannable line: "原盘封装 · 3 个标题 · 分集 · 起始第 5 集"."""
    if request is None:
        return ""
    parts = ["原盘封装",
             f"{len(request.titles)} 个标题" if request.titles else "按默认勾选"]
    if request.series is not None:
        parts.append("分集" if request.series else "整片")
    # a box set's second volume starts at 7 without anyone having said
    # "series" — the number is worth showing either way
    if request.episode_start not in (None, 1):
        parts.append(f"从第 {request.episode_start} 集起")
    mode = request.effective_output_mode()
    if mode == "custom":
        parts.append(f"输出到 {request.output_dir}")
    elif mode == "inside":
        parts.append("放进光盘自己的文件夹")
    return " · ".join(parts)


def describe(request: Optional[JobRequest]) -> str:
    """One scannable line: "ja → 简体中文 · 双语 · 语音识别"."""
    if request is None:
        return ""
    modes = {"bilingual": "双语", "translation_only": "纯译文",
             "original_only": "纯原文", "bilingual_split": "双文件"}
    source = "片源字幕" if request.text_source == "subtitle" else "语音识别"
    src = request.source_language or "auto"
    parts = [f"{src} → {request.target_language}",
             modes.get(request.output_mode, request.output_mode), source]
    if request.embed_subtitle:
        parts.append("内嵌视频")
    return " · ".join(parts)


# ----------------------------------------------------------- reading a file

def _lenient_settings(raw: object) -> Tuple[Optional[AppSettings], List[str]]:
    """Validate a stored snapshot, dropping only the fields that no longer fit.

    A setting that fails validation reverts to its default — a value the
    user could have chosen anyway — which is a far better outcome than
    losing the whole entry over one out-of-range number. Returns the
    settings (or None if even that fails) and the paths that were dropped.
    """
    if not isinstance(raw, dict):
        return None, []
    try:
        return AppSettings.model_validate(raw), []
    except ValidationError as exc:
        dropped: List[str] = []
        patched = json.loads(json.dumps(raw))  # cheap deep copy
        for err in exc.errors():
            loc = [str(p) for p in err.get("loc", ())]
            if not loc:
                continue
            node = patched
            for part in loc[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, dict) and loc[-1] in node:
                node.pop(loc[-1])
                dropped.append(".".join(loc))
        try:
            return AppSettings.model_validate(patched), dropped
        except ValidationError:
            return None, dropped


def _entry_from_raw(raw: dict) -> QueueEntry:
    """One stored entry, never raising.

    Settings are repaired where possible; a request that will not validate
    is fatal to that entry and nothing else. The asymmetry is deliberate: a
    defaulted *setting* is a value the user might have picked, while a
    defaulted *request* field means translating the wrong thing — losing
    embed_subtitle would quietly write a sidecar instead of a video.
    """
    raw = dict(raw)
    raw_settings = raw.pop("settings", None)
    settings, dropped = _lenient_settings(raw_settings)
    note = raw.get("note") or ""
    if dropped:
        note = (note + " " if note else "") + \
            "以下设置项已回退为默认值：" + "、".join(dropped)
    elif raw_settings is not None and settings is None:
        note = (note + " " if note else "") + "设置快照无法读取，将使用当前设置"
    try:
        entry = QueueEntry.model_validate(raw)
    except ValidationError as exc:
        # Keep a placeholder rather than dropping it: a queue that silently
        # loses entries is worse than one that shows a broken row.
        return QueueEntry(
            id=str(raw.get("id") or uuid.uuid4().hex[:12]),
            kind=str(raw.get("kind") or "job"),
            status="failed",
            title=str(raw.get("title") or ""),
            error=f"这条记录无法读取（可能来自别的版本）：{exc.error_count()} 处不符",
        )
    entry.settings = settings
    entry.note = note
    return entry


def _payload(entry: QueueEntry):
    """The request an entry runs, whatever its kind."""
    if entry.kind == "disc":
        return entry.disc
    if entry.kind == "encode":
        return entry.encode
    if entry.kind == "audio":
        return entry.audio
    return entry.request


# --------------------------------------------------------------- the store

class QueueStore:
    """The persisted list. No threads here — see QueueManager for the runner."""

    def __init__(self):
        self.lock = threading.RLock()
        self.entries: List[QueueEntry] = []
        self.paused = False
        # 列队页的「压制让出 CPU」。存在这里只为跨重启记住；正在跑的压制读的是
        # cpuyield 里的现值，这里每次读到或改了都推过去
        self.cpu_yield = False
        # 列队页的「压制内存上限」（GB，0 = 自动），同理推给 memguard
        self.memory_limit_gb = 0.0
        self._loaded_from: Optional[Path] = None

    # -- disk ------------------------------------------------------------

    def load(self) -> None:
        """Read the file, repairing anything left running by a crash."""
        path = queue_path()
        self._loaded_from = path
        self.entries = []
        self.paused = False
        self.cpu_yield = False
        cpuyield.set_enabled(False)
        self.memory_limit_gb = 0.0
        memguard.set_limit(0)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            # Only reachable through disk corruption, given the atomic
            # write. Keep the evidence instead of overwriting it.
            try:
                os.replace(path, path.with_name(path.name + ".bad"))
            except OSError:
                pass
            return
        self.paused = bool(raw.get("paused", False))
        self.cpu_yield = bool(raw.get("cpu_yield", False))
        cpuyield.set_enabled(self.cpu_yield)
        try:
            self.memory_limit_gb = max(float(raw.get("memory_limit_gb", 0) or 0), 0.0)
        except (TypeError, ValueError):
            self.memory_limit_gb = 0.0
        memguard.set_limit(self.memory_limit_gb)
        for item in raw.get("entries", []) or []:
            if isinstance(item, dict):
                self.entries.append(_entry_from_raw(item))
        self._recover_interrupted()
        self.save()

    def _recover_interrupted(self) -> None:
        """Anything still marked running died with the process.

        Repaired on load rather than at shutdown, so that one mechanism
        covers both a clean stop and a SIGKILL — and the crash path is the
        one the tests exercise.
        """
        for entry in self.entries:
            if entry.status == "running":
                entry.status = "queued"
                entry.job_id = ""
                entry.started_at = 0.0
                entry.interrupted += 1
                entry.note = INTERRUPTED_NOTE

    def save(self) -> None:
        with self.lock:
            self._prune()
            payload = json.dumps(
                {"version": 1, "paused": self.paused, "cpu_yield": self.cpu_yield,
                 "memory_limit_gb": self.memory_limit_gb,
                 "entries": [e.model_dump() for e in self.entries]},
                ensure_ascii=False, indent=2, default=str,
            )
            path = queue_path()
            part = path.with_name(path.name + ".part")
            try:
                part.write_text(payload, encoding="utf-8")
                os.replace(part, path)
            except BaseException:
                part.unlink(missing_ok=True)
                raise

    def _prune(self) -> None:
        finished = [e for e in self.entries if e.status in TERMINAL]
        if len(finished) <= KEEP_FINISHED:
            return
        finished.sort(key=lambda e: e.finished_at)
        drop = {id(e) for e in finished[: len(finished) - KEEP_FINISHED]}
        self.entries = [e for e in self.entries if id(e) not in drop]

    # -- queries ---------------------------------------------------------

    def get(self, entry_id: str) -> Optional[QueueEntry]:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def next_queued(self) -> Optional[QueueEntry]:
        for entry in self.entries:
            if entry.status == "queued":
                return entry
        return None

    def counts(self) -> dict:
        out = {k: 0 for k in ("queued", "running", "done", "failed", "cancelled")}
        for entry in self.entries:
            if entry.status in out:
                out[entry.status] += 1
        return out

    # -- mutations -------------------------------------------------------

    def add(self, request: JobRequest, settings: AppSettings,
            group_id: str = "", group_title: str = "") -> QueueEntry:
        with self.lock:
            if sum(1 for e in self.entries if e.status == "queued") >= MAX_QUEUED:
                raise ValueError(f"列队已满（最多 {MAX_QUEUED} 条等待中的任务）")
            entry = QueueEntry(
                id=uuid.uuid4().hex[:12],
                title=request.video_path,
                created_at=time.time(),
                request=request,
                settings=settings,
                group_id=group_id,
                group_title=group_title,
            )
            self.entries.append(entry)
            self.save()
            return entry

    def add_disc(self, request: DiscRequest, settings: AppSettings,
                 group_id: str = "", group_title: str = "",
                 then: Optional[JobRequest] = None,
                 then_encode: Optional[EncodeRequest] = None) -> QueueEntry:
        with self.lock:
            if sum(1 for e in self.entries if e.status == "queued") >= MAX_QUEUED:
                raise ValueError(f"列队已满（最多 {MAX_QUEUED} 条等待中的任务）")
            entry = QueueEntry(
                id=uuid.uuid4().hex[:12],
                kind="disc",
                title=request.path,
                created_at=time.time(),
                disc=request,
                settings=settings,
                group_id=group_id,
                group_title=group_title,
                then=then,
                then_encode=then_encode,
            )
            self.entries.append(entry)
            self.save()
            return entry

    def add_encode(self, request: EncodeRequest, settings: AppSettings,
                   group_id: str = "", group_title: str = "",
                   then: Optional[JobRequest] = None,
                   origin: str = "", origin_kind: str = "") -> QueueEntry:
        with self.lock:
            if sum(1 for e in self.entries if e.status == "queued") >= MAX_QUEUED:
                raise ValueError(f"列队已满（最多 {MAX_QUEUED} 条等待中的任务）")
            entry = QueueEntry(
                id=uuid.uuid4().hex[:12],
                kind="encode",
                title=request.source,
                created_at=time.time(),
                encode=request,
                settings=settings,
                group_id=group_id,
                group_title=group_title,
                then=then,
                origin=origin,
                origin_kind=origin_kind,
            )
            self.entries.append(entry)
            self.save()
            return entry

    def add_audio(self, request: AudioRequest, settings: AppSettings,
                  group_id: str = "", group_title: str = "") -> QueueEntry:
        with self.lock:
            if sum(1 for e in self.entries if e.status == "queued") >= MAX_QUEUED:
                raise ValueError(f"列队已满（最多 {MAX_QUEUED} 条等待中的任务）")
            entry = QueueEntry(
                id=uuid.uuid4().hex[:12],
                kind="audio",
                title=request.source,
                created_at=time.time(),
                audio=request,
                settings=settings,
                group_id=group_id,
                group_title=group_title,
            )
            self.entries.append(entry)
            self.save()
            return entry

    def remove(self, entry_id: str) -> bool:
        with self.lock:
            entry = self.get(entry_id)
            if entry is None:
                return False
            if entry.status == "running":
                raise ValueError("这条任务正在运行，请先取消它")
            self.entries = [e for e in self.entries if e.id != entry_id]
            self.save()
            return True

    def clear_finished(self) -> int:
        with self.lock:
            before = len(self.entries)
            self.entries = [e for e in self.entries if e.status not in TERMINAL]
            self.save()
            return before - len(self.entries)

    def reorder(self, ids: List[str]) -> None:
        """Rearrange the queued entries into the given order.

        Takes the full list of queued ids and refuses a mismatch, rather
        than accepting an index: a browser holding a stale list would
        otherwise scramble the queue instead of being told to refresh.
        """
        with self.lock:
            queued = [e for e in self.entries if e.status == "queued"]
            if {e.id for e in queued} != set(ids) or len(ids) != len(queued):
                raise ValueError("列队已经变化，请刷新后重试")
            by_id = {e.id: e for e in queued}
            ordered = iter([by_id[i] for i in ids])
            self.entries = [
                next(ordered) if e.status == "queued" else e for e in self.entries
            ]
            self.save()

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self.paused = paused
            self.save()

    def set_cpu_yield(self, on: bool) -> None:
        with self.lock:
            self.cpu_yield = bool(on)
            cpuyield.set_enabled(self.cpu_yield)
            self.save()

    def set_memory_limit(self, gb: float) -> None:
        with self.lock:
            self.memory_limit_gb = max(float(gb), 0.0)
            memguard.set_limit(self.memory_limit_gb)
            self.save()


# -------------------------------------------------------------- the runner

POLL_SECONDS = 0.5


class QueueManager:
    """Runs the queue: one entry at a time, in order, surviving restarts.

    The core is a single `step()` rather than a blocking loop, so tests can
    drive the whole state machine synchronously — no sleeps, no races. The
    thread is only `while not stopped: step(); wait(POLL_SECONDS)`.

    It never touches pipeline._run_slot. Two separate guarantees compose:
    the semaphore gives "one heavy job at a time", this gives "one queue
    entry started at a time". A job launched from the 开始翻译 button
    therefore jumps ahead of the queue, which is what 立即 should mean.
    """

    def __init__(self, store: Optional[QueueStore] = None):
        self.store = store or QueueStore()
        self._current: Optional[str] = None     # entry id
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self.store.load()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the runner. Never waits for the job itself.

        A running job is a daemon thread and dies with the process; its
        entry stays `running` on disk and is re-queued on the next start.
        Said plainly: shutting the server down does not cancel the running
        job, it loses it.
        """
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001 — a runner must not die
                print(f"[queue] {exc}")
            self._wake.wait(POLL_SECONDS)
            self._wake.clear()

    def nudge(self) -> None:
        """Wake the runner now instead of at the next tick."""
        self._wake.set()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the state machine -----------------------------------------------

    def step(self) -> None:
        """One tick: start the next entry, or check on the running one."""
        if self._current is None:
            self._start_next()
        else:
            self._poll_current()

    def _start_next(self) -> None:
        from app.services import series
        from app.services.pipeline import manager as job_manager

        with self.store.lock:
            if self.store.paused:
                return
            entry = self.store.next_queued()
            payload = None
            if entry is not None:
                payload = _payload(entry)
            if entry is None or payload is None:
                if entry is not None:            # unreadable request
                    entry.status = "failed"
                    entry.error = entry.error or "这条任务的参数无法读取"
                    entry.finished_at = time.time()
                    self.store.save()
                return
            # Marked running and persisted BEFORE anything is started, so a
            # crash in between is recoverable rather than invisible.
            entry.status = "running"
            entry.started_at = time.time()
            # The note is transient — it said "this will start over" and it
            # just did. The interrupted count is not, and is left alone.
            entry.note = ""
            self.store.save()
            entry_id, kind, settings = entry.id, entry.kind, entry.settings

        # The glossary store is memory-only, so a batch that spans a restart
        # starts a fresh one. Same trade as re-running an interrupted job.
        if kind == "job" and payload.series_id and series.get(payload.series_id) is None:
            series.create(payload.series_id)

        try:
            if kind == "disc":
                job = job_manager.create_disc(payload, settings=settings)
            elif kind == "encode":
                job = job_manager.create_encode(payload, settings=settings)
            elif kind == "audio":
                job = job_manager.create_audio(payload, settings=settings)
            else:
                job = job_manager.create(payload, settings=settings)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the queue
            with self.store.lock:
                current = self.store.get(entry_id)
                if current is not None:
                    current.status = "failed"
                    current.error = str(exc)
                    current.finished_at = time.time()
                    self.store.save()
            return
        with self.store.lock:
            current = self.store.get(entry_id)
            if current is not None:
                current.job_id = job.id
                self.store.save()
        self._current = entry_id

    def _queue_encodes(self, entry: QueueEntry) -> None:
        """加入列队并压制、做字幕: queue an encode of every MKV *entry* wrote,
        right behind it — so a batch encodes each disc before remuxing the
        next, and only one disc's lossless files take up space at a time.
        Each gets the request frozen when the button was pressed
        (entry.then_encode, source filled in) and entry.then, so the encode
        queues its own subtitles when it is done.

        Idempotent for the reasons _queue_subtitles is, plus one: a file
        that already carries our encode tag is an encode already (a retried
        disc lists titles whose lossless file was swapped for it).
        Caller holds store.lock and saves."""
        from app.services import encode

        try:
            template = entry.then_encode
            taken = {e.encode.source for e in self.store.entries
                     if e.kind == "encode" and e.encode is not None
                     and e.status in ("queued", "running", "done")}
            wanted, skipped = [], 0
            for name in entry.result_files:
                path = Path(name)
                if name in taken or not path.is_file() \
                        or encode.read_tags(path)["encode"] is not None:
                    skipped += 1
                    continue
                wanted.append(name)
            room = MAX_QUEUED - sum(1 for e in self.store.entries if e.status == "queued")
            left_out = wanted[max(room, 0):]
            now = time.time()
            added = [QueueEntry(
                id=uuid.uuid4().hex[:12],
                kind="encode",
                title=name,
                created_at=now,
                encode=template.model_copy(update={"source": name}),
                settings=entry.settings,
                group_id=entry.group_id or entry.id,
                group_title=entry.group_title or entry.title,
                then=entry.then,
                origin=entry.id,
                origin_kind="disc",
            ) for name in wanted[:max(room, 0)]]
            at = self.store.entries.index(entry) + 1
            self.store.entries[at:at] = added
            note = f"已自动加入 {len(added)} 条压制任务（紧跟在这张盘后面）"
            if skipped:
                note += f"，{skipped} 个已压制过或已在列队里，跳过"
            if left_out:
                note += f"；列队已满，另有 {len(left_out)} 个没有加入"
            entry.note = note
        except Exception as exc:  # noqa: BLE001 — the queue must go on
            entry.note = f"自动加入压制任务失败：{exc}"

    def _queue_subtitles(self, entry: QueueEntry,
                         files: Optional[List[str]] = None) -> None:
        """加入列队并做字幕: queue a translation of every MKV *entry* wrote
        (or *files*), at the end of the queue, with the request frozen when
        its button was pressed (entry.then) and the same settings snapshot.

        A failed remux still hands over what it did write — one broken
        title must not cost the others their subtitles, the same rule the
        remux itself follows. Idempotent, because it runs again for the
        same files: a retried remux reports the titles it skipped as
        already written, and a crash before the queue is saved runs this
        again. So a file is skipped when a translation of it is already
        queued, running or done, or a subtitle in the target language is
        already beside it (the check a translation batch makes).
        Caller holds store.lock and saves."""
        from app.core.media import subtitle_state

        try:
            template = entry.then
            taken = {e.request.video_path for e in self.store.entries
                     if e.kind == "job" and e.request is not None
                     and e.status in ("queued", "running", "done")}
            wanted, skipped = [], 0
            for name in (entry.result_files if files is None else files):
                path = Path(name)
                if name in taken or not path.is_file() or subtitle_state(
                        path.parent, path.stem,
                        template.target_language)[0] == "done":
                    skipped += 1
                    continue
                wanted.append(name)
            room = MAX_QUEUED - sum(1 for e in self.store.entries if e.status == "queued")
            left_out = wanted[max(room, 0):]
            now = time.time()
            for name in wanted[:max(room, 0)]:
                self.store.entries.append(QueueEntry(
                    id=uuid.uuid4().hex[:12],
                    title=name,
                    created_at=now,
                    request=template.model_copy(update={"video_path": name}),
                    settings=entry.settings,
                    group_id=entry.group_id or entry.id,
                    group_title=entry.group_title or entry.title,
                    origin=entry.id,
                    origin_kind=entry.kind,
                ))
            note = f"已自动加入 {len(wanted) - len(left_out)} 条字幕任务（排在列队最后）"
            if skipped:
                note += f"，{skipped} 个已有字幕或已在列队里，跳过"
            if left_out:
                note += f"；列队已满，另有 {len(left_out)} 个没有加入"
        except Exception as exc:  # noqa: BLE001 — the queue must go on
            note = f"自动加入字幕任务失败：{exc}"
        # an encode's note already says what it made; this goes after it
        entry.note = f"{entry.note}；{note}" if entry.note else note

    def _poll_current(self) -> None:
        from app.services.pipeline import manager as job_manager

        entry_id = self._current
        with self.store.lock:
            entry = self.store.get(entry_id or "")
            if entry is None:                     # removed under us
                self._current = None
                return
            try:
                job = job_manager.get(entry.job_id)
            except KeyError:
                entry.status = "failed"
                entry.error = "任务记录已丢失"
                entry.finished_at = time.time()
                self.store.save()
                self._current = None
                return
            if job.status.stage not in TERMINAL:
                return
            entry.status = job.status.stage
            entry.error = job.status.error or ""
            entry.result_srt = job.status.srt_filename or ""
            entry.result_srt_original = job.status.original_srt_filename or ""
            entry.result_video = job.status.video_filename or ""
            entry.result_in_place = bool(job.status.srt_in_place)
            entry.result_files = list(job.status.outputs)
            entry.finished_at = time.time()
            if entry.kind in ("encode", "audio") and entry.status == "done":
                # 已压制：片名.mkv（3.2 GB，原 21 GB 的 15%…）— the one line
                # worth keeping once the job has left memory
                entry.note = job.status.message or ""
            self._follow_up(entry)
            # one write for the entry's end and the entries it queued: two
            # writes, and a crash between them would lose the follow-ups for
            # good — a finished entry is never polled again
            self.store.save()
        self._current = None
        # Swept after every entry, not only at startup: a queue left
        # running for days would otherwise never come back under its limit.
        from app.core.cache import prune_checkpoints
        try:
            prune_checkpoints()
        except OSError:
            pass          # housekeeping must never stop the queue

    def _follow_up(self, entry: QueueEntry) -> None:
        """Queue what a finished entry was told to lead to. Caller holds
        store.lock and saves — in the same write as the entry's end.

        A disc hands every MKV it wrote on, done or failed (a broken title
        must not cost the others): to an encode first when there is one
        (then_encode), else straight to subtitles. An encode hands on its
        result once done; one that replaces a lossless file hands on that
        name even when it failed — the lossless file is still there under
        it, and gets its subtitles like any other. Cancelled leads nowhere.
        """
        if entry.status not in ("done", "failed"):
            return
        if entry.kind == "disc" and entry.result_files:
            if entry.then_encode is not None:
                self._queue_encodes(entry)
            elif entry.then is not None:
                self._queue_subtitles(entry)
        elif entry.kind == "encode" and entry.then is not None:
            if entry.status == "done":
                files = entry.result_files
            elif entry.encode is not None and entry.encode.replace_source:
                files = [entry.encode.source]
            else:
                files = []
            if files:
                self._queue_subtitles(entry, files)

    # -- commands --------------------------------------------------------

    def cancel(self, entry_id: str) -> str:
        """Cancel an entry. Queued ones stop instantly, having never started."""
        from app.services.pipeline import manager as job_manager

        with self.store.lock:
            entry = self.store.get(entry_id)
            if entry is None:
                raise KeyError(entry_id)
            if entry.status in TERMINAL:
                return entry.status
            if entry.status == "queued":
                # No Job was ever created, so there is nothing to wait for.
                entry.status = "cancelled"
                entry.finished_at = time.time()
                self.store.save()
                return "cancelled"
            job_id = entry.job_id
        if job_id:
            try:
                job_manager.cancel(job_id)
            except KeyError:
                pass
        return "running"

    def retry(self, entry_id: str, fresh: bool = False) -> QueueEntry:
        """Queue a finished entry again, as a new entry at the tail."""
        with self.store.lock:
            entry = self.store.get(entry_id)
            if entry is None:
                raise KeyError(entry_id)
            if entry.status not in TERMINAL:
                raise ValueError("这条任务还没有结束")
            payload = _payload(entry)
            if payload is None:
                raise ValueError("这条任务的参数无法读取，无法重试")
            # Retry means "run that same thing again", so the snapshot is
            # reused; fresh=True re-reads current settings, and is forced
            # when the stored snapshot could not be loaded.
            settings = snapshot() if (fresh or entry.settings is None) \
                else entry.settings
            if entry.kind == "disc":
                # the same request, so the same checkpoint: titles finished
                # last time are skipped, not written again as 片名.2.mkv
                # …and the same follow-up: _queue_subtitles skips the MKVs
                # whose translation is already queued or done
                # …and a remux → encode → subtitles chain keeps both steps;
                # the titles already handed on are recognised by their tags
                new = self.store.add_disc(payload, settings,
                                          group_id=entry.group_id,
                                          group_title=entry.group_title,
                                          then=entry.then,
                                          then_encode=entry.then_encode)
            elif entry.kind == "encode":
                # a finished encode finds its own file and does nothing
                # (encode.existing_encode); a failed one runs again
                new = self.store.add_encode(payload, settings,
                                            group_id=entry.group_id,
                                            group_title=entry.group_title,
                                            then=entry.then,
                                            origin=entry.origin,
                                            origin_kind=entry.origin_kind)
            elif entry.kind == "audio":
                # a finished one finds its own file (audioextract.existing)
                new = self.store.add_audio(payload, settings,
                                           group_id=entry.group_id,
                                           group_title=entry.group_title)
            else:
                new = self.store.add(payload, settings,
                                     group_id=entry.group_id,
                                     group_title=entry.group_title)
        self.nudge()
        return new


queue_manager = QueueManager()

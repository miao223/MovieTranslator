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
from app.models.schemas import AppSettings, JobRequest, QueueEntry

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
_NOT_OUTPUT_AFFECTING = ("server", "mcp", "work_dir", "model_cache_dir")


def settings_hash(settings: Optional[AppSettings]) -> str:
    """A short fingerprint of the output-affecting settings."""
    if settings is None:
        return ""
    data = settings.model_dump()
    for key in _NOT_OUTPUT_AFFECTING:
        data.pop(key, None)
    for key in ("api_key", "vision_api_key", "audio_api_key"):
        data.get("llm", {}).pop(key, None)
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def describe(request: Optional[JobRequest]) -> str:
    """One scannable line: "ja → 简体中文 · 双语 · 语音识别"."""
    if request is None:
        return ""
    modes = {"bilingual": "双语", "translation_only": "纯译文",
             "original_only": "纯原文"}
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


# --------------------------------------------------------------- the store

class QueueStore:
    """The persisted list. No threads here — see QueueManager for the runner."""

    def __init__(self):
        self.lock = threading.RLock()
        self.entries: List[QueueEntry] = []
        self.paused = False
        self._loaded_from: Optional[Path] = None

    # -- disk ------------------------------------------------------------

    def load(self) -> None:
        """Read the file, repairing anything left running by a crash."""
        path = queue_path()
        self._loaded_from = path
        self.entries = []
        self.paused = False
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
                {"version": 1, "paused": self.paused,
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
            if entry is None or entry.request is None:
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
            entry_id, request, settings = entry.id, entry.request, entry.settings

        # The glossary store is memory-only, so a batch that spans a restart
        # starts a fresh one. Same trade as re-running an interrupted job.
        if request.series_id and series.get(request.series_id) is None:
            series.create(request.series_id)

        try:
            job = job_manager.create(request, settings=settings)
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
            entry.result_video = job.status.video_filename or ""
            entry.result_in_place = bool(job.status.srt_in_place)
            entry.finished_at = time.time()
            self.store.save()
        self._current = None

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
            if entry.request is None:
                raise ValueError("这条任务的参数无法读取，无法重试")
            # Retry means "run that same thing again", so the snapshot is
            # reused; fresh=True re-reads current settings, and is forced
            # when the stored snapshot could not be loaded.
            settings = snapshot() if (fresh or entry.settings is None) \
                else entry.settings
            new = self.store.add(entry.request, settings,
                                 group_id=entry.group_id,
                                 group_title=entry.group_title)
        self.nudge()
        return new


queue_manager = QueueManager()

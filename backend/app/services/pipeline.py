"""Job orchestration: state machine, background execution, SSE progress.

Whisper is CPU-bound, so only one job runs at a time (global semaphore).
Intermediate artifacts are written to the per-job cache dir for debugging
and are wiped on the next application startup.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from app.core import config
from app.core.cache import (
    checkpoint_dir, checkpoints_enabled, job_dir, prune_checkpoints)
from app.core.debuglog import DebugLog, open_debug_log
from app.core.joblog import JobLogWriter, _settings_lines as joblog_settings_lines
from app.models.schemas import (
    AppSettings,
    JobRequest,
    JobStatus,
    ProgressEvent,
    SubtitleLine,
)
from app.services import (
    asr, audio, lyrics, mux, ocr, refine, segmenter, series, subsource,
    subtitle, vet,
)
from app.services.translator import Translator

# overall progress ranges per stage: (start%, end%)
STAGE_RANGES = {
    "extracting": (0.0, 10.0),
    # reading a subtitle covers the whole span the other two stages use,
    # because it replaces both — and takes seconds rather than an hour
    "importing": (0.0, 60.0),
    "transcribing": (10.0, 60.0),
    "refining": (60.0, 72.0),
    "translating": (72.0, 95.0),
    "composing": (95.0, 100.0),
}


def _hms(seconds: float) -> str:
    whole = int(max(seconds, 0))
    hours, minutes = divmod(whole // 60, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分"
    return f"{minutes} 分" if minutes else f"{whole} 秒"


def _eta(started: float, fraction: float, recoding: bool) -> str:
    """How long the re-encode has taken and has left, as text.

    The progress bar cannot express this: composing owns 95->100% because
    it used to be a file copy, and widening it now would mean re-numbering
    every literal percentage in _execute. A re-encode running for hours
    behind a bar that barely moves needs the real figure somewhere, so it
    goes in the message.
    """
    if not recoding or fraction <= 0.01:
        return ""
    elapsed = time.monotonic() - started
    if elapsed < 5:  # too early for the estimate to mean anything
        return ""
    return f"（已用 {_hms(elapsed)}，约剩 {_hms(elapsed / fraction - elapsed)}）"


_run_slot = threading.Semaphore(1)  # one CPU-heavy job at a time

# Credentials are the one thing a frozen snapshot does NOT keep. Everything
# else about a queued job is settled when it is enqueued, but a key that
# expired or was rotated since then should repair the queue rather than
# break it — so these three are read live, at the moment the work starts.
# jobqueue blanks them before writing, so a snapshot never carries one and
# this is the only path by which a key reaches a queued job at all.
LIVE_KEY_FIELDS = ("api_key", "vision_api_key", "audio_api_key")


def checkpoint_key(job: Job, settings: AppSettings) -> str:
    """Identifies this exact piece of work, for resuming it.

    Every reason a half-finished result would be wrong to continue from is
    folded into the name: change the source file, the settings, or the
    program, and the work lands in a different directory and starts clean.
    Nothing is compared afterwards, so there is no check that can be
    forgotten — a key that matches cannot be stale.

    Deliberately not the job id: that is new every run, which is precisely
    what stopped anything being reused.
    """
    from app.core.joblog import APP_VERSION
    from app.services.jobqueue import settings_hash

    video = Path(job.request.video_path)
    try:
        stat = video.stat()
        stamp = f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        stamp = "missing"
    raw = "|".join([
        str(video.resolve()), stamp, settings_hash(settings), APP_VERSION,
        # the request decides what is produced, not just how
        job.request.model_dump_json(),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _settings_for(job: Job) -> AppSettings:
    """The settings this job runs with: its own snapshot, or the live ones.

    Resolved here rather than where the job was dequeued: a job can sit
    parked on _run_slot for hours after its turn came up, and "run time"
    has to mean the moment the work actually starts.
    """
    live = config.load_settings()
    if job.settings is None:
        return live
    return job.settings.model_copy(update={
        "llm": job.settings.llm.model_copy(
            update={f: getattr(live.llm, f) for f in LIVE_KEY_FIELDS}),
    })


class Job:
    def __init__(self, request: JobRequest, audio_only: Optional[bool] = None,
                 settings: Optional[AppSettings] = None):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        # A source with no picture cannot be muxed into a video and has no
        # frames to translate. Worked out here rather than only in
        # JobManager.create so that a Job built directly (as the tests do)
        # answers the question correctly too; create passes its own probe in
        # so the container is opened once per job, not twice.
        self.audio_only = (
            audio_only if audio_only is not None
            else not audio.has_picture(request.video_path)
        )
        # Frozen settings, or None for "read them live when you start".
        # A Job field and deliberately not a JobRequest field, for the same
        # reason audio_only is not one: JobRequest is the outward schema and
        # a client must not be able to hand us the settings to run under —
        # a LAN browser's copy has its API keys masked to ********.
        self.settings = settings
        self.checkpoint = ""      # set by _workdir_for when resuming is on
        self.status = JobStatus(id=self.id, video_path=request.video_path)
        self.cancel_event = threading.Event()
        self.events: List[ProgressEvent] = []
        self.subscribers: List[queue.Queue] = []
        self.lock = threading.Lock()
        self.srt_path: Optional[Path] = None
        # everything published below is mirrored into a downloadable file
        self.logfile = JobLogWriter(self.id, request.video_path)

    # -------------------------------------------------------- events

    def publish(self, stage: str, progress: float, message: str = "", log: str = ""):
        self.status.stage = stage  # type: ignore[assignment]
        self.status.progress = round(progress, 1)
        if message:
            self.status.message = message
        event = ProgressEvent(
            stage=stage, progress=self.status.progress, message=message, log=log
        )
        self.logfile.event(stage, self.status.progress, message, log)
        with self.lock:
            self.events.append(event)
            for q in self.subscribers:
                q.put(event)

    def subscribe(self) -> Iterator[ProgressEvent]:
        q: queue.Queue = queue.Queue()
        with self.lock:
            history = list(self.events)
            self.subscribers.append(q)
        try:
            yield from history
            if self.status.stage in ("done", "failed", "cancelled"):
                return
            while True:
                event = q.get()
                yield event
                if event.stage in ("done", "failed", "cancelled"):
                    return
        finally:
            with self.lock:
                if q in self.subscribers:
                    self.subscribers.remove(q)


# Finished jobs are kept so the UI can still read their status, but not
# forever: every Job holds its whole event list (about half a megabyte for
# a feature film). A few jobs a day by hand is nothing; a queue running
# unattended for a week is hundreds of megabytes that never come back.
KEEP_FINISHED_JOBS = 100


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def _evict_old(self) -> None:
        """Drop the oldest finished jobs, sparing any a live batch still reads.

        BatchManager.status walks its members with job_manager.get(), so a
        job belonging to an unfinished batch must stay reachable. Queue
        entries need no such protection: each one records its own outcome
        and never looks a finished job up again.
        """
        finished = [j for j in self.jobs.values()
                    if j.status.stage in ("done", "failed", "cancelled")]
        if len(finished) <= KEEP_FINISHED_JOBS:
            return
        from app.services.batch import batch_manager

        spoken_for = {
            jid
            for batch in batch_manager.batches.values()
            for jid in batch.job_ids
            if any(self.jobs[j].status.stage not in ("done", "failed", "cancelled")
                   for j in batch.job_ids if j in self.jobs)
        }
        droppable = [j for j in finished if j.id not in spoken_for]
        for job in droppable[: len(finished) - KEEP_FINISHED_JOBS]:
            self.jobs.pop(job.id, None)

    def create(self, request: JobRequest,
               settings: Optional[AppSettings] = None) -> Job:
        video = Path(request.video_path)
        if not video.is_file():
            raise FileNotFoundError(f"片源文件不存在: {video}")
        audio_only = not audio.has_picture(video)
        if request.frame_only and audio_only:
            # embed_subtitle degrades quietly on an audio source, but frame
            # translation IS the whole output here — running it would mean
            # producing nothing at all.
            raise ValueError("纯音频片源没有画面，无法使用「仅补充画面翻译」")
        # audio sources never reach the encoder (see _embed_subtitle), so the
        # check below must not reject them before that degradation happens
        if (request.embed_subtitle and not audio_only
                and request.embed.video_codec != mux.COPY):
            # Checked here rather than at mux time: the mux runs after an
            # hour of transcription and translation, and "this machine has
            # no such encoder" is knowable right now.
            usable = {enc["id"] for enc in mux.available_encoders()}
            if request.embed.video_codec not in usable:
                raise ValueError(
                    f"本机无法使用编码器 {request.embed.video_codec}。"
                    f"可用的有：{'、'.join(sorted(usable)) or '（无）'}"
                )
        job = Job(request, audio_only=audio_only, settings=settings)
        self._evict_old()
        self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> None:
        self.get(job_id).cancel_event.set()

    # ---------------------------------------------------------- pipeline

    def _resume_lines(self, job: Job, path: Path, what: str):
        """Lines a previous attempt already produced, or None.

        Only ever reads from the checkpoint directory, whose name encodes
        the source file, the settings and the program version — so a file
        found here was produced by the same work this run is doing. The
        check is the location, not the contents.
        """
        if not job.checkpoint or not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            lines = [SubtitleLine.model_validate(item) for item in raw]
        except (OSError, ValueError) as exc:
            job.publish(job.status.stage, job.status.progress,
                        log=f"⚠ 上次的{what}结果读不出来（{exc}），本次重做")
            return None
        if not lines:
            return None
        job.publish(job.status.stage, job.status.progress,
                    log=f"↻ 沿用上次的{what}结果（{len(lines)} 条），跳过这一步")
        return lines

    def _resume_meta(self, workdir: Path) -> dict:
        try:
            return json.loads((workdir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _workdir_for(self, job: Job) -> Path:
        """Where this job writes — a resumable place when that is allowed.

        Keyed on the work rather than on the job id, so a second attempt at
        the same film under the same settings finds the first attempt's
        results sitting there. With resuming switched off it is the old
        per-job directory, which the next startup wipes.
        """
        if not checkpoints_enabled():
            return job_dir(job.id)
        job.checkpoint = checkpoint_key(job, _settings_for(job))
        return checkpoint_dir(job.checkpoint)

    def _stage_progress(self, job: Job, stage: str):
        lo, hi = STAGE_RANGES[stage]

        def cb(fraction: float, log: str = ""):
            job.publish(stage, lo + (hi - lo) * min(max(fraction, 0.0), 1.0), log=log)

        return cb

    def _run(self, job: Job) -> None:
        req = job.request
        workdir = self._workdir_for(job)
        job.publish("pending", 0, message="排队中…")
        # Acquired with a timeout rather than `with _run_slot:` so that a job
        # waiting its turn can still be cancelled. Blocking outright put the
        # cancel check on the far side of the wait, so cancelling a queued
        # job did nothing observable until it reached the front of the line —
        # which, behind a two-hour film, is not cancelling.
        while not _run_slot.acquire(timeout=0.2):
            if job.cancel_event.is_set():
                job.publish("cancelled", job.status.progress, message="任务已取消")
                return
        try:
            try:
                if job.cancel_event.is_set():  # cancelled while queued
                    raise InterruptedError
                self._execute(job, req, workdir)
            except InterruptedError:
                job.publish("cancelled", job.status.progress, message="任务已取消")
            except Exception as exc:  # noqa: BLE001 — surface anything to the UI
                job.status.error = str(exc)
                tb = traceback.format_exc()
                print(tb)
                job.publish("failed", job.status.progress,
                            message=f"失败: {exc}", log=tb)
        finally:
            _run_slot.release()

    def _download_model_with_progress(self, job: Job, settings, check_cancel) -> float:
        """Download the whisper model, mapping progress into 10–20%.

        Returns the progress value where transcription should start (20.0).
        """
        from app.services import model_download

        size = settings.asr.model_size
        job.publish(
            "transcribing", 10,
            message=f"下载语音识别模型 {size}…",
            log=f"本地未找到模型 {size}，开始下载（仅首次需要）",
        )
        model_download.start_download(size, settings.network)
        last_logged = -10.0
        last_bytes = -1
        stalled_since = time.monotonic()
        stall_warned = 0
        while True:
            check_cancel()
            st = model_download.get_status(size)
            if st["status"] == "failed":
                raise RuntimeError(f"模型下载失败: {st.get('error')}")
            pct = float(st.get("progress") or 0.0)
            done_bytes = st.get("downloaded_bytes") or 0
            mb = done_bytes // 1048576
            total_mb = (st.get("total_bytes") or 0) // 1048576
            logline = ""
            if pct - last_logged >= 5 or st["status"] == "done":
                last_logged = pct
                logline = f"模型下载 {pct:.0f}%（{mb}/{total_mb} MB）"
            # stall watchdog: no bytes for 60s almost always means the machine
            # cannot reach HuggingFace — tell the user what to do about it
            if done_bytes != last_bytes:
                last_bytes = done_bytes
                stalled_since = time.monotonic()
                stall_warned = 0
            elif st["status"] == "downloading":
                stalled = int(time.monotonic() - stalled_since)
                if stalled >= 60 and stalled // 60 > stall_warned:
                    stall_warned = stalled // 60
                    logline = (
                        f"⚠ 模型下载已 {stalled} 秒无进展——大概率无法直连 "
                        "HuggingFace。建议：取消任务后，到「设置 → 网络」启用"
                        "「模型下载走代理」，或用设置页「立即下载」重试，"
                        "或改用「本地模型目录」离线导入。"
                    )
            job.publish(
                "transcribing", 10 + pct * 0.1,
                message=f"下载模型 {size}: {pct:.0f}%", log=logline,
            )
            if st["status"] == "done":
                return 20.0
            time.sleep(1)

    def _frame_lines(
        self, job: Job, req: JobRequest, settings, workdir: Path
    ) -> List[SubtitleLine]:
        """The frame translations, or nothing when there is no picture.

        Not left to per-task failure: an audio file with album art *has* a
        video stream, so every task would cheerfully translate the cover.
        """
        if job.audio_only:
            job.publish(
                "translating", 94,
                log=f"⚠ 纯音频片源没有画面，已跳过 {len(req.frame_tasks)} 条画面翻译"
                    "（不影响字幕生成）",
            )
            return []
        return self._translate_frames(job, req, settings, workdir)

    def _translate_frames(
        self, job, req, settings, workdir: Path,
        progress_lo: float = 93.0, progress_hi: float = 95.0,
    ) -> List[SubtitleLine]:
        """Translate on-screen text at the requested timestamps.

        Every task is independently fault-tolerant: a bad timestamp, a
        text-only model or an empty frame only produces a warning log.
        """
        from app.services import frame as frame_svc
        from app.services import vision

        results: List[SubtitleLine] = []
        total = len(req.frame_tasks)
        for i, task in enumerate(req.frame_tasks, start=1):
            label = task.time.strip()
            job.publish(
                "translating",
                progress_lo + (progress_hi - progress_lo) * i / total,
                message=f"画面翻译 {i}/{total}…",
            )
            try:
                if job.cancel_event.is_set():
                    raise InterruptedError
                seconds = frame_svc.parse_time(task.time)
                jpg = workdir / f"frame_{i}.jpg"
                frame_svc.extract_frame(req.video_path, seconds, jpg)
                text = vision.translate_frame(
                    jpg,
                    target_language=req.target_language,
                    note=task.note,
                    llm=settings.llm,
                    network=settings.network,
                )
                if text is None:
                    job.publish(
                        "translating", job.status.progress,
                        log=f"画面翻译 [{label}]：模型判断画面无可翻译文字，已跳过",
                    )
                    continue
                results.append(
                    SubtitleLine(
                        index=0,
                        start=round(seconds, 3),
                        end=round(seconds + task.duration, 3),
                        text="",
                        translation=text,
                        is_frame=True,
                    )
                )
                job.publish(
                    "translating", job.status.progress,
                    log=f"画面翻译 [{label}] 完成：{text[:60]}",
                )
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001 — per-task tolerance by design
                job.publish(
                    "translating", job.status.progress,
                    log=f"⚠ 画面翻译 [{label}] 失败（不影响正常字幕）: {exc}",
                )
        return results

    def _execute_frame_only(self, job: Job, req: JobRequest, workdir: Path) -> None:
        """Supplement mode: translate frames and merge into the existing
        same-stem subtitle file (.ass preferred over .srt) in place."""
        settings = _settings_for(job)
        video = Path(req.video_path)
        # 纯原文的产物带语言后缀（片名.ja.srt），两个精确候选认不出来——所以
        # 在它们之后再兜一层 glob，否则「先跑纯原文、之后补画面翻译」是死路
        target = next(
            (p for p in (
                video.with_suffix(".ass"), video.with_suffix(".srt"),
                *sorted(video.parent.glob(f"{video.stem}.*.ass")),
                *sorted(video.parent.glob(f"{video.stem}.*.srt")),
            ) if p.is_file()),
            None,
        )
        if target is None:
            raise RuntimeError(
                "未找到视频同名的 .srt / .ass 字幕文件，补充模式需要已有字幕；"
                "请先执行完整翻译"
            )
        if not req.frame_tasks:
            raise RuntimeError("补充模式至少需要一条画面翻译时间点")
        if req.embed_subtitle:
            # this mode edits an existing subtitle file in place; there is no
            # new subtitle to embed, and silently doing nothing would read as
            # a bug when no .mkv appears
            job.publish(
                "translating", 0,
                log="补充模式只修改已有字幕文件，本次忽略「合成带字幕的新视频」",
            )

        job.publish("translating", 10, message=f"补充画面翻译 → {target.name}")
        frame_lines = self._translate_frames(
            job, req, settings, workdir, progress_lo=20.0, progress_hi=90.0
        )
        if not frame_lines:
            raise RuntimeError("所有画面翻译均失败，未修改字幕文件（原因见日志）")

        job.publish("composing", 95, message="合并进字幕文件…")
        cues = [(l.start, l.end, l.translation) for l in frame_lines]
        original = target.read_text(encoding="utf-8", errors="replace")
        if target.suffix.lower() == ".ass":
            updated = subtitle.insert_frame_cues_ass(original, cues)
        else:
            updated = subtitle.insert_frame_cues_srt(original, cues)
        # atomic replace so a crash can never corrupt the user's subtitles
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(updated, encoding="utf-8")
        os.replace(tmp, target)

        job.srt_path = target
        job.status.srt_filename = str(target)
        job.status.srt_in_place = True
        job.publish(
            "done", 100,
            message=f"已补充 {len(frame_lines)} 条画面翻译到: {target}",
        )

    def _write_diagnostics(self, job: Job, req: JobRequest, settings) -> None:
        """Header that lets one log file explain a failure on its own."""
        job.logfile.write_environment()
        job.logfile.write_request(req, audio_only=job.audio_only)
        job.logfile.write_settings(settings)
        job.logfile.write_media(req.video_path)
        job.logfile.section("进度", [])

    def _execute(self, job: Job, req: JobRequest, workdir: Path) -> None:
        settings = _settings_for(job)
        self._write_diagnostics(job, req, settings)
        if job.audio_only and req.embed_subtitle:
            # said now, not at 95%: someone waiting for a video should not
            # learn an hour later that there was never going to be one
            job.publish(
                "extracting", 0,
                log="⚠ 纯音频片源没有画面，本次将忽略「合成带字幕的新视频」，"
                    "改为在音频所在目录生成字幕文件",
            )
        if job.audio_only and req.frame_tasks:
            job.publish(
                "extracting", 0,
                log=f"⚠ 纯音频片源没有画面，将跳过 {len(req.frame_tasks)} 条画面翻译",
            )
        if req.frame_only:
            self._execute_frame_only(job, req, workdir)
            return

        # Series mode: the names the earlier episodes of this batch settled
        # on, merged ahead of this job's own glossary. settings.prompts is
        # the instance config caches, so it is copied, never written to.
        shared, glossary = series.for_job(req.series_id, settings.prompts.glossary)

        # deep diagnostics: written next to the subtitle we are about to
        # produce, so the user finds it without hunting for a cache dir
        video = Path(req.video_path)
        debug = open_debug_log(
            video.parent / f"{video.stem}.srt", workdir, settings.debug_mode
        )
        if debug.enabled:
            job.publish(
                "extracting", 0,
                log=f"调试模式已开启，诊断日志: {debug.path}",
            )
            debug.section("任务与设置")
            debug.kv("视频", req.video_path)
            debug.kv("源语言 / 目标语言", f"{req.source_language} → {req.target_language}")
            debug.kv("输出模式", req.output_mode)
            debug.kv("剧情简介", req.synopsis.strip() or "（未填写）")
            debug.lines(joblog_settings_lines(settings))

        def check_cancel():
            if job.cancel_event.is_set():
                raise InterruptedError

        # 1-2. get the original text ---------------------------------------
        # Either from a subtitle the release already carries, or from the
        # audio. The rest of the pipeline cannot tell which it got.
        lines: List[SubtitleLine] = []
        detected = ""
        segments = None  # whisper's raw output, for the lyric cross-check
        vet_usage = {"calls": 0, "prompt": 0, "completion": 0, "cached": 0}
        source_kind = "asr"
        if req.text_source == "subtitle":
            got = self._import_subtitle(job, req, settings, debug)
            if got is not None:
                lines, detected, source_kind = got
        if source_kind == "asr":
            done = self._resume_lines(job, workdir / "transcript.json", "转写")
            if done is not None:
                # The detected language is not recoverable from the lines,
                # and everything downstream needs it — the subtitle's name
                # suffix in original_only mode, and the prompts. It is
                # stored beside the transcript for exactly this reason.
                lines = done
                detected = self._resume_meta(workdir).get("language", "") or detected
                job.publish("transcribing", 60,
                            message=f"沿用上次的转写，共 {len(lines)} 条")
            else:
                lines, detected, segments = self._transcribe(
                    job, req, settings, workdir, debug, check_cancel, vet_usage
                )
        (workdir / "transcript.json").write_text(
            json.dumps([l.model_dump() for l in lines], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        if job.checkpoint:
            (workdir / "meta.json").write_text(
                json.dumps({"language": detected or ""}, ensure_ascii=False),
                encoding="utf-8")

        # 3. preprocess the transcript --------------------------------------
        # rejoin sentences the ASR cut mid-phrase, in the source language and
        # with mechanical verification — a line holding only a fragment has
        # no counterpart in the target language, and the translator fills it
        # with the next line's content, shifting everything after it
        refine_usage = {"calls": 0, "prompt": 0, "completion": 0, "cached": 0}
        if source_kind == "subtitle" and settings.prompts.refine_enabled:
            # Every job this pass does — rejoin broken sentences, fix
            # homophones, restore missing full stops — is a repair of speech
            # recognition. A subtitle written by a person already has all
            # three right, and re-deciding its line breaks would only undo
            # work done against the picture. OCR output is a different
            # matter: a machine produced those characters, so it runs.
            job.publish(
                "refining", 60,
                log="使用已有字幕作为原文，已跳过转写预处理"
                    "（它是为修复语音识别缺陷设计的，对人工字幕只会打乱断句）",
            )
        elif settings.prompts.refine_enabled and (
                done := self._resume_lines(
                    job, workdir / "transcript_refined.json", "转写预处理")):
            lines = done
            job.publish("refining", 72,
                        message=f"沿用上次的预处理结果，共 {len(lines)} 条字幕")
        elif settings.prompts.refine_enabled:
            job.publish(
                "refining", 60,
                message=("OCR 结果校对中（纠正形近字误识别）…" if source_kind == "ocr"
                         else "转写预处理中（断句整理 + 识别纠错）…"),
            )
            lines = refine.refine_lines(
                lines,
                settings.llm,
                settings.subtitle,
                language_hint=detected or "",
                source=source_kind,
                log=lambda msg: job.publish("refining", job.status.progress, log=msg),
                progress=lambda f: job.publish(
                    "refining", 60 + 12 * min(max(f, 0.0), 1.0),
                    message=f"转写预处理中… {min(max(f, 0.0), 1.0):.0%}",
                ),
                should_cancel=job.cancel_event.is_set,
                network=settings.network,
                glossary=glossary,
                debug=debug,
                usage=refine_usage,
            )
            (workdir / "transcript_refined.json").write_text(
                json.dumps([l.model_dump() for l in lines], ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            job.publish("refining", 72, message=f"预处理完成，共 {len(lines)} 条字幕")


        self._finish(job, req, settings, workdir, video, debug, lines, detected,
                     glossary, shared, refine_usage, vet_usage, segments)

    # ------------------------------------------------------------ sources

    def _import_subtitle(self, job: Job, req: JobRequest, settings, debug: DebugLog):
        """Read the release's own subtitle. None means "transcribe instead".

        Returns (lines, language, "subtitle" | "ocr") — the third value says
        whether a person typed those characters or a recogniser did, which
        is what decides whether the proofreading pass runs.

        Returns None only for a batch job (see
        JobRequest.subtitle_fallback_asr): someone who picked a track by hand
        for a single film is expecting a result in seconds, and silently
        starting an hour of speech recognition instead would be the surprise
        this whole feature exists to avoid.
        """
        job.publish("importing", 0, message="读取已有字幕…")
        try:
            tracks = subsource.all_tracks(req.video_path)
            if tracks:
                job.publish(
                    "importing", 0,
                    log=f"检测到 {len(tracks)} 条可用字幕：\n"
                        + "\n".join("  " + subsource.describe_track(t) for t in tracks),
                )
            track = subsource.pick_track(
                tracks, req.subtitle_track, req.subtitle_file, req.subtitle_language
            )
            stats: dict = {}
            if track["text"]:
                kind = "subtitle"
                lines = subsource.read_cues(
                    req.video_path, track,
                    log=lambda msg: job.publish("importing", 30, log=msg),
                    stats=stats,
                )
            else:
                kind = "ocr"
                lines = self._ocr_subtitle(job, req, settings, track, debug, stats)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 — the message is the point
            if not req.subtitle_fallback_asr:
                raise
            job.publish(
                "importing", 0,
                log=f"⚠ 无法使用已有字幕（{exc}），本文件改用语音识别",
            )
            return None

        detected = stats.get("language") or (
            subsource.iso2(track["language"]) if track["language"] else ""
        )
        source = "来自轨道标签" if track["language"] else "OCR 时判定"
        if req.source_language != "auto":
            detected, source = req.source_language, "用户指定"
        elif not detected:
            detected = subsource.detect_language(lines)
            source = "按文本判定" if detected else "未能判定"
        job.publish(
            "importing", 50,
            log=f"字幕语言: {detected or '未知'}（{source}）",
        )
        if (detected and req.output_mode != "original_only" and detected
                == subsource.iso2(mux.language_of(req.target_language)[1])):
            # 纯原文除外：那个模式下 target_language 只管画面翻译，对白本来就
            # 要保持这份字幕的语言——命中正是它的定义，不是可疑信号
            # the most likely explanation is a previous run of this program
            job.publish(
                "importing", 50,
                log="⚠ 这份字幕的语言与目标语言相同，请确认它不是本软件上次生成的结果",
            )
        if debug.enabled:
            debug.section("原文来源（已有字幕）")
            debug.kv("选用", subsource.describe_track(track))
            debug.kv("读取方式", "OCR 识别" if kind == "ocr" else "直接读取文字")
            debug.kv("条数", len(lines))
            debug.kv("语言", f"{detected or '未知'}（{source}）")
            if kind == "subtitle":
                debug.kv("丢弃/合并", f"无文字内容 {stats.get('blank', 0)} 条，"
                                      f"重叠重复 {stats.get('folded', 0)} 条")
                debug.kv("已标记歌词", stats.get("lyric", 0))
                debug.kv("时间轴位移", f"{stats.get('offset', 0.0):.2f}s")
            debug.line("\n全部字幕轨：")
            debug.lines("  " + subsource.describe_track(t)
                        for t in subsource.all_tracks(req.video_path))
            debug.line("\n前 20 条：")
            debug.lines(
                f"[{l.index:4d}] {l.start:8.2f} → {l.end:8.2f} | {l.text}"
                for l in lines[:20]
            )
        job.publish(
            "importing", 60,
            message=f"已读取 {len(lines)} 条字幕（语言: {detected or '未知'}），跳过语音识别",
        )
        return lines, detected, kind

    def _ocr_subtitle(self, job: Job, req: JobRequest, settings, track: dict,
                      debug: DebugLog, stats: dict) -> List[SubtitleLine]:
        """Recognise a graphic subtitle track — Blu-ray PGS and the like."""
        job.publish(
            "importing", 5,
            message="识别图形字幕（OCR）…",
            log=f"{subsource.describe_track(track)} 是图形字幕，"
                f"将用 OCR 识别（引擎：{settings.ocr.engine}）",
        )
        language = settings.ocr.language.strip()
        if not language and req.source_language != "auto":
            language = req.source_language
        if not language and track["language"]:
            language = subsource.iso2(track["language"])
        return ocr.read_cues(
            req.video_path, track, settings.ocr,
            llm=settings.llm, network=settings.network, language=language,
            log=lambda msg: job.publish("importing", job.status.progress, log=msg),
            # OCR of a whole film runs for minutes, so it gets the bulk of
            # the stage rather than a single jump at the end
            progress=lambda f: job.publish(
                "importing", 5 + 50 * min(max(f, 0.0), 1.0),
                message=f"识别图形字幕… {min(max(f, 0.0), 1.0):.0%}",
            ),
            should_cancel=job.cancel_event.is_set,
            debug=debug,
            stats=stats,
        )

    def _transcribe(self, job: Job, req: JobRequest, settings, workdir: Path,
                    debug: DebugLog, check_cancel, vet_usage: dict):
        """Extract the audio and recognise the speech in it.

        Returns (lines, language, raw segments) — the segments only so the
        debug log can grade our lyric marking against whisper's own ♪.
        """
        # 1. extract audio ------------------------------------------------
        job.publish("extracting", 0, message="提取音频…")
        wav = workdir / "audio.wav"
        if wav.exists() and wav.stat().st_size > 0:
            # Left by a run that was interrupted later on. The directory
            # name already guarantees it came from this same source file
            # under these same settings, so there is nothing to re-check.
            job.publish("extracting", 10,
                        message="沿用上次已提取的音频",
                        log=f"↻ 沿用上次提取的音频 audio.wav"
                            f"（{wav.stat().st_size / 1048576:.1f} MB），跳过提取")
            return self._transcribe_from(job, req, settings, workdir, wav,
                                         debug, check_cancel, vet_usage)
        extract_cb = self._stage_progress(job, "extracting")

        def extract_progress(fraction: float):
            check_cancel()
            extract_cb(fraction)

        tracks = audio.list_tracks(req.video_path)
        track = audio.pick_track(tracks, req.audio_track, req.audio_language)
        job.publish(
            "extracting", 0,
            log=f"检测到 {len(tracks)} 条音轨：\n"
                + "\n".join("  " + audio.describe_track(t) for t in tracks),
        )
        if req.audio_track is not None and track["index"] != req.audio_track:
            job.publish(
                "extracting", 0,
                log=f"⚠ 指定的音轨 #{req.audio_track} 不存在，改用 {audio.describe_track(track)}",
            )
        else:
            reason = (
                "用户指定" if req.audio_track is not None
                else f"匹配语言偏好 {req.audio_language}" if req.audio_language
                else "片源默认音轨" if track["default"]
                else "第一条音轨"
            )
            job.publish(
                "extracting", 0,
                log=f"选用 {audio.describe_track(track)}（{reason}）",
            )
        if debug.enabled:
            # a sparse transcript on a multi-track disc rip is the wrong
            # track more often than it is a quiet film, and that question
            # cannot be answered from the transcript alone
            debug.section("音轨")
            debug.kv("选用", audio.describe_track(track))
            debug.line("\n全部音轨：")
            debug.lines("  " + audio.describe_track(t) for t in tracks)
        audio.extract_audio(
            req.video_path, wav,
            progress=extract_progress,
            track_index=track["index"],
            log=lambda msg: job.publish("extracting", job.status.progress, log=msg),
        )
        job.publish(
            "extracting", 10,
            log=f"音频提取完成: audio.wav（{wav.stat().st_size / 1048576:.1f} MB）",
        )
        return self._transcribe_from(job, req, settings, workdir, wav,
                                     debug, check_cancel, vet_usage)

    def _transcribe_from(self, job: Job, req: JobRequest, settings, workdir: Path,
                         wav: Path, debug: DebugLog, check_cancel,
                         vet_usage: dict):
        """Everything after the audio exists — split out so a resumed run
        can join here with the audio it already had."""
        # 2a. download the ASR model first if it's missing, with progress --
        asr_lo = 10.0
        if (settings.asr.engine == "local"
                and not settings.asr.model_path.strip()
                and not asr.is_model_cached(settings.asr.model_size)):
            asr_lo = self._download_model_with_progress(job, settings, check_cancel)

        # 2b. transcribe ----------------------------------------------------
        span = 60.0 - asr_lo
        job.publish("transcribing", asr_lo, message="语音识别中…")

        def asr_progress(fraction: float):
            job.publish(
                "transcribing",
                asr_lo + span * min(max(fraction, 0.0), 1.0),
                message=f"语音识别中… {min(max(fraction, 0.0), 1.0):.0%}",
            )

        def asr_log(msg: str):
            job.publish("transcribing", job.status.progress, log=msg)

        language = None if req.source_language == "auto" else req.source_language
        asr_usage = {"calls": 0, "prompt": 0, "completion": 0}
        segments, detected = asr.transcribe(
            str(wav),
            settings.asr,
            language=language,
            progress=asr_progress,
            log=asr_log,
            should_cancel=job.cancel_event.is_set,
            network=settings.network,
            debug=debug,
            llm=settings.llm,  # only the api engine uses it
            # ...and its tokens are reported on their own line, not folded
            # into the preprocessing total: recognition is not preprocessing
            usage=asr_usage,
        )
        if not segments:
            raise RuntimeError("未识别到任何语音内容")

        # 2b-2. vet what the second pass recovered ---------------------------
        # It re-transcribes exactly where the VAD heard nothing, which is
        # where whisper emits its subtitle-file training residue. Only a
        # model holding the film's own transcript can tell that apart from
        # the real dialogue the pass also recovers, so ask one — before
        # segmentation, while each recovered segment is still a whole
        # sentence with its provenance intact.
        if any(s.recovered for s in segments):
            job.publish("transcribing", job.status.progress, message="二次识别结果复核中…")
            segments = vet.vet_recovered(
                segments,
                settings.llm,
                language_hint=detected or "",
                synopsis=req.synopsis,
                mark_lyrics=settings.prompts.mark_lyrics,
                log=lambda msg: job.publish("transcribing", job.status.progress, log=msg),
                should_cancel=job.cancel_event.is_set,
                network=settings.network,
                debug=debug,
                usage=vet_usage,
            )
            if not segments:
                raise RuntimeError("未识别到任何语音内容")

        lines = segmenter.segment_lines(segments, settings.subtitle, debug=debug)
        if segmenter.is_effectively_unpunctuated(lines):
            # everything that decides where a sentence ends reads punctuation;
            # without it the segmenter only repairs fragments and leaves whole
            # sentences to the refine pass, which restores the punctuation first
            job.publish(
                "transcribing", job.status.progress,
                log=f"⚠ 语音识别几乎未输出句末标点（{len(lines)} 条），"
                    "已跳过整句合并，改由转写预处理补回标点后再合并"
                    + ("" if settings.prompts.refine_enabled
                       else "；但转写预处理已关闭，字幕会偏碎，建议开启"),
            )
        job.publish(
            "transcribing", 60,
            message=f"识别完成，共 {len(lines)} 条字幕（语言: {detected}）",
        )
        return lines, detected, segments

    # ------------------------------------------------------------ the rest

    def _finish(self, job: Job, req: JobRequest, settings, workdir: Path,
                video: Path, debug: DebugLog, lines: List[SubtitleLine],
                detected: str, glossary: str, shared, refine_usage: dict,
                vet_usage: dict, segments=None) -> None:
        """Mark lyrics, translate, compose, and hand back the result."""
        # 纯原文：跳过的只有对白翻译这一步，它前后的每一件事照旧——转写预处理、
        # 歌词识别、二次识别复核、OCR 校对都已经跑完了，这个模式把这条流水线
        # 当成一个高质量的转写器，而不是「把翻译关掉」。
        translate = req.output_mode != "original_only"
        # 4. mark what is sung rather than spoken -------------------------
        # After refine, because merging and splitting lines would tear a
        # ♪ … ♪ pair apart; before translation, so the lyrics can be
        # translated as lyrics. It only annotates — a failure marks nothing.
        lyrics_usage = {"calls": 0, "prompt": 0, "completion": 0, "cached": 0}
        if settings.prompts.mark_lyrics and lines:
            job.publish("refining", 72, message="歌词识别中…")
            lines = lyrics.mark_lyrics(
                lines,
                settings.llm,
                language_hint=detected or "",
                log=lambda msg: job.publish("refining", job.status.progress, log=msg),
                should_cancel=job.cancel_event.is_set,
                network=settings.network,
                debug=debug,
                usage=lyrics_usage,
            )
            lyrics.apply_marks(lines)
            self._debug_lyric_agreement(debug, lines, segments)

        # 3. translate ----------------------------------------------------
        translator = None
        # The most expensive thing here to lose. Resuming skips the series
        # glossary too, which is why the season's table is saved per
        # episode rather than at the end of the batch.
        if translate and (done := self._resume_lines(
                job, workdir / "translation.json", "翻译")):
            lines = done
            translate = False
            job.publish("translating", 95,
                        message=f"沿用上次的译文，共 {len(lines)} 条")
        if translate:
            job.publish("translating", 72, message="AI 翻译中（全局上下文）…")

            def tr_log(msg: str):
                job.publish("translating", job.status.progress, log=msg)

            if shared and len(shared):
                tr_log(f"剧集模式：沿用本批已确定的 {len(shared)} 条译名")

            translator = Translator(
                settings.llm,
                target_language=req.target_language,
                synopsis=req.synopsis,
                log=tr_log,
                progress=lambda f: job.publish(
                    "translating", 72 + 23 * min(max(f, 0.0), 1.0),
                    message=f"AI 翻译中… {min(max(f, 0.0), 1.0):.0%}",
                ),
                should_cancel=job.cancel_event.is_set,
                prompts=settings.prompts.model_copy(update={"glossary": glossary}),
                max_line_chars=settings.subtitle.max_chars_per_line,
                network=settings.network,
                debug=debug,
            )
            translator.translate(lines)
            if shared and translator.glossary_text:
                added, clashes = shared.learn(
                    translator.glossary_text, settings.prompts.glossary
                )
                # Persisted per episode, not at the end of the batch: a
                # season that spans a restart should carry the names it has
                # already settled on into the episodes still to come.
                series.save()
                tr_log(
                    f"剧集模式：本集新增 {added} 条译名，累计 {len(shared)} 条"
                    + (
                        f"\n⚠ {len(clashes)} 条与已确定的译法不一致，已沿用先前译法"
                        "（如需改用新译法，请在设置的术语表里写死）：\n  "
                        + "\n  ".join(clashes[:20])
                        if clashes else ""
                    )
                )
        else:
            # 阶段名会直接映射成前端的「AI 翻译」标签，所以这里不发 translating
            # 事件：让一个一次都不调翻译模型的任务顶着那个标签，比进度条从 72
            # 跳到 95 严重得多。进度停在 72（预处理刚做完，这就是真实状态）。
            job.publish(
                "refining", 72,
                log="纯原文模式：本次不调用翻译模型，对白保持原文。"
                    "转写预处理 / 歌词识别 / 二次识别复核 / OCR 校对均已照常执行"
                    + ("；画面翻译仍会译成目标语言" if req.frame_tasks else ""),
            )
            if shared:
                # 剧集模式写入侧的唯一输入是译名表，而纯原文不产生译名表；读取
                # 侧（合并后喂给转写预处理的 glossary）照旧生效，开关不是白开
                job.publish(
                    "refining", 72,
                    log="剧集模式：纯原文不产生新的译名表，本集不会写入本批对照表"
                        "（设置里的术语表仍然用于转写预处理）",
                )
        if settings.prompts.mark_lyrics or any(l.is_lyric for l in lines):
            # the model was asked to keep the ♪; a marker it dropped or moved
            # onto the neighbouring line would be worse than none, so the
            # flag — not the model's output — has the last word. The `any`
            # covers lyrics marking being off while an imported subtitle
            # brought its own ♪ — those were stripped on the way in and have
            # to be written back, or the source's own marks would vanish.
            #
            # 纯原文同样要跑，所以这一段刻意留在 `if translate` 之外：♪ 是
            # apply_marks 写进文本的（mark_lyrics 只置 is_lyric 标志），把它
            # 顺手塞进翻译分支里就会静默丢掉全部歌词标记，包括 subsource 从
            # 片源自带字幕里剥下来、承诺要写回去的那些。
            lyrics.apply_marks(lines)
        preprocess = _preprocess_usage(vet_usage, lyrics_usage, refine_usage)
        if translate:
            job.publish(
                "translating", 95,
                log=preprocess + "\n" + translator.report_usage(),
            )
        else:
            job.publish("refining", 72, log=preprocess)
        # 纯原文下每行的 translation 是空串——那正是这个模式的数据表达，不必
        # 换个文件名：这份快照的意义是「送进 composing 的那份行集」
        (workdir / "translation.json").write_text(
            json.dumps([l.model_dump() for l in lines], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

        # 3b. on-screen text translation (画面翻译) — best-effort per task,
        # failures must never affect the speech subtitles
        if req.frame_tasks:
            frame_lines = self._frame_lines(job, req, settings, workdir)
            if frame_lines:
                lines = sorted(lines + frame_lines, key=lambda l: l.start)

        # 4. compose subtitle file -----------------------------------------
        styled = settings.subtitle.style_enabled
        ext = ".ass" if styled else ".srt"
        # 译文模式下 sidecar 恒为 ""，下面两处文件名与改动前逐字节相同
        sidecar = _naming(req, detected)[0]
        job.publish("composing", 95, message=f"生成 {ext[1:].upper()} 字幕…")
        if styled:
            srt_text = subtitle.build_ass(lines, settings.subtitle, mode=req.output_mode)
        else:
            srt_text = subtitle.build_srt(lines, settings.subtitle, mode=req.output_mode)
        # embed mode keeps the subtitle in the work dir — the download button
        # and the debug artifacts still want it, the video folder does not
        muxed = self._embed_subtitle(job, req, workdir, video, ext, srt_text,
                                     detected=detected) \
            if req.embed_subtitle else None
        if muxed is None:
            target = video.parent / f"{video.stem}{sidecar}{ext}"
            try:
                target.write_text(srt_text, encoding="utf-8")
                job.srt_path = target
                job.status.srt_in_place = True
            except OSError as exc:
                # video dir not writable (read-only share etc.): keep it in the
                # work dir and let the UI offer a download instead
                job.srt_path = workdir / f"{video.stem}{sidecar}{ext}"
                job.srt_path.write_text(srt_text, encoding="utf-8")
                job.status.srt_in_place = False
                job.publish(
                    "composing", 99,
                    log=f"⚠ 无法写入视频所在目录（{exc}），字幕已保存到工作目录，可用下载按钮获取",
                )
        job.status.srt_filename = str(job.srt_path)
        self._debug_final(debug, lines, settings)
        job.publish(
            "done", 100,
            message=(f"完成，带字幕的视频已生成: {muxed}" if muxed
                     else f"完成，字幕已保存: {job.srt_path}"),
        )

    def _embed_subtitle(
        self, job: Job, req: JobRequest, workdir: Path,
        video: Path, ext: str, srt_text: str, detected: str = "",
    ) -> Optional[Path]:
        """Mux the subtitle into a new video. Returns None to fall back.

        A failure here must never cost the hour of transcription and
        translation behind it, so anything short of a cancellation degrades
        to writing the subtitle file — the behaviour with the switch off.

        A source with no picture is one such failure, decided before anything
        is written: muxing it would produce a video file with nothing to watch.

        *detected* is the language of the original text; only 纯原文 uses it.
        There the track is in the film's own language, and the file name, the
        stream tag and the track title all follow it rather than the
        translation target (see _naming).
        """
        if job.audio_only:
            job.publish(
                "composing", 95,
                log="⚠ 纯音频片源没有画面，无法合成带字幕的视频，已改为生成字幕文件",
            )
            return None
        sidecar, suffix, lang, title = _naming(req, detected)
        # 工作目录里这一份也带后缀：它就是「下载字幕」按钮吐给用户的文件名
        job.srt_path = workdir / f"{video.stem}{sidecar}{ext}"
        job.srt_path.write_text(srt_text, encoding="utf-8")
        job.status.srt_in_place = False
        out = mux.output_path(video, req.target_language, req.embed.container,
                              suffix=suffix)
        recoding = req.embed.video_codec != mux.COPY
        how = (f"重编码为 {mux.encoder_label(req.embed.video_codec)}"
               if recoding else "复制音视频流，不重编码")
        job.publish(
            "composing", 95,
            message=f"合成带字幕的视频（{how}）…",
            log=f"开始合成 {out.name}（{how}）",
        )
        if recoding:
            job.publish(
                "composing", 95,
                log="⚠ 重编码要把整部影片重新压一遍：软件编码常需数小时，"
                    "画质只减不增。中途可以取消，不会留下半成品。",
            )
        started = time.monotonic()
        try:
            mux.embed(
                video, job.srt_path, out,
                target_language=req.target_language,
                opts=req.embed,
                track_title=title,
                track_language=lang,
                log=lambda msg: job.publish("composing", job.status.progress, log=msg),
                progress=lambda f: job.publish(
                    "composing", 95 + 5 * min(max(f, 0.0), 1.0),
                    # The bar only has 95->100 to give, while a re-encode is
                    # the longest step of the whole job. The honest fix is to
                    # put the real number and an estimate in the text.
                    message=f"合成带字幕的视频… {min(max(f, 0.0), 1.0):.0%}"
                            + _eta(started, f, recoding),
                ),
                should_cancel=job.cancel_event.is_set,
            )
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 — any failure falls back
            job.publish(
                "composing", 95,
                log=f"⚠ 合成视频失败（{exc}），已改为在视频所在目录生成字幕文件",
            )
            return None
        job.status.video_filename = str(out)
        return out

    @staticmethod
    def _debug_lyric_agreement(debug: DebugLog, lines, segments) -> None:
        """Score our lyric judgement against whisper's own ♪ prefixes.

        Whisper's marking is worthless as an answer — measured across one
        film it appeared in 0 of 434 first-pass segments and 90 of 216
        second-pass ones, tracking the decoding mode rather than the singing.
        But it is an *independent* opinion, and that is enough to grade this
        pass without a human subtitle: the agreements are the confident part,
        and the disagreements are the list worth listening to.

        *segments* is None when the text came from a subtitle rather than
        from whisper: there is no second opinion to compare against, and the
        subtitle's own ♪ have already been believed outright.
        """
        if not debug.enabled or segments is None:
            return
        from app.core.debuglog import fmt_time

        marked = [(s.start, s.end) for s in segments if lyrics.MARK in s.text]
        both, ours, theirs, ours_alone, theirs_alone = lyrics.whisper_agreement(
            lines, marked
        )
        debug.section("歌词识别 vs whisper 自带的 ♪（交叉比对）")
        debug.kv("whisper 标了 ♪ 的 segment", len(marked))
        debug.kv("两边都判为歌词", both)
        debug.kv("只有本程序判为歌词", ours)
        debug.kv("只有 whisper 标了 ♪", theirs)
        debug.line(
            "\nwhisper 的 ♪ 只在关闭 VAD 的二次识别里出现，本身不可信；"
            "\n它的价值在于是一个独立意见——下面两份分歧清单就是需要人工听的部分。\n"
        )
        for title, rows in (
            ("只有本程序判为歌词（可能是它标对了 whisper 漏了，也可能是误标）", ours_alone),
            ("只有 whisper 标了 ♪（可能是本程序漏标）", theirs_alone),
        ):
            debug.line(f"\n{title}：{len(rows)} 行")
            debug.lines(f"  [{l.index}] {fmt_time(l.start)} {l.text}" for l in rows)

    @staticmethod
    def _debug_final(debug: DebugLog, lines: List[SubtitleLine], settings) -> None:
        """Final cues plus the checks that only apply to the finished file."""
        if not debug.enabled:
            return
        from app.core.debuglog import fmt_cue, percentiles

        wrap = settings.subtitle.max_chars_per_line
        overflow = [
            l for l in lines
            if any(
                len(part) > wrap
                for text in (l.text, l.translation)
                for part in subtitle.wrap_display_text(text, wrap)
            )
        ]
        debug.section("最终字幕")
        debug.kv("条数", len(lines))
        debug.kv("cue 时长(s)", percentiles([l.end - l.start for l in lines]))
        debug.kv("相邻间隔(s)", percentiles(
            [b.start - a.end for a, b in zip(lines, lines[1:])]
        ))
        debug.kv(f"换行后仍超过 {wrap} 字的 cue", f"{len(overflow)} / {len(lines)}")
        debug.kv("未完句结尾占比", f"{segmenter.open_ended_ratio(lines):.0%}")
        debug.kv("碎片 cue", f"{segmenter.fragment_count(lines)} / {len(lines)}")
        debug.line("\n最终结果（原文 ⇒ 译文）：")
        debug.lines(
            fmt_cue(l.index, l.start, l.end, f"{l.text}  ⇒  {l.translation}")
            for l in lines
        )





def _preprocess_usage(vet_usage: dict, lyrics_usage: dict, refine_usage: dict) -> str:
    """翻译之外那几个 LLM 环节的 token 汇总——纯原文模式下它就是全部。"""
    return (
        (_usage_line("二次识别复核", vet_usage) + "\n" if vet_usage["calls"] else "")
        + (_usage_line("歌词识别", lyrics_usage) + "\n" if lyrics_usage["calls"] else "")
        + _usage_line("转写预处理", refine_usage)
    )


def _naming(req: JobRequest, detected: str) -> tuple[str, str, str, str]:
    """这个任务的产物怎么命名：(字幕文件后缀, 内嵌视频后缀, 轨道标签, 轨道标题)。

    译文模式沿用历史命名——字幕一直是与片源同名的 片名.srt。**那一侧一个字都
    不能改**，改了所有已经指向它的播放器 / 媒体库全部对不上。

    纯原文两侧都带语言后缀（片名.ja.srt / 片名.ja.mkv）：它的产物极可能和一份
    译文字幕落在同一个目录里，不加后缀就是互相覆盖。
    """
    if req.output_mode == "original_only":
        suffix, tag, title = mux.source_language_of(detected)
        return f".{suffix}", suffix, tag, title
    suffix, tag = mux.language_of(req.target_language)
    return "", suffix, tag, f"{req.target_language}字幕"


def _usage_line(label: str, usage: dict) -> str:
    if not usage["calls"]:
        return f"{label}未记录 token 用量（服务端未返回 usage）"
    return (
        f"{label}共 {usage['calls']} 次请求：输入 {usage['prompt']:,} tokens"
        f"（其中缓存命中 {usage['cached']:,}）、输出 {usage['completion']:,} tokens"
    )
manager = JobManager()

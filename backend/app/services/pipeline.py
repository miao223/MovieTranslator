"""Job orchestration: state machine, background execution, SSE progress.

Whisper is CPU-bound, so only one job runs at a time (global semaphore).
Intermediate artifacts are written to the per-job cache dir for debugging
and are wiped on the next application startup.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from app.core import config
from app.core.cache import job_dir
from app.models.schemas import (
    JobRequest,
    JobStatus,
    ProgressEvent,
    SubtitleLine,
)
from app.services import asr, audio, segmenter, subtitle
from app.services.translator import Translator

# overall progress ranges per stage: (start%, end%)
STAGE_RANGES = {
    "extracting": (0.0, 10.0),
    "transcribing": (10.0, 60.0),
    "translating": (60.0, 95.0),
    "composing": (95.0, 100.0),
}

_run_slot = threading.Semaphore(1)  # one CPU-heavy job at a time


class Job:
    def __init__(self, request: JobRequest):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = JobStatus(id=self.id, video_path=request.video_path)
        self.cancel_event = threading.Event()
        self.events: List[ProgressEvent] = []
        self.subscribers: List[queue.Queue] = []
        self.lock = threading.Lock()
        self.srt_path: Optional[Path] = None

    # -------------------------------------------------------- events

    def publish(self, stage: str, progress: float, message: str = "", log: str = ""):
        self.status.stage = stage  # type: ignore[assignment]
        self.status.progress = round(progress, 1)
        if message:
            self.status.message = message
        event = ProgressEvent(
            stage=stage, progress=self.status.progress, message=message, log=log
        )
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


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def create(self, request: JobRequest) -> Job:
        video = Path(request.video_path)
        if not video.is_file():
            raise FileNotFoundError(f"视频文件不存在: {video}")
        job = Job(request)
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

    def _stage_progress(self, job: Job, stage: str):
        lo, hi = STAGE_RANGES[stage]

        def cb(fraction: float, log: str = ""):
            job.publish(stage, lo + (hi - lo) * min(max(fraction, 0.0), 1.0), log=log)

        return cb

    def _run(self, job: Job) -> None:
        req = job.request
        workdir = job_dir(job.id)
        job.publish("pending", 0, message="排队中…")
        with _run_slot:
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

    def _execute(self, job: Job, req: JobRequest, workdir: Path) -> None:
        settings = config.load_settings()

        def check_cancel():
            if job.cancel_event.is_set():
                raise InterruptedError

        # 1. extract audio ------------------------------------------------
        job.publish("extracting", 0, message="提取音频…")
        wav = workdir / "audio.wav"
        extract_cb = self._stage_progress(job, "extracting")

        def extract_progress(fraction: float):
            check_cancel()
            extract_cb(fraction)

        audio.extract_audio(req.video_path, wav, progress=extract_progress)
        job.publish(
            "extracting", 10,
            log=f"音频提取完成: audio.wav（{wav.stat().st_size // 1048576} MB）",
        )

        # 2a. download the ASR model first if it's missing, with progress --
        asr_lo = 10.0
        if not settings.asr.model_path.strip() and not asr.is_model_cached(
            settings.asr.model_size
        ):
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
        segments, detected = asr.transcribe(
            str(wav),
            settings.asr,
            language=language,
            progress=asr_progress,
            log=asr_log,
            should_cancel=job.cancel_event.is_set,
            network=settings.network,
        )
        if not segments:
            raise RuntimeError("未识别到任何语音内容")
        lines = segmenter.segment_lines(segments, settings.subtitle)
        (workdir / "transcript.json").write_text(
            json.dumps([l.model_dump() for l in lines], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        job.publish(
            "transcribing", 60,
            message=f"识别完成，共 {len(lines)} 条字幕（语言: {detected}）",
        )

        # 3. translate ----------------------------------------------------
        job.publish("translating", 60, message="AI 翻译中（全局上下文）…")

        def tr_log(msg: str):
            job.publish("translating", job.status.progress, log=msg)

        translator = Translator(
            settings.llm,
            target_language=req.target_language,
            synopsis=req.synopsis,
            log=tr_log,
            progress=lambda f: job.publish(
                "translating", 60 + 35 * min(max(f, 0.0), 1.0),
                message=f"AI 翻译中… {min(max(f, 0.0), 1.0):.0%}",
            ),
            should_cancel=job.cancel_event.is_set,
            prompts=settings.prompts,
            max_line_chars=settings.subtitle.max_chars_per_line,
            network=settings.network,
        )
        translator.translate(lines)
        (workdir / "translation.json").write_text(
            json.dumps([l.model_dump() for l in lines], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

        # 4. compose SRT ----------------------------------------------------
        job.publish("composing", 95, message="生成 SRT…")
        srt_text = subtitle.build_srt(lines, settings.subtitle, mode=req.output_mode)
        video = Path(req.video_path)
        target = video.parent / f"{video.stem}.srt"
        try:
            target.write_text(srt_text, encoding="utf-8")
            job.srt_path = target
            job.status.srt_in_place = True
        except OSError as exc:
            # video dir not writable (read-only share etc.): keep it in the
            # work dir and let the UI offer a download instead
            job.srt_path = workdir / f"{video.stem}.srt"
            job.srt_path.write_text(srt_text, encoding="utf-8")
            job.status.srt_in_place = False
            job.publish(
                "composing", 99,
                log=f"⚠ 无法写入视频所在目录（{exc}），字幕已保存到工作目录，可用下载按钮获取",
            )
        job.status.srt_filename = str(job.srt_path)
        job.publish("done", 100, message=f"完成，字幕已保存: {job.srt_path}")


manager = JobManager()

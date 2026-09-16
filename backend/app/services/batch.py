"""Batch translation: scan a directory and queue one job per video.

No extra queueing machinery is needed — every job's thread blocks on the
pipeline's global run slot, so batch members execute sequentially in
creation order.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.media import scan_media
from app.models.schemas import AppSettings, BatchRequest, BatchStatus, JobRequest
from app.services import series
from app.services.pipeline import manager as job_manager

TERMINAL = {"done", "failed", "cancelled"}


@dataclass
class Batch:
    id: str
    directory: str
    job_ids: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed_to_create: List[str] = field(default_factory=list)
    # series mode: the id of the glossary these jobs share, "" when off
    series_id: str = ""


class BatchManager:
    def __init__(self):
        self.batches: Dict[str, Batch] = {}

    def create(self, req: BatchRequest,
               settings: Optional[AppSettings] = None) -> BatchStatus:
        """Fan the directory out into one job per file.

        *settings* is the frozen snapshot every file of this batch runs
        with. One snapshot for the whole batch on purpose: editing settings
        while a season is half done used to change the parameters from that
        episode on, while the series glossary assumes all of them match.
        """
        videos, skipped, _shadowed, _with_source = scan_media(
            req.directory, req.recursive, req.skip_existing_srt,
            req.target_language,
        )
        if not videos:
            raise ValueError("目录中没有需要翻译的视频或音频文件")
        batch = Batch(
            id=uuid.uuid4().hex[:12],
            directory=req.directory,
            skipped=[str(p) for p in skipped],
        )
        if req.series_mode:
            # jobs run one at a time in creation order, so the glossary is
            # written by whichever episode is running and read by the next
            batch.series_id = batch.id
            series.create(batch.series_id)
        for video in videos:
            try:
                job = job_manager.create(
                    JobRequest(
                        video_path=str(video),
                        audio_language=req.audio_language,
                        text_source=req.text_source,
                        subtitle_language=req.subtitle_language,
                        # a season where one episode ships without a subtitle
                        # should translate that episode, not stop the batch
                        subtitle_fallback_asr=True,
                        source_language=req.source_language,
                        target_language=req.target_language,
                        synopsis=req.synopsis,
                        output_mode=req.output_mode,
                        embed_subtitle=req.embed_subtitle,
                        embed=req.embed,
                        series_id=batch.series_id,
                    ),
                    settings=settings,
                )
                batch.job_ids.append(job.id)
            except Exception as exc:  # noqa: BLE001 — one bad file must not kill the batch
                batch.failed_to_create.append(f"{video}: {exc}")
        self.batches[batch.id] = batch
        return self.status(batch.id)

    def status(self, batch_id: str) -> BatchStatus:
        if batch_id not in self.batches:
            raise KeyError(batch_id)
        batch = self.batches[batch_id]
        jobs = [job_manager.get(jid).status for jid in batch.job_ids]
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0}
        current = ""
        for status in jobs:
            if status.stage == "done":
                counts["done"] += 1
            elif status.stage == "failed":
                counts["failed"] += 1
            elif status.stage == "cancelled":
                counts["cancelled"] += 1
            elif status.stage == "pending":
                counts["pending"] += 1
            else:
                counts["running"] += 1
                current = status.id
        shared = series.get(batch.series_id)
        return BatchStatus(
            id=batch.id,
            directory=batch.directory,
            total=len(jobs),
            current_job_id=current,
            jobs=jobs,
            skipped=batch.skipped + batch.failed_to_create,
            glossary=shared.render() if shared else "",
            **counts,
        )

    def cancel(self, batch_id: str) -> None:
        if batch_id not in self.batches:
            raise KeyError(batch_id)
        for jid in self.batches[batch_id].job_ids:
            job = job_manager.get(jid)
            if job.status.stage not in TERMINAL:
                job.cancel_event.set()


batch_manager = BatchManager()

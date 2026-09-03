"""A release shared as its audio track alone, used as a source.

Two features need a picture and so must be off for these files: muxing the
subtitle into a new video, and translating on-screen text. The first degrades
quietly (a mixed folder must not stop over one mp3), the second is refused up
front (its whole output *is* the picture).
"""

from pathlib import Path

import pytest

from app.models.schemas import JobRequest, SubtitleSettings
from app.services import audio, subtitle
from app.services.pipeline import Job, manager
from tests.test_audio_tracks import make_audio_file, make_multitrack_video


@pytest.fixture
def audio_file(tmp_path):
    return make_audio_file(tmp_path / "ep1.m4a")


@pytest.fixture
def video_file(tmp_path):
    return make_multitrack_video(tmp_path / "film.mkv")


@pytest.fixture(autouse=True)
def logs_in_tmp(tmp_path, monkeypatch):
    """Every job here writes a log file; none of them may land in the
    user's real log folder."""
    from app.core import joblog

    monkeypatch.setattr(joblog, "_base_dir", lambda: tmp_path / "logs")


@pytest.fixture
def job_factory():
    def make(source: Path, **kwargs):
        return Job(JobRequest(video_path=str(source), **kwargs))

    return make


# ------------------------------------------------------------ has_picture


def test_has_picture_answers_by_content_not_by_name(tmp_path, video_file):
    """The suffix decides what gets listed; this decides what gets run.

    Both directions of the lie are real: a release can ship its audio inside
    an .mka, and a great many audio files carry album art — which is a video
    stream, so `streams.video` would call them films.
    """
    assert audio.has_picture(video_file) is True
    assert audio.has_picture(make_audio_file(tmp_path / "a.m4a")) is False
    assert audio.has_picture(make_audio_file(tmp_path / "solo.mka")) is False
    assert audio.has_picture(
        make_audio_file(tmp_path / "art.mp3", codec="mp3", cover=True)
    ) is False


def test_an_unreadable_file_falls_back_to_the_extension(tmp_path):
    """Nothing to probe yet: the pipeline should report the real problem
    later, not fail inside a question about pictures."""
    assert audio.has_picture(tmp_path / "gone.mp3") is False
    assert audio.has_picture(tmp_path / "gone.mkv") is True


def test_a_job_works_it_out_for_itself(audio_file, video_file, job_factory):
    """Jobs are built directly in plenty of places, not only by create(); a
    default of False there would let every one of them think it has a picture."""
    assert job_factory(audio_file).audio_only is True
    assert job_factory(video_file).audio_only is False


# ------------------------------------------------------- create() refusals


def test_frame_only_is_refused_for_an_audio_source(audio_file, video_file):
    """Unlike embed, there is nothing here to degrade to: the whole product
    of this mode is translated on-screen text."""
    with pytest.raises(ValueError, match="纯音频"):
        manager.create(JobRequest(video_path=str(audio_file), frame_only=True))
    # the same request against a film gets past this check (and fails later
    # for its own reason: there is no subtitle to supplement)
    job = manager.create(JobRequest(video_path=str(video_file), frame_only=True))
    job.cancel_event.set()


def test_the_api_reports_that_refusal_as_a_400(audio_file):
    from tests.conftest import local_client

    from app.main import app

    with local_client(app) as c:
        r = c.post("/api/jobs", json={
            "video_path": str(audio_file), "frame_only": True,
        })
    assert r.status_code == 400 and "纯音频" in r.json()["detail"]


def test_an_encoder_this_machine_lacks_still_refuses_a_film(video_file):
    """Guarding the pair below: the encoder check must keep working where
    it matters, or skipping it for audio has quietly disabled it."""
    with pytest.raises(ValueError, match="编码器"):
        manager.create(JobRequest(
            video_path=str(video_file), embed_subtitle=True,
            embed={"video_codec": "h264_nonexistent"},
        ))


def test_an_audio_source_is_not_refused_over_an_encoder_it_never_reaches(
    audio_file, monkeypatch
):
    """It degrades to a subtitle file, so it must not be turned away first —
    that would break the promise a mixed folder relies on."""
    from app.services import pipeline

    monkeypatch.setattr(pipeline.JobManager, "_run", lambda self, job: None)
    job = manager.create(JobRequest(
        video_path=str(audio_file), embed_subtitle=True,
        embed={"video_codec": "h264_nonexistent"},
    ))
    assert job.audio_only is True


def test_the_job_says_so_before_the_hour_of_work_not_after(
    audio_file, tmp_path, job_factory, monkeypatch
):
    """Someone waiting for a new video should not find out at 95% that there
    was never going to be one — the same reason the encoder check sits in
    create() rather than at mux time."""
    from app.services import pipeline

    def stop_here(*args, **kwargs):
        raise InterruptedError

    monkeypatch.setattr(pipeline.JobManager, "_transcribe", stop_here)
    job = job_factory(audio_file, embed_subtitle=True, frame_tasks=[
        {"time": "00:10", "note": ""},
    ])
    with pytest.raises(InterruptedError):
        manager._execute(job, job.request, tmp_path)

    logs = [e.log for e in job.events]
    assert any("忽略「合成带字幕的新视频」" in l for l in logs)
    assert any("将跳过 1 条画面翻译" in l for l in logs)
    assert job.logfile.path.read_text(encoding="utf-8").count("纯音频") >= 2


# ---------------------------------------------------------- the mux stage


def test_the_embed_stage_degrades_to_a_subtitle_file(
    audio_file, tmp_path, job_factory
):
    """No picture to put the subtitle over, so the switch is ignored and the
    file lands beside the audio — with the reason on the record."""
    job = job_factory(audio_file, embed_subtitle=True, target_language="简体中文")
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = manager._embed_subtitle(
        job, job.request, workdir, audio_file, ".srt",
        subtitle.build_srt(_lines(), SubtitleSettings()),
    )

    assert out is None  # _finish writes the subtitle itself from here
    assert job.status.video_filename == ""
    assert list(audio_file.parent.glob("*.mkv")) == []
    assert list(audio_file.parent.glob("*.part")) == []
    assert any("纯音频" in e.log for e in job.events)


def test_a_film_in_the_same_stage_is_still_muxed(video_file, tmp_path, job_factory):
    """The guard is about the source, not about the stage."""
    job = job_factory(video_file, embed_subtitle=True, target_language="简体中文")
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = manager._embed_subtitle(
        job, job.request, workdir, video_file, ".srt",
        subtitle.build_srt(_lines(), SubtitleSettings()),
    )
    assert out is not None and out.is_file()


# ------------------------------------------------------- frame translation


def test_frame_tasks_are_skipped_rather_than_left_to_fail(
    tmp_path, job_factory, monkeypatch
):
    """Not left to per-task failure: an audio file with album art has a video
    stream, so every task would cheerfully translate the cover instead."""
    from app.services import pipeline

    def boom(*args, **kwargs):
        raise AssertionError("frames must not be translated for an audio source")

    monkeypatch.setattr(pipeline.JobManager, "_translate_frames", boom)
    art = make_audio_file(tmp_path / "art.mp3", codec="mp3", cover=True)
    job = job_factory(art, frame_tasks=[
        {"time": "00:10", "note": ""}, {"time": "00:20", "note": ""},
    ])
    assert job.audio_only is True

    assert manager._frame_lines(job, job.request, None, tmp_path) == []
    assert any("已跳过 2 条画面翻译" in e.log for e in job.events)


def test_a_film_still_gets_its_frames_translated(
    video_file, tmp_path, job_factory, monkeypatch
):
    """The guard must not have turned the feature off for everyone."""
    from app.services import pipeline

    called = []
    monkeypatch.setattr(
        pipeline.JobManager, "_translate_frames",
        lambda self, *a: called.append(a) or [],
    )
    job = job_factory(video_file, frame_tasks=[{"time": "00:10", "note": ""}])
    manager._frame_lines(job, job.request, None, tmp_path)
    assert called


def test_extracting_a_frame_from_album_art_is_refused(tmp_path):
    """The cover is a video stream. Answering with it would look like the
    frame translation worked."""
    from app.services import frame

    art = make_audio_file(tmp_path / "art.mp3", codec="mp3", cover=True)
    with pytest.raises(ValueError, match="画面流"):
        frame.extract_frame(art, 1.0, tmp_path / "out.jpg")


def _lines():
    from app.models.schemas import SubtitleLine

    return [SubtitleLine(index=1, start=0.0, end=1.0, text="hi", translation="嗨")]

"""Per-job log files: content, retention, secrecy and download endpoints."""

import pytest

from app.core import joblog


@pytest.fixture
def logdir(tmp_path, monkeypatch):
    """Redirect the log folder so tests never touch the real cache dir."""
    monkeypatch.setattr(joblog, "_base_dir", lambda: tmp_path)
    return tmp_path / joblog.LOG_DIR_NAME


def test_writer_records_header_and_events(logdir):
    w = joblog.JobLogWriter("abc123", r"D:\Movies\film.mkv")
    w.event("extracting", 0.0, "提取音频…", "")
    w.event("failed", 20.0, "失败: boom", "Traceback…\n  line 2")
    text = w.path.read_text(encoding="utf-8")

    assert "任务 ID       : abc123" in text
    assert r"D:\Movies\film.mkv" in text
    assert "[extracting     0.0%] 提取音频…" in text
    # multi-line logs (tracebacks) keep one prefixed line each
    assert "[failed        20.0%] Traceback…" in text
    assert "[failed        20.0%]   line 2" in text


def test_api_key_is_never_written(logdir):
    from app.models.schemas import AppSettings

    settings = AppSettings()
    settings.llm.api_key = "sk-super-secret-value"
    w = joblog.JobLogWriter("k1", "x.mkv")
    w.write_settings(settings)
    text = w.path.read_text(encoding="utf-8")

    assert "sk-super-secret-value" not in text
    assert "API key: 已配置" in text


def test_media_section_lists_every_track(logdir, tmp_path):
    from tests.test_audio_tracks import make_multitrack_video

    video = make_multitrack_video(tmp_path / "dual.mkv")
    w = joblog.JobLogWriter("m1", str(video))
    w.write_media(str(video))
    text = w.path.read_text(encoding="utf-8")

    assert "容器格式" in text and "matroska" in text
    assert "音轨 #1 日语" in text and "音轨 #2 英语" in text
    assert "视频流" in text


def test_media_section_survives_an_unreadable_file(logdir, tmp_path):
    broken = tmp_path / "not-a-video.mkv"
    broken.write_bytes(b"garbage")
    w = joblog.JobLogWriter("m2", str(broken))
    w.write_media(str(broken))  # must not raise
    assert "探测失败" in w.path.read_text(encoding="utf-8")


def test_pruning_keeps_only_the_newest(logdir):
    import os
    import time

    for i in range(6):
        joblog.JobLogWriter(f"job{i}", "x.mkv")
        os.utime(joblog.find_log(f"job{i}"), (time.time() + i, time.time() + i))
    joblog.prune_logs(keep=3)
    remaining = sorted(p.name.split("_")[1] for p in logdir.glob("*.log"))
    assert remaining == ["job3.log", "job4.log", "job5.log"]


def test_logs_survive_the_cache_wipe(logdir, tmp_path, monkeypatch):
    from app.core import cache

    monkeypatch.setattr(cache, "_base_dir", lambda: tmp_path)
    joblog.JobLogWriter("keepme", "x.mkv")
    (tmp_path / "jobs").mkdir(exist_ok=True)
    (tmp_path / "jobs" / "scratch.wav").write_bytes(b"x")

    cache.clear_cache()

    assert not (tmp_path / "jobs" / "scratch.wav").exists()
    assert joblog.find_log("keepme") is not None


# ------------------------------------------------------------- endpoints


def test_log_endpoints(logdir):
    from fastapi.testclient import TestClient

    from app.main import app

    w = joblog.JobLogWriter("endpoint1", "x.mkv")
    w.event("done", 100.0, "完成", "")

    with TestClient(app) as client:
        listing = client.get("/api/logs").json()
        assert listing["dir"] == str(logdir)
        assert any(f["name"] == w.path.name for f in listing["files"])

        by_job = client.get("/api/logs/job/endpoint1")
        assert by_job.status_code == 200 and "完成" in by_job.text

        assert client.get("/api/logs/job/nosuchjob").status_code == 404

        by_name = client.get(f"/api/logs/file/{w.path.name}")
        assert by_name.status_code == 200

        # a traversal attempt never yields a file from outside the folder
        escaped = client.get("/api/logs/file/..%2F..%2Fetc%2Fpasswd")
        assert "root:" not in escaped.text


def test_download_log_rejects_paths_outside_the_folder(logdir):
    """The route argument is reduced to a bare filename before use."""
    from fastapi import HTTPException

    from app.api.routes import download_log

    for name in ("../../etc/passwd", "/etc/passwd", "..\\..\\windows\\win.ini"):
        with pytest.raises(HTTPException) as err:
            download_log(name)
        assert err.value.status_code == 404

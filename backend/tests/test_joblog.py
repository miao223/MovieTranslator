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


def test_the_access_token_is_never_written_either(logdir):
    """The log is the file users are told to send to the developer, so a
    token in it is the same mistake as a key in it."""
    from app.models.schemas import AppSettings

    settings = AppSettings()
    settings.server.lan_access = True
    settings.server.access_token = "tok-super-secret-value"
    settings.mcp.enabled = True
    w = joblog.JobLogWriter("k2", "x.mkv")
    w.write_settings(settings)
    text = w.path.read_text(encoding="utf-8")

    assert "tok-super-secret-value" not in text
    # but the posture itself must be visible: it explains a lot of reports
    assert "局域网访问    : 开" in text and "需要令牌" in text
    assert "MCP 服务      : 开" in text


def test_the_shipped_defaults_are_closed(logdir):
    """Doing nothing must leave the app exactly where it was: loopback
    only, no MCP. This is the assertion that catches a careless default."""
    from app.models.schemas import AppSettings

    settings = AppSettings()
    assert settings.server.lan_access is False
    assert settings.mcp.enabled is False
    assert settings.server.require_token is True
    assert settings.prompts.mark_lyrics is True  # this one ships on

    w = joblog.JobLogWriter("k3", "x.mkv")
    w.write_settings(settings)
    text = w.path.read_text(encoding="utf-8")
    assert "局域网访问    : 关（仅本机 127.0.0.1）" in text
    assert "MCP 服务      : 关" in text


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
    from tests.conftest import local_client

    from app.main import app

    w = joblog.JobLogWriter("endpoint1", "x.mkv")
    w.event("done", 100.0, "完成", "")

    with local_client(app) as client:
        listing = client.get("/api/logs").json()
        assert listing["dir"] == str(logdir)
        assert any(f["name"] == w.path.name for f in listing["files"])

        by_job = client.get("/api/logs/job/endpoint1")
        assert by_job.status_code == 200 and "完成" in by_job.text

        assert client.get("/api/logs/job/nosuchjob").status_code == 404

        by_name = client.get(f"/api/logs/file/{w.path.name}")
        assert by_name.status_code == 200

        # a traversal attempt never yields a file from outside the folder,
        # and says so rather than handing back the page shell — an /api
        # path that answers 200 with HTML reaches api.js as a JSON syntax
        # error instead of "no such endpoint"
        escaped = client.get("/api/logs/file/..%2F..%2Fetc%2Fpasswd")
        assert "root:" not in escaped.text
        assert escaped.status_code == 404
        assert client.get("/api/nosuchendpoint").status_code == 404
        # …while a client-side route still gets the shell
        assert client.get("/settings").status_code == 200


def test_download_log_rejects_paths_outside_the_folder(logdir):
    """The route argument is reduced to a bare filename before use."""
    from fastapi import HTTPException

    from app.api.routes import download_log

    for name in ("../../etc/passwd", "/etc/passwd", "..\\..\\windows\\win.ini"):
        with pytest.raises(HTTPException) as err:
            download_log(name)
        assert err.value.status_code == 404


def test_version_endpoint_reports_the_running_build():
    """The header reads this to show which build answered the request."""
    from tests.conftest import local_client

    from app.core.joblog import APP_VERSION
    from app.main import app

    with local_client(app) as client:
        body = client.get("/api/version").json()
    assert body["version"] == APP_VERSION
    assert body["version"][0].isdigit()

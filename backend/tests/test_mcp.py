"""The MCP surface: what a remote client can and cannot make this do.

Tools are exercised through a real in-memory MCP session rather than by
calling the functions directly — the schema the SDK derives from the
signature is half the contract, and a client only ever sees that.
"""

import json

import pytest

from app.models.schemas import JobStatus
from app.services import mcp_server
from tests.conftest import local_client, remote_client


@pytest.fixture
def mcp():
    server = mcp_server.build()
    assert server is not None, "mcp SDK should be installed"
    return server


async def call(server, name: str, **args) -> dict:
    """Invoke a tool the way a client would, and unwrap the text payload."""
    result = await server.call_tool(name, args)
    payload = result[0] if isinstance(result, tuple) else result
    if isinstance(payload, dict):
        return payload
    text = payload[0].text if isinstance(payload, list) else payload
    return json.loads(text)


class FakeJob:
    def __init__(self, job_id="abc123", stage="done", srt=None):
        self.id = job_id
        self.srt_path = srt
        self.status = JobStatus(
            id=job_id, stage=stage, progress=100.0, video_path="/v/film.mkv",
            srt_filename=str(srt) if srt else "",
        )


# ------------------------------------------------------------------ shape


@pytest.mark.anyio
async def test_the_advertised_tools_are_the_agreed_set(mcp):
    """Settings are deliberately absent: a client that could rewrite the
    ASR model or the LLM key from a sentence of prose is a far larger
    blast radius than one that can only run jobs."""
    names = {t.name for t in await mcp.list_tools()}
    assert names == {
        "list_videos", "list_audio_tracks", "get_server_status",
        "translate_video", "translate_directory",
        "get_job", "get_batch", "cancel_job", "cancel_batch",
        "get_subtitle", "get_job_log",
    }
    assert not any("setting" in n for n in names)


@pytest.mark.anyio
async def test_every_tool_explains_itself(mcp):
    """The description is the whole basis on which a client picks a tool."""
    for tool in await mcp.list_tools():
        assert tool.description, tool.name


# ------------------------------------------------------------- job control


@pytest.mark.anyio
async def test_translate_video_returns_at_once_with_a_job_id(mcp, monkeypatch):
    """A film takes hours; the tool must hand back a handle, not block."""
    seen = {}

    def fake_create(request):
        seen["request"] = request
        return FakeJob(stage="pending")

    monkeypatch.setattr(mcp_server.manager, "create", fake_create)
    out = await call(
        mcp, "translate_video", video_path="/v/film.mkv",
        source_language="ja", synopsis="一部恐怖片", audio_track=2,
    )
    assert out["job_id"] == "abc123" and out["finished"] is False
    assert seen["request"].source_language == "ja"
    assert seen["request"].audio_track == 2
    assert seen["request"].synopsis == "一部恐怖片"


@pytest.mark.anyio
async def test_a_missing_file_comes_back_as_a_message_not_a_crash(mcp, monkeypatch):
    def boom(request):
        raise FileNotFoundError("视频文件不存在: /v/gone.mkv")

    monkeypatch.setattr(mcp_server.manager, "create", boom)
    out = await call(mcp, "translate_video", video_path="/v/gone.mkv")
    assert "不存在" in out["error"]


@pytest.mark.anyio
async def test_a_bad_output_mode_is_rejected_before_a_job_starts(mcp, monkeypatch):
    def fail(request):  # pragma: no cover — must not be reached
        raise AssertionError("job should not have been created")

    monkeypatch.setattr(mcp_server.manager, "create", fail)
    out = await call(mcp, "translate_video", video_path="/v/f.mkv", output_mode="srt")
    assert "output_mode" in out["error"]


@pytest.mark.anyio
async def test_unknown_ids_report_instead_of_raising(mcp, monkeypatch):
    def missing(job_id):
        raise KeyError(job_id)

    monkeypatch.setattr(mcp_server.manager, "get", missing)
    monkeypatch.setattr(mcp_server.manager, "cancel", missing)
    assert "没有这个任务" in (await call(mcp, "get_job", job_id="nope"))["error"]
    assert "没有这个任务" in (await call(mcp, "cancel_job", job_id="nope"))["error"]


# ---------------------------------------------------------------- results


@pytest.mark.anyio
async def test_get_subtitle_returns_the_file_contents(mcp, monkeypatch, tmp_path):
    srt = tmp_path / "film.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n你好\n", encoding="utf-8")
    monkeypatch.setattr(
        mcp_server.manager, "get", lambda job_id: FakeJob(srt=srt)
    )
    out = await call(mcp, "get_subtitle", job_id="abc123")
    assert "你好" in out["content"]
    assert out["truncated"] is False


@pytest.mark.anyio
async def test_asking_for_a_subtitle_too_early_says_which_stage(mcp, monkeypatch):
    monkeypatch.setattr(
        mcp_server.manager, "get",
        lambda job_id: FakeJob(stage="transcribing"),
    )
    out = await call(mcp, "get_subtitle", job_id="abc123")
    assert out["stage"] == "transcribing" and "尚未完成" in out["error"]


@pytest.mark.anyio
async def test_a_huge_subtitle_is_capped_and_says_so(mcp, monkeypatch, tmp_path):
    srt = tmp_path / "big.srt"
    srt.write_text("x" * (mcp_server.MAX_SUBTITLE_CHARS + 500), encoding="utf-8")
    monkeypatch.setattr(mcp_server.manager, "get", lambda job_id: FakeJob(srt=srt))
    out = await call(mcp, "get_subtitle", job_id="abc123")
    assert len(out["content"]) == mcp_server.MAX_SUBTITLE_CHARS
    assert out["truncated"] is True
    assert out["total_chars"] == mcp_server.MAX_SUBTITLE_CHARS + 500


@pytest.mark.anyio
async def test_server_status_never_leaks_the_api_key(mcp, settings_file):
    settings_file(llm__api_key="sk-real", llm__model="deepseek-chat")
    out = await call(mcp, "get_server_status")
    assert "sk-real" not in json.dumps(out, ensure_ascii=False)
    assert out["llm_model"] == "deepseek-chat"


@pytest.mark.anyio
async def test_list_videos_reports_what_a_batch_would_pick_up(mcp, tmp_path):
    (tmp_path / "a.mkv").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "b.srt").write_text("done", encoding="utf-8")
    out = await call(mcp, "list_videos", directory=str(tmp_path))
    assert [p.rsplit("/", 1)[-1] for p in out["videos"]] == ["a.mkv"]
    assert out["total"] == 1 and len(out["skipped"]) == 1


# ------------------------------------------------------------- the mount


def test_the_endpoint_is_reachable_and_not_eaten_by_the_spa(settings_file):
    """The SPA catch-all is registered after /mcp on purpose; if that order
    ever flips, this initialize call comes back as index.html."""
    from app.main import app

    settings_file(mcp__enabled=True)
    with local_client(app) as client:
        r = client.post(
            "/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
    assert r.status_code == 200
    assert r.json()["result"]["serverInfo"]["name"] == mcp_server.SERVER_NAME


def test_the_endpoint_is_absent_while_the_switch_is_off(settings_file):
    """Off means off for this machine too — loopback is exempt from the
    token, not from the switch."""
    from app.main import app

    settings_file(server__lan_access=True, server__require_token=False)
    with remote_client(app) as client:
        assert client.post("/mcp").status_code == 404
    with local_client(app) as client:
        assert client.post("/mcp").status_code == 404


def test_another_machine_is_served_under_its_own_address(settings_file):
    """The SDK's DNS-rebinding guard checks Host against an allowlist that
    cannot cover whatever address the user's network hands out — left on,
    every LAN client gets an unexplained 421. This is that regression."""
    from app.main import app

    settings_file(
        server__lan_access=True, server__require_token=False, mcp__enabled=True
    )
    with remote_client(app) as client:
        r = client.post(
            "http://192.168.1.9:8760/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
    assert r.status_code == 200, r.text


def test_a_web_page_cannot_reach_the_tools(settings_file):
    """No MCP client is a browser, so an Origin header means a page is
    reaching into this machine. Refused even on loopback — that is exactly
    the address a rebinding attack aims at."""
    from app.main import app

    settings_file(mcp__enabled=True)
    with local_client(app) as client:
        r = client.post("/mcp", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403

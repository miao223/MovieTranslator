from pathlib import Path

import pytest

from app.models.schemas import SubtitleLine, SubtitleSettings
from app.services.frame import extract_frame, parse_time
from app.services.subtitle import build_ass, build_srt
from app.services.vision import translate_frame


# --------------------------------------------------------------- parse_time


def test_parse_time_formats():
    assert parse_time("85") == 85.0
    assert parse_time("23:45") == 23 * 60 + 45
    assert parse_time("1:23:45") == 3600 + 23 * 60 + 45
    assert parse_time(" 0:05 ") == 5.0


@pytest.mark.parametrize("bad", ["", "a:b", "1:2:3:4", "-5", "1::2"])
def test_parse_time_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_time(bad)


# ------------------------------------------------------------ extract_frame


def make_test_video(path: Path, seconds: int = 3):
    """Synthesize a small mp4 with pyav (black frames, 8 fps)."""
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        import numpy as np

        for _ in range(seconds * 8):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((48, 64, 3), dtype="uint8"), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_extract_frame_produces_jpeg(tmp_path):
    video = tmp_path / "v.mp4"
    make_test_video(video)
    out = tmp_path / "f.jpg"
    extract_frame(video, 1.0, out)
    assert out.is_file() and out.stat().st_size > 100
    assert out.read_bytes()[:2] == b"\xff\xd8"  # JPEG magic


def test_extract_frame_beyond_duration_raises(tmp_path):
    video = tmp_path / "v.mp4"
    make_test_video(video, seconds=2)
    with pytest.raises(ValueError, match="超出视频时长"):
        extract_frame(video, 9999, tmp_path / "f.jpg")


# ------------------------------------------------------------------ vision


class FakeVisionClient:
    def __init__(self, reply):
        self.reply = reply
        self.chat = self
        self.completions = self
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class Msg:
            pass

        m = Msg(); m.content = self.reply
        ch = Msg(); ch.message = m
        r = Msg(); r.choices = [ch]
        return r


def test_translate_frame_returns_text(tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"\xff\xd8fake")
    fake = FakeVisionClient("屏幕短信：马上到家")
    out = translate_frame(img, "简体中文", note="手机短信", client=fake)
    assert out == "屏幕短信：马上到家"
    content = fake.last_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "text" and "手机短信" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_translate_frame_no_text_marker(tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"x")
    assert translate_frame(img, "简体中文", client=FakeVisionClient("[无文字]")) is None
    assert translate_frame(img, "简体中文", client=FakeVisionClient("")) is None


# ------------------------------------------------------ an7 cue rendering


def test_frame_cue_in_srt_and_ass():
    lines = [
        SubtitleLine(index=1, start=0.0, end=2.0, text="Hi", translation="嗨"),
        SubtitleLine(
            index=2, start=1.0, end=6.0, text="", translation="短信：马上到家",
            is_frame=True,
        ),
    ]
    settings = SubtitleSettings()
    srt = build_srt(lines, settings, mode="bilingual")
    blocks = srt.strip().split("\n\n")
    assert blocks[1].splitlines()[2] == "{\\an7}短信：马上到家"
    assert "Hi" not in blocks[1]  # frame cue has no original line

    ass = build_ass(lines, settings, mode="bilingual")
    assert "{\\an7}短信：马上到家" in ass


# --------------------------------------------------- the vision self-test


def vision_test(monkeypatch, reply, settings_file):
    """Run POST /api/settings/test-vision against a scripted endpoint."""
    from app.main import app as fastapi_app
    from tests.conftest import local_client

    class Client:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.sent = None

        def create(self, model, messages, temperature, **kw):
            self.sent = messages
            if isinstance(reply, Exception):
                raise reply

            class Obj:
                pass

            msg, choice, resp = Obj(), Obj(), Obj()
            msg.content = reply
            choice.message = msg
            resp.choices = [choice]
            return resp

    client = Client()
    monkeypatch.setattr("app.services.translator.make_vision_client",
                        lambda *a, **k: client)
    settings_file()
    body = local_client(fastapi_app).post(
        "/api/settings/test-vision",
        json={"base_url": "http://x/v1", "model": "m", "vision_model": "vl"},
    ).json()
    return body, client


def test_the_vision_self_test_sends_a_picture_and_checks_it_was_read(
    monkeypatch, settings_file
):
    """A text ping proves nothing here — a text-only model answers it and
    then fails on the first subtitle. So the probe is a picture."""
    body, client = vision_test(monkeypatch, "MOVIE 42", settings_file)
    assert body["ok"] and body["read_it"]
    assert body["model"] == "vl"
    content = client.sent[0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_model_that_answers_but_cannot_read_is_not_called_working(
    monkeypatch, settings_file
):
    body, _ = vision_test(monkeypatch, "这是一张图片。", settings_file)
    assert body["ok"] and not body["read_it"]


def test_the_vision_self_test_reports_a_failure_verbatim(monkeypatch, settings_file):
    body, _ = vision_test(monkeypatch, RuntimeError("connection refused"),
                          settings_file)
    assert not body["ok"] and "connection refused" in body["error"]

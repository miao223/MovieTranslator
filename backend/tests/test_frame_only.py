import time as time_mod
from pathlib import Path

from app.services.subtitle import insert_frame_cues_ass, insert_frame_cues_srt

SRT = """1
00:00:01,000 --> 00:00:03,000
Hello there.
你好。

2
00:00:10,000 --> 00:00:12,000
Bye.
再见。
"""


def test_srt_insert_middle_and_renumber():
    out = insert_frame_cues_srt(SRT, [(5.0, 10.0, "短信：马上到家")])
    blocks = out.strip().split("\n\n")
    assert len(blocks) == 3
    assert blocks[0].splitlines()[0] == "1"
    assert blocks[1].splitlines()[0] == "2"
    assert blocks[1].splitlines()[1] == "00:00:05,000 --> 00:00:10,000"
    assert blocks[1].splitlines()[2] == "{\\an7}短信：马上到家"
    assert blocks[2].splitlines()[0] == "3"
    # original text preserved verbatim
    assert "Hello there.\n你好。" in out
    assert "Bye.\n再见。" in out


def test_srt_insert_at_ends():
    out = insert_frame_cues_srt(SRT, [(0.0, 0.5, "开头"), (100.0, 105.0, "结尾")])
    blocks = out.strip().split("\n\n")
    assert "{\\an7}开头" in blocks[0]
    assert "{\\an7}结尾" in blocks[-1]
    assert [b.splitlines()[0] for b in blocks] == ["1", "2", "3", "4"]


ASS = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,56

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello
Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,Bye
"""


def test_ass_insert_sorted_header_untouched():
    out = insert_frame_cues_ass(ASS, [(5.0, 10.0, "短信内容")])
    lines = out.splitlines()
    dialogues = [l for l in lines if l.startswith("Dialogue:")]
    assert len(dialogues) == 3
    assert "{\\an7}短信内容" in dialogues[1]  # between the two existing
    assert out.startswith("[Script Info]")
    assert "Style: Default,Arial,56" in out  # header verbatim


def test_ass_append_fallback_without_dialogues():
    header_only = "[Script Info]\nScriptType: v4.00+\n"
    out = insert_frame_cues_ass(header_only, [(1.0, 6.0, "文字")])
    assert out.rstrip().endswith("{\\an7}文字")


# ------------------------------------------------ frame_only pipeline


def test_frame_only_pipeline_merges_in_place(tmp_path, monkeypatch):
    from app.models.schemas import FrameTask, JobRequest
    from app.services import vision
    from app.services.pipeline import manager
    from tests.test_frame_translation import make_test_video

    video = tmp_path / "movie.mp4"
    make_test_video(video, seconds=3)
    srt = tmp_path / "movie.srt"
    srt.write_text(SRT, encoding="utf-8")

    monkeypatch.setattr(
        vision, "translate_frame", lambda *a, **k: "屏幕文字：测试译文"
    )
    job = manager.create(JobRequest(
        video_path=str(video),
        frame_only=True,
        frame_tasks=[FrameTask(time="0:01"), FrameTask(time="99:00")],  # 2nd fails
    ))
    for _ in range(100):
        if job.status.stage in ("done", "failed"):
            break
        time_mod.sleep(0.1)
    assert job.status.stage == "done", job.status.error
    out = srt.read_text(encoding="utf-8")
    assert "{\\an7}屏幕文字：测试译文" in out
    assert "Hello there." in out and "Bye." in out  # originals intact
    assert job.status.srt_in_place is True


def test_frame_only_fails_without_subtitle_file(tmp_path):
    from app.models.schemas import FrameTask, JobRequest
    from app.services.pipeline import manager
    from tests.test_frame_translation import make_test_video

    video = tmp_path / "nosub.mp4"
    make_test_video(video, seconds=2)
    job = manager.create(JobRequest(
        video_path=str(video), frame_only=True,
        frame_tasks=[FrameTask(time="1")],
    ))
    for _ in range(100):
        if job.status.stage in ("done", "failed"):
            break
        time_mod.sleep(0.1)
    assert job.status.stage == "failed"
    assert "未找到" in (job.status.error or "")

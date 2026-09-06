"""纯原文：跳过对白翻译，其余一切照旧。

这个模式不是「把翻译关掉」，而是把整条流水线当成一个高质量的转写 / 字幕
提取器——转写预处理、歌词识别、二次识别复核、图形字幕 OCR 与校对一个都不
少，只是最后不把原文换成译文。所以这里守的是两件事：

* 翻译模型一次都不许被调用，阶段名也不许说自己在翻译；
* 跳过翻译不得顺手带走别的东西——尤其是把 ♪ 写进文本的那一步。

素材用片源已有的字幕（text_source="subtitle"），由生产代码的封装器现造，
所以整条链路是真的跑了一遍，而且不需要 whisper 模型。
"""

import time as time_mod
from pathlib import Path

import pytest

from app.models.schemas import FrameTask, JobRequest, SubtitleLine
from tests.test_subsource import video_with_subs

CUES = [
    SubtitleLine(index=1, start=0.5, end=1.2, text="Hello there"),
    SubtitleLine(index=2, start=1.4, end=1.9, text="Goodbye my friend"),
]


def run(job, seconds: float = 20.0):
    """Drive a job to completion the way the other pipeline tests do."""
    for _ in range(int(seconds * 10)):
        if job.status.stage in ("done", "failed"):
            break
        time_mod.sleep(0.1)
    return job


@pytest.fixture
def quiet(settings_file):
    """预处理里会调 LLM 的两个环节关掉——本文件测的是翻译那一步。

    转写预处理在 text_source="subtitle" 下本来就跳过（人工字幕的断句是对着
    画面排的，不该重排），剩下的只有歌词识别。
    """
    return settings_file(prompts__mark_lyrics=False)


def test_a_pure_original_job_never_calls_the_translator(tmp_path, monkeypatch, quiet):
    """纯原文的全部定义就是这一条。

    Translator 被换成会爆炸的替身：只要管线还构造它，任务就会失败。顺带钉住
    产物的名字——译文一直叫 片名.srt，两份必须能并排放在同一个目录里。
    """
    from app.services import pipeline

    video = video_with_subs(tmp_path, CUES)

    def boom(*args, **kwargs):
        raise AssertionError("纯原文模式不得构造翻译器")

    monkeypatch.setattr(pipeline, "Translator", boom)
    job = run(pipeline.manager.create(JobRequest(
        video_path=str(video), text_source="subtitle",
        output_mode="original_only", source_language="ja",
    )))

    assert job.status.stage == "done", job.status.error
    out = Path(job.status.srt_filename)
    assert out.name == "withsubs.ja.srt"          # 带源语言后缀
    text = out.read_text(encoding="utf-8")
    assert "Hello there" in text and "Goodbye my friend" in text
    # 译文模式的名字仍空着：两份字幕能并排放，不会互相覆盖
    assert not (video.parent / "withsubs.srt").exists()
    # 阶段名不许说谎：一次都没发过 translating（前端会把它显示成「AI 翻译」）
    assert all(e.stage != "translating" for e in job.events)


def test_a_lyric_keeps_its_marks_when_nothing_is_translated(tmp_path, monkeypatch, quiet):
    """♪ 是 apply_marks 写进文本的，而它挂在翻译之后。

    mark_lyrics 只置 is_lyric 标志；顺手把 apply_marks 塞进「跳过翻译」的分支
    里，歌词标记就会静默消失——包括这里这种片源自己带的（读进来时被剥掉，
    说好了要写回去）。这是本模式最容易破的一条不变量。
    """
    from app.services import pipeline

    lyric = [SubtitleLine(index=1, start=0.5, end=1.5, text="♪ Crawl out through the fallout ♪")]
    video = video_with_subs(tmp_path, lyric)

    monkeypatch.setattr(pipeline, "Translator", lambda *a, **k: pytest.fail("不该翻译"))
    job = run(pipeline.manager.create(JobRequest(
        video_path=str(video), text_source="subtitle",
        output_mode="original_only", source_language="en",
    )))

    assert job.status.stage == "done", job.status.error
    text = Path(job.status.srt_filename).read_text(encoding="utf-8")
    assert "♪ Crawl out through the fallout ♪" in text


def test_the_frame_translation_still_targets_the_target_language(
    tmp_path, monkeypatch, quiet
):
    """对白原文 + 画面译文共存：招牌和标题仍然译成目标语言。

    靠的是 build_srt 里 is_frame 排在 mode 判断之前。这也是「目标语言」下拉
    在纯原文下仍然可用的唯一理由。
    """
    from app.services import pipeline, vision

    video = video_with_subs(tmp_path, CUES)
    monkeypatch.setattr(vision, "translate_frame", lambda *a, **k: "青木市立图书馆")
    monkeypatch.setattr(pipeline, "Translator", lambda *a, **k: pytest.fail("不该翻译"))
    job = run(pipeline.manager.create(JobRequest(
        video_path=str(video), text_source="subtitle",
        output_mode="original_only", source_language="ja",
        frame_tasks=[FrameTask(time="0:01")],
    )))

    assert job.status.stage == "done", job.status.error
    text = Path(job.status.srt_filename).read_text(encoding="utf-8")
    assert "{\\an7}青木市立图书馆" in text      # 画面文字：译文
    assert "Hello there" in text                # 对白：原文


# ------------------------------------------------------------- 命名契约


def test_the_translated_side_of_the_naming_is_untouched():
    """译文字幕一直叫 片名.srt（后缀为空）——改了所有指向它的媒体库都对不上。"""
    from app.services.pipeline import _naming

    req = JobRequest(video_path="/v/film.mkv", target_language="简体中文")
    assert _naming(req, "ja") == ("", "zh", "chi", "简体中文字幕")


def test_an_undetected_language_falls_back_instead_of_guessing():
    """判不出来就退回 orig / und，不拿目标语言冒充，也不猜一个。"""
    from app.services.pipeline import _naming

    req = JobRequest(video_path="/v/film.mkv", target_language="简体中文",
                     output_mode="original_only")
    assert _naming(req, "ja") == (".ja", "ja", "jpn", "日语字幕")
    assert _naming(req, "") == (".orig", "orig", "und", "原文字幕")

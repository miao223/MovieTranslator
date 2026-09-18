"""双文件：同样的内容，写成两个单语字幕文件。

bilingual 把原文和译文叠在同一条 cue 上；这个模式把它们分开放，
片名.zh.srt 与 片名.en.srt 并排落在片源旁边，播放器里当成两条可选字幕。
所以这里守的是三件事：

* 两份内容就是现成的 translation_only 与 original_only —— subtitle.py 不为
  这个模式增加任何分支，"bilingual_split" 根本不会传进 builder；
* 两份都带语言后缀。译文那一份因此**不再**叫 片名.srt，这是刻意的：另一半
  就在旁边，不带后缀两份会互相覆盖；
* 谁都不许写到它自己的原文上，也不许写到对方头上。

素材用片源已有的字幕（text_source="subtitle"），由生产代码的封装器现造，
翻译器换成一个只填 translation 的替身——本文件测的是产物落在哪里、内容是
什么，不是翻译质量。
"""

import time as time_mod
from pathlib import Path

import pytest

from app.models.schemas import FrameTask, JobRequest, SubtitleLine, SubtitleSettings
from app.services import subtitle
from tests.conftest import local_client
from tests.test_subsource import video_with_subs, write_subs

CUES = [
    SubtitleLine(index=1, start=0.5, end=1.2, text="Hello there"),
    SubtitleLine(index=2, start=1.4, end=1.9, text="Goodbye my friend"),
]

TRANSLATIONS = {"Hello there": "你好啊", "Goodbye my friend": "再见了朋友"}


class FakeTranslator:
    """填上译文就完事，不发任何请求。"""

    glossary_text = ""

    def __init__(self, *args, **kwargs):
        pass

    def translate(self, lines):
        for line in lines:
            if not line.is_frame:
                line.translation = TRANSLATIONS.get(line.text, "译：" + line.text)

    def report_usage(self):
        return "（测试替身，未调用模型）"


def run(job, seconds: float = 20.0):
    for _ in range(int(seconds * 10)):
        if job.status.stage in ("done", "failed"):
            break
        time_mod.sleep(0.1)
    return job


@pytest.fixture
def quiet(settings_file):
    """歌词识别会调 LLM，本文件与它无关。"""
    return settings_file(prompts__mark_lyrics=False)


@pytest.fixture
def split(monkeypatch, quiet):
    from app.services import pipeline

    monkeypatch.setattr(pipeline, "Translator", FakeTranslator)
    return pipeline


def start(pipeline, video, **kwargs):
    fields = dict(video_path=str(video), text_source="subtitle",
                  output_mode="bilingual_split", source_language="en")
    fields.update(kwargs)
    return run(pipeline.manager.create(JobRequest(**fields)))


# ------------------------------------------------------- 两份产物


def test_the_two_files_are_exactly_the_two_single_file_modes(tmp_path, split):
    """内容不是新写的：一份是 translation_only，一份是 original_only。

    这条断言是「subtitle.py 一个分支都不用加」的记录——它一旦红了，说明有人
    在 builder 里给这个模式开了自己的路。
    """
    job = start(split, video_with_subs(tmp_path, CUES))
    assert job.status.stage == "done", job.status.error

    lines = [SubtitleLine(index=i, start=c.start, end=c.end, text=c.text,
                          translation=TRANSLATIONS[c.text])
             for i, c in enumerate(CUES, 1)]
    settings = SubtitleSettings()
    assert Path(job.status.srt_filename).read_text(encoding="utf-8") == \
        subtitle.build_srt(lines, settings, mode="translation_only")
    assert Path(job.status.original_srt_filename).read_text(encoding="utf-8") == \
        subtitle.build_srt(lines, settings, mode="original_only")


def test_both_files_carry_a_language_suffix(tmp_path, split):
    """译文那一份在这个模式下**不**叫 片名.srt。

    别处的规矩是「译文一直与片源同名」，改了所有指向它的媒体库都对不上。这里
    是唯一的例外，而理由就在旁边：另一半也要落在同一个目录里。
    """
    video = video_with_subs(tmp_path, CUES)
    job = start(split, video)
    assert job.status.stage == "done", job.status.error

    assert Path(job.status.srt_filename).name == "withsubs.zh.srt"
    assert Path(job.status.original_srt_filename).name == "withsubs.en.srt"
    assert not (video.parent / "withsubs.srt").exists()
    assert job.status.srt_in_place is True


def test_the_original_keeps_its_own_language_not_the_target(tmp_path, split):
    """原文那一份按**识别出来的语言**命名，不是翻译目标。"""
    job = start(split, video_with_subs(tmp_path, CUES), source_language="auto")
    assert job.status.stage == "done", job.status.error
    # 片源那条轨标着 English，所以后缀是 en 而不是 zh，也不是 orig
    assert Path(job.status.original_srt_filename).name == "withsubs.en.srt"


def test_one_language_on_both_sides_still_makes_two_files(tmp_path, split):
    """源语言与目标语言撞车时，让位的必须是原文那一份。

    译文才是媒体库要认的那个名字；orig 正是判不出语言时用的那个词。
    """
    job = start(split, video_with_subs(tmp_path, CUES),
                target_language="English", source_language="en")
    assert job.status.stage == "done", job.status.error

    translation = Path(job.status.srt_filename)
    original = Path(job.status.original_srt_filename)
    assert translation.name == "withsubs.en.srt"
    assert original.name == "withsubs.orig.srt"
    assert translation != original and original.read_text(encoding="utf-8")


def test_neither_file_is_written_over_the_subtitle_it_came_from(tmp_path, split):
    """原文读自 film.en.srt，产物也叫 film.en.srt —— 它不许覆盖来源。

    而且不能用译文那一招让开（改叫 .zh）：那是把英文原文写进中文译文的名字里。
    """
    video = video_with_subs(tmp_path, CUES)
    sidecar = write_subs(video.parent / f"{video.stem}.en.srt", CUES, False)
    before = sidecar.read_bytes()

    job = start(split, video, subtitle_file=str(sidecar))
    assert job.status.stage == "done", job.status.error

    assert sidecar.read_bytes() == before          # 来源一个字节都没动
    original = Path(job.status.original_srt_filename)
    assert original.name == "withsubs.en.2.srt"
    assert Path(job.status.srt_filename).name == "withsubs.zh.srt"


# ------------------------------------------------------- 画面翻译


def test_the_frame_translation_goes_into_the_translation_only(
    tmp_path, split, monkeypatch
):
    """画面文字只有译文（frame 任务的 text 是空的），所以它只进译文那一份。

    两份合起来正好等于 bilingual 那一份会有的内容——那就是这个模式的定义。
    把一条中文招牌塞进英文原文字幕里，既没有原文可对照，也和「补充模式」并进
    译文的做法自相矛盾。
    """
    from app.services import vision

    monkeypatch.setattr(vision, "translate_frame", lambda *a, **k: "青木市立图书馆")
    job = start(split, video_with_subs(tmp_path, CUES),
                frame_tasks=[FrameTask(time="0:01")])
    assert job.status.stage == "done", job.status.error

    translation = Path(job.status.srt_filename).read_text(encoding="utf-8")
    original = Path(job.status.original_srt_filename).read_text(encoding="utf-8")
    assert "{\\an7}青木市立图书馆" in translation
    assert "青木市立图书馆" not in original
    assert "Hello there" in original


def test_a_supplement_run_merges_into_the_translation(tmp_path, split, monkeypatch):
    """补充模式在两份之间挑一份，必须挑译文那一份。

    候选是 glob 出来的，按字母序 film.en.srt 排在 film.zh.srt 前面——不先点名
    译文，{\\an7} 那条 cue 就会并进原文里。
    """
    from app.services import vision

    video = video_with_subs(tmp_path, CUES)
    job = start(split, video)
    assert job.status.stage == "done", job.status.error
    original = Path(job.status.original_srt_filename)
    before = original.read_bytes()

    monkeypatch.setattr(vision, "translate_frame", lambda *a, **k: "青木市立图书馆")
    again = run(split.manager.create(JobRequest(
        video_path=str(video), frame_only=True,
        frame_tasks=[FrameTask(time="0:01")],
    )))
    assert again.status.stage == "done", again.status.error

    assert Path(again.status.srt_filename).name == "withsubs.zh.srt"
    assert "青木市立图书馆" in Path(again.status.srt_filename).read_text(encoding="utf-8")
    assert original.read_bytes() == before          # 原文那一份没被动过


# ------------------------------------------------------- 结果与命名契约


def test_the_job_hands_back_both_paths(tmp_path, split):
    """界面只靠 original_srt_filename 决定要不要显示第二个下载按钮。"""
    job = start(split, video_with_subs(tmp_path, CUES))
    assert job.status.stage == "done", job.status.error
    assert job.status.srt_filename and job.status.original_srt_filename
    assert job.status.srt_filename != job.status.original_srt_filename


def test_the_other_modes_hand_back_nothing_extra(tmp_path, split):
    """三个老模式一份都不多产。"""
    job = start(split, video_with_subs(tmp_path, CUES), output_mode="bilingual")
    assert job.status.stage == "done", job.status.error
    assert job.status.original_srt_filename == ""
    assert Path(job.status.srt_filename).name == "withsubs.srt"


def test_the_two_file_mode_names_the_translation_after_the_target():
    """命名契约：译文带目标语言后缀，原文带源语言后缀，两份都在同一个目录。"""
    from app.services.pipeline import _naming, _products

    req = JobRequest(video_path="/v/film.mkv", target_language="简体中文",
                     output_mode="bilingual_split")
    assert _naming(req, "ja") == (".zh", "zh", "chi", "简体中文字幕")
    assert [p.sidecar for p in _products(req, "ja")] == [".zh", ".ja"]
    assert [p.mode for p in _products(req, "ja")] == [
        "translation_only", "original_only"]
    # 判不出源语言就退回 orig，绝不拿目标语言冒充
    assert [p.sidecar for p in _products(req, "")] == [".zh", ".orig"]


def test_each_half_can_be_downloaded_on_its_own(tmp_path, split):
    """两个下载按钮走同一个端点，靠 part 区分；不带 part 时链接与从前相同。"""
    from app.main import app

    job = start(split, video_with_subs(tmp_path, CUES))
    assert job.status.stage == "done", job.status.error

    with local_client(app) as client:
        first = client.get(f"/api/jobs/{job.id}/result")
        second = client.get(f"/api/jobs/{job.id}/result", params={"part": "original"})
        bad = client.get(f"/api/jobs/{job.id}/result", params={"part": "nonsense"})

    assert first.status_code == 200 and "你好啊" in first.text
    assert second.status_code == 200 and "Hello there" in second.text
    assert "你好啊" not in second.text
    assert bad.status_code == 400


def test_asking_for_an_original_that_does_not_exist_says_so(tmp_path, split):
    """三个老模式没有第二份，端点不能拿译文冒充它。"""
    from app.main import app

    job = start(split, video_with_subs(tmp_path, CUES), output_mode="bilingual")
    assert job.status.stage == "done", job.status.error

    with local_client(app) as client:
        assert client.get(f"/api/jobs/{job.id}/result").status_code == 200
        assert client.get(f"/api/jobs/{job.id}/result",
                          params={"part": "original"}).status_code == 404


def test_a_failure_writing_the_second_file_keeps_the_first(
    tmp_path, split, monkeypatch
):
    """两个文件之间没有原子性，所以约定是「谁都不丢」。

    第二份写不下去时，已经写成的第一份**留在原地**——把它删掉只是把「少一
    个文件」换成「两手空空」。兜底那一份进工作目录，能下载，而 srt_in_place
    诚实地变成 False：界面那句「视频目录不可写，请下载保存」是这时唯一安全
    的说法。
    """
    video = video_with_subs(tmp_path, CUES)
    real = Path.write_text

    def refuse(self, *args, **kwargs):
        if self.name == "withsubs.en.srt" and self.parent == video.parent:
            raise OSError("Read-only file system")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", refuse)
    job = start(split, video)

    assert job.status.stage == "done", job.status.error
    # 译文仍在片源旁边，一个字都没少
    translation = Path(job.status.srt_filename)
    assert translation == video.parent / "withsubs.zh.srt" and translation.is_file()
    # 原文退到了工作目录，仍然能取到
    original = Path(job.status.original_srt_filename)
    assert original.parent != video.parent and "Hello there" in original.read_text(
        encoding="utf-8")
    assert job.status.srt_in_place is False

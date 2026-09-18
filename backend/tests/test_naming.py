"""产物叫什么名字，以及绝不覆盖谁。

两条规矩合在一起：

* **每一份字幕都说得出自己是什么语言**——后缀就是文件里的内容。片名.srt 这个
  名字本程序不再产出，它从前同时是「中文译文」和「双语」两种东西的名字。
* **片源目录里任何已经存在的文件都不会被覆盖**——名字被占就按编号让路。

这里跑的是真管线（翻译器换成只填译文的替身），所以钉的是「文件真的落在哪里」，
而不是命名函数的返回值——那部分在 test_original_only.py 的命名契约一节。
"""

from pathlib import Path

import pytest

from app.models.schemas import JobRequest
from tests.test_split_output import CUES, FakeTranslator, quiet, run  # noqa: F401
from tests.test_subsource import video_with_subs


@pytest.fixture
def naming(monkeypatch, quiet):  # noqa: F811
    from app.services import pipeline

    monkeypatch.setattr(pipeline, "Translator", FakeTranslator)
    return pipeline


def start(pipeline, video, **kwargs):
    fields = dict(video_path=str(video), text_source="subtitle",
                  source_language="en")
    fields.update(kwargs)
    return run(pipeline.manager.create(JobRequest(**fields)))


def test_a_bilingual_job_names_both_languages_in_reading_order(tmp_path, naming):
    """默认原文在上，所以是 en-zh；片名.srt 不再出现。"""
    video = video_with_subs(tmp_path, CUES)
    job = start(naming, video)

    assert job.status.stage == "done", job.status.error
    assert Path(job.status.srt_filename).name == "withsubs.en-zh.srt"
    assert not (video.parent / "withsubs.srt").exists()
    text = Path(job.status.srt_filename).read_text(encoding="utf-8")
    assert "Hello there" in text and "你好啊" in text     # 两种语言都在里面


def test_putting_the_translation_on_top_swaps_the_name(tmp_path, naming,
                                                       settings_file):
    """哪一行在上，哪个语言就在前——名字跟着排版走。"""
    settings_file(prompts__mark_lyrics=False,
                  subtitle__bilingual_layout="translation_top")
    job = start(naming, video_with_subs(tmp_path, CUES))

    assert job.status.stage == "done", job.status.error
    assert Path(job.status.srt_filename).name == "withsubs.zh-en.srt"


def test_the_second_run_steps_aside_and_leaves_the_first_alone(tmp_path, naming):
    """重跑撞上自己上一次的成品，也一样让路。

    这是新规则最常撞到的情形：以前直接覆盖，现在第一次那份原封不动，新的那份
    带编号落在旁边——而**工作目录里的副本不参与这条规则**，否则每次续跑都会在
    自己的目录里堆一层。
    """
    video = video_with_subs(tmp_path, CUES)
    first = start(naming, video)
    assert first.status.stage == "done", first.status.error
    before = Path(first.status.srt_filename).read_bytes()

    second = start(naming, video)
    assert second.status.stage == "done", second.status.error

    assert Path(first.status.srt_filename).name == "withsubs.en-zh.srt"
    assert Path(second.status.srt_filename).name == "withsubs.en-zh.2.srt"
    assert Path(first.status.srt_filename).read_bytes() == before
    assert sorted(p.name for p in video.parent.glob("withsubs*.srt")) == [
        "withsubs.en-zh.2.srt", "withsubs.en-zh.srt"]


def test_the_work_directory_is_not_subject_to_the_rule(tmp_path, naming):
    """让路只管片源目录。

    内嵌模式下工作目录里也留一份字幕（下载按钮取的就是它）。两次跑同一份请求
    用的是同一个续跑目录，编号一旦进去，每次续跑都会在自己的目录里堆一层——
    而那是本程序自己的东西，本来就该被重写。片源目录那一侧照常让路。
    """
    from app.core.cache import checkpoint_dir

    video = video_with_subs(tmp_path, CUES)
    first = start(naming, video, embed_subtitle=True)
    assert first.status.stage == "done", first.status.error
    second = start(naming, video, embed_subtitle=True)
    assert second.status.stage == "done", second.status.error

    # 同一份请求 → 同一个工作目录，里面始终只有没编号的那一份
    assert first.checkpoint and first.checkpoint == second.checkpoint
    workdir = checkpoint_dir(first.checkpoint)
    assert [p.name for p in sorted(workdir.glob("withsubs*.srt"))] == [
        "withsubs.en-zh.srt"]

    # 而片源目录里的成品让了路
    assert sorted(p.name for p in video.parent.glob("withsubs*.mkv")) == [
        "withsubs.mkv", "withsubs.zh.2.mkv", "withsubs.zh.mkv"]


def test_a_third_party_subtitle_of_the_same_name_is_not_touched(tmp_path, naming):
    """用户自己放的 片名.zh.srt（比如下载来的）也是「已经存在的文件」。"""
    video = video_with_subs(tmp_path, CUES)
    theirs = video.parent / "withsubs.zh.srt"
    theirs.write_text("1\n00:00:01,000 --> 00:00:02,000\n别人的字幕\n",
                      encoding="utf-8")
    before = theirs.read_bytes()

    job = start(naming, video, output_mode="translation_only")
    assert job.status.stage == "done", job.status.error

    assert theirs.read_bytes() == before
    assert Path(job.status.srt_filename).name == "withsubs.zh.2.srt"


def test_the_supplement_merges_into_our_own_subtitle_not_a_strangers(
    tmp_path, naming, monkeypatch
):
    """目录里同时有第三方的 片名.srt 和我们刚生成的译文时，画面译文进后者。

    {\\an7} 那条 cue 本身就是译文；并进一份下载来的英文字幕里既没人看得到，
    也把别人的文件改了。
    """
    from app.services import vision
    from tests.test_subsource import write_subs

    video = video_with_subs(tmp_path, CUES)
    theirs = write_subs(video.parent / "withsubs.srt", CUES, False)
    before = theirs.read_bytes()

    job = start(naming, video, output_mode="translation_only")
    assert job.status.stage == "done", job.status.error

    monkeypatch.setattr(vision, "translate_frame", lambda *a, **k: "青木市立图书馆")
    from app.models.schemas import FrameTask

    again = run(naming.manager.create(JobRequest(
        video_path=str(video), frame_only=True,
        frame_tasks=[FrameTask(time="0:01")],
    )))
    assert again.status.stage == "done", again.status.error

    assert Path(again.status.srt_filename).name == "withsubs.zh.srt"
    assert theirs.read_bytes() == before

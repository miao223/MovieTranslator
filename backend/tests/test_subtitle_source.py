"""字幕文件本身就是翻译对象。

从前字幕只能作为**视频的附属品**被翻译——`video_path` 必须指向一个视频。可手里
常常只有一份 `.srt`（或蓝光拆出来的 `.sup`、DVD 的 `.idx`+`.sub`），片子在别处。

读取层其实早就支持了：`subsource.read_cues` 与 `ocr.bitmap_cues` 都写着
`source = Path(track["path"] or video_path)`。所以这里守的是那条**通路**：

* 产物剥掉源文件自己的语言后缀（`film.en.srt` → `film.zh.srt`，不是 `film.en.zh.srt`）；
* 源文件一个字节都不许动；
* 图形字幕（`.sup` / VobSub）照旧走 OCR，`.sub` 这个歧义后缀按旁边有没有 `.idx` 决定；
* 没有画面的那几个开关，该降级的降级、该拒绝的拒绝。
"""

import time as time_mod
from pathlib import Path

import pytest

from app.models.schemas import FrameTask, JobRequest, SubtitleLine
from tests import pgs
from tests.test_subsource import write_subs

CUES = [
    SubtitleLine(index=1, start=0.5, end=1.2, text="Hello there"),
    SubtitleLine(index=2, start=1.4, end=1.9, text="Goodbye my friend"),
]

MICRODVD = ("{1}{1}25.000\n{25}{50}Hello there\n"
            "{75}{100}Goodbye my friend\n{125}{150}See you later\n")


class FakeTranslator:
    glossary_text = ""

    def __init__(self, *args, **kwargs):
        pass

    def translate(self, lines):
        for line in lines:
            line.translation = "译：" + line.text

    def report_usage(self):
        return "（测试替身）"


def run(job, seconds: float = 20.0):
    for _ in range(int(seconds * 10)):
        if job.status.stage in ("done", "failed"):
            break
        time_mod.sleep(0.1)
    return job


@pytest.fixture
def subs_only(monkeypatch, settings_file):
    """翻译器换成替身，会调 LLM 的预处理环节全关——这里测的是通路不是质量。"""
    from app.services import pipeline

    settings_file(prompts__mark_lyrics=False, prompts__refine_enabled=False)
    monkeypatch.setattr(pipeline, "Translator", FakeTranslator)
    return pipeline


def start(pipeline, source, **kwargs):
    return run(pipeline.manager.create(JobRequest(video_path=str(source), **kwargs)))


# --------------------------------------------------------------- 文字字幕


def test_a_bare_srt_is_translated_on_its_own(tmp_path, subs_only):
    """只给一份 .srt，没有视频，也能跑完整条管线。"""
    source = write_subs(tmp_path / "film.srt", CUES)
    before = source.read_bytes()

    job = start(subs_only, source, output_mode="translation_only")

    assert job.status.stage == "done", job.status.error
    out = Path(job.status.srt_filename)
    assert out.name == "film.zh.srt"
    assert "译：Hello there" in out.read_text(encoding="utf-8")
    assert source.read_bytes() == before          # 源文件一个字节没动


def test_the_language_suffix_is_replaced_not_stacked(tmp_path, subs_only):
    """film.en.srt 的产物是 film.zh.srt —— 与直接翻视频得到的名字一模一样。"""
    source = write_subs(tmp_path / "film.en.srt", CUES)

    job = start(subs_only, source, output_mode="translation_only")

    assert job.status.stage == "done", job.status.error
    assert Path(job.status.srt_filename).name == "film.zh.srt"
    assert not (tmp_path / "film.en.zh.srt").exists()


def test_a_dotted_release_name_keeps_every_dot(tmp_path, subs_only):
    """发行版片名里全是点，只有最后那一段才可能是语言。"""
    source = write_subs(tmp_path / "Movie.2019.1080p.en.srt", CUES)

    job = start(subs_only, source, output_mode="translation_only")

    assert job.status.stage == "done", job.status.error
    assert Path(job.status.srt_filename).name == "Movie.2019.1080p.zh.srt"


def test_a_bilingual_run_names_both_languages(tmp_path, subs_only):
    """命名规则与视频那条路共用一套。"""
    source = write_subs(tmp_path / "film.en.srt", CUES)

    job = start(subs_only, source)          # 默认双语

    assert job.status.stage == "done", job.status.error
    assert Path(job.status.srt_filename).name == "film.en-zh.srt"


def test_the_source_is_never_written_over(tmp_path, subs_only):
    """纯原文读 film.en.srt，产物也叫 film.en.srt —— 它必须让路。"""
    source = write_subs(tmp_path / "film.en.srt", CUES)
    before = source.read_bytes()

    job = start(subs_only, source, output_mode="original_only", source_language="en")

    assert job.status.stage == "done", job.status.error
    assert Path(job.status.srt_filename).name == "film.en.2.srt"
    assert source.read_bytes() == before


def test_a_microdvd_sub_is_read_as_text(tmp_path, subs_only):
    """.sub 旁边没有 .idx，那它就是 MicroDVD/SubViewer 这类文字格式。"""
    source = tmp_path / "film.sub"
    source.write_text(MICRODVD, encoding="utf-8")

    job = start(subs_only, source, output_mode="translation_only")

    assert job.status.stage == "done", job.status.error
    text = Path(job.status.srt_filename).read_text(encoding="utf-8")
    assert "译：Hello there" in text and "译：See you later" in text


# --------------------------------------------------------------- 图形字幕


def test_a_standalone_pgs_goes_through_ocr(tmp_path, subs_only, monkeypatch):
    """单独一份 .sup 照旧走 OCR —— bitmap_cues 本来就打开 track["path"]。"""
    from app.services import ocr

    monkeypatch.setattr(
        ocr, "_read_rapidocr",
        lambda cues, *a, **k: [("Hello there", 0.95) for _ in cues])
    source = pgs.write_sup(tmp_path / "film.sup", [(0.5, 1.2, "Hello there")])

    job = start(subs_only, source, output_mode="translation_only")

    assert job.status.stage == "done", job.status.error
    assert Path(job.status.srt_filename).name == "film.zh.srt"
    assert "译：Hello there" in Path(job.status.srt_filename).read_text(encoding="utf-8")


def test_a_graphic_file_is_recognised_by_its_codec_not_its_suffix(tmp_path):
    """文字/图形之分按打开后的 codec 判，后缀只用来决定列不列。"""
    from app.services import subsource

    text = subsource.file_track(write_subs(tmp_path / "a.srt", CUES))
    graphic = subsource.file_track(
        pgs.write_sup(tmp_path / "b.sup", [(0.5, 1.2, "Hi")]))

    assert text["text"] is True and text["codec"] == "SRT"
    assert graphic["text"] is False and "PGS" in graphic["codec"]


# ------------------------------------------------------------- VobSub 的一对


def test_a_sub_is_opened_through_its_idx(tmp_path):
    """VobSub 是 .idx + .sub 一对，而 ffmpeg 的解复用器注册在 .idx 上。

    真正的解码是 ffmpeg 的事（这台机器造不出一对合法的 VobSub——vobsub 只有
    解复用器没有复用器），本项目要负责的是**指到哪个文件**。
    """
    from app.services import subsource

    sub = tmp_path / "film.sub"
    sub.write_text(MICRODVD, encoding="utf-8")
    assert subsource.vobsub_index(sub) is None        # 没有 .idx：它就是文字

    idx = tmp_path / "film.idx"
    idx.write_text("# VobSub index file\n", encoding="utf-8")
    assert subsource.vobsub_index(sub) == idx         # 有了就改指它
    # 别的后缀一概不参与这条规则
    assert subsource.vobsub_index(tmp_path / "film.srt") is None


def test_an_unreadable_sub_says_the_idx_is_missing(tmp_path):
    """读不出来的 .sub 最可能的原因就是它的 .idx 没跟过来，得说出口。"""
    from app.services import subsource

    broken = tmp_path / "film.sub"
    broken.write_bytes(b"\x00\x01\x02not a subtitle at all")

    with pytest.raises(ValueError) as caught:
        subsource.file_track(broken)
    assert ".idx" in str(caught.value) and "film.idx" in str(caught.value)


# --------------------------------------------------------------- 没有画面


def test_embedding_degrades_and_frame_only_is_refused(tmp_path, subs_only):
    """两条相反的处理各自成立：能降级的降级，降级等于白跑的直接拒绝。"""
    source = write_subs(tmp_path / "film.srt", CUES)

    job = start(subs_only, source, output_mode="translation_only",
                embed_subtitle=True)
    assert job.status.stage == "done", job.status.error
    assert job.status.video_filename == ""            # 没有视频产出
    assert Path(job.status.srt_filename).name == "film.zh.srt"
    said = "\n".join(e.log for e in job.events if e.log)
    assert "字幕文件" in said and "纯音频" not in said  # 说的是字幕不是音频

    with pytest.raises(ValueError, match="字幕文件"):
        subs_only.manager.create(JobRequest(
            video_path=str(source), frame_only=True,
            frame_tasks=[FrameTask(time="0:01")]))


def test_asking_for_speech_recognition_reads_the_file_instead(tmp_path, subs_only):
    """字幕文件没有音频可听。不是拒绝而是改道——批量里「原文来源」是整批
    共用的一个开关，为个别文件停下整批不值得——但必须说出来。"""
    source = write_subs(tmp_path / "film.srt", CUES)

    job = start(subs_only, source, text_source="asr",
                output_mode="translation_only")

    assert job.status.stage == "done", job.status.error
    said = "\n".join(e.log for e in job.events if e.log)
    assert "语音识别" in said and "已改为直接读取" in said

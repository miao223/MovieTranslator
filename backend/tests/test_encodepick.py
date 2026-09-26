"""按画面自动选编码：拼图、问模型、按实测表对照，以及做不成时退回表单。

模型一律用假的（tests/test_ocr.FakeVision）：它要花钱，而且不是这里会出错的
地方——这里测的是拼图对不对、回答读不读得出、对照表对不对、失败时退不退得回去。
真模型在本机上的实测记在 CLAUDE.md。
"""

from __future__ import annotations

import io
import json
from fractions import Fraction

import pytest
from PIL import Image

from app.models.schemas import AppSettings, EncodeOptions, LLMSettings
from app.services import encodepick
from app.services.encodepick import PickError, Verdict
from tests.mediagen import make_source
from tests.test_ocr import FakeVision


def answer(content="live", grain="light", reason="真人，细颗粒"):
    return json.dumps({"content": content, "grain": grain, "reason": reason}, ensure_ascii=False)


# ------------------------------------------------------------------ 拼图


def test_the_sheet_is_six_frames_above_three_crops_at_their_own_pixels(tmp_path):
    source = make_source(tmp_path / "f.mkv", frames=240, size=(1280, 720))
    shots, facts = encodepick.grab(source)
    assert len(shots) == 6 and not facts["interlaced"] and not facts["hdr"]
    # a crop is never scaled: 640×360 of the frame's own pixels
    assert {shot.crop.size for shot in shots} == {(640, 360)}
    sheet = encodepick.contact_sheet(shots)
    assert sheet.size == (1920, 1080)
    image = Image.open(io.BytesIO(encodepick.jpeg(sheet)))
    assert image.format == "JPEG" and image.size == (1920, 1080)


def test_the_frames_come_from_inside_the_film_not_its_ends(tmp_path):
    """12%–82% of the running time: the ends are black, logos and credits."""
    source = make_source(tmp_path / "f.mkv", frames=240)     # ten seconds
    _, facts = encodepick.grab(source)
    times = facts["times"]
    assert times == sorted(times) and len(set(times)) == 6
    assert 1.1 <= times[0] and times[-1] <= 8.4


def test_an_anamorphic_picture_is_shown_at_its_display_shape(tmp_path):
    # 720×480 with 32:27 pixels is shown 853 wide: 16:9, filling the tile
    source = make_source(tmp_path / "dvd.mkv", frames=48, size=(720, 480), sar=Fraction(32, 27))
    shots, _ = encodepick.grab(source)
    assert shots[0].overview.size == (640, 360)
    square = make_source(tmp_path / "sq.mkv", frames=48, size=(720, 480))
    shots, _ = encodepick.grab(square)
    assert shots[0].overview.size == (540, 360)       # 3:2 inside a 16:9 tile


def test_an_interlaced_source_is_said_so_to_the_model(tmp_path):
    source = make_source(tmp_path / "i.mkv", frames=96, interlaced=True)
    _, facts = encodepick.grab(source)
    assert facts["interlaced"]
    assert "梳齿" in encodepick.build_prompt(facts)
    assert "HDR" in encodepick.build_prompt({"hdr": True})
    assert "梳齿" not in encodepick.build_prompt({})


def test_a_file_with_no_picture_cannot_be_judged(tmp_path):
    from tests.test_audio_tracks import make_audio_file

    audio = make_audio_file(tmp_path / "a.mka")
    with pytest.raises(PickError, match="没有画面"):
        encodepick.grab(audio)


# ------------------------------------------------------------------ 回答


@pytest.mark.parametrize("reply, expected", [
    (answer("animation", "none"), ("animation", "none")),
    ("```json\n" + answer("live", "heavy") + "\n```", ("live", "heavy")),
    ('好的：{"content": "Anime", "grain": "clean", "reason": ""}', ("animation", "none")),
    ('{"content": "live-action", "grain": "high"}', ("live", "heavy")),
])
def test_the_answer_is_read_in_the_shapes_models_give_it(reply, expected):
    verdict = encodepick.parse(reply)
    assert (verdict.content, verdict.grain) == expected


@pytest.mark.parametrize("reply", [
    "这是一部动画", '{"content": "documentary", "grain": "none"}',
    '{"content": "live", "grain": "medium-ish"}', "{not json}", "",
])
def test_anything_else_is_not_an_answer(reply):
    assert encodepick.parse(reply) is None


def test_the_model_is_asked_once_more_and_then_given_up_on():
    llm = LLMSettings(model="main", vision_model="eyes")
    fake = FakeVision(["我看不太清", answer("animation", "none")])
    verdict = encodepick.ask(b"jpeg", {}, llm, client=fake)
    assert verdict.content == "animation" and len(fake.calls) == 2
    first = fake.calls[0][0]["content"]
    assert first[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    with pytest.raises(PickError, match="读不出判断"):
        encodepick.ask(b"jpeg", {}, llm, client=FakeVision(["不知道"]))


# ------------------------------------------------------------------ 对照


@pytest.mark.parametrize("content, grain, codec, preset, quality, tune", [
    ("animation", "none", "libsvtav1", "veryslow", 25, ""),
    ("animation", "light", "libsvtav1", "veryslow", 25, ""),
    ("animation", "heavy", "libx265", "slow", 18, ""),      # old grainy anime: SVT would scrub it
    ("live", "none", "libx265", "slow", 18, ""),
    ("live", "light", "libx265", "slow", 18, ""),
    ("live", "heavy", "libx265", "slow", 18, "grain"),
])
def test_the_measured_table(content, grain, codec, preset, quality, tune):
    form = EncodeOptions(video_codec="libx264", preset="fast", quality=30, rate_control="bitrate",
                         audio_codec="flac", container="mp4", max_height=720, auto_pick=True)
    picked = encodepick.apply(Verdict(content, grain, ""), form)
    assert (picked.video_codec, picked.preset, picked.quality, picked.tune) == (
        codec, preset, quality, tune)
    assert picked.rate_control == "quality" and not picked.auto_pick
    # only the picture's choices change
    assert (picked.audio_codec, picked.container, picked.max_height) == ("flac", "mp4", 720)


def test_a_judgement_that_cannot_be_made_falls_back_to_the_form_and_says_so(tmp_path):
    source = make_source(tmp_path / "f.mkv", frames=48)
    form = EncodeOptions(quality=21, auto_pick=True)
    lines = []
    picked = encodepick.decide(source, form, AppSettings(), log=lines.append,
                               client=FakeVision([ConnectionError("relay down")]))
    assert picked == form.model_copy(update={"auto_pick": False})
    assert lines[0].startswith("⚠ 按画面自动选编码没做成（relay down）") and "CRF 21" in lines[0]


def test_a_judgement_is_logged_with_what_it_chose(tmp_path):
    source = make_source(tmp_path / "f.mkv", frames=48)
    lines = []
    picked = encodepick.decide(source, EncodeOptions(auto_pick=True), AppSettings(),
                               log=lines.append,
                               client=FakeVision([answer("animation", "none", "赛璐璐风格")]))
    assert picked.video_codec == "libsvtav1"
    assert lines == ["按画面自动选编码：动画，画面干净（赛璐璐风格） → "
                     "AV1 10bit（SVT-AV1） · CRF 25 · veryslow · 无损音轨→E-AC-3"]

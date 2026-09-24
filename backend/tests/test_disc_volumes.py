"""Box sets, the batch mode's disc finder, and where the MKVs go.

A season on several discs must come out as one season: VOL02's first
episode is E07 when VOL01 has six, under the one name the volumes share —
whether VOL02 was scanned alone or together with its folder. The discs are
metadata-only (tests/discgen.simple_bd): what is tested is naming, which
needs the playlists' real lengths, not their pictures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.schemas import AppSettings
from app.services.disc import DiscError, open_disc
from app.services.disc import report as disc_report
from app.services.disc.fs import find_discs
from app.services.disc.plan import Answer, plan, report_of
from app.services.disc.volumes import parse, volume_set
from tests import discgen as g
from tests import udfimg


def season(root: Path, first_clip: int, episodes: int = 3, extras=()) -> Path:
    """A series volume: *episodes* near-equal episodes, a play-all of them,
    and extras of the given lengths."""
    pls = {}
    clips = [f"{first_clip + i:05d}" for i in range(episodes)]
    for i, clip in enumerate(clips):
        pls[f"{i + 1:05d}"] = [(clip, 0, 1420 + i)]
    pls["00000"] = [(clip, 0, 1420 + i) for i, clip in enumerate(clips)]
    for k, seconds in enumerate(extras):
        pls[f"{50 + k:05d}"] = [(f"{900 + k:05d}", 0, seconds)]
    return g.simple_bd(root, pls)


def film(root: Path, minutes: float = 100) -> Path:
    return g.simple_bd(root, {"00800": [("00001", 0, minutes * 60)],
                              "00001": [("00002", 0, 300)]})


def outputs(item, *categories):
    return [v.output for t, v in sorted(item.analysis.verdicts.items())
            if v.category in categories]


# ------------------------------------------------------------ the names

@pytest.mark.parametrize("name, expected", [
    ("SHOW_VOL01", ("SHOW", 1, "VOL01")),
    ("Show Vol.2 [BDMV][FRA]", ("Show", 2, "Vol2")),
    ("Show Volume 10", ("Show", 10, "Volume10")),
    ("THE_SHOW_S1_D2", ("THE_SHOW_S1", 2, "D2")),
    ("Show.S02.BD1", ("Show.S02", 1, "BD1")),
    ("DISC1", ("", 1, "DISC1")),
    ("アニメ 第3巻", ("アニメ", 3, "第3巻")),
])
def test_a_volume_number_is_read_off_the_name(name, expected):
    parsed = parse(name)
    assert (parsed.display, parsed.number, parsed.token) == expected


@pytest.mark.parametrize("name", [
    "Rocky 3",                                # a sequel, not a volume
    "Film (2001)", "FILM2_BD", "Movie.2019.1080p",
    "怪奇館 館の巻 (1992)",                    # 巻 without a number
])
def test_a_bare_number_is_not_a_volume(name):
    assert parse(name) is None


def test_the_volume_word_is_part_of_what_must_match():
    """A show's BD1 and DVD1 are the same episodes twice, not two volumes."""
    assert parse("Show BD1").key != parse("Show DVD1").key
    assert parse("Show Vol 1").key == parse("SHOW_VOL02").key


def test_the_volumes_of_a_box_are_found_side_by_side(tmp_path):
    box = tmp_path / "Show [BD]"
    v2 = season(box / "SHOW_VOL02", 21)
    v1 = season(box / "SHOW_VOL01", 11)
    alone = film(box / "Other Film")
    found = volume_set(v2)
    assert found.name == "SHOW" and found.folder == box
    assert [v.root for v in found.volumes] == [v1, v2]
    assert found.index(v2) == 2 and found.index(alone) == 0
    assert volume_set(v1).id == found.id
    assert volume_set(alone) is None


def test_sequels_side_by_side_are_not_a_set(tmp_path):
    film(tmp_path / "Rocky 3")
    assert volume_set(film(tmp_path / "Rocky 4")) is None


def test_the_same_volume_twice_is_not_a_set(tmp_path):
    """A folder and an image of it: nothing can be numbered from that."""
    v1 = season(tmp_path / "SHOW_VOL01", 11)
    (tmp_path / "SHOW_VOL01.iso").write_bytes(udfimg.build(udfimg.disc_files(v1)))
    season(tmp_path / "SHOW_VOL02", 21)
    assert volume_set(v1) is None


def test_bare_volume_names_take_the_folders_name(tmp_path):
    box = tmp_path / "The Show S1"
    season(box / "DISC1", 11)
    found = volume_set(season(box / "DISC2", 21))
    assert found.name == "The Show S1"
    assert [v.token for v in found.volumes] == ["DISC1", "DISC2"]


# ------------------------------------------------------------ numbering

def test_a_later_volume_numbers_on_from_the_ones_before(tmp_path):
    box = tmp_path / "Show [BD]"
    v1 = season(box / "SHOW_VOL01", 11, extras=(90,))
    v2 = season(box / "SHOW_VOL02", 21, extras=(95, 100))
    settings = AppSettings()
    (alone,) = plan([(str(v2), Answer())], settings)
    assert (alone.name, alone.episode_start, alone.extra_start) == ("SHOW", 4, 2)
    assert outputs(alone, "episode") == ["SHOW.E04.mkv", "SHOW.E05.mkv", "SHOW.E06.mkv"]
    assert outputs(alone, "extra") == ["SHOW.花絮02.mkv", "SHOW.花絮03.mkv"]
    # what is numbered per disc must not collide across the volumes
    assert outputs(alone, "duplicate") == ["SHOW.VOL02.00000.mkv"]
    assert "接着第 1 卷的 3 集，从第 4 集编起" in alone.note
    # scanned with its folder, the same disc gets the same names
    first, second = plan([(str(v1), Answer()), (str(v2), Answer())], settings)
    assert outputs(second, "episode", "extra", "duplicate") == \
        outputs(alone, "episode", "extra", "duplicate")
    assert outputs(first, "episode", "extra") == [
        "SHOW.E01.mkv", "SHOW.E02.mkv", "SHOW.E03.mkv", "SHOW.花絮01.mkv"]
    report = report_of(second, settings, "beside", "")
    assert report.volume.index == 2 and report.volume.members == ["SHOW_VOL01", "SHOW_VOL02"]
    assert report.output_dir == str(box)          # a season lands together


def test_the_users_numbers_carry_through_the_volumes_after(tmp_path):
    box = tmp_path / "Show"
    v1 = season(box / "SHOW_VOL01", 11)
    v2 = season(box / "SHOW_VOL02", 21)
    first, second = plan([(str(v1), Answer(episode_start=14, name="The Show")),
                          (str(v2), Answer(name="The Show"))], AppSettings())
    assert outputs(first, "episode")[0] == "The Show.E14.mkv"
    assert outputs(second, "episode") == [
        "The Show.E17.mkv", "The Show.E18.mkv", "The Show.E19.mkv"]


def test_a_volume_that_is_a_film_keeps_its_own_names(tmp_path):
    """The bonus disc of a box: in the shared name its feature would be
    written as SHOW.mkv, and the volume after it must not count it."""
    box = tmp_path / "Show"
    season(box / "SHOW_VOL01", 11)
    bonus = film(box / "SHOW_VOL02", minutes=50)
    v3 = season(box / "SHOW_VOL03", 31)
    bonus_item, third = plan([(str(bonus), Answer()), (str(v3), Answer())], AppSettings())
    assert not bonus_item.chained and bonus_item.name == "SHOW_VOL02"
    assert outputs(bonus_item, "main") == ["SHOW_VOL02.mkv"]
    assert "判断为电影" in bonus_item.note
    assert third.episode_start == 4 and outputs(third, "episode")[0] == "SHOW.E04.mkv"
    assert "接着第 1 卷的 3 集" in third.note


def test_a_disc_on_its_own_is_numbered_on_its_own(tmp_path):
    (item,) = plan([(str(season(tmp_path / "Some Show", 11)), Answer())], AppSettings())
    assert item.volume is None and item.name == "Some Show" and item.episode_start == 1
    assert outputs(item, "episode")[0] == "Some Show.E01.mkv"


# ------------------------------------------------------------ where to

def test_the_three_places_the_mkvs_can_go(tmp_path):
    folder = open_disc(str(film(tmp_path / "lib" / "Film (1992)")))
    image_path = tmp_path / "lib" / "Other (1996).iso"
    image_path.write_bytes(udfimg.build(udfimg.disc_files(film(tmp_path / "src" / "x"))))
    image = open_disc(str(image_path))
    lib = tmp_path / "lib"
    assert disc_report.output_dir_for(folder) == lib              # beside, the default
    assert disc_report.output_dir_for(image) == lib
    assert disc_report.output_dir_for(folder, "inside") == lib / "Film (1992)"
    # an image has no folder of its own: it gets one, named after it
    assert disc_report.output_dir_for(image, "inside") == lib / "Other (1996)"
    assert disc_report.output_dir_for(folder, "custom", str(tmp_path / "out")) == tmp_path / "out"
    assert disc_report.output_dir_for(folder, "custom", " ") is None
    # free space of a folder that does not exist yet is its parent's
    assert disc_report.free_bytes(lib / "Other (1996)") is not None


# ------------------------------------------------------------ the finder

def test_the_batch_mode_finds_every_disc_but_not_inside_one(tmp_path):
    lib = tmp_path / "lib"
    a = film(lib / "Film A")
    box_1 = season(lib / "Show" / "SHOW_VOL01", 11)
    box_2 = season(lib / "Show" / "SHOW_VOL10", 21)
    image = lib / "Film B.iso"
    image.write_bytes(b"x")
    (lib / "Film A" / "BDMV" / "STREAM" / "stray.iso").write_bytes(b"x")   # inside a disc
    (lib / ".hidden").mkdir()
    film(lib / ".hidden" / "Secret")
    (lib / "notes").mkdir()
    (lib / "notes" / "readme.txt").write_text("x")
    assert find_discs(str(lib)) == [a, image, box_1, box_2]    # VOL10 after VOL01
    assert find_discs(str(lib), recursive=False) == [a, image]
    assert find_discs(str(box_1)) == [box_1]                   # a disc picked as the folder
    assert find_discs(str(box_1 / "BDMV")) == [box_1]
    assert find_discs(str(lib / "notes")) == []
    with pytest.raises(DiscError):
        find_discs(str(lib / "missing"))

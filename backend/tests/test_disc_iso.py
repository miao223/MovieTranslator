"""Disc images: a Blu-ray or DVD read straight out of its .iso.

The images are written by tests/udfimg.py in both layouts a real disc uses
— plain UDF (DVD) and UDF 2.50 with a metadata partition (Blu-ray) — with
fragmented, out-of-order files and overflowing descriptor lists, so the
reader's hard paths run on every test. The promise under test: a disc in an
image analyses and remuxes exactly like the same disc in a folder.
"""

from __future__ import annotations

import pytest

from app.services.disc import DiscError, open_disc
from app.services.disc.analyze import analyze
from app.services.disc.remux import remux_title
from app.services.disc.udf import UdfImage
from tests import discgen as g
from tests import udfimg
from tests.discgen import C, Item, P
from tests.test_mux import _frames

HARD = dict(extent_blocks=3, scramble=True, aed_after=2, efe=True,
            embed_dirs=True, wide_names=True)


def _bd_folder(tmp_path):
    a = g.make_m2ts(1.5, tmp=tmp_path)
    b = g.make_m2ts(1.5, start=100.0, tmp=tmp_path)
    pls = {"00800": g.mpls([Item("00001", a.start, a.end),
                            Item("00002", b.start, b.end, connection=5)],
                           marks=[(1, 0, a.start), (1, 1, b.start)]),
           "00001": g.mpls([Item("00001", a.start, a.end)])}
    root = g.write_bd(tmp_path / "folder", pls,
                      {"00001": g.clpi(a.start, a.end), "00002": g.clpi(b.start, b.end)},
                      streams={"00001": a.data, "00002": b.data})
    return root, a.frames + b.frames


@pytest.mark.parametrize("options", [dict(metadata=True), dict(metadata=True, **HARD),
                                     dict(metadata=False), dict(metadata=False, **HARD)])
def test_every_file_reads_back_byte_for_byte(tmp_path, options):
    files = {
        "BDMV/index.bdmv": b"INDX" + bytes(300),
        "BDMV/STREAM/00001.m2ts": bytes(range(256)) * 97 + b"tail",
        "BDMV/STREAM/00002.m2ts": b"",
        "BDMV/PLAYLIST/00800.mpls": b"MPLS" + bytes(range(200)),
        "VIDEO_TS/VTS_01_1.VOB": bytes(g.SECTOR * 11 + 17),
        "日本語/名前.txt": "ファイル".encode(),
    }
    iso = tmp_path / "disc.iso"
    iso.write_bytes(udfimg.build(files, **options))
    from app.services.disc.udf import IsoFS

    fs = IsoFS(UdfImage(iso))
    for path, data in files.items():
        assert fs.read(path) == data, path
        assert fs.size(path) == len(data)
    assert fs.listdir("BDMV") == ["PLAYLIST", "STREAM", "index.bdmv"]
    assert fs.lookup("bdmv/stream/00001.M2TS") == "BDMV/STREAM/00001.m2ts"
    assert fs.lookup("BDMV/nope") is None


def test_a_blu_ray_image_remuxes_like_the_folder_it_came_from(tmp_path):
    folder, frames = _bd_folder(tmp_path)
    iso = tmp_path / "Some Film (2001).iso"
    iso.write_bytes(udfimg.build(udfimg.disc_files(folder), metadata=True, **HARD))
    disc = open_disc(str(iso))
    assert (disc.kind, disc.source, disc.name) == ("bd", "iso", "Some Film (2001)")
    a = analyze(disc)
    assert a.verdicts["00800.mpls"].category == "main"
    assert a.verdicts["00001.mpls"].category == "duplicate"
    main = next(t for t in disc.titles if t.id == "00800.mpls")
    result = remux_title(disc, main, tmp_path / "out.mkv")
    assert _frames(result.path) == frames


def test_a_dvd_image_remuxes_like_the_folder_it_came_from(tmp_path):
    vob = g.make_vob(2.0, subtitles=[0.5])
    n = len(vob.data) // g.SECTOR
    folder = g.write_dvd(tmp_path / "folder", g.vmg_ifo([(1, 1, 1, 1)]),
                         {1: g.vts_ifo([P([C(0, n - 1, 2.0)])], [[(1, 1)]])},
                         vobs={1: [vob.data]})
    iso = tmp_path / "DVD Film.iso"
    iso.write_bytes(udfimg.build(udfimg.disc_files(folder), metadata=False, extent_blocks=5))
    disc = open_disc(str(iso))
    assert (disc.kind, disc.source, disc.name) == ("dvd", "iso", "DVD Film")
    result = remux_title(disc, disc.titles[0], tmp_path / "dvd.mkv")
    assert _frames(result.path) == vob.frames


def test_something_that_is_not_a_udf_image_says_so(tmp_path):
    fake = tmp_path / "fake.iso"
    fake.write_bytes(bytes(g.SECTOR * 400))
    with pytest.raises(DiscError, match="不是 UDF 镜像"):
        open_disc(str(fake))


def test_an_image_without_a_disc_in_it_says_so(tmp_path):
    iso = tmp_path / "data.iso"
    iso.write_bytes(udfimg.build({"docs/readme.txt": b"hello"}))
    with pytest.raises(DiscError, match="不是蓝光或 DVD 的镜像"):
        open_disc(str(iso))

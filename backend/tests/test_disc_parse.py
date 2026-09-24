"""Reading disc metadata: every parser against its writer in discgen.py."""

from __future__ import annotations

import pytest

from app.services.disc import DiscError, bdmv, dvd, open_disc
from app.services.disc.fs import ConcatReader, Extent, locate, sub_extents
from tests import discgen as g
from tests.discgen import C, Item, P, S, T


# ------------------------------------------------------------------ Blu-ray

def test_a_playlist_reads_back_items_streams_and_chapters():
    streams = (
        S("video", 0x1011, 0x24, fmt=8, rate=1),               # HEVC 2160p
        S("audio", 0x1100, 0x83, "eng", fmt=6, rate=1),        # TrueHD
        S("audio", 0x1101, 0x80, "jpn", fmt=3, rate=1),        # LPCM
        S("audio", 0x1102, 0x81, "fra", fmt=6, rate=1, stream_type=2),
        S("pg", 0x1200, 0x90, "zho"),
        S("textst", 0x1800, 0x92, "eng"),
        S("ig", 0x1400, 0x91, "eng"),
    )
    items = [Item("00001", T(10), T(610), streams=streams),
             Item("00002", T(0), T(300), angles=("00003",), connection=5,
                  streams=streams, secondary=2, dolby_vision=1)]
    pl = bdmv.parse_mpls(g.mpls(items, marks=[(1, 0, T(10)), (1, 0, T(310)),
                                               (2, 1, T(5)), (1, 1, T(0))]), "00800")
    assert [(i.clip, i.in_time, i.out_time) for i in pl.items] == [
        ("00001", T(10), T(610)), ("00002", 0, T(300))]
    assert pl.items[1].angles == ["00003"] and pl.items[1].connection == 5
    assert pl.items[1].secondary == 2 and pl.items[1].dolby_vision == 1
    groups = [(s.group, s.pid, s.language) for s in pl.items[0].streams]
    assert groups == [
        ("video", 0x1011, ""), ("audio", 0x1100, "eng"), ("audio", 0x1101, "jpn"),
        ("audio", 0x1102, "fra"), ("pg", 0x1200, "zho"), ("textst", 0x1800, "eng"),
        ("ig", 0x1400, "eng")]
    assert pl.items[0].streams[3].stream_type == 2
    assert [(m.kind, m.item) for m in pl.marks] == [(1, 0), (1, 0), (2, 1), (1, 1)]
    # entry marks become chapters on the playlist's own timeline; the link
    # point (type 2) does not
    assert bdmv._chapters(pl) == [0.0, 300.0, 600.0]


def test_clip_info_reads_presentation_times_languages_and_the_ep_map():
    ep = [(0, 0), (T(10) & ~0xFF, 5000), (T(700) & ~0xFF, 400_000)]
    info = bdmv.parse_clpi(g.clpi(T(0), T(1200), ep=ep, packets=900_000), "00001")
    assert (info.start, info.end, info.source_packets) == (0, T(1200), 900_000)
    assert info.languages[0x1100] == "jpn" and info.languages[0x1200] == "jpn"
    assert info.ep[0x1011] == ep
    assert info.spn_at(T(300)) == 5000        # the last entry point at or before
    assert info.spn_at(T(700)) == 400_000


def test_movie_objects_name_the_playlists_the_menu_plays():
    objects = bdmv.parse_movie_objects(g.movie_objects([[800], [1, 2], []]))
    assert objects == [{800}, {1, 2}, set()]
    titles, bdj = bdmv.parse_index(g.index_bdmv([("hdmv", 0), ("hdmv", 1)]))
    assert titles == [("hdmv", 0), ("hdmv", 1)] and not bdj
    _, bdj = bdmv.parse_index(g.index_bdmv([("bdj", 0)]))
    assert bdj


def test_a_bd_folder_loads_with_label_reachability_and_problems(tmp_path):
    vc1 = (S("video", 0x1011, 0xEA, fmt=6, rate=1), S("audio", 0x1100, 0x81, "eng"))
    g.write_bd(
        tmp_path / "Film (2001)",
        playlists={"00800": g.mpls([Item("00001", 0, T(5400))]),
                   "00801": g.mpls([Item("00002", 0, T(120), streams=vc1)])},
        clips={"00001": g.clpi(0, T(5400)), "00002": g.clpi(0, T(120), streams=vc1)},
        streams={"00001": bytes(192 * 40)},
        mobj=g.movie_objects([[800]]),
        meta_title="FILM",
    )
    disc = open_disc(str(tmp_path / "Film (2001)"))
    assert (disc.kind, disc.name, disc.label) == ("bd", "Film (2001)", "FILM")
    main, extra = disc.titles
    assert main.id == "00800.mpls" and main.reachable and main.duration == 5400
    assert main.size is not None and not main.problems
    assert not extra.reachable
    assert any("VC-1" in p for p in extra.problems)
    assert any("缺少片段文件" in p for p in extra.problems)


def test_a_metadata_only_copy_analyses_but_says_it_cannot_be_remuxed(tmp_path):
    g.simple_bd(tmp_path / "Meta", {"00800": [("00001", 0, 5400)]})
    disc = open_disc(str(tmp_path / "Meta"))
    assert disc.analysis_only
    assert disc.titles[0].size is None and not disc.titles[0].problems
    assert "只有元数据" in disc.warnings[0]


def test_an_encrypted_clip_is_recognised_and_a_clear_one_is_not():
    clear = b"".join(bytes(4) + b"\x47" + bytes(187) for _ in range(32))
    scrambled = clear[:192 + 16] + bytes(range(256)) * 23
    assert not bdmv.looks_encrypted(clear)
    assert bdmv.looks_encrypted(scrambled[:6144])


# ---------------------------------------------------------------------- DVD

def _dvd(tmp_path, titles, pgcs, ptts, **kw):
    return g.write_dvd(tmp_path / "Some Film", g.vmg_ifo(titles), {1: g.vts_ifo(pgcs, ptts, **kw)},
                       vobs={1: [bytes(g.SECTOR * 4)]})


def test_a_dvd_title_is_its_cells_with_chapters_streams_and_palette(tmp_path):
    cells = [C(0, 99, 600.0), C(100, 199, 900.0, cell_id=2), C(200, 299, 300.0, cell_id=3)]
    pgc = P(cells, programs=(1, 2, 3), audio=(0x8000, 0x8100),
            subp=(0x80000100 | (1 << 16), 0x80000000 | (2 << 16)))
    root = _dvd(tmp_path, [(1, 3, 1, 1)], [pgc], [[(1, 1), (1, 2), (1, 3)]],
                audio=((0, 6, "ja", 1), (4, 2, "en", 3)),
                subp=(("zh", 1), ("en", 9)))
    disc = open_disc(str(root))
    assert disc.kind == "dvd" and disc.name == "Some Film"
    title = disc.titles[0]
    assert title.id == "title01" and title.duration == 1800.0
    assert title.chapters == [0.0, 600.0, 1500.0]
    kinds = [(s.kind, s.codec, s.key, s.language) for s in title.streams]
    assert kinds == [
        ("video", "MPEG-2", 0x1E0, ""),
        ("audio", "AC-3", 0x80, "jpn"),
        ("audio", "LPCM", 0xA1, "eng"),
        # 16:9: the widescreen stream id of each subpicture entry
        ("subtitle", "VobSub", 0x21, "chi"),
        ("subtitle", "VobSub", 0x22, "eng"),
    ]
    assert title.streams[2].commentary and title.streams[4].forced
    assert title.plan.palette[0] == 0x108080
    assert [s.start for s in title.segments] == [0, 100, 200]
    assert [s.end for s in title.segments] == [100, 200, 300]


def test_a_title_plays_its_chain_from_its_first_program_onwards(tmp_path):
    """Chapter 2 of a title may start mid-chain; the cells before its first
    program are not part of it, the cells after its last named one are."""
    cells = [C(0, 9, 30.0), C(10, 99, 600.0, cell_id=2), C(100, 199, 600.0, cell_id=3)]
    root = _dvd(tmp_path, [(1, 1, 1, 1)], [P(cells, programs=(1, 2))], [[(1, 2)]])
    title = open_disc(str(root)).titles[0]
    assert [s.start for s in title.segments] == [10, 100]
    assert title.duration == 1200.0


def test_only_angle_one_of_an_angle_block_plays_and_the_title_is_refused(tmp_path):
    cells = [C(0, 99, 60.0), C(100, 199, 60.0, block_mode=1, block_type=1),
             C(200, 299, 60.0, block_mode=3, block_type=1), C(300, 399, 60.0)]
    root = _dvd(tmp_path, [(2, 1, 1, 1)], [P(cells)], [[(1, 1)]])
    title = open_disc(str(root)).titles[0]
    assert [s.start for s in title.segments] == [0, 100, 300]
    assert title.duration == 180.0
    assert any("多角度" in p for p in title.problems)


def test_ifo_languages_become_matroska_codes():
    assert dvd.iso639_2("ja") == "jpn"
    assert dvd.iso639_2("zh") == "chi"
    assert dvd.iso639_2("hr") == "hrv"
    assert dvd.iso639_2("qq") == "und"
    assert dvd.iso639_2("\x00\x00") == ""


def test_a_scrambled_vob_is_recognised():
    pack = bytearray(g.SECTOR)
    pack[0:4] = b"\x00\x00\x01\xba"
    pack[4] = 0x44                          # MPEG-2 pack header
    pack[13] = 0xF8                         # no stuffing
    pes = b"\x00\x00\x01\xe0" + (100).to_bytes(2, "big") + bytes([0x80, 0x80, 5])
    pack[14:14 + len(pes)] = pes
    assert not dvd.looks_scrambled(bytes(pack))
    pack[20] = 0x80 | (1 << 4)              # PES_scrambling_control = 01
    assert dvd.looks_scrambled(bytes(pack))


# ------------------------------------------------------- finding the disc

def test_any_path_inside_a_disc_finds_the_disc(tmp_path):
    root = g.simple_bd(tmp_path / "Movie.2019.1080p", {"00800": [("00001", 0, 60)]})
    for path in (root, root / "BDMV", root / "BDMV" / "PLAYLIST",
                 root / "BDMV" / "PLAYLIST" / "00800.mpls", root / "BDMV" / "index.bdmv"):
        found = locate(str(path))
        assert found.kind == "bd" and found.root == root
        # the full folder name, dots and all — not Path.stem
        assert found.name == "Movie.2019.1080p"


def test_case_does_not_matter(tmp_path):
    root = tmp_path / "Film"
    g.simple_bd(root, {"00800": [("00001", 0, 60)]})
    (root / "BDMV").rename(root / "bdmv")
    (root / "bdmv" / "PLAYLIST" / "00800.mpls").rename(root / "bdmv" / "PLAYLIST" / "00800.MPLS")
    disc = open_disc(str(root))
    assert [t.id for t in disc.titles] == ["00800.mpls"]


def test_a_dvd_is_found_from_its_folder_its_video_ts_or_bare_ifos(tmp_path):
    cells = [C(0, 99, 600.0)]
    for in_video_ts in (True, False):
        root = tmp_path / f"dvd{in_video_ts}"
        g.write_dvd(root, g.vmg_ifo([(1, 1, 1, 1)]),
                    {1: g.vts_ifo([P(cells)], [[(1, 1)]])}, in_video_ts=in_video_ts)
        for path in ([root, root / "VIDEO_TS", root / "VIDEO_TS" / "VTS_01_0.IFO"]
                     if in_video_ts else [root, root / "VIDEO_TS.IFO"]):
            found = locate(str(path))
            assert found.kind == "dvd" and found.root == root


def test_a_generic_disc_folder_is_named_after_its_parent(tmp_path):
    root = g.simple_bd(tmp_path / "Some Film" / "DISC1", {"00800": [("00001", 0, 60)]})
    assert locate(str(root)).name == "Some Film DISC1"


def test_a_box_set_folder_names_the_discs_in_it(tmp_path):
    g.simple_bd(tmp_path / "Box" / "Disc A", {"00800": [("00001", 0, 60)]})
    g.simple_bd(tmp_path / "Box" / "Disc B", {"00800": [("00001", 0, 60)]})
    with pytest.raises(DiscError, match="2 张原盘.*Disc A、Disc B"):
        locate(str(tmp_path / "Box"))


def test_a_folder_without_a_disc_says_so(tmp_path):
    with pytest.raises(DiscError, match="没有找到 BDMV 或 VIDEO_TS"):
        locate(str(tmp_path))


# ------------------------------------------------------------- the reader

def test_concat_reader_stitches_ranges_of_several_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(bytes(range(100)))
    b.write_bytes(bytes(range(100, 200)))
    extents = [Extent(str(a), 10, 50), Extent(str(b), 0, 100)]
    with ConcatReader(extents) as fh:
        assert fh.read(5) == bytes(range(10, 15))
        fh.seek(45)
        assert fh.read(10) == bytes(range(55, 60)) + bytes(range(100, 105))
        fh.seek(-3, 2)
        assert fh.read() == bytes(range(197, 200))
        assert fh.consumed == 18
    part = sub_extents(extents, 40, 20)
    assert part == [Extent(str(a), 50, 10), Extent(str(b), 0, 10)]

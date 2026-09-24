"""Which titles of a disc to export: the two problems this feature exists for.

1. "The whole film came out, and then the film again in pieces."
2. Extras: listed, numbered, ticked or not by a setting.

Every scenario is a small disc built by tests/discgen.py; the numbers are
chosen to sit clearly on one side of each threshold, and the tests that
exist to pin a threshold say so.
"""

from __future__ import annotations

from app.services.disc import open_disc
from app.services.disc.analyze import analyze
from tests import discgen as g
from tests.discgen import C, Item, P, S, T


def bd(tmp_path, playlists, name="Film", mobj=None, streams=None):
    """playlists: name -> [(clip, in s, out s[, connection])]."""
    clips: dict = {}
    pls = {}
    for pl, items in playlists.items():
        built = []
        for spec in items:
            clip, a, b = spec[:3]
            conn = spec[3] if len(spec) > 3 else 1
            built.append(Item(clip, T(a), T(b), connection=conn,
                              streams=streams or g.DEFAULT_STREAMS))
            clips[clip] = max(clips.get(clip, 0.0), b)
        pls[pl] = g.mpls(built, chapters_every=300)
    root = g.write_bd(tmp_path / name, pls,
                      {c: g.clpi(0, T(s)) for c, s in clips.items()}, mobj=mobj)
    return open_disc(str(root))


def v(analysis, title_id):
    return analysis.verdicts[title_id]


# ---------------------------------------------------- 1. whole + pieces

def test_the_film_and_its_segments_keep_the_film(tmp_path):
    """The reported problem: a film stored as three clips, offered both as
    one playlist and as a playlist per clip. Exported naively that is the
    film twice."""
    disc = bd(tmp_path, {
        "00800": [("00001", 0, 2100), ("00002", 0, 2820, 5), ("00003", 0, 2280, 5)],
        "00001": [("00001", 0, 2100)],
        "00002": [("00002", 0, 2820)],
        "00003": [("00003", 0, 2280)],
    })
    a = analyze(disc)
    assert a.mode == "movie"
    assert v(a, "00800.mpls").category == "main" and v(a, "00800.mpls").selected
    for piece in ("00001.mpls", "00002.mpls", "00003.mpls"):
        assert v(a, piece).category == "duplicate"
        assert v(a, piece).duplicate_of == "00800.mpls"
        assert not v(a, piece).selected
    assert v(a, "00800.mpls").output == "Film.mkv"


def test_equal_pieces_of_a_seamless_film_are_still_not_episodes(tmp_path):
    """Three pieces of exactly the same length would pass for episodes on
    length alone. They play on from each other seamlessly, which episodes
    strung together by a play-all do not — so this stays a film."""
    disc = bd(tmp_path, {
        "00800": [("00001", 0, 2400), ("00002", 0, 2400, 5), ("00003", 0, 2400, 6)],
        "00001": [("00001", 0, 2400)],
        "00002": [("00002", 0, 2400)],
        "00003": [("00003", 0, 2400)],
    })
    a = analyze(disc)
    assert a.mode == "movie" and v(a, "00800.mpls").category == "main"
    assert a.series_choice and not a.series_default
    # …and the user can still say otherwise
    a = analyze(disc, series=True)
    assert a.mode == "series"
    assert [v(a, f"0000{i}.mpls").output for i in (1, 2, 3)] == [
        "Film.E01.mkv", "Film.E02.mkv", "Film.E03.mkv"]
    assert v(a, "00800.mpls").category == "duplicate"


def test_a_series_play_all_keeps_the_episodes_in_their_own_order(tmp_path):
    """Episodes share the opening and ending clips; the play-all is all of
    them end to end. Playlist numbers do not follow the episodes here — the
    order comes from each episode's own clip."""
    op, ed = ("00090", 0, 90), ("00091", 0, 90)
    disc = bd(tmp_path, {
        "00001": [op, ("00013", 0, 1300), ed],
        "00002": [op, ("00011", 0, 1320), ed],
        "00003": [op, ("00012", 0, 1310), ed],
        "00010": [op, ("00011", 0, 1320), ed, op, ("00012", 0, 1310), ed,
                  op, ("00013", 0, 1300), ed],
    }, name="Show")
    a = analyze(disc)
    assert a.mode == "series"
    assert v(a, "00010.mpls").category == "duplicate"
    assert "全部播放" in v(a, "00010.mpls").reason
    assert [(v(a, t).number, v(a, t).output) for t in ("00002.mpls", "00003.mpls", "00001.mpls")] \
        == [(1, "Show.E01.mkv"), (2, "Show.E02.mkv"), (3, "Show.E03.mkv")]
    assert all(v(a, t).selected for t in ("00001.mpls", "00002.mpls", "00003.mpls"))


def test_episode_numbers_start_where_the_user_says(tmp_path):
    op = ("00090", 0, 90)
    disc = bd(tmp_path, {"00001": [op, ("00011", 0, 1300)],
                         "00002": [op, ("00012", 0, 1300)]}, name="Show Vol.2")
    a = analyze(disc, episode_start=5, name="Show")
    assert [v(a, t).output for t in ("00001.mpls", "00002.mpls")] == [
        "Show.E05.mkv", "Show.E06.mkv"]


def test_identical_playlists_keep_the_one_the_menu_plays(tmp_path):
    items = [("00001", 0, 5400)]
    disc = bd(tmp_path, {"00800": items, "00801": items},
              mobj=g.movie_objects([[801]]))
    a = analyze(disc)
    assert v(a, "00801.mpls").category == "main"
    assert v(a, "00800.mpls").category == "duplicate"
    assert v(a, "00800.mpls").duplicate_of == "00801.mpls"
    assert "完全相同" in v(a, "00800.mpls").reason


def test_a_repeated_clip_is_a_menu_loop(tmp_path):
    disc = bd(tmp_path, {"00800": [("00001", 0, 5400)],
                         "00005": [("00009", 0, 30)] * 5})
    a = analyze(disc)
    assert v(a, "00005.mpls").category == "loop" and not v(a, "00005.mpls").selected


def test_a_piece_of_an_episode_is_still_just_a_piece(tmp_path):
    op = ("00090", 0, 90)
    disc = bd(tmp_path, {"00001": [op, ("00011", 0, 1300)],
                         "00002": [op, ("00012", 0, 1300)],
                         "00003": [op, ("00013", 0, 1300)],
                         "00050": [op]})
    a = analyze(disc)
    assert v(a, "00050.mpls").category == "duplicate"


# ------------------------------------------------------ different cuts

def test_a_shorter_cut_inside_the_longer_one_is_another_version(tmp_path):
    """Everything the theatrical cut shows is in the extended cut, but not
    in one stretch — it skips a scene. That is a different edit, not a
    piece of the film, and it is offered unticked."""
    a_, b_, x_, c_ = ("00001", 0, 3000), ("00002", 0, 600), ("00003", 0, 600), ("00004", 0, 3000)
    disc = bd(tmp_path, {"00800": [a_, b_, x_, c_], "00801": [a_, b_, c_]})
    a = analyze(disc)
    assert v(a, "00800.mpls").category == "main"
    theatrical = v(a, "00801.mpls")
    assert theatrical.category == "variant" and not theatrical.selected
    assert "跳过" in theatrical.reason and theatrical.output == "Film.版本2.mkv"


def test_a_cut_with_scenes_of_its_own_is_another_version(tmp_path):
    a_, c_ = ("00001", 0, 3000), ("00004", 0, 3000)
    disc = bd(tmp_path, {"00800": [a_, ("00002", 0, 900), c_],
                         "00801": [a_, ("00005", 0, 300), c_]})
    a = analyze(disc)
    other = v(a, "00801.mpls")
    assert other.category == "variant" and not other.selected
    assert "共享" in other.reason


# ------------------------------------------------------------ 2. extras

def test_extras_are_numbered_in_disc_order_and_ticked_by_the_setting(tmp_path):
    disc = bd(tmp_path, {
        "00800": [("00001", 0, 6000)],
        "00003": [("00013", 0, 1500)],
        "00001": [("00011", 0, 300)],
        "00002": [("00012", 0, 720)],
        "00009": [("00019", 0, 12)],
    })
    a = analyze(disc)
    assert a.mode == "movie" and not a.series_choice   # the film dwarfs the rest
    assert [(v(a, t).category, v(a, t).output) for t in
            ("00001.mpls", "00002.mpls", "00003.mpls")] == [
        ("extra", "Film.花絮01.mkv"), ("extra", "Film.花絮02.mkv"),
        ("extra", "Film.花絮03.mkv")]
    assert all(v(a, t).selected for t in ("00001.mpls", "00002.mpls", "00003.mpls"))
    assert v(a, "00009.mpls").category == "short" and not v(a, "00009.mpls").selected

    off = analyze(disc, export_extras=False)
    assert not any(v(off, t).selected for t in ("00001.mpls", "00002.mpls", "00003.mpls"))
    # unticking never renumbers: the names stay what they were
    assert v(off, "00002.mpls").output == "Film.花絮02.mkv"

    longer = analyze(disc, min_seconds=400)
    assert v(longer, "00001.mpls").category == "short"
    assert v(longer, "00002.mpls").output == "Film.花絮01.mkv"


def test_titles_that_cannot_be_written_cannot_be_ticked(tmp_path):
    vc1 = (S("video", 0x1011, 0xEA, fmt=6, rate=1), S("audio", 0x1100, 0x81, "eng"))
    disc = bd(tmp_path, {"00800": [("00001", 0, 5400)]}, streams=vc1)
    a = analyze(disc)
    verdict = v(a, "00800.mpls")
    assert verdict.category == "main" and not verdict.selectable and not verdict.selected
    assert verdict.reason.startswith("不支持：VC-1")


def test_an_encrypted_disc_has_nothing_to_tick(tmp_path):
    disc = bd(tmp_path, {"00800": [("00001", 0, 5400)], "00001": [("00011", 0, 300)]})
    disc.encrypted = True
    a = analyze(disc)
    assert not any(x.selectable or x.selected for x in a.verdicts.values())
    assert any("加密" in w for w in a.warnings)


# ------------------------------------------------- 整片 or 分集, when unclear

def test_two_comparable_long_titles_are_a_question_for_the_user(tmp_path):
    disc = bd(tmp_path, {"00001": [("00011", 0, 3000)], "00002": [("00012", 0, 2700)]})
    a = analyze(disc)
    assert a.mode == "movie" and a.series_choice and not a.series_default
    assert v(a, "00001.mpls").category == "main"
    assert v(a, "00002.mpls").category == "extra"
    a = analyze(disc, series=True)
    assert [v(a, t).category for t in ("00001.mpls", "00002.mpls")] == ["episode", "episode"]


def test_a_lone_play_all_is_cut_up_only_when_two_signs_agree(tmp_path):
    items = [("00011", 0, 1400), ("00012", 0, 1420), ("00013", 0, 1410)]
    # a series-like name and three even pieces: cut into episodes
    a = analyze(bd(tmp_path, {"00001": items}, name="Show Vol.1"))
    assert a.mode == "series" and a.series_default
    assert v(a, "00001.mpls").category == "duplicate"
    assert [v(a, f"00001.mpls#{k}").output for k in (1, 2, 3)] == [
        "Show Vol.1.E01.mkv", "Show Vol.1.E02.mkv", "Show Vol.1.E03.mkv"]
    # the same pieces under a film's name: kept whole, but asked
    disc = bd(tmp_path, {"00001": items}, name="Film")
    a = analyze(disc)
    assert a.mode == "movie" and a.series_choice and not a.series_default
    assert v(a, "00001.mpls").category == "main"
    a = analyze(disc, series=True)
    assert a.mode == "series" and "00001.mpls#2" in a.verdicts


def test_a_seamless_film_is_never_offered_as_a_play_all(tmp_path):
    items = [("00011", 0, 1400), ("00012", 0, 1420, 5), ("00013", 0, 1410, 5)]
    a = analyze(bd(tmp_path, {"00001": items}, name="Show Vol.1"))
    assert a.mode == "movie" and not a.series_choice


# ------------------------------------------------------------------- DVD

def test_dvd_chapter_titles_are_pieces_of_the_film(tmp_path):
    cells = [C(0, 999, 2000.0), C(1000, 1999, 2400.0, cell_id=2),
             C(2000, 2999, 1800.0, cell_id=3)]
    pgcs = [P(cells, programs=(1, 2, 3))] + [P([c]) for c in cells]
    ptts = [[(1, 1), (1, 2), (1, 3)], [(2, 1)], [(3, 1)], [(4, 1)]]
    titles = [(1, 3, 1, 1), (1, 1, 1, 2), (1, 1, 1, 3), (1, 1, 1, 4)]
    root = g.write_dvd(tmp_path / "Film", g.vmg_ifo(titles), {1: g.vts_ifo(pgcs, ptts)})
    a = analyze(open_disc(str(root)))
    assert v(a, "title01").category == "main"
    assert [v(a, f"title0{n}").category for n in (2, 3, 4)] == ["duplicate"] * 3


def test_a_dvd_series_with_a_play_all_title(tmp_path):
    eps = [C(1000 * k, 1000 * k + 999, 1500.0 + k, cell_id=k + 1) for k in range(4)]
    pgcs = [P(eps, programs=(1, 2, 3, 4))] + [P([c]) for c in eps]
    ptts = [[(1, 1), (1, 2), (1, 3), (1, 4)]] + [[(k, 1)] for k in range(2, 6)]
    titles = [(1, 4, 1, 1)] + [(1, 1, 1, k) for k in range(2, 6)]
    root = g.write_dvd(tmp_path / "Show", g.vmg_ifo(titles), {1: g.vts_ifo(pgcs, ptts)})
    a = analyze(open_disc(str(root)))
    assert a.mode == "series"
    assert v(a, "title01").category == "duplicate"
    assert [v(a, f"title0{n}").number for n in (2, 3, 4, 5)] == [1, 2, 3, 4]


def test_a_very_short_title_is_not_contained_in_something_it_shares_nothing_with(tmp_path):
    """The two-second allowance for rounding at clip boundaries once made
    every title shorter than two seconds look like a piece of the film."""
    disc = bd(tmp_path, {"00800": [("00001", 0, 5400)], "00002": [("00012", 0, 1.5)]})
    a = analyze(disc, min_seconds=0)
    assert v(a, "00002.mpls").category == "extra"


def test_a_title_ticked_against_the_advice_still_gets_a_name(tmp_path):
    disc = bd(tmp_path, {"00800": [("00001", 0, 5400)], "00001": [("00001", 0, 5400)],
                         "00009": [("00019", 0, 12)]})
    a = analyze(disc)
    # identical content: the lower number represents it, the other is named by its own
    assert v(a, "00800.mpls").category == "duplicate"
    assert v(a, "00800.mpls").output == "Film.00800.mkv"
    assert v(a, "00009.mpls").output == "Film.00009.mkv"


def test_a_silent_title_is_not_an_extra(tmp_path):
    """Measured on a series Blu-ray: its one title with no audio at all was a
    minute of menu background, and was listed as 花絮01."""
    disc = bd(tmp_path, {"00800": [("00001", 0, 5400)], "00009": [("00019", 0, 60)],
                         "00003": [("00013", 0, 300)]})
    silent = next(t for t in disc.titles if t.id == "00009.mpls")
    silent.streams = [s for s in silent.streams if s.kind == "video"]
    a = analyze(disc)
    assert v(a, "00009.mpls").category == "silent" and not v(a, "00009.mpls").selected
    assert v(a, "00003.mpls").output == "Film.花絮01.mkv"


def test_a_slideshow_is_not_an_extra(tmp_path):
    """Blu-ray marks slideshows in the clip info (application type 3,
    browsable slideshow) — seven ten-second stills on a real disc."""
    disc = bd(tmp_path, {"00800": [("00001", 0, 5400)], "00015": [("00081", 0, 60)]})
    next(t for t in disc.titles if t.id == "00015.mpls").still = True
    a = analyze(disc)
    assert v(a, "00015.mpls").category == "still" and not v(a, "00015.mpls").selected

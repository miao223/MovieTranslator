"""Multi-track videos: enumeration, selection and extraction of the right track."""

import fractions
from pathlib import Path

import numpy as np
import pytest

from app.services.audio import (
    canon_language,
    describe_track,
    extract_audio,
    language_name,
    list_tracks,
    pick_track,
)

SR = 16_000


def make_multitrack_video(path: Path, tracks=(("jpn", "日本語", 0.0), ("eng", "English dub", 0.6))):
    """Synthesize an mkv with one video stream and N audio tracks.

    Each track carries a constant-amplitude tone (amplitude given per track),
    so a test can tell from the extracted WAV *which* track was decoded.
    """
    import av

    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=8)
        video.width, video.height = 64, 48
        video.pix_fmt = "yuv420p"

        audio_streams = []
        for lang, title, _amp in tracks:
            st = container.add_stream("aac", rate=SR)
            st.layout = "mono"
            st.metadata["language"] = lang
            st.metadata["title"] = title
            audio_streams.append(st)

        for i in range(16):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts = i
            for packet in video.encode(frame):
                container.mux(packet)

        for st, (_lang, _title, amp) in zip(audio_streams, tracks):
            pts = 0
            for _ in range(16):
                t = (np.arange(1024) + pts) / SR
                samples = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
                af = av.AudioFrame.from_ndarray(
                    samples.reshape(1, -1), format="fltp", layout="mono"
                )
                af.sample_rate = SR
                af.time_base = fractions.Fraction(1, SR)
                af.pts = pts
                pts += 1024
                for packet in st.encode(af):
                    container.mux(packet)
            for packet in st.encode(None):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
    return path


@pytest.fixture(scope="module")
def multitrack(tmp_path_factory):
    return make_multitrack_video(tmp_path_factory.mktemp("mt") / "dual.mkv")


# --------------------------------------------------------------- languages


def test_canon_language_folds_variants():
    assert canon_language("ja") == canon_language("jpn") == canon_language("JAP") == "jpn"
    assert canon_language("zho") == canon_language("zh") == "chi"
    assert canon_language("") == ""


def test_language_name_falls_back_to_the_code():
    assert language_name("ja") == "日语"
    assert language_name("") == "未标注语言"
    assert language_name("xyz") == "xyz"  # unknown tag still displays something


# -------------------------------------------------------------- list_tracks


def test_list_tracks_reads_metadata(multitrack):
    tracks = list_tracks(multitrack)
    assert len(tracks) == 2
    assert [t["language"] for t in tracks] == ["jpn", "eng"]
    assert [t["language_name"] for t in tracks] == ["日语", "英语"]
    assert tracks[0]["title"] == "日本語"
    assert tracks[0]["codec"] == "AAC"
    assert tracks[0]["channel_name"] == "单声道"
    # indices are container-wide (stream 0 is the video), not 0-based per type
    assert [t["index"] for t in tracks] == [1, 2]


def test_list_tracks_rejects_audioless_file(tmp_path):
    from tests.test_frame_translation import make_test_video

    silent = make_test_video(tmp_path / "silent.mp4", seconds=1) or tmp_path / "silent.mp4"
    with pytest.raises(ValueError, match="没有音频流"):
        list_tracks(silent)


def test_describe_track_is_readable(multitrack):
    text = describe_track(list_tracks(multitrack)[0])
    assert "音轨 #1" in text and "日语" in text and "日本語" in text


# --------------------------------------------------------------- pick_track


def _tracks(*specs):
    return [
        {"index": i, "language": lang, "default": default}
        for i, lang, default in specs
    ]


def test_pick_track_explicit_index_wins():
    tracks = _tracks((1, "jpn", True), (2, "eng", False))
    assert pick_track(tracks, index=2, language="jpn")["index"] == 2


def test_pick_track_falls_back_when_index_is_gone():
    """A stale index (path changed after picking) must not fail the job."""
    tracks = _tracks((1, "jpn", False), (2, "eng", True))
    assert pick_track(tracks, index=7)["index"] == 2  # the default track


def test_pick_track_by_language_accepts_aliases():
    tracks = _tracks((1, "eng", True), (2, "jpn", False))
    assert pick_track(tracks, language="ja")["index"] == 2
    assert pick_track(tracks, language="jpn")["index"] == 2


def test_pick_track_language_miss_falls_back_to_default():
    tracks = _tracks((1, "eng", False), (2, "ger", True))
    assert pick_track(tracks, language="jpn")["index"] == 2


def test_pick_track_without_hints_prefers_default_then_first():
    assert pick_track(_tracks((1, "eng", False), (2, "jpn", True)))["index"] == 2
    assert pick_track(_tracks((1, "eng", False), (2, "jpn", False)))["index"] == 1


# ------------------------------------------------------------ extract_audio


def _peak(wav: Path) -> float:
    import wave

    with wave.open(str(wav), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return float(np.abs(data).max()) / 32768 if len(data) else 0.0


def test_extract_audio_decodes_the_requested_track(multitrack, tmp_path):
    """Track 1 is silent, track 2 carries a tone — the WAV proves which ran."""
    quiet = extract_audio(multitrack, tmp_path / "t1.wav", track_index=1)
    loud = extract_audio(multitrack, tmp_path / "t2.wav", track_index=2)
    assert _peak(quiet) < 0.05
    assert _peak(loud) > 0.3


def test_extract_audio_rejects_unknown_track(multitrack, tmp_path):
    with pytest.raises(ValueError, match="音轨 #9"):
        extract_audio(multitrack, tmp_path / "bad.wav", track_index=9)


# -------------------------------------------------------------------- API


def test_audio_tracks_endpoint(multitrack):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        r = client.get("/api/media/audio-tracks", params={"path": str(multitrack)})
        assert r.status_code == 200
        body = r.json()
        assert [t["index"] for t in body] == [1, 2]
        assert body[1]["language_name"] == "英语"

        bad = client.get("/api/media/audio-tracks", params={"path": "/no/such/file.mkv"})
        assert bad.status_code == 400

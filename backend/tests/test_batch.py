from pathlib import Path

import pytest

from app.core.media import scan_videos


def make_tree(root: Path):
    (root / "a.mkv").write_bytes(b"x")
    (root / "b.mp4").write_bytes(b"x")
    (root / "b.srt").write_text("1", encoding="utf-8")  # b has subs already
    (root / "notes.txt").write_text("x", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.avi").write_bytes(b"x")
    hidden = root / ".hidden"
    hidden.mkdir()
    (hidden / "d.mkv").write_bytes(b"x")


def test_scan_recursive_with_skip(tmp_path):
    make_tree(tmp_path)
    videos, skipped = scan_videos(tmp_path, recursive=True, skip_existing_srt=True)
    names = [v.name for v in videos]
    assert names == ["a.mkv", "c.avi"]  # b skipped (srt), d hidden, txt ignored
    assert [s.name for s in skipped] == ["b.mp4"]


def test_scan_non_recursive_no_skip(tmp_path):
    make_tree(tmp_path)
    videos, skipped = scan_videos(tmp_path, recursive=False, skip_existing_srt=False)
    assert [v.name for v in videos] == ["a.mkv", "b.mp4"]
    assert skipped == []


def test_scan_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_videos(tmp_path / "nope", True, True)


def test_batch_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    make_tree(tmp_path)
    c = TestClient(app)

    scan = c.get(
        "/api/batch/scan",
        params={"path": str(tmp_path), "recursive": True, "skip_existing": True},
    ).json()
    assert scan["total"] == 2 and len(scan["skipped"]) == 1

    r = c.post("/api/batch", json={"directory": str(tmp_path)})
    assert r.status_code == 200
    b = r.json()
    assert b["total"] == 2
    batch_id = b["id"]

    # cancel immediately; fake videos would fail extraction anyway — either
    # way every job must reach a terminal state without blocking the batch
    c.post(f"/api/batch/{batch_id}/cancel")
    import time

    for _ in range(100):
        st = c.get(f"/api/batch/{batch_id}").json()
        if st["pending"] + st["running"] == 0:
            break
        time.sleep(0.1)
    assert st["pending"] + st["running"] == 0
    assert st["done"] + st["failed"] + st["cancelled"] == 2

    # empty dir → 400
    empty = tmp_path / "empty"
    empty.mkdir()
    assert c.post("/api/batch", json={"directory": str(empty)}).status_code == 400


def test_batch_passes_audio_language_to_each_job(tmp_path, monkeypatch):
    """Batches select tracks by language tag, so every job must carry it."""
    from app.models.schemas import BatchRequest, JobStatus
    from app.services import batch as batch_mod

    make_tree(tmp_path)
    seen = []
    fakes = {}

    class FakeJob:
        def __init__(self, req):
            seen.append(req)
            self.id = f"job{len(seen)}"
            self.status = JobStatus(id=self.id)
            fakes[self.id] = self

    monkeypatch.setattr(batch_mod.job_manager, "create", FakeJob)
    monkeypatch.setattr(batch_mod.job_manager, "get", lambda jid: fakes[jid])
    batch_mod.batch_manager.create(
        BatchRequest(directory=str(tmp_path), audio_language="jpn")
    )
    assert len(seen) == 2
    assert {r.audio_language for r in seen} == {"jpn"}
    assert all(r.audio_track is None for r in seen)

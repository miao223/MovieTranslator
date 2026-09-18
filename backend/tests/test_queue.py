"""The persistent queue: what it freezes, what it does not, and what survives.

Every runner test drives `queue_manager.step()` by hand. The thread is only
a `while` loop around that one call, so testing the state machine directly
means no sleeps and no races — and the assertions are about the decision
rather than about timing.
"""

from __future__ import annotations

import json

import pytest

from app.models.schemas import AppSettings, JobRequest, JobStatus
from app.services import jobqueue
from app.services.jobqueue import QueueManager, QueueStore


def req(path: str = "/film.mkv", **kwargs) -> JobRequest:
    return JobRequest(video_path=path, target_language="简体中文", **kwargs)


class FakeJob:
    """Stands in for pipeline.Job: an id, a status, nothing running."""

    def __init__(self, request, settings=None):
        self.id = f"job-{request.video_path}"
        self.request = request
        self.settings = settings
        self.status = JobStatus(id=self.id, video_path=request.video_path)
        self.cancelled = False


class FakeManager:
    """Stands in for pipeline.manager, recording what it was asked to run."""

    def __init__(self, fail_on=()):
        self.jobs = {}
        self.created = []
        self.fail_on = set(fail_on)

    def create(self, request, settings=None):
        if request.video_path in self.fail_on:
            raise FileNotFoundError(f"片源文件不存在: {request.video_path}")
        job = FakeJob(request, settings)
        self.jobs[job.id] = job
        self.created.append(job)
        return job

    def get(self, job_id):
        return self.jobs[job_id]

    def cancel(self, job_id):
        self.jobs[job_id].cancelled = True
        self.jobs[job_id].status.stage = "cancelled"

    def finish(self, index=-1, stage="done", srt="out.srt"):
        job = self.created[index]
        job.status.stage = stage
        job.status.srt_filename = srt


@pytest.fixture
def runner(settings_file, monkeypatch):
    """A queue manager wired to a fake pipeline, on a temp settings dir."""
    settings_file()
    fake = FakeManager()
    import app.services.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "manager", fake)
    manager = QueueManager()
    manager.store.load()
    return manager, fake


# --------------------------------------------------- the headline promise


def test_settings_are_frozen_at_enqueue_not_at_run(settings_file):
    """A, change settings, B — and A is untouched by the change.

    This is the whole feature. Before it, both jobs read the settings when
    they started, so whatever you changed while A waited its turn silently
    applied to A as well.
    """
    settings_file(llm__model="model-A")
    store = QueueStore()
    store.load()
    store.add(req("/a.mkv"), jobqueue.snapshot())

    settings_file(llm__model="model-B")
    store.add(req("/b.mkv"), jobqueue.snapshot())

    assert [e.settings.llm.model for e in store.entries] == ["model-A", "model-B"]
    assert jobqueue.settings_hash(store.entries[0].settings) \
        != jobqueue.settings_hash(store.entries[1].settings)


def test_a_queued_job_runs_with_its_snapshot_but_a_live_api_key(settings_file):
    """Frozen everything, except the credential.

    A key that expired or was rotated while the job waited should repair
    the queue, not break it — so it is the one thing read at run time.
    """
    from app.services.pipeline import Job, _settings_for

    settings_file(llm__model="model-A", llm__api_key="sk-old")
    frozen = jobqueue.snapshot()
    settings_file(llm__model="model-B", llm__api_key="sk-new")

    job = Job(req(), audio_only=True, settings=frozen)
    resolved = _settings_for(job)

    assert resolved.llm.model == "model-A"      # frozen
    assert resolved.llm.api_key == "sk-new"     # live
    # and the shared snapshot was not mutated on the way through
    assert frozen.llm.api_key == ""


def test_a_job_with_no_snapshot_still_reads_settings_live(settings_file):
    """The immediate path keeps working: settings=None means "read them now"."""
    from app.services.pipeline import Job, _settings_for

    settings_file(llm__model="model-live")
    job = Job(req(), audio_only=True)
    assert _settings_for(job).llm.model == "model-live"


def test_a_snapshot_never_contains_a_credential(settings_file):
    """queue.json sits next to settings.json and must not duplicate secrets."""
    settings_file(llm__api_key="sk-SECRET", server__access_token="TOK-SECRET")
    store = QueueStore()
    store.load()
    store.add(req(), jobqueue.snapshot())

    written = jobqueue.queue_path().read_text(encoding="utf-8")
    assert "sk-SECRET" not in written
    assert "TOK-SECRET" not in written


def test_the_fingerprint_ignores_settings_that_cannot_change_the_output(settings_file):
    """Toggling LAN access must not relabel every later entry."""
    settings_file(llm__model="m")
    before = jobqueue.settings_hash(jobqueue.snapshot())
    settings_file(llm__model="m", server__lan_access=True, work_dir="/tmp/elsewhere")
    assert jobqueue.settings_hash(jobqueue.snapshot()) == before


# ------------------------------------------------------------- the runner


def test_entries_run_in_order_and_only_one_at_a_time(runner):
    manager, fake = runner
    manager.store.add(req("/a.mkv"), jobqueue.snapshot())
    manager.store.add(req("/b.mkv"), jobqueue.snapshot())

    manager.step()
    assert [j.request.video_path for j in fake.created] == ["/a.mkv"]

    manager.step()      # A still running: B must not start
    assert len(fake.created) == 1

    fake.finish(stage="done")
    manager.step()      # notices A finished
    manager.step()      # starts B
    assert [j.request.video_path for j in fake.created] == ["/a.mkv", "/b.mkv"]
    assert manager.store.entries[0].status == "done"
    assert manager.store.entries[0].result_srt == "out.srt"


def test_every_entry_runs_with_its_own_snapshot(runner, settings_file):
    manager, fake = runner
    settings_file(llm__model="model-A")
    manager.store.add(req("/a.mkv"), jobqueue.snapshot())
    settings_file(llm__model="model-B")
    manager.store.add(req("/b.mkv"), jobqueue.snapshot())

    manager.step()
    fake.finish(stage="done")
    manager.step()
    manager.step()

    assert [j.settings.llm.model for j in fake.created] == ["model-A", "model-B"]


def test_cancelling_a_queued_entry_is_instant_and_creates_no_job(runner):
    """The improvement over today: a waiting entry stops without being run.

    Before the queue, cancelling something that had not started did nothing
    observable until it reached the front of the line — which, behind a
    two-hour film, is not cancelling.
    """
    manager, fake = runner
    entry = manager.store.add(req("/a.mkv"), jobqueue.snapshot())

    assert manager.cancel(entry.id) == "cancelled"
    assert entry.status == "cancelled"
    manager.step()
    assert fake.created == []


def test_cancelling_a_running_entry_forwards_to_the_job(runner):
    manager, fake = runner
    entry = manager.store.add(req("/a.mkv"), jobqueue.snapshot())
    manager.step()

    manager.cancel(entry.id)
    assert fake.created[0].cancelled is True
    manager.step()
    assert manager.store.get(entry.id).status == "cancelled"


def test_a_failed_start_does_not_stop_the_queue(runner):
    """An unattended queue must not be stopped by one missing file."""
    manager, fake = runner
    fake.fail_on.add("/gone.mkv")
    manager.store.add(req("/gone.mkv"), jobqueue.snapshot())
    manager.store.add(req("/b.mkv"), jobqueue.snapshot())

    manager.step()      # first one fails at creation
    manager.step()      # second one still starts

    assert manager.store.entries[0].status == "failed"
    assert "片源文件不存在" in manager.store.entries[0].error
    assert [j.request.video_path for j in fake.created] == ["/b.mkv"]


def test_pause_stops_new_entries_and_never_touches_the_running_one(runner):
    manager, fake = runner
    manager.store.add(req("/a.mkv"), jobqueue.snapshot())
    manager.store.add(req("/b.mkv"), jobqueue.snapshot())
    manager.step()

    manager.store.set_paused(True)
    fake.finish(stage="done")
    manager.step()      # records A as done...
    manager.step()      # ...and must not start B

    assert manager.store.entries[0].status == "done"    # was not interrupted
    assert len(fake.created) == 1

    manager.store.set_paused(False)
    manager.step()
    assert len(fake.created) == 2


def test_pause_survives_a_restart(settings_file):
    """Silently resuming after a restart would burn tokens unattended."""
    settings_file()
    store = QueueStore()
    store.load()
    store.set_paused(True)

    again = QueueStore()
    again.load()
    assert again.paused is True


def test_retry_reuses_the_snapshot_and_fresh_retry_takes_a_new_one(runner, settings_file):
    manager, fake = runner
    settings_file(llm__model="model-A")
    entry = manager.store.add(req("/a.mkv"), jobqueue.snapshot())
    entry.status = "failed"

    settings_file(llm__model="model-B")
    same = manager.retry(entry.id)
    assert same.settings.llm.model == "model-A"

    fresh = manager.retry(entry.id, fresh=True)
    assert fresh.settings.llm.model == "model-B"
    assert fresh.id != entry.id != same.id


def test_retry_refuses_an_entry_that_has_not_finished(runner):
    manager, _ = runner
    entry = manager.store.add(req(), jobqueue.snapshot())
    with pytest.raises(ValueError, match="还没有结束"):
        manager.retry(entry.id)


# ----------------------------------------------------------- persistence


def test_the_queue_survives_a_restart(settings_file):
    settings_file(llm__model="model-A")
    store = QueueStore()
    store.load()
    store.add(req("/a.mkv"), jobqueue.snapshot())
    store.add(req("/b.mkv"), jobqueue.snapshot())

    again = QueueStore()
    again.load()
    assert [e.title for e in again.entries] == ["/a.mkv", "/b.mkv"]
    assert again.entries[0].settings.llm.model == "model-A"
    assert again.entries[0].request.target_language == "简体中文"


def test_an_entry_that_was_running_when_the_process_died_is_queued_again(settings_file):
    settings_file()
    store = QueueStore()
    store.load()
    entry = store.add(req("/a.mkv"), jobqueue.snapshot())
    entry.status = "running"
    entry.job_id = "job-gone"
    entry.started_at = 123.0
    store.save()

    again = QueueStore()
    again.load()
    assert again.entries[0].status == "queued"
    assert again.entries[0].job_id == ""
    assert "中断" in again.entries[0].note


def test_the_interruption_is_still_visible_after_it_runs_again(settings_file):
    """Being restarted must not erase the fact that it was interrupted.

    The note says "this will start over" and is cleared the moment it does,
    which is right for the note and wrong as the only record: something
    that keeps killing the server shows up as a count that climbs, and
    nowhere else.
    """
    settings_file()
    store = QueueStore()
    store.load()
    entry = store.add(req("/a.mkv"), jobqueue.snapshot())

    for expected in (1, 2):                    # crash, restart, crash again
        entry.status = "running"
        store.save()
        store = QueueStore()
        store.load()
        entry = store.entries[0]
        assert entry.interrupted == expected
        assert "中断" in entry.note

    entry.status = "running"                   # ...and now it gets to run
    entry.note = ""
    assert entry.interrupted == 2              # the count survives the restart


def test_a_failed_write_leaves_the_old_file_intact(settings_file, monkeypatch):
    """What atomicity actually means: the previous content survives."""
    settings_file()
    store = QueueStore()
    store.load()
    store.add(req("/a.mkv"), jobqueue.snapshot())
    before = jobqueue.queue_path().read_text(encoding="utf-8")

    monkeypatch.setattr(jobqueue.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        store.add(req("/b.mkv"), jobqueue.snapshot())

    assert jobqueue.queue_path().read_text(encoding="utf-8") == before
    assert not jobqueue.queue_path().with_suffix(".json.part").exists()


def test_a_missing_file_is_simply_an_empty_queue(settings_file):
    settings_file()
    store = QueueStore()
    store.load()
    assert store.entries == []
    assert store.paused is False


# ------------------------------------------------------ schema evolution


def test_an_older_snapshot_loads_with_defaults_for_new_fields(settings_file):
    settings_file()
    jobqueue.queue_path().write_text(json.dumps({
        "version": 1, "paused": False,
        "entries": [{"id": "e1", "title": "/a.mkv", "status": "queued",
                     "request": {"video_path": "/a.mkv", "target_language": "中文"},
                     "settings": {"llm": {"model": "old-model"}}}],
    }), encoding="utf-8")

    store = QueueStore()
    store.load()
    assert store.entries[0].settings.llm.model == "old-model"
    assert store.entries[0].settings.asr.engine == AppSettings().asr.engine


def test_a_newer_snapshot_ignores_fields_this_version_does_not_know(settings_file):
    settings_file()
    jobqueue.queue_path().write_text(json.dumps({
        "version": 99, "paused": False,
        "entries": [{"id": "e1", "title": "/a.mkv", "status": "queued",
                     "request": {"video_path": "/a.mkv", "target_language": "中文"},
                     "settings": {"llm": {"model": "m"}, "from_the_future": 1}}],
    }), encoding="utf-8")

    store = QueueStore()
    store.load()
    assert store.entries[0].status == "queued"
    assert store.entries[0].settings.llm.model == "m"


def test_an_out_of_range_setting_reverts_to_its_default_and_the_entry_survives(settings_file):
    """A repaired setting is a value the user could have chosen anyway.

    Losing a whole queue entry over one out-of-range number would be a far
    worse outcome, so settings are repaired — while a bad *request* is
    fatal to that entry, because it would mean translating the wrong thing.
    """
    settings_file()
    jobqueue.queue_path().write_text(json.dumps({
        "version": 1, "paused": False,
        "entries": [{"id": "e1", "title": "/a.mkv", "status": "queued",
                     "request": {"video_path": "/a.mkv", "target_language": "中文"},
                     "settings": {"llm": {"model": "m", "batch_size": 999999}}}],
    }), encoding="utf-8")

    store = QueueStore()
    store.load()
    entry = store.entries[0]
    assert entry.status == "queued"
    assert entry.settings.llm.model == "m"                       # kept
    assert entry.settings.llm.batch_size == AppSettings().llm.batch_size
    assert "batch_size" in entry.note


def test_an_unreadable_entry_degrades_instead_of_disappearing(settings_file):
    """One broken row must not take the rest of the file with it."""
    settings_file()
    jobqueue.queue_path().write_text(json.dumps({
        "version": 1, "paused": False,
        "entries": [
            {"id": "bad", "title": "/broken.mkv", "status": "nonsense-status"},
            {"id": "ok", "title": "/fine.mkv", "status": "queued",
             "request": {"video_path": "/fine.mkv", "target_language": "中文"}},
        ],
    }), encoding="utf-8")

    store = QueueStore()
    store.load()
    assert [e.id for e in store.entries] == ["bad", "ok"]
    assert store.entries[0].status == "failed"
    assert store.entries[0].title == "/broken.mkv"    # still identifiable
    assert store.entries[1].status == "queued"


def test_a_corrupt_file_is_kept_as_evidence_not_overwritten(settings_file):
    settings_file()
    jobqueue.queue_path().write_text("{ this is not json", encoding="utf-8")

    store = QueueStore()
    store.load()
    assert store.entries == []
    assert jobqueue.queue_path().with_name("queue.json.bad").exists()


# ----------------------------------------------------------- list handling


def test_reorder_rejects_a_stale_list_instead_of_scrambling(settings_file):
    settings_file()
    store = QueueStore()
    store.load()
    a = store.add(req("/a.mkv"), jobqueue.snapshot())
    b = store.add(req("/b.mkv"), jobqueue.snapshot())

    store.reorder([b.id, a.id])
    assert [e.title for e in store.entries] == ["/b.mkv", "/a.mkv"]

    with pytest.raises(ValueError, match="刷新"):
        store.reorder([a.id])          # a browser holding an old list


def test_a_running_entry_cannot_be_removed(settings_file):
    settings_file()
    store = QueueStore()
    store.load()
    entry = store.add(req(), jobqueue.snapshot())
    entry.status = "running"
    with pytest.raises(ValueError, match="先取消"):
        store.remove(entry.id)


def test_pruning_keeps_the_newest_finished_and_never_drops_a_waiting_one(settings_file):
    settings_file()
    store = QueueStore()
    store.load()
    for i in range(jobqueue.KEEP_FINISHED + 10):
        entry = store.add(req(f"/done{i}.mkv"), jobqueue.snapshot())
        entry.status = "done"
        entry.finished_at = float(i)
    waiting = store.add(req("/waiting.mkv"), jobqueue.snapshot())
    store.save()

    kept = [e.title for e in store.entries]
    assert len([e for e in store.entries if e.status == "done"]) == jobqueue.KEEP_FINISHED
    assert "/waiting.mkv" in kept
    assert "/done0.mkv" not in kept          # oldest went first
    assert waiting.status == "queued"


# ------------------------------------------------------------- endpoints


@pytest.fixture
def client(settings_file, monkeypatch, tmp_path):
    """A local browser talking to a queue backed by the temp settings dir."""
    from app.main import app
    from tests.conftest import local_client

    settings_file()
    fresh = QueueManager()
    fresh.store.load()
    import app.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "queue_manager", fresh)
    return local_client(app), fresh


def a_film(tmp_path, name="film.mkv"):
    path = tmp_path / name
    path.write_bytes(b"not really a video")
    return str(path)


def test_enqueueing_freezes_the_settings_as_they_are_now(client, settings_file, tmp_path):
    api, manager = client
    settings_file(llm__model="model-A")
    api.post("/api/queue/jobs", json={"video_path": a_film(tmp_path, "a.mkv"),
                                      "target_language": "中文"})
    settings_file(llm__model="model-B")
    api.post("/api/queue/jobs", json={"video_path": a_film(tmp_path, "b.mkv"),
                                      "target_language": "中文"})

    assert [e.settings.llm.model for e in manager.store.entries] == \
        ["model-A", "model-B"]


def test_the_list_reports_which_entries_use_other_settings(client, settings_file, tmp_path):
    api, _ = client
    settings_file(llm__model="model-A")
    api.post("/api/queue/jobs", json={"video_path": a_film(tmp_path, "a.mkv"),
                                      "target_language": "中文"})
    settings_file(llm__model="model-B")
    api.post("/api/queue/jobs", json={"video_path": a_film(tmp_path, "b.mkv"),
                                      "target_language": "中文"})

    view = api.get("/api/queue").json()
    assert [e["settings_differs"] for e in view["entries"]] == [True, False]
    assert view["entries"][0]["summary"].startswith("auto → 中文")


def test_a_client_cannot_choose_the_settings_to_freeze(settings_file, monkeypatch, tmp_path):
    """The snapshot is taken server-side, always.

    A LAN browser holds API keys masked to ********; if a client could hand
    us settings to freeze, that mask would be frozen into a job that then
    cannot authenticate.
    """
    from app.main import app
    from tests.conftest import remote_client

    settings_file(llm__model="real-model", llm__api_key="sk-real",
                  server__lan_access=True, server__require_token=False)
    fresh = QueueManager()
    fresh.store.load()
    import app.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "queue_manager", fresh)

    api = remote_client(app)
    api.post("/api/queue/jobs", json={
        "video_path": a_film(tmp_path), "target_language": "中文",
        "settings": {"llm": {"model": "injected", "api_key": "********"}},
    })

    entry = fresh.store.entries[0]
    assert entry.settings.llm.model == "real-model"
    assert entry.settings.llm.api_key == ""


def test_a_directory_becomes_one_entry_per_file_sharing_one_snapshot(client, tmp_path):
    api, manager = client
    folder = tmp_path / "season"
    folder.mkdir()
    for name in ("ep1.mkv", "ep2.mkv", "ep3.mkv"):
        (folder / name).write_bytes(b"x")

    body = api.post("/api/queue/batch", json={"directory": str(folder),
                                              "target_language": "中文"}).json()
    assert body["count"] == 3
    hashes = {jobqueue.settings_hash(e.settings) for e in manager.store.entries}
    assert len(hashes) == 1                       # one snapshot for the season
    groups = {e.group_id for e in manager.store.entries}
    assert len(groups) == 1 and manager.store.entries[0].group_title == str(folder)


def test_a_queued_season_shares_one_series_id(client, tmp_path):
    api, manager = client
    folder = tmp_path / "season"
    folder.mkdir()
    for name in ("ep1.mkv", "ep2.mkv"):
        (folder / name).write_bytes(b"x")

    api.post("/api/queue/batch", json={"directory": str(folder),
                                       "target_language": "中文",
                                       "series_mode": True})
    ids = {e.request.series_id for e in manager.store.entries}
    assert len(ids) == 1 and ids != {""}


def test_enqueueing_a_missing_file_is_refused_at_the_door(client, tmp_path):
    api, manager = client
    resp = api.post("/api/queue/jobs", json={"video_path": str(tmp_path / "nope.mkv"),
                                             "target_language": "中文"})
    assert resp.status_code == 400
    assert "不存在" in resp.json()["detail"]
    assert manager.store.entries == []


def test_clearing_finished_does_not_touch_the_waiting(client, tmp_path):
    api, manager = client
    api.post("/api/queue/jobs", json={"video_path": a_film(tmp_path, "a.mkv"),
                                      "target_language": "中文"})
    api.post("/api/queue/jobs", json={"video_path": a_film(tmp_path, "b.mkv"),
                                      "target_language": "中文"})
    manager.store.entries[0].status = "done"

    assert api.delete("/api/queue/finished").json() == {"removed": 1}
    assert [e.title.endswith("b.mkv") for e in manager.store.entries] == [True]


def test_the_settings_of_an_entry_can_be_read_back_without_any_key(client, settings_file, tmp_path):
    api, manager = client
    settings_file(llm__model="model-A", llm__api_key="sk-secret")
    api.post("/api/queue/jobs", json={"video_path": a_film(tmp_path),
                                      "target_language": "中文"})
    entry_id = manager.store.entries[0].id

    body = api.get(f"/api/queue/{entry_id}/settings").json()
    text = "\n".join(body["lines"])
    assert "model-A" in text
    assert "sk-secret" not in text


def test_pause_is_reported_back_and_persisted(client):
    api, manager = client
    assert api.post("/api/queue/pause", json={"paused": True}).json() == {"paused": True}
    assert manager.store.paused is True
    again = QueueStore()
    again.load()
    assert again.paused is True


def test_an_immediate_job_never_touches_the_queue_file(client, tmp_path, monkeypatch):
    """开始翻译 stays what it was: it runs now and is not queue state."""
    api, manager = client
    import app.api.routes as routes_mod

    monkeypatch.setattr(routes_mod.manager, "create",
                        lambda req, settings=None: FakeJob(req, settings))
    api.post("/api/jobs", json={"video_path": a_film(tmp_path),
                                "target_language": "中文"})
    assert manager.store.entries == []


def test_an_immediate_job_is_also_snapshotted_at_submit(client, tmp_path, monkeypatch):
    """Same rule for all three entry points: what you saw is what runs."""
    api, _ = client
    import app.api.routes as routes_mod
    seen = {}

    def capture(req, settings=None):
        seen["settings"] = settings
        return FakeJob(req, settings)

    monkeypatch.setattr(routes_mod.manager, "create", capture)
    api.post("/api/jobs", json={"video_path": a_film(tmp_path),
                                "target_language": "中文"})
    assert seen["settings"] is not None
    assert seen["settings"].llm.api_key == ""      # blanked, read live later


def test_finished_jobs_do_not_pile_up_forever(settings_file, monkeypatch, tmp_path):
    """A week of unattended queue must not become hundreds of megabytes.

    Every Job keeps its whole event list, so "never evict" is fine for a
    few jobs a day and not fine for a queue that runs for days.
    """
    from app.services import pipeline as pipeline_mod

    settings_file()
    manager = pipeline_mod.JobManager()
    monkeypatch.setattr(pipeline_mod.threading, "Thread",
                        lambda *a, **k: type("T", (), {"start": lambda s: None})())
    monkeypatch.setattr(pipeline_mod.audio, "has_picture", lambda p: True)
    film = tmp_path / "f.mkv"
    film.write_bytes(b"x")

    for _ in range(pipeline_mod.KEEP_FINISHED_JOBS + 20):
        job = manager.create(req(str(film)))
        job.status.stage = "done"

    assert len(manager.jobs) <= pipeline_mod.KEEP_FINISHED_JOBS + 1


def test_a_live_batch_keeps_its_members_reachable(settings_file, monkeypatch, tmp_path):
    """BatchManager.status looks every member up; eviction must not break it."""
    from app.services import pipeline as pipeline_mod
    from app.services.batch import Batch, batch_manager

    settings_file()
    manager = pipeline_mod.JobManager()
    monkeypatch.setattr(pipeline_mod.threading, "Thread",
                        lambda *a, **k: type("T", (), {"start": lambda s: None})())
    monkeypatch.setattr(pipeline_mod.audio, "has_picture", lambda p: True)
    film = tmp_path / "f.mkv"
    film.write_bytes(b"x")

    member = manager.create(req(str(film)))
    member.status.stage = "done"
    still_going = manager.create(req(str(film)))          # keeps the batch live
    batch = Batch(id="b1", directory=str(tmp_path),
                  job_ids=[member.id, still_going.id])
    monkeypatch.setitem(batch_manager.batches, "b1", batch)

    for _ in range(pipeline_mod.KEEP_FINISHED_JOBS + 20):
        manager.create(req(str(film))).status.stage = "done"

    assert member.id in manager.jobs


# ------------------------------------------------------- the three gaps


SRT_EN = ("1\n00:00:01,000 --> 00:00:03,000\n"
          "What are you doing here at this hour?\n")
SRT_ZH = "1\n00:00:01,000 --> 00:00:03,000\n你这个时候来干什么？\n"
SRT_JA = "1\n00:00:01,000 --> 00:00:03,000\nこんな時間に何をしているの？\n"


def test_an_existing_subtitle_is_judged_by_its_language_not_its_name(tmp_path):
    """"Already has a subtitle" is two different situations.

    A subtitle in the target language means the film is done. A subtitle in
    any other language is not a result at all — it is material, and better
    material than speech recognition. The filename cannot tell them apart:
    this program's own original_only output and a subtitle downloaded from
    anywhere else are both film.ja.srt.
    """
    from app.core.media import scan_media

    (tmp_path / "done.mkv").write_bytes(b"x")
    (tmp_path / "done.srt").write_text(SRT_ZH, encoding="utf-8")
    (tmp_path / "japanese.mkv").write_bytes(b"x")
    (tmp_path / "japanese.ja.srt").write_text(SRT_JA, encoding="utf-8")
    (tmp_path / "english.mkv").write_bytes(b"x")
    (tmp_path / "english.srt").write_text(SRT_EN, encoding="utf-8")
    (tmp_path / "bare.mkv").write_bytes(b"x")

    scan = scan_media(tmp_path, target_language="简体中文")

    assert sorted(p.name for p in scan.skipped) == ["done.mkv"]
    # 那两份外语字幕现在**直接被翻译**，顶替掉它们伺候的片子——更快、不占 GPU，
    # 代价是这两部拿不到内嵌视频，所以被顶替的要单独说出来
    assert sorted(p.name for p in scan.to_translate) == [
        "bare.mkv", "english.srt", "japanese.ja.srt"]
    assert sorted(p.name for p in scan.replaced) == ["english.mkv", "japanese.mkv"]
    assert not scan.with_source          # 不再只是「报告」，已经改用它了


def test_without_a_target_language_any_subtitle_still_counts_as_done(tmp_path):
    """Nothing to compare against means the old, conservative rule."""
    from app.core.media import scan_media

    (tmp_path / "a.mkv").write_bytes(b"x")
    (tmp_path / "a.srt").write_text(SRT_EN, encoding="utf-8")
    scan = scan_media(tmp_path)
    assert [p.name for p in scan.skipped] == ["a.mkv"]
    assert not scan.to_translate and not scan.with_source


def test_both_halves_of_a_split_job_together_mean_the_film_is_done(tmp_path):
    """One finished subtitle settles it, whichever sibling turns up first.

    双文件模式 leaves film.en.srt and film.zh.srt side by side, and the glob
    is alphabetical — so the original is read first and it is not the target
    language. Answering from it called a finished film untranslated, and the
    batch re-ran it every single scan.
    """
    from app.core.media import scan_media

    (tmp_path / "pair.mkv").write_bytes(b"x")
    (tmp_path / "pair.en.srt").write_text(SRT_EN, encoding="utf-8")
    (tmp_path / "pair.zh.srt").write_text(SRT_ZH, encoding="utf-8")

    scan = scan_media(tmp_path, target_language="简体中文")
    assert [p.name for p in scan.skipped] == ["pair.mkv"]
    assert not scan.to_translate and not scan.with_source


def test_an_original_on_its_own_is_still_only_material(tmp_path):
    """The guard against over-correcting: looking at every sibling must not
    turn "no translation here" into "done"."""
    from app.core.media import scan_media

    (tmp_path / "half.mkv").write_bytes(b"x")
    (tmp_path / "half.en.srt").write_text(SRT_EN, encoding="utf-8")
    (tmp_path / "half.ja.srt").write_text(SRT_JA, encoding="utf-8")

    scan = scan_media(tmp_path, target_language="简体中文")
    # 两份都不是目标语言，所以这一组没做完；翻的是字幕（文字优先，同为文字
    # 时按名字），被顶替的视频进 replaced
    assert [p.name for p in scan.to_translate] == ["half.en.srt"]
    assert [p.name for p in scan.replaced] == ["half.mkv"]
    assert not scan.skipped


def test_a_bilingual_subtitle_counts_as_done_for_either_of_its_languages(tmp_path):
    """片名.en-zh.srt 里有目标语言，这部片就是做完的。

    配对后缀是本程序双语模式的产物，两种排版下前后顺序相反（en-zh / zh-en），
    第三方的更是毫无规律——所以判据是「哪一半是目标语言」，不是「哪一半在前」。
    """
    from app.core.media import subtitle_state

    (tmp_path / "pair.en-zh.srt").write_text(SRT_ZH, encoding="utf-8")
    assert subtitle_state(tmp_path, "pair", "简体中文")[0] == "done"
    assert subtitle_state(tmp_path, "pair", "English")[0] == "done"
    # 两种语言都不是目标：那它就是材料，不是成品
    assert subtitle_state(tmp_path, "pair", "Français")[0] == "source"


def test_a_dotted_release_name_is_asked_about_correctly(tmp_path):
    """按 (目录, 主干) 提问，而不是按一个 Path。

    Path("/x/Movie.2019.1080p").with_suffix(".srt") 得到的是 Movie.2019.srt
    ——发行版片名里全是点，一旦调用方手里只有主干（字幕文件那条路就是如此），
    按 Path 提问就会去看隔壁那部片的字幕。
    """
    from app.core.media import base_stem, split_language_tag, subtitle_state

    stem = "Movie.2019.1080p"
    (tmp_path / f"{stem}.zh.srt").write_text(SRT_ZH, encoding="utf-8")
    assert subtitle_state(tmp_path, stem, "简体中文")[0] == "done"
    # 而语言后缀是从最后一段剥的，1080p 不是语言
    assert split_language_tag(f"{stem}.en.srt") == (stem, "en")
    assert base_stem(f"{stem}.srt") == stem


def test_a_copy_that_stepped_aside_is_not_mistaken_for_a_result(tmp_path):
    """片名.zh.2.srt 是防覆盖留下的副本，不是成品。

    它必须落在语言后缀的判定之外：认它就等于把一份「让路的备份」当成这部片
    已经翻完的证据。
    """
    from app.core.media import subtitle_state

    (tmp_path / "again.zh.2.srt").write_text(SRT_ZH, encoding="utf-8")
    assert subtitle_state(tmp_path, "again", "简体中文")[0] == "none"


def test_a_subtitle_that_is_not_ours_by_name_is_left_alone(tmp_path):
    """film.backup.srt is nobody's language tag, so it is not consulted."""
    from app.core.media import scan_media

    (tmp_path / "d.mkv").write_bytes(b"x")
    (tmp_path / "d.backup.srt").write_text(SRT_EN, encoding="utf-8")
    scan = scan_media(tmp_path, target_language="简体中文")
    # d.backup.srt 认领到 d.mkv 这一组，但 backup 不是语言后缀，所以它不能
    # 当原文——既不让这部片显示为已完成，也不会自己变成一个任务
    assert [p.name for p in scan.to_translate] == ["d.mkv"] and not scan.skipped
    assert not scan.replaced


def test_a_season_glossary_survives_a_restart(settings_file):
    """Series mode assumes the whole season agrees; a restart used to reset it."""
    from app.services import series

    settings_file()
    series._store.clear()
    shared = series.create("season-1")
    shared.learn("佐藤健一 → 佐藤健一\nユキ → 雪")
    series.save()

    series._store.clear()                 # a restart
    series.load()
    again = series.get("season-1")
    assert again is not None
    assert "佐藤健一" in again.render() and "雪" in again.render()


def test_loading_never_overwrites_a_table_already_in_memory(settings_file):
    """Disk is the fallback for what this process lacks, not an overwrite.

    A table in memory has been added to since it was last written, so
    loading over it would throw the newer names away.
    """
    from app.services import series

    settings_file()
    series._store.clear()
    shared = series.create("season-2")
    series.save()                         # written while still empty
    shared.learn("ユキ → 雪")              # ...learned afterwards, not saved

    series.load()
    assert "雪" in series.get("season-2").render()


def test_an_abandoned_glossary_is_eventually_forgotten(settings_file):
    from app.services import series

    settings_file()
    series._store.clear()
    old = series.create("ancient")
    old.learn("あ → 阿")
    old.touched = 0.0                     # long ago
    series.save()

    series._store.clear()
    series.load()
    assert series.get("ancient") is None


# --------------------------------------------------- resuming, and its cost


def test_resuming_reuses_the_work_and_a_changed_setting_does_not(settings_file, tmp_path):
    """The directory name is the whole validity check.

    Same film, same settings → same key → last time's work is found. Change
    anything that would make that work wrong to continue from, and the key
    moves, so nothing has to be compared afterwards and no check can be
    forgotten.
    """
    from app.services.pipeline import Job, checkpoint_key

    settings_file(llm__model="model-A")
    film = tmp_path / "f.mkv"
    film.write_bytes(b"x")

    first = checkpoint_key(Job(req(str(film)), audio_only=True), jobqueue.snapshot())
    same = checkpoint_key(Job(req(str(film)), audio_only=True), jobqueue.snapshot())
    assert first == same

    settings_file(llm__model="model-B")               # 设置变了
    assert checkpoint_key(Job(req(str(film)), audio_only=True),
                          jobqueue.snapshot()) != first

    settings_file(llm__model="model-A")
    film.write_bytes(b"different content")            # 片源变了
    assert checkpoint_key(Job(req(str(film)), audio_only=True),
                          jobqueue.snapshot()) != first


def test_a_different_request_is_different_work(settings_file, tmp_path):
    """Same film and settings, but translated into another language."""
    from app.services.pipeline import Job, checkpoint_key

    settings_file()
    film = tmp_path / "f.mkv"
    film.write_bytes(b"x")
    snap = jobqueue.snapshot()
    a = Job(JobRequest(video_path=str(film), target_language="简体中文"), audio_only=True)
    b = Job(JobRequest(video_path=str(film), target_language="English"), audio_only=True)
    assert checkpoint_key(a, snap) != checkpoint_key(b, snap)


def test_the_program_version_invalidates_kept_work(settings_file, tmp_path, monkeypatch):
    from app.core import joblog
    from app.services.pipeline import Job, checkpoint_key

    settings_file()
    film = tmp_path / "f.mkv"
    film.write_bytes(b"x")
    snap = jobqueue.snapshot()
    before = checkpoint_key(Job(req(str(film)), audio_only=True), snap)
    monkeypatch.setattr(joblog, "APP_VERSION", "99.0.0")
    assert checkpoint_key(Job(req(str(film)), audio_only=True), snap) != before


def test_old_and_oversized_checkpoints_are_swept(settings_file, tmp_path):
    """Disk must not grow without bound: age first, then total size."""
    import os
    from app.core import cache

    # work_dir, not just settings_path: cache._base_dir() falls back to the
    # real user cache directory, so without this the test writes into the
    # developer's own ~/.cache and leaves files behind.
    settings_file(work_dir=str(tmp_path), checkpoint_days=7, checkpoint_max_gb=1)
    root = cache.checkpoints_root()

    stale = cache.checkpoint_dir("stale")
    (stale / "audio.wav").write_bytes(b"x" * 10)
    old = __import__("time").time() - 30 * 86400
    os.utime(stale / "audio.wav", (old, old))

    fresh = cache.checkpoint_dir("fresh")
    (fresh / "audio.wav").write_bytes(b"x" * 10)

    cache.prune_checkpoints()
    assert not stale.exists()
    assert fresh.exists()


def test_turning_resuming_off_keeps_nothing_at_all(settings_file, tmp_path):
    from app.core import cache

    settings_file(work_dir=str(tmp_path), checkpoint_days=0)
    kept = cache.checkpoint_dir("doomed")
    (kept / "audio.wav").write_bytes(b"x" * 10)

    assert cache.checkpoints_enabled() is False
    cache.prune_checkpoints()
    assert not kept.exists()


def test_the_size_cap_drops_the_oldest_first(settings_file, tmp_path):
    import os
    import time
    from app.core import cache

    settings_file(work_dir=str(tmp_path), checkpoint_days=90, checkpoint_max_gb=0)
    assert cache.checkpoints_enabled() is False

    # and with a real cap, the oldest goes first
    settings_file(work_dir=str(tmp_path), checkpoint_days=90, checkpoint_max_gb=1)
    root = cache.checkpoints_root()
    now = time.time()
    for i, name in enumerate(("oldest", "newest")):
        d = cache.checkpoint_dir(name)
        (d / "audio.wav").write_bytes(b"x" * 1024)
        os.utime(d / "audio.wav", (now - (10 - i) * 3600,) * 2)
    monkey = 1500          # a cap below the total but above one of them
    from unittest import mock
    with mock.patch.object(cache, "_limits", lambda: (90, monkey)):
        cache.prune_checkpoints()
    assert not (root / "oldest").exists()
    assert (root / "newest").exists()


def test_nothing_in_the_users_folder_is_ever_written_over(tmp_path):
    """名字被占就让路，无一例外。

    这条规则以前只管一种情况——产物正好和它自己读进来的那份旁挂字幕同名。
    用户下载的同名字幕、上一次跑出来的成品，照写不误。现在片源目录里任何
    已经存在的文件都不会被覆盖：让路的永远是新写的那一份。
    """
    from app.services.pipeline import output_target

    film = tmp_path / "film.mkv"

    # 名字空着就用它，什么都没让
    target, taken = output_target(film, ".zh", ".srt")
    assert target == tmp_path / "film.zh.srt" and taken is None

    # 上一次的成品在那儿：让路，并说出让给了谁
    body = "1\n00:00:01,000 --> 00:00:02,000\n上一次的译文\n"
    (tmp_path / "film.zh.srt").write_text(body, encoding="utf-8")
    target, taken = output_target(film, ".zh", ".srt")
    assert target.name == "film.zh.2.srt" and taken == tmp_path / "film.zh.srt"

    # 再来一次就 .3，编号从原名重新算起，不会变成 .2.2
    (tmp_path / "film.zh.2.srt").write_text(body, encoding="utf-8")
    assert output_target(film, ".zh", ".srt")[0].name == "film.zh.3.srt"

    # 而那份先来的，一个字节都没动
    assert (tmp_path / "film.zh.srt").read_text(encoding="utf-8") == body


def test_the_subtitle_a_job_read_from_is_just_another_file_it_will_not_touch(tmp_path):
    """原文读自 film.ja.srt、产物也叫 film.ja.srt —— 老规则里的那个特例。

    现在它不再需要特例：通用规则已经把它包含在内，而且让出来的名字带着自己的
    语言（film.ja.2.srt），不会像从前那样改叫 film.zh.srt —— 那是把日语原文
    写进中文译文的名字里，没人会发现。
    """
    from app.services.pipeline import output_target

    film = tmp_path / "film.mkv"
    source = tmp_path / "film.ja.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n", encoding="utf-8")

    target, taken = output_target(film, ".ja", ".srt")
    assert target.name == "film.ja.2.srt" and taken == source
    assert target.name != "film.zh.srt"

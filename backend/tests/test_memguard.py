"""压制的内存保护：整机快没内存、或者压制自己超了上限，就停下这次压制。

守卫的判断全在假的读数上测（整机多少、可用多少、本进程多少由测试给），
所以断言的是规则本身，不看这台机器此刻有多少空闲内存。真读数另有一条
Linux 用例看一眼数字是否合理。
"""

from __future__ import annotations

import sys

import pytest

from app.services import memguard
from app.services.memguard import GIB, LowMemory, MemoryGuard


@pytest.fixture(autouse=True)
def _auto_limit(monkeypatch):
    monkeypatch.setattr(memguard, "_limit_gb", 0.0)
    monkeypatch.setattr(memguard, "_active", None)


class Machine:
    """A fake machine whose numbers the test moves by hand."""

    def __init__(self, total=32 * GIB, available=16 * GIB, ours=100 << 20):
        self.total, self.available, self.ours = total, available, ours
        self.t = 0.0
        self.slept = []

    def machine(self):
        return self.total, self.available

    def process(self):
        return self.ours

    def clock(self):
        return self.t

    def sleep(self, seconds):
        assert len(self.slept) < 10_000, "start() keeps waiting without end"
        self.slept.append(seconds)
        self.t += seconds


def guard(m, lines=None, machine=None, release=None):
    lines = [] if lines is None else lines
    return MemoryGuard(lines.append, read_machine=machine or m.machine,
                       read_process=m.process, clock=m.clock, sleep=m.sleep,
                       release=release or (lambda: None)), lines


def later(m, seconds=memguard.CHECK_SECONDS):
    m.t += seconds


# ------------------------------------------------------------------ 线


def test_the_floor_is_a_tenth_of_memory_within_one_and_four_gb():
    assert memguard.floor_bytes(32 * GIB) == pytest.approx(3.2 * GIB, rel=1e-6)
    assert memguard.floor_bytes(4 * GIB) == 1 * GIB
    assert memguard.floor_bytes(128 * GIB) == 4 * GIB


def test_the_limit_is_half_the_memory_until_the_queue_page_says_otherwise():
    assert memguard.cap_bytes(32 * GIB) == 16 * GIB
    memguard.set_limit(6)
    assert memguard.cap_bytes(32 * GIB) == 6 * GIB


def test_proc_files_are_read_in_bytes():
    text = "MemTotal:       32749000 kB\nMemFree: 1 kB\nMemAvailable:   17200000 kB\n"
    assert memguard.parse_kib(text, "MemTotal", "MemAvailable") == (
        32749000 * 1024, 17200000 * 1024)
    assert memguard.parse_kib("VmRSS: 5 kB\n", "VmRSS", "VmSwap") is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc")
def test_this_machine_can_be_read():
    total, available = memguard.machine()
    assert 0 < available < total
    assert memguard.process() > 10 << 20


# ------------------------------------------------------------------ 守卫


def test_an_encode_that_fits_is_left_alone_and_its_peak_is_logged():
    m = Machine()
    g, lines = guard(m)
    g.start(lambda: False)
    assert "压制最多用 16.0 GB（自动：物理内存的一半）" in lines[0]
    for used in (0.8, 1.1, 1.0):
        m.ours = (100 << 20) + int(used * GIB)
        later(m)
        g.check()
    g.close()
    assert lines[-1] == "压制占用内存峰值 1.1 GB（整机 32.0 GB）"


def test_it_checks_every_two_seconds_not_every_packet():
    m = Machine()
    reads = []
    g, _ = guard(m, machine=lambda: reads.append(1) or m.machine())
    g.start(lambda: False)
    before = len(reads)
    for _ in range(1000):
        g.check()           # the clock does not move: one packet after another
    assert len(reads) == before


def test_an_encode_that_outgrows_its_limit_is_stopped_and_told_what_to_change():
    m = Machine()
    memguard.set_limit(4)
    g, _ = guard(m)
    g.start(lambda: False)
    m.ours += int(4.5 * GIB)
    later(m)
    with pytest.raises(LowMemory, match="到了 4.5 GB，超过上限 4.0 GB") as caught:
        g.check()
    assert "x265 比 SVT-AV1 省一半多" in str(caught.value)


def test_the_limit_is_read_live_so_raising_it_saves_a_running_encode():
    m = Machine()
    memguard.set_limit(4)
    g, _ = guard(m)
    g.start(lambda: False)
    m.ours += int(4.5 * GIB)
    memguard.set_limit(8)
    later(m)
    g.check()


def test_only_what_the_encode_added_counts_not_the_server_or_a_loaded_model():
    m = Machine(ours=6 * GIB)     # a whisper model already in memory
    memguard.set_limit(4)
    g, _ = guard(m)
    g.start(lambda: False)
    m.ours += 3 * GIB
    later(m)
    g.check()
    assert g.used == 3 * GIB


def test_a_machine_running_out_stops_the_encode_whoever_ate_the_memory():
    m = Machine()
    g, _ = guard(m)
    g.start(lambda: False)
    m.ours += 1 * GIB
    m.available = 2 * GIB         # someone else grew by 14 GB
    later(m)
    with pytest.raises(LowMemory, match=r"整机可用内存只剩 2.0 GB（保护线 3.2 GB）") as caught:
        g.check()
    assert "压制本身占 1.0 GB" in str(caught.value) and "重试" in str(caught.value)


def test_short_of_memory_before_it_starts_it_waits_instead_of_failing():
    """Failing here would fail the next entry too, and the next: one tight
    moment should not turn the whole queue into a column of failures."""
    m = Machine(available=2 * GIB)
    said = []
    g, lines = guard(m)

    def sleep(seconds):
        m.sleep(seconds)
        if m.t >= 200:
            m.available = 12 * GIB

    g._sleep = sleep
    g.start(lambda: False, waiting=said.append)
    assert "等内存腾出来再开始压制" in lines[0]
    assert any("回到 12.0 GB，开始压制" in line for line in lines)
    assert 3 <= len(said) <= 4     # once a minute, not every five seconds


def test_waiting_for_memory_can_be_cancelled():
    m = Machine(available=1 * GIB)
    g, _ = guard(m)
    stop = {"now": False}

    def sleep(seconds):
        m.sleep(seconds)
        stop["now"] = m.t > 30

    g._sleep = sleep
    with pytest.raises(InterruptedError):
        g.start(lambda: stop["now"])


def test_memory_is_handed_back_before_measuring():
    """What an earlier job freed but kept would be reused unseen, and the encode
    would look smaller than it is. (Handing back after the encode is the job
    runner's, once a stopped encode's traceback has let go of the encoder:
    test_encode_api covers it.)"""
    m = Machine(ours=4 * GIB)          # an earlier encode's leftovers
    calls = []

    def release():
        calls.append(m.t)
        m.ours = 200 << 20

    g, _ = guard(m, release=release)
    g.start(lambda: False)
    assert calls == [0.0] and g.base == 200 << 20
    m.ours += 1 * GIB
    later(m)
    g.check()
    assert g.used == 1 * GIB           # not 0 GB, as it would read over the leftovers
    g.close()
    assert calls == [0.0]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc")
def test_trim_gives_back_what_a_worker_thread_freed():
    import threading

    survivors = []

    def churn():
        # 120 MB in this thread's own arena; one block in 64 lives on, so the
        # freed rest is stranded between them — the shape an encoder's worker
        # threads leave behind — and glibc keeps it
        blocks = [bytearray(3000) for _ in range(40_000)]
        survivors.extend(blocks[::64])
        del blocks

    before = memguard.process()
    worker = threading.Thread(target=churn)
    worker.start()
    worker.join()
    kept = memguard.process() - before
    assert kept > 60 << 20, "the freed memory was not held: nothing to test"
    memguard.trim()
    assert memguard.process() - before < kept / 3
    assert len(survivors) == 625


def test_where_memory_cannot_be_read_the_encode_runs_unguarded_and_says_so():
    m = Machine()
    g, lines = guard(m, machine=lambda: None)
    g.start(lambda: False)
    m.ours += 40 * GIB
    later(m)
    g.check()
    assert lines == ["⚠ 这个系统上读不到内存用量：压制不受内存保护"]


def test_the_queue_page_sees_the_running_encode_and_then_the_machine():
    m = Machine()
    g, _ = guard(m)
    g.start(lambda: False)
    m.ours += 2 * GIB
    later(m)
    g.check()
    status = memguard.status()
    assert status["active"] and status["used_gb"] == pytest.approx(2.0)
    assert status["limit_gb"] == pytest.approx(16.0) and status["floor_gb"] == pytest.approx(3.2)
    g.close()
    assert memguard.status()["active"] is False

"""压制让路：列队页的开关开着时，压制给别的程序让出 CPU。

控制器的测试在一台假机器上跑（World）：时间、整机忙闲、本进程用掉的 CPU
全由它记账，所以「限速之后压制实际用了多少」是可以精确断言的数，不看这台
机器此刻有多忙。优先级那一半在 Linux 上用真线程测：编码器开出来的原生线程
必须降，别的线程一个都不许碰。
"""

from __future__ import annotations

import os
import sys
import threading
from fractions import Fraction

import av
import pytest

from app.services import cpuyield


class World:
    """一台假机器：*cpus* 个逻辑核，别的程序占 *others(t)*，本进程醒着就吃满。

    *drain* 是每个包留给后台线程的活（CPU 秒）：真的 x265 在读包循环往下走、
    甚至睡下之后，还在编前面喂进去的帧。这部分在睡的时候照样烧 CPU，控制器
    必须按实测记账才收敛。
    """

    def __init__(self, cpus=12, others=lambda t: 0.0, drain=0.0):
        self.t = 0.0
        self.cpus = cpus
        self.others = others
        self.drain = drain
        self.backlog = 0.0
        self.busy = 0.0
        self.ours = 0.0
        self.slept = []
        self.reads = 0

    def clock(self):
        return self.t

    def cpu(self):
        return self.ours

    def system(self):
        self.reads += 1
        return self.busy, self.t * self.cpus

    def _pass(self, seconds, ours):
        self.busy += self.others(self.t) * self.cpus * seconds + ours
        self.ours += ours
        self.t += seconds

    def work(self, seconds):
        self._pass(seconds, self.cpus * seconds)
        self.backlog += self.drain

    def sleep(self, seconds):
        # pace() once spun here on waits too small to move the clock, and the
        # simulation that found it grew to 19 GB before anyone noticed: fail
        # at once instead
        assert seconds >= cpuyield.MIN_SLEEP, f"pace() asked for a {seconds}s sleep"
        assert len(self.slept) < 100_000, "pace() keeps sleeping without end"
        self.slept.append(seconds)
        burn = min(self.backlog, self.cpus * seconds)
        self.backlog -= burn
        self._pass(seconds, burn)


class FakePriority:
    def __init__(self, restorable=True):
        self.lowered = 0
        self.restored = 0
        self.restorable = restorable

    def lower(self):
        self.lowered += 1
        return 1

    def restore(self):
        self.restored += 1
        return self.restorable


def governor(world, on=True, priority=None, system=None, cancelled=None):
    switch = {"on": on}
    lines = []
    g = cpuyield.Governor(lines.append, is_on=lambda: switch["on"],
                          system=system or world.system, cpu=world.cpu,
                          clock=world.clock, sleep=world.sleep, cancelled=cancelled,
                          priority=priority or FakePriority())
    return g, switch, lines


def encode(world, g, seconds, packet=0.02):
    """The read loop: a packet's worth of full-machine work, then pace()."""
    end = world.t + seconds
    while world.t < end:
        world.work(packet)
        g.pace()


def share(world, start_t, start_ours, cpus=12):
    return (world.ours - start_ours) / ((world.t - start_t) * cpus)


# ------------------------------------------------------------------ 控制器


def test_switched_off_it_measures_nothing_lowers_nothing_sleeps_never():
    world = World(others=lambda t: 0.9)
    priority = FakePriority()
    g, _, lines = governor(world, on=False, priority=priority)
    g.start()
    encode(world, g, 60)
    g.close()
    assert world.slept == [] and world.reads == 0
    assert priority.lowered == 0 and priority.restored == 0
    assert lines == []


def test_a_quiet_machine_encodes_at_full_speed_but_at_the_lowest_priority():
    world = World(others=lambda t: 0.10)
    priority = FakePriority()
    g, _, lines = governor(world, priority=priority)
    g.start()
    encode(world, g, 60)
    g.close()
    assert world.slept == [] and g.cap == 1.0
    assert priority.lowered > 0 and priority.restored == 1
    assert "一直没超过 20%" in lines[-1]


def test_busy_others_get_half_of_what_is_left_and_the_encode_gets_the_rest():
    """别的程序占 50%：压制最多用剩下 50% 的一半，即 25%。

    每个包还给后台线程留两个 CPU 秒的活（一个 1080p 的帧喂进 x265 差不多就是
    这样）：比一段睡眠挣回的份额还多。睡一小段就回去的写法在这里会被「每个包
    都欠一点」拖到 38%（份额 25%）。
    """
    world = World(others=lambda t: 0.50, drain=2.0)
    g, _, lines = governor(world)
    g.start()
    encode(world, g, 10)
    assert g.cap == pytest.approx(0.25)
    t0, ours0 = world.t, world.ours
    encode(world, g, 120)
    assert share(world, t0, ours0) == pytest.approx(0.25, abs=0.02)
    assert max(world.slept) <= cpuyield.MAX_SLEEP
    assert "让路中：其他程序占 50%，压制最多用 25%" in g.note()
    assert any("压制开始限速" in line for line in lines)


def test_however_busy_the_others_the_encode_keeps_a_tenth():
    world = World(others=lambda t: 0.95, drain=2.0)
    g, _, _ = governor(world)
    g.start()
    encode(world, g, 10)
    t0, ours0 = world.t, world.ours
    encode(world, g, 120)
    assert g.cap == cpuyield.FLOOR
    assert share(world, t0, ours0) == pytest.approx(cpuyield.FLOOR, abs=0.02)


def test_a_cancel_is_not_kept_waiting_behind_the_debt():
    world = World(others=lambda t: 0.9, drain=50.0)
    stop = {"now": False}
    g, _, _ = governor(world, cancelled=lambda: stop["now"])
    g.start()
    encode(world, g, 5)
    world.work(0.02)            # one more packet: ~50 CPU-s owed, minutes at 10%
    stop["now"] = True
    slept = len(world.slept)
    g.pace()
    assert len(world.slept) == slept


def test_it_speeds_back_up_ten_seconds_after_the_others_calm_down():
    world = World(others=lambda t: 0.6 if t < 30 else 0.05)
    g, _, lines = governor(world)
    g.start()
    encode(world, g, 32)
    assert g.cap < 1.0
    encode(world, g, 5)       # 30 + 7: still inside the hold
    assert g.cap < 1.0
    encode(world, g, 10)      # 30 + 17: the 3 s window and the 10 s hold are past
    assert g.cap == 1.0 and g.note() == ""
    g.close()
    assert g.episodes == 1
    assert lines[-1].startswith("压制让路：限速 1 次，累计") and "最高 60%" in lines[-1]


def test_a_one_second_spike_is_not_a_busy_machine():
    world = World(others=lambda t: 0.5 if 20 <= t < 21 else 0.0)
    g, _, _ = governor(world)
    g.start()
    encode(world, g, 40)
    assert world.slept == [] and g.episodes == 0


def test_switching_it_off_mid_encode_restores_priority_and_stops_waiting():
    world = World(others=lambda t: 0.5)
    priority = FakePriority()
    g, switch, lines = governor(world, priority=priority)
    g.start()
    encode(world, g, 20)
    assert g.cap < 1.0
    switch["on"] = False
    encode(world, g, 2)
    slept = len(world.slept)
    encode(world, g, 20)
    assert len(world.slept) == slept and g.cap == 1.0
    assert priority.restored == 1 and "已关闭：恢复正常优先级" in lines[-1]
    g.close()
    assert priority.restored == 1   # not twice


def test_switching_it_on_mid_encode_takes_hold_within_seconds():
    world = World(others=lambda t: 0.5)
    priority = FakePriority()
    g, switch, lines = governor(world, on=False, priority=priority)
    g.start()
    encode(world, g, 20)
    assert priority.lowered == 0 and world.slept == []
    switch["on"] = True
    encode(world, g, 5)
    assert priority.lowered > 0 and g.cap < 1.0 and world.slept
    assert lines[0].startswith("压制让路已开启")


def test_without_the_right_to_raise_priority_again_it_says_so():
    world = World()
    g, switch, lines = governor(world, priority=FakePriority(restorable=False))
    g.start()
    encode(world, g, 3)
    switch["on"] = False
    encode(world, g, 2)
    assert "没有权限把优先级调回来" in lines[-1]


def test_where_the_machine_cannot_be_measured_only_the_priority_drops():
    world = World(others=lambda t: 0.9)
    priority = FakePriority()
    g, _, lines = governor(world, priority=priority, system=lambda: None)
    g.start()
    encode(world, g, 30)
    g.close()
    assert world.slept == [] and priority.lowered > 0
    assert sum("读不到整机的 CPU 占用" in line for line in lines) == 1
    assert g.status()["supported"] is False


def test_the_queue_page_sees_the_running_encode_and_then_nothing():
    world = World(others=lambda t: 0.5)
    g, _, _ = governor(world)
    g.start()
    encode(world, g, 10)
    status = cpuyield.status()
    assert status["active"] and status["limited"]
    assert status["others"] == pytest.approx(0.5) and status["cap"] == pytest.approx(0.25)
    g.close()
    assert cpuyield.status()["active"] is False


# ---------------------------------------------------------------- 读整机占用


def test_proc_stat_counts_steal_as_neither_busy_nor_available():
    #          user nice system idle iowait irq softirq steal guest guest_nice
    line = "cpu  100  50   30     700  20     5   5       90    40    0"
    assert cpuyield.parse_proc_stat(line) == (190, 910)
    assert cpuyield.parse_proc_stat("intr 1 2 3") is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/stat")
def test_this_machine_can_be_measured():
    busy, total = cpuyield.read_system()
    assert 0 < busy < total


# ------------------------------------------------------------ Windows 优先级


class FakeKernel32:
    def __init__(self, cls):
        self.cls = cls
        self.calls = []

    def GetCurrentProcess(self):
        return -1

    def GetPriorityClass(self, handle):
        return self.cls

    def SetPriorityClass(self, handle, cls):
        self.calls.append(cls)
        self.cls = cls
        return 1


def test_windows_drops_the_process_to_below_normal_and_back():
    k = FakeKernel32(0x20)   # NORMAL
    process = cpuyield.WindowsProcess(k)
    assert process.lower() == 1 and k.cls == cpuyield.BELOW_NORMAL_PRIORITY_CLASS
    assert process.lower() == 0   # once
    assert process.restore() and k.cls == 0x20


def test_windows_never_raises_a_process_started_lower():
    k = FakeKernel32(cpuyield.IDLE_PRIORITY_CLASS)   # start /low
    process = cpuyield.WindowsProcess(k)
    assert process.lower() == 0 and process.restore()
    assert k.calls == []


# ------------------------------------------------------------ Linux 优先级


def _nice(tid):
    with open(f"/proc/self/task/{tid}/stat") as f:
        stat = f.read()
    return int(stat[stat.rindex(")") + 2:].split()[16])


def _tids():
    return {int(t) for t in os.listdir("/proc/self/task")}


def _open_encoder():
    ctx = av.CodecContext.create("libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = 320, 240, "yuv420p"
    ctx.time_base, ctx.framerate = Fraction(1, 24), 24
    ctx.thread_type = "AUTO"
    ctx.open()
    return ctx


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux 的逐线程 nice")
def test_only_the_threads_the_encode_opened_are_lowered():
    """Run in a thread of its own: a nice value an ordinary user cannot raise
    back dies with the thread, instead of slowing the rest of the session."""
    seen = {}

    def job():
        me = threading.get_native_id()
        natural = _nice(me)
        threads = cpuyield.LinuxThreads()
        waiting = threading.Event()
        # another job waiting for its turn: a Python thread, born mid-encode
        other = threading.Thread(target=waiting.wait)
        other.start()
        others = {other.native_id: _nice(other.native_id),
                  threading.main_thread().native_id: _nice(threading.main_thread().native_id)}
        before = _tids()
        first = _open_encoder()
        opened = _tids() - before
        threads.lower()
        seen["first"] = {_nice(t) for t in opened}
        seen["me"] = _nice(me)
        # an encoder opened after the drop inherits it...
        before = _tids()
        second = _open_encoder()
        inherited = _tids() - before
        seen["inherited"] = {_nice(t) for t in inherited}
        threads.lower()
        seen["others"] = all(_nice(t) == n for t, n in others.items())
        seen["restored"] = threads.restore()
        # ...and is put back where this thread was, not where it found it
        seen["after"] = {_nice(t) for t in opened | inherited | {me}}
        seen["natural"] = natural
        waiting.set()
        other.join()
        del first, second

    worker = threading.Thread(target=job)
    worker.start()
    worker.join()
    assert seen["first"] == {cpuyield.NICE} and seen["me"] == cpuyield.NICE
    assert seen["inherited"] == {cpuyield.NICE}
    assert seen["others"], "a thread the encode did not open was touched"
    if seen["restored"]:
        assert seen["after"] == {seen["natural"]}
    else:   # an ordinary user: lowering is one-way
        assert seen["after"] == {cpuyield.NICE}

"""压制让路（列队页的开关）：别的程序要用 CPU 的时候，压制往后退。

用户定的规则：**别的程序占用 CPU 低于 20% 时照常全速；高于 20% 就限速，
把 CPU 让给别的程序。** 落到实现是两件事：

* **开着就一直是最低优先级**（Linux nice 19 / Windows「低于正常」）。操作系统
  先满足别的程序，压制只拿剩下的——机器闲着的时候剩下的就是全部，所以这一条
  不影响速度。不做成「超线才降」，因为 **Linux 上普通用户只能降低优先级、不能
  再调回来**：降一次就回不去，随负载来回切换根本做不成。
* **超过 20% 时限速：压制最多只用剩余空闲 CPU 的一半，至少 10%。** 优先级只管
  「谁先用」，管不到同一个物理核上的超线程、内存带宽、全核负载压低的睿频——
  这些只有少用才能缓解。回落到 20% 以下要持续 10 秒才恢复全速，浏览网页那种
  一阵一阵的占用才不会让它来回切。

「别的程序」＝整机 CPU 占用减去本进程自己的（`time.process_time()` 是本进程
所有线程的合计）：压制自己吃满 CPU 不会把自己限住。按最近 3 秒的平均判断，
一秒的尖峰不算数。

限速的办法是在压制的读包循环里睡（`Governor.pace`），并且按本进程**实测**的
CPU 时间记账（令牌桶）：停止喂帧之后，x265 的工作线程还会把手上的几帧做完，
按「睡了多久」算份额会偏高，按「实际用了多少」算才收敛到目标。

只管压制任务（含原盘全流程里的压制），不管封装和翻译。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from functools import lru_cache
from typing import Callable, Deque, Dict, Optional, Set, Tuple

THRESHOLD = 0.20   # 别的程序超过它就限速（用户定的）
HOLD = 10.0        # 回落到线下要持续这么久才恢复全速
SAMPLE = 1.0       # 多久量一次
WINDOW = 3.0       # 按最近这么多秒的平均判断：一秒的尖峰不算数
FLOOR = 0.10       # 限速时至少给压制留这么多
BURST = 0.25       # 最多攒几秒的份额：越小，每次连着跑的时间越短
MAX_SLEEP = 0.5    # 睡的时候每隔多久看一眼取消和开关
MIN_SLEEP = 0.005  # 欠得比这还少就不睡了，记着账等下一个包
NICE = 19          # Linux 的最低优先级

# Windows 的优先级类（GetPriorityClass 的返回值），按高低排
IDLE_PRIORITY_CLASS = 0x40
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
_CLASS_RANK = {IDLE_PRIORITY_CLASS: 0, BELOW_NORMAL_PRIORITY_CLASS: 1, 0x20: 2,
               0x8000: 3, 0x80: 4, 0x100: 5}

# 列队页那个开关的现值：queue.json 持久化它（jobqueue.QueueStore），
# 正在跑的压制每秒读一次，所以开关随时生效
_enabled = False
_active: Optional["Governor"] = None


def enabled() -> bool:
    return _enabled


def set_enabled(on: bool) -> None:
    global _enabled
    _enabled = bool(on)


# ------------------------------------------------------ 整机的 CPU 占用


def parse_proc_stat(line: str) -> Optional[Tuple[int, int]]:
    """/proc/stat 的 `cpu` 行 → 整机累计的（忙，总）时钟滴答。

    忙 = user + nice + system + irq + softirq；总 = 忙 + idle + iowait。steal
    （虚拟机被宿主拿走的时间）两边都不算：那不是这台机器上的程序在用，让路也
    让不给它。guest 已经含在 user/nice 里。
    """
    fields = line.split()
    if not fields or fields[0] != "cpu":
        return None
    try:
        values = [int(v) for v in fields[1:8]]
    except ValueError:
        return None
    user, nice, system, idle, iowait, irq, softirq = (values + [0] * 7)[:7]
    busy = user + nice + system + irq + softirq
    return busy, busy + idle + iowait


def _linux_times() -> Optional[Tuple[float, float]]:
    try:
        with open("/proc/stat", encoding="ascii") as f:
            ticks = parse_proc_stat(f.readline())
        hz = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        return None
    if ticks is None or hz <= 0:
        return None
    return ticks[0] / hz, ticks[1] / hz


@lru_cache(maxsize=1)
def _kernel32():
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    # 句柄是 64 位的：不声明类型的话 ctypes 按 32 位 int 传，GetCurrentProcess
    # 的伪句柄 -1 会变成一个无效句柄
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetCurrentProcess.argtypes = []
    k.GetPriorityClass.restype = wintypes.DWORD
    k.GetPriorityClass.argtypes = [wintypes.HANDLE]
    k.SetPriorityClass.restype = wintypes.BOOL
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.GetSystemTimes.restype = wintypes.BOOL
    k.GetSystemTimes.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
    return k


def _windows_times() -> Optional[Tuple[float, float]]:
    """GetSystemTimes：所有核累计的空闲、内核（含空闲）、用户时间，单位 100ns。"""
    import ctypes
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not _kernel32().GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                      ctypes.byref(user)):
        return None
    i, k, u = ((t.dwHighDateTime << 32 | t.dwLowDateTime) / 1e7 for t in (idle, kernel, user))
    return k - i + u, k + u


def read_system() -> Optional[Tuple[float, float]]:
    """整机累计的（忙，总）CPU 秒；这个系统读不到时 None。"""
    if sys.platform.startswith("linux"):
        return _linux_times()
    if sys.platform == "win32":
        try:
            return _windows_times()
        except (OSError, AttributeError, ValueError):
            return None
    return None


@lru_cache(maxsize=1)
def supported() -> bool:
    return read_system() is not None


# ------------------------------------------------------------ 优先级


class LinuxThreads:
    """Linux：nice 是逐线程的（NPTL），新线程继承创建它的那个线程的 nice。

    只降「这次压制开出来的线程」：本线程（任务线程），加上创建本对象之后新
    出现、又不是 Python 线程的那些——解码、编码、滤镜的工作线程都是 FFmpeg /
    x265 在本线程里开出来的原生线程（实测 x265 一开就是 15 个、SVT-AV1 78 个，
    名字都叫 python，认不出来，只能按出现的时间认）。开压之前就在的线程（网页
    服务、列队、已加载的模型），以及这期间新开的 Python 线程（另一个在等位的
    任务、处理请求的线程）一概不碰：普通用户降下去的优先级是调不回来的，碰了
    就等于永久改了别的任务。

    本线程降了以后，它再开的线程天生就是最低优先级；每秒再扫一遍，是为了开关
    在压制中途才打开的情况。
    """

    def __init__(self):
        self.me = threading.get_native_id()
        self.before = self._tids() - {self.me}
        try:
            self.natural = os.getpriority(os.PRIO_PROCESS, self.me)
        except OSError:
            self.natural = 0
        self.lowered: Dict[int, int] = {}   # tid → 原来的 nice

    @staticmethod
    def _tids() -> Set[int]:
        try:
            return {int(name) for name in os.listdir("/proc/self/task")}
        except OSError:
            return set()

    def lower(self) -> int:
        """把还没降的目标线程降到最低优先级，返回这次降了几个。"""
        dummy = getattr(threading, "_DummyThread", ())
        python = {t.native_id for t in threading.enumerate()
                  if not isinstance(t, dummy)} - {self.me}
        count = 0
        for tid in ((self._tids() - self.before - python) | {self.me}) - self.lowered.keys():
            try:
                was = os.getpriority(os.PRIO_PROCESS, tid)
                if was < NICE:
                    os.setpriority(os.PRIO_PROCESS, tid, NICE)
            except OSError:   # 线程刚好退出了
                continue
            # 已经是最低的，多半是降过的本线程开出来、继承来的：它本来该和本线程
            # 原来一样。按读到的值记，关掉开关时就「恢复」成最低优先级了
            self.lowered[tid] = was if was < NICE else min(was, self.natural)
            count += 1
        return count

    def restore(self) -> bool:
        """调回原来的优先级。没有权限时返回 False（Linux 普通用户只能降、不能升）。"""
        ok = True
        alive = self._tids()
        for tid, was in self.lowered.items():
            if tid not in alive:
                continue
            try:
                os.setpriority(os.PRIO_PROCESS, tid, was)
            except PermissionError:
                ok = False
            except OSError:
                pass
        self.lowered.clear()
        return ok


class WindowsProcess:
    """Windows：把整个进程降到「低于正常」。

    不像 Linux 那样逐线程：Windows 的新线程不继承创建者的优先级，逐线程降就得
    不停地追编码器开出来的新线程；而进程的优先级类随时调得回来，网页服务跟着
    降一会儿也无妨（它几乎不用 CPU）。本来就不高于「低于正常」的（比如用
    start /low 启动的）不动。
    """

    def __init__(self, kernel32=None):
        self._k = kernel32
        self._before: Optional[int] = None

    def _api(self):
        return self._k if self._k is not None else _kernel32()

    def lower(self) -> int:
        if self._before is not None:
            return 0
        k = self._api()
        handle = k.GetCurrentProcess()
        before = k.GetPriorityClass(handle)
        if not before or _CLASS_RANK.get(before, 2) <= _CLASS_RANK[BELOW_NORMAL_PRIORITY_CLASS]:
            return 0
        if not k.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS):
            return 0
        self._before = before
        return 1

    def restore(self) -> bool:
        if self._before is None:
            return True
        k = self._api()
        ok = bool(k.SetPriorityClass(k.GetCurrentProcess(), self._before))
        self._before = None
        return ok


class NoPriority:
    def lower(self) -> int:
        return 0

    def restore(self) -> bool:
        return True


def _priority():
    try:
        if sys.platform.startswith("linux") and os.path.isdir("/proc/self/task"):
            return LinuxThreads()
        if sys.platform == "win32":
            return WindowsProcess()
    except OSError:
        pass
    return NoPriority()


# ------------------------------------------------------------ 控制器


def _span(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes >= 60:
        return f"{minutes // 60} 小时 {minutes % 60} 分"
    if minutes:
        return f"{minutes} 分 {int(seconds % 60)} 秒"
    return f"{int(seconds)} 秒"


class Governor:
    """一次压制的让路控制：任务线程里创建并 start()，读包循环里每个包调一次
    pace()，结束时（成功、失败、取消都一样）close()。

    开关关着的时候 pace() 只是一次时钟读取和一次比较：几个小时的循环里每个包
    都要调它。
    """

    def __init__(self, log: Optional[Callable[[str], None]] = None, *,
                 is_on: Optional[Callable[[], bool]] = None,
                 system: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
                 cpu: Optional[Callable[[], float]] = None,
                 clock: Optional[Callable[[], float]] = None,
                 sleep: Optional[Callable[[float], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None,
                 priority=None):
        self._log = log or (lambda _m: None)
        self._cancelled = cancelled or (lambda: False)
        self._is_on = is_on or enabled
        self._system = system or read_system
        self._cpu = cpu or time.process_time
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._priority = priority if priority is not None else _priority()
        self.on = False
        self.measurable = True
        self.others: Optional[float] = None   # 别的程序的 CPU 占用（0–1）
        self.cap = 1.0                        # 压制此刻最多能用的份额（1 = 不限）
        self.capacity = float(os.cpu_count() or 1)   # 整机每秒能给的 CPU 秒
        self._history: Deque[Tuple[float, float, float, float]] = deque()
        self._next = 0.0
        self._over_until = 0.0
        self._credit = 0.0
        self._mark: Optional[Tuple[float, float]] = None
        self._since: Optional[float] = None   # 这一段限速从什么时候开始
        self.episodes = 0
        self.limited_seconds = 0.0
        self.peak = 0.0

    # -- 对外 --------------------------------------------------------------

    def start(self) -> "Governor":
        global _active
        _active = self
        # 开关开着就先降本线程：之后开出来的解码、编码线程天生就是最低优先级
        self._tick(self._clock())
        return self

    def pace(self) -> None:
        """限速时睡到欠的账还清为止。

        不能一次只睡一小段就回去：一个 1080p 的包喂进 x265，后台线程要花将近
        一个 CPU 秒去编它，比一小段睡眠挣回的份额还多——那样每个包都欠一点，
        实际占用就由「一段睡多久」决定，而不是由份额决定。分段睡只是为了中途
        看一眼取消和开关。
        """
        while True:
            now = self._clock()
            if now >= self._next:
                self._tick(now)
            if self.cap >= 1.0:
                return
            wait = self._limit(now)
            if wait < MIN_SLEEP or self._cancelled():
                return
            self._sleep(min(wait, MAX_SLEEP))

    def close(self) -> None:
        global _active
        if _active is self:
            _active = None
        self._end_episode(self._clock())
        if self.episodes:
            self._log(f"压制让路：限速 {self.episodes} 次，累计 {_span(self.limited_seconds)}，"
                      f"其他程序的 CPU 占用最高 {self.peak:.0%}")
        elif self.on and self.measurable and self.others is not None:
            self._log(f"压制让路：其他程序的 CPU 占用一直没超过 {THRESHOLD:.0%}，没有限速")
        if self.on:
            self._priority.restore()

    def note(self) -> str:
        """进度消息的尾巴：正在限速时说一声，否则空。"""
        if self.cap >= 1.0 or self.others is None:
            return ""
        return f" · 让路中：其他程序占 {self.others:.0%}，压制最多用 {self.cap:.0%}"

    def status(self) -> dict:
        return {"supported": self.measurable and supported(), "active": self.on,
                "limited": self.cap < 1.0, "others": self.others, "cap": self.cap}

    # -- 内部 --------------------------------------------------------------

    def _tick(self, now: float) -> None:
        self._next = now + SAMPLE
        on = bool(self._is_on())
        if on != self.on:
            self._switch(on, now)
        if not on:
            return
        self._priority.lower()
        sample = self._system()
        if sample is None:
            if self.measurable:
                self.measurable = False
                self._log("⚠ 这个系统上读不到整机的 CPU 占用：压制让路只降低了优先级，不会限速")
            return
        history = self._history
        history.append((now, sample[0], sample[1], self._cpu()))
        while len(history) > 2 and now - history[1][0] >= WINDOW:
            history.popleft()
        if len(history) < 2:
            return
        then, busy0, total0, ours0 = history[0]
        _, busy1, total1, ours1 = history[-1]
        span = total1 - total0
        if span <= 0 or now <= then:
            return
        self.capacity = span / (now - then)
        others = min(max((busy1 - busy0 - (ours1 - ours0)) / span, 0.0), 1.0)
        self.others = others
        self.peak = max(self.peak, others)
        if others > THRESHOLD:
            self._over_until = now + HOLD
        if now < self._over_until:
            if self._since is None:
                self._since = now
                self.episodes += 1
                if self.episodes == 1:
                    self._log(f"其他程序的 CPU 占用 {others:.0%}，超过 {THRESHOLD:.0%}：压制开始限速"
                              f"（之后的限速只在进度里标出）")
            self.cap = max(FLOOR, (1.0 - others) / 2)
        else:
            self._end_episode(now)
            self.cap = 1.0
            self._mark = None   # 旧账不带进下一段

    def _limit(self, now: float) -> float:
        """令牌桶：每秒进账 cap × 整机容量的 CPU 秒，按本进程实际用掉的扣。
        返回还要睡多少秒才能还清。"""
        rate = self.cap * self.capacity
        cpu = self._cpu()
        if self._mark is None:
            self._mark, self._credit = (now, cpu), 0.0
            return 0.0
        wall0, cpu0 = self._mark
        self._mark = (now, cpu)
        self._credit = min(self._credit + (now - wall0) * rate - (cpu - cpu0), BURST * rate)
        return -self._credit / rate if self._credit < 0 else 0.0

    def _end_episode(self, now: float) -> None:
        if self._since is not None:
            self.limited_seconds += now - self._since
            self._since = None

    def _switch(self, on: bool, now: float) -> None:
        self.on = on
        self._history.clear()
        if on:
            self._log(f"压制让路已开启：压制以最低优先级运行，其他程序的 CPU 占用超过 "
                      f"{THRESHOLD:.0%} 时限速")
            return
        self._end_episode(now)
        self.cap, self.others, self._mark, self._over_until = 1.0, None, None, 0.0
        if self._priority.restore():
            self._log("压制让路已关闭：恢复正常优先级，不再限速")
        else:
            self._log("压制让路已关闭，不再限速；但没有权限把优先级调回来（Linux 普通用户"
                      "只能降、不能升），这次压制剩下的部分仍是最低优先级——机器空闲时"
                      "这不影响速度")


def status() -> dict:
    """列队页显示用：正在受管的那个压制的状态，没有就是空闲。"""
    governor = _active
    if governor is None:
        return {"supported": supported(), "active": False, "limited": False,
                "others": None, "cap": 1.0}
    return governor.status()

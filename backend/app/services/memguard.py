"""压制的内存保护：压制绝不能把整台机器的内存吃光。

实测（2026-09-25，Ryzen 3600 12 线程，默认 CRF、音频转 E-AC-3，峰值是整个进程的）：

    片源            x265 medium   x265 veryslow   SVT-AV1 medium   SVT-AV1 veryslow   x264 medium
    DVD 480i        0.35 GB       —               —                —                  —
    1080p           1.06 GB       1.48 GB         2.44 GB          3.65 GB            0.93 GB
    4K 10bit HEVC   3.33 GB       4.92 GB         6.44 GB          10.18 GB           3.38 GB

整段 1080p 压完，内存在前几秒升到位，之后平着走（1057 MB 封顶），不是随时长增长
的——所以超没超上限在开压几秒内就见分晓。**线程数几乎管不了内存**：x265 减线程只省
5–10%，SVT-AV1 把并行度压到最低也只省 20%（它的 lp 是「并行级别」，核越多默认级别
越高，4K 能到 10 GB），真正决定内存的是编码器、分辨率和档位。所以这里只设线、不调
线程：

* **整机可用内存低于保护线（物理内存的 10%，至少 1 GB、最多 4 GB）就停下压制。**
  不管是谁吃的——压制是本程序能立刻还给系统的最大一块；等内核 OOM 来挑，挑中的可能
  是整个网站或者别的服务。暂停没用：编码器占着的内存不会因为暂停而释放。
* **压制自己用的超过上限就停下**（列队页可设，默认物理内存的一半）。「自己用的」是
  本进程从开压那一刻起涨了多少（常驻 + 换出），网页服务和之前加载的模型不算在内。
* **开压之前整机就已经低于保护线，先等**：那时开压只会马上被停下，而列队会接着开下一
  条、再被停下——一次内存紧张不该把整条列队跑成一排失败。

停下的任务在列队里点「重试」即可，片源一个字节不动（和磁盘空间不足那条同一个处理）。

**压制前后都把空闲内存还给系统（`trim`，gc + glibc 的 `malloc_trim`）**：编码器的几十个
线程各有自己的分配区，释放的内存 glibc 不会主动还——实测 4K SVT-AV1 压完，进程还占着
4.1 GB，trim 之后 1.0 GB（x265 4K：1.2 → 0.5 GB）。常驻的网站后端不还的话，压一次 4K
就一直多占几个 GB。开压前先还一次，「压制自己用了多少」才量得准：否则新的压制复用了上次
留下的内存，涨幅看着只有 0.2 GB，实际用了 1 GB。压完那一次在任务线程的最外层做
（`pipeline._run`），不在这里：被停下的压制，异常往外传的时候它的调用栈还攥着编码器——
实测在那之前 trim，进程停在 1.6 GB 下不来。

**上限是事后检查，挡不住一次性的大分配**：SVT-AV1 在打开编码器时一下子就要好几 GB
（4K 上限设 2 GB，实测进程先冲到 5.6 GB，7 秒后被停下）。它保证的是压制不会一直占着
超额的内存、机器不会被拖进 OOM，不是「一个字节都不会超」。
"""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from functools import lru_cache
from typing import Callable, Optional, Tuple

GIB = 1 << 30
FLOOR_SHARE = 0.10        # 整机可用内存的保护线：物理内存的 10%……
FLOOR_MIN = 1 * GIB       # ……至少 1 GB
FLOOR_MAX = 4 * GIB       # ……最多 4 GB
CHECK_SECONDS = 2.0       # 压制中多久查一次（读两个 /proc 文件，几十微秒）
WAIT_SECONDS = 5.0        # 开压前等内存时多久看一次……
WAIT_REPORT_SECONDS = 60  # ……多久在进度里说一次（每条进度都进任务日志和内存里的事件表）


class LowMemory(RuntimeError):
    """停下压制的原因，原样成为任务的失败信息（和 encode.LowDiskSpace 一样）。"""


# 列队页的「压制内存上限」（GB，0 = 自动：物理内存的一半）。queue.json 持久化它，
# 正在跑的压制每次检查都读现值
_limit_gb = 0.0
_active: Optional["MemoryGuard"] = None


def limit_gb() -> float:
    return _limit_gb


def set_limit(gb: float) -> None:
    global _limit_gb
    _limit_gb = max(float(gb), 0.0)


def floor_bytes(total: int) -> int:
    return int(min(max(total * FLOOR_SHARE, FLOOR_MIN), FLOOR_MAX))


def cap_bytes(total: int) -> int:
    return int(_limit_gb * GIB) if _limit_gb > 0 else total // 2


def gb(n: float) -> str:
    return f"{n / GIB:.1f} GB"


# ----------------------------------------------------------- 读内存


def parse_kib(text: str, *fields: str) -> Optional[Tuple[int, ...]]:
    """/proc/meminfo 或 /proc/self/status 里几个 `Name:  123 kB` 字段，换成字节。"""
    found = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in fields:
            try:
                found[name] = int(rest.split()[0]) * 1024
            except (IndexError, ValueError):
                return None
    if len(found) != len(fields):
        return None
    return tuple(found[f] for f in fields)


@lru_cache(maxsize=1)
def _windows():
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GlobalMemoryStatusEx.restype = wintypes.BOOL
    k.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetCurrentProcess.argtypes = []
    k.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    k.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                          wintypes.DWORD]
    return ctypes, k, MemoryStatus, Counters


def machine() -> Optional[Tuple[int, int]]:
    """整机的（物理内存，可用内存）字节数；读不到时 None。

    Linux 用 MemAvailable：它把可回收的页缓存算进去了，正是「不换页还能再给出去
    多少」；free 一栏会把缓存当成被占用，一台正常的机器看着也像快满了。
    """
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="ascii") as f:
                return parse_kib(f.read(), "MemTotal", "MemAvailable")
        if sys.platform == "win32":
            ctypes, k, MemoryStatus, _ = _windows()
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if k.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (OSError, AttributeError, ValueError):
        pass
    return None


def process() -> Optional[int]:
    """本进程占的内存字节数：Linux 是常驻 + 被换出的（只看常驻的话，内存一紧张、
    页被换出去，数字反而往下掉）；Windows 是私有提交量。读不到时 None。"""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/status", encoding="ascii") as f:
                got = parse_kib(f.read(), "VmRSS", "VmSwap")
            return sum(got) if got else None
        if sys.platform == "win32":
            ctypes, k, _, Counters = _windows()
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            if k.K32GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(counters),
                                         counters.cb):
                return int(counters.PrivateUsage)
    except (OSError, AttributeError, ValueError):
        pass
    return None


@lru_cache(maxsize=1)
def _malloc_trim():
    if not sys.platform.startswith("linux"):
        return None
    try:
        fn = ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):   # not glibc (musl): nothing to call
        return None
    fn.argtypes = [ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return fn


def trim() -> None:
    """把已经释放、却还留在进程里的内存还给系统（glibc 各线程的分配区不会自己还）。

    先 gc：编码器、滤镜图挂在闭包和环状引用上，不收一遍就还没释放，trim 也就
    还不回去。其他系统上只做 gc：Windows 的堆会自己把大块还回去。
    """
    gc.collect()
    fn = _malloc_trim()
    if fn is not None:
        fn(0)


# ----------------------------------------------------------- 守卫


class MemoryGuard:
    """一次压制的内存保护：任务线程里创建；start() 等到整机内存够了才返回；压制循环
    里每个包调 check()（自己按 CHECK_SECONDS 节流）；结束时 close() 记下峰值。"""

    def __init__(self, log: Optional[Callable[[str], None]] = None, *,
                 read_machine: Optional[Callable[[], Optional[Tuple[int, int]]]] = None,
                 read_process: Optional[Callable[[], Optional[int]]] = None,
                 clock: Optional[Callable[[], float]] = None,
                 sleep: Optional[Callable[[float], None]] = None,
                 release: Optional[Callable[[], None]] = None):
        self._log = log or (lambda _m: None)
        self._machine = read_machine or machine
        self._process = read_process or process
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._release = release or trim
        self.total: Optional[int] = None
        self.available: Optional[int] = None
        self.base: Optional[int] = None
        self.used = 0
        self.peak = 0
        self._next = 0.0

    # -- 对外 --------------------------------------------------------------

    def start(self, cancelled: Callable[[], bool],
              waiting: Optional[Callable[[str], None]] = None) -> None:
        global _active
        _active = self
        if not self._read_machine():
            self._log("⚠ 这个系统上读不到内存用量：压制不受内存保护")
        elif self.available < self.floor:
            message = (f"整机可用内存只剩 {gb(self.available)}，低于保护线 {gb(self.floor)}："
                       f"等内存腾出来再开始压制")
            self._log(message)
            said = self._clock()
            while self.available < self.floor:
                if cancelled():
                    raise InterruptedError
                if waiting and self._clock() - said >= WAIT_REPORT_SECONDS:
                    said = self._clock()
                    waiting(f"等内存：整机可用 {gb(self.available)}，低于保护线 {gb(self.floor)}")
                self._sleep(WAIT_SECONDS)
                self._read_machine()
            self._log(f"整机可用内存回到 {gb(self.available)}，开始压制")
        if self.total:
            self._log(f"内存保护：压制最多用 {gb(self.cap)}"
                      f"{'（自动：物理内存的一半）' if _limit_gb <= 0 else '（列队页设的上限）'}；"
                      f"整机可用内存低于 {gb(self.floor)} 时停下")
        # what an earlier job freed but kept would otherwise be reused unseen
        # and the encode would look smaller than it is
        self._release()
        self.base = self._process()
        self._next = self._clock() + CHECK_SECONDS

    def check(self) -> None:
        now = self._clock()
        if now < self._next:
            return
        self._next = now + CHECK_SECONDS
        current = self._process()
        if current is not None and self.base is not None:
            self.used = max(current - self.base, 0)
            self.peak = max(self.peak, self.used)
        if not self._read_machine():
            return
        if self.used > self.cap:
            raise LowMemory(
                f"压制占用的内存到了 {gb(self.used)}，超过上限 {gb(self.cap)}，已停止。"
                f"调高列队页的「压制内存上限」，或者换省内存的做法：x265 比 SVT-AV1 省一半多，"
                f"快一档、降分辨率也省（4K 的 SVT-AV1 要 6–10 GB）")
        if self.available < self.floor:
            raise LowMemory(
                f"整机可用内存只剩 {gb(self.available)}（保护线 {gb(self.floor)}），为免内存耗尽"
                f"已停止压制；压制本身占 {gb(self.used)}。腾出内存后在列队里点「重试」")

    def close(self) -> None:
        """Log the peak. Memory is handed back later, by the job runner
        (pipeline._run): while a stopped encode's exception is still on its
        way out, its traceback holds the encoder and the filter graph."""
        global _active
        if _active is self:
            _active = None
        if self.base is not None and self.peak:
            self._log(f"压制占用内存峰值 {gb(self.peak)}"
                      + (f"（整机 {gb(self.total)}）" if self.total else ""))

    @property
    def floor(self) -> int:
        return floor_bytes(self.total or 0)

    @property
    def cap(self) -> int:
        return cap_bytes(self.total or 0)

    def status(self) -> dict:
        return {**_describe(self.total, self.available), "active": True,
                "used_gb": self.used / GIB, "peak_gb": self.peak / GIB}

    # -- 内部 --------------------------------------------------------------

    def _read_machine(self) -> bool:
        got = self._machine()
        if got is None:
            return False
        self.total, self.available = got
        return True


def _describe(total: Optional[int], available: Optional[int]) -> dict:
    if not total:
        return {"supported": False, "total_gb": 0.0, "available_gb": 0.0, "floor_gb": 0.0,
                "limit_gb": 0.0, "auto_gb": 0.0}
    return {"supported": True, "total_gb": total / GIB, "available_gb": (available or 0) / GIB,
            "floor_gb": floor_bytes(total) / GIB, "limit_gb": cap_bytes(total) / GIB,
            "auto_gb": total // 2 / GIB}


def status() -> dict:
    """列队页显示用：正在跑的压制用了多少；没有压制时只报整机。"""
    guard = _active
    if guard is not None:
        return guard.status()
    got = machine()
    return {**_describe(*(got or (None, None))), "active": False, "used_gb": 0.0,
            "peak_gb": 0.0}

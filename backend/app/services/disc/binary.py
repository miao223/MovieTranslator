"""A bounds-checked big-endian cursor, shared by every disc parser.

MPLS/CLPI/index/MovieObject (Blu-ray) and IFO (DVD) are all big-endian
tables of fixed-width fields with length prefixes. A parser that walks off
the end of a truncated or foreign file must say so in words — a bare
struct.error from the middle of a playlist tells nobody which file was bad.
"""

from __future__ import annotations

import struct


class DiscError(Exception):
    """The disc (or one of its metadata files) cannot be read as expected.

    The message is shown to the user as is, so it is written in Chinese and
    names the file.
    """


class Reader:
    def __init__(self, data: bytes, name: str = "", pos: int = 0):
        self.data = data
        self.name = name
        self.pos = pos

    def _take(self, n: int) -> bytes:
        end = self.pos + n
        if n < 0 or end > len(self.data):
            raise DiscError(
                f"{self.name or '光盘文件'}已损坏或不完整："
                f"在第 {self.pos} 字节处需要 {n} 字节，文件只有 {len(self.data)} 字节")
        chunk = self.data[self.pos:end]
        self.pos = end
        return chunk

    def u8(self) -> int:
        return self._take(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self._take(2))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self._take(8))[0]

    def raw(self, n: int) -> bytes:
        return self._take(n)

    def text(self, n: int) -> str:
        return self._take(n).decode("ascii", "replace")

    def skip(self, n: int) -> None:
        self._take(n)

    def seek(self, pos: int) -> "Reader":
        if not 0 <= pos <= len(self.data):
            raise DiscError(
                f"{self.name or '光盘文件'}已损坏：指向第 {pos} 字节，"
                f"文件只有 {len(self.data)} 字节")
        self.pos = pos
        return self

    def at(self, pos: int) -> "Reader":
        """A second cursor over the same bytes, starting at *pos*."""
        return Reader(self.data, self.name).seek(pos)

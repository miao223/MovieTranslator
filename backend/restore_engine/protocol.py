"""The pipe between the app and the engine worker.

Every message is a 4-byte big-endian length, one byte saying what it is,
and the payload: ``J`` + UTF-8 JSON (control, logs, errors), ``F`` + a frame
(8-byte header: width, height, channels, bytes per sample — then the raw
pixels, row-major RGB). Frames go in order and come back in the same order,
one out for one in; timestamps never cross the pipe, the app keeps them.
Imported by the app as well as the worker, so: standard library and numpy.
"""

from __future__ import annotations

import json
import struct
from typing import BinaryIO, Optional, Tuple

import numpy as np

HEADER = struct.Struct(">HHBBxx")     # width, height, channels, bytes per sample


def send(stream: BinaryIO, kind: bytes, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload) + 1))
    stream.write(kind)
    stream.write(payload)
    stream.flush()


def send_json(stream: BinaryIO, message: dict) -> None:
    send(stream, b"J", json.dumps(message, ensure_ascii=False).encode("utf-8"))


def send_frame(stream: BinaryIO, array: np.ndarray) -> None:
    h, w, c = array.shape
    width = array.dtype.itemsize
    stream.write(struct.pack(">I", 1 + HEADER.size + array.nbytes))
    stream.write(b"F")
    stream.write(HEADER.pack(w, h, c, width))
    stream.write(np.ascontiguousarray(array).data)
    stream.flush()


def _read_exact(stream: BinaryIO, n: int) -> Optional[bytes]:
    chunks, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def receive(stream: BinaryIO) -> Tuple[Optional[bytes], object]:
    """(kind, decoded payload); (None, None) once the other side has gone."""
    head = _read_exact(stream, 4)
    if head is None:
        return None, None
    (length,) = struct.unpack(">I", head)
    body = _read_exact(stream, length)
    if body is None:
        return None, None
    kind, payload = body[:1], body[1:]
    if kind == b"J":
        return kind, json.loads(payload.decode("utf-8"))
    if kind == b"F":
        w, h, c, width = HEADER.unpack_from(payload)
        dtype = np.uint8 if width == 1 else np.uint16   # native order on both ends: one machine
        array = np.frombuffer(payload, dtype=dtype, offset=HEADER.size).reshape(h, w, c)
        return kind, array
    raise ValueError(f"unknown message kind {kind!r}")

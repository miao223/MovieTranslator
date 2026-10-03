"""Model files: downloaded once into the models folder, checked by sha256.

A file whose hash does not match is deleted and fetched again — never used.
``gdrive:<id>`` is a Google Drive file (RealViformer's authors publish there
only), fetched through the same public download endpoint gdown uses.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Callable, List

LogFn = Callable[[str], None]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _url(url: str) -> str:
    if url.startswith("gdrive:"):
        return ("https://drive.usercontent.google.com/download?export=download&confirm=t&id="
                + url[len("gdrive:"):])
    return url


def _download(url: str, target: Path, log: LogFn) -> None:
    part = target.with_name(target.name + ".part")
    request = urllib.request.Request(_url(url), headers={"User-Agent": "MovieTranslator"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            pct = int(done * 100 / total) if total else -1
            if total and pct // 10 != shown:
                shown = pct // 10
                log(f"下载 {target.name}：{pct}%（{done / 1e6:.0f} / {total / 1e6:.0f} MB）")
    os.replace(part, target)


def fetch(files: List[dict], folder: Path, log: LogFn) -> List[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in files:
        target = folder / item["file"]
        if target.exists() and not _sha256(target).startswith(item["sha256"]):
            log(f"⚠ {target.name} 校验不对，重新下载")
            target.unlink()
        if not target.exists():
            errors = []
            for url in item["urls"]:
                try:
                    _download(url, target, log)
                    break
                except Exception as exc:  # noqa: BLE001 — try the next mirror
                    errors.append(f"{url}：{exc}")
            else:
                raise RuntimeError(f"下载不了 {target.name}：" + "；".join(errors))
            if not _sha256(target).startswith(item["sha256"]):
                target.unlink()
                raise RuntimeError(f"{target.name} 下载下来校验不对（文件被改过或下载不完整）")
        paths.append(str(target))
    return paths

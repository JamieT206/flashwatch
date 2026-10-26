"""Shared helpers for the downloaders: logging, atomic writes, retries, HTTP fetch."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from flashwatch.config import LOGS


def get_logger(name: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"fw.{name}")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
        for h in (logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def retry(fn, log, what: str, tries: int = 4, wait: float = 30.0):
    """Call fn(); on failure wait (doubling) and retry. Returns None after the last failure."""
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - downloads fail in many ways; log and move on
            log.warning("%s: attempt %d/%d failed: %s: %s", what, i, tries, type(e).__name__, str(e)[:300])
            if i < tries:
                time.sleep(wait * 2 ** (i - 1))
    log.error("%s: giving up after %d attempts", what, tries)
    return None


def http_download(url: str, target: Path, log, chunk: int = 1 << 20) -> bool:
    """Stream url to target via a .part file, resuming a partial download. True on success."""
    import requests

    if target.exists():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 404:
            log.warning("404 %s", url)
            return False
        if r.status_code == 416:            # partial file already complete
            os.replace(part, target)
            return True
        r.raise_for_status()
        mode = "ab" if done and r.status_code == 206 else "wb"
        with open(part, mode) as f:
            for block in r.iter_content(chunk):
                f.write(block)
    os.replace(part, target)
    return True

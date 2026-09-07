from __future__ import annotations

import os
from pathlib import Path

_HEAD = 1_048_576
_TAIL = 1_048_576

_ORDERED_MARKERS = (
    ("google", ("synthid", "gemini", "google llc", "google deepmind")),
    ("openai", ("openai",)),
    ("microsoft", ("invismark", "microsoft corporation")),
    ("meta", ("trainedalgorithmicmedia", "content seal")),
)


def sniff_vendor(path: str | Path) -> str | None:
    p = Path(path)
    try:
        size = p.stat().st_size
        with open(p, "rb") as fh:
            head = fh.read(_HEAD)
            if size > _HEAD:
                fh.seek(size - min(_TAIL, size - _HEAD), os.SEEK_SET)
                tail = fh.read()
            else:
                tail = b""
    except OSError:
        return None
    blob = (head + tail).lower()
    for vendor, markers in _ORDERED_MARKERS:
        if any(m.encode() in blob for m in markers):
            return vendor
    return None

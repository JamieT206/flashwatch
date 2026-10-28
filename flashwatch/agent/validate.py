"""Number checker for agent drafts: every figure in the text must come from a tool output.

A number in the draft passes if some number in the tool results rounds to it, either as written
(0.123 -> "0.12") or as a percentage (0.123 -> "12%" or "12.3%"). Small counting words and calendar
numbers (years, days of the month, "week 2") are exempt. Failures go back to the model to fix
(at most 2 redrafts), then the sentences still carrying them are dropped (HANDOVER 6.5).
"""

from __future__ import annotations

import re
from typing import Iterable

NUM = re.compile(r"(?<![\w.])(-?\d+(?:,\d{3})*(?:\.\d+)?)(\s*%)?")
EXEMPT_CONTEXT = re.compile(r"(week|weeks|day|days|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*$", re.I)


def numbers_in(obj) -> list[float]:
    """Every number anywhere in a nested tool result (numbers inside strings included)."""
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(m.group(1).replace(",", "")) for m in NUM.finditer(obj)]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out += numbers_in(v)
    return out


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def supported(text_num: str, pct: bool, sources: Iterable[float]) -> bool:
    v = float(text_num.replace(",", ""))
    d = _decimals(text_num)
    tol = 0.5 * 10 ** -d + 1e-9
    for s in sources:
        if abs(s - v) <= tol:
            return True
        if pct and abs(100 * s - v) <= tol:                 # 0.123 written as 12% / 12.3%
            return True
    return False


def unsupported(text: str, tool_results: list) -> list[str]:
    """Figures in text that no tool result supports."""
    sources = numbers_in(tool_results)
    bad = []
    for m in NUM.finditer(text):
        num, pct = m.group(1), bool(m.group(2))
        v = float(num.replace(",", ""))
        before = text[max(0, m.start() - 12):m.start()]
        if not pct and "." not in num and (0 <= v <= 10 or 1900 <= v <= 2100 or EXEMPT_CONTEXT.search(before)):
            continue                                         # counts, years, dates, "week 2"
        if not supported(num, pct, sources):
            bad.append(m.group(0).strip())
    return bad


def drop_sentences(text: str, bad: list[str]) -> str:
    """Last resort: remove the sentences that still carry unsupported figures."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(s for s in sentences if not any(b in s for b in bad))

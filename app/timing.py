"""Server-Timing for slow endpoints: how long each part took, readable in the browser's network tab."""

from __future__ import annotations

import time


class Stopwatch:
    """Marks how long after the start each named part finished."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.marks: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        self.marks.append((name, (time.perf_counter() - self.start) * 1000))

    def header(self) -> str:
        """``espn;dur=512.3, news;dur=40.1, total;dur=530.0``."""
        parts = [*self.marks, ("total", (time.perf_counter() - self.start) * 1000)]
        return ", ".join(f"{name};dur={ms:.1f}" for name, ms in parts)

"""Time sources.

Every component asks a Clock for "now" instead of calling time.time() directly,
so the exact same pipeline runs live (WallClock) and on recorded ticks
(ReplayClock, advanced by the replayer to each frame's receive time).
"""
from __future__ import annotations

import time


class Clock:
    def now(self) -> float:
        raise NotImplementedError


class WallClock(Clock):
    def now(self) -> float:
        return time.time()


class ReplayClock(Clock):
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance_to(self, t: float) -> None:
        # never go backwards: recorded frames can be a few µs out of order
        if t > self._t:
            self._t = t

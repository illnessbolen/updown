"""Timed callbacks that behave the same live and in replay.

Execution is a chain of delays (book refresh, order travel time, maker timeout).
Live, they are asyncio timers; in a replay they sit in a heap and fire in time
order between recorded frames, with the ReplayClock advanced to each one — so
a backtest sees the same sequence of states a live run would.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from typing import Callable, List, Tuple

from .clock import Clock, ReplayClock

log = logging.getLogger("latarb.scheduler")


def _safe(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001 - a failing callback must not take the loop down
        log.exception("scheduled callback failed")


class Scheduler:
    def call_at(self, ts: float, fn: Callable[[], None]) -> None:
        raise NotImplementedError


class LoopScheduler(Scheduler):
    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def call_at(self, ts: float, fn: Callable[[], None]) -> None:
        asyncio.get_running_loop().call_later(max(0.0, ts - self.clock.now()), _safe, fn)


class ReplayScheduler(Scheduler):
    def __init__(self, clock: ReplayClock) -> None:
        self.clock = clock
        self._heap: List[Tuple[float, int, Callable[[], None]]] = []
        self._seq = itertools.count()

    def call_at(self, ts: float, fn: Callable[[], None]) -> None:
        heapq.heappush(self._heap, (max(ts, self.clock.now()), next(self._seq), fn))

    def every(self, period: float, fn: Callable[[], None], start: float) -> None:
        def tick(t=start):
            fn()
            self.call_at(t + period, lambda: tick(t + period))
        self.call_at(start, tick)

    def run_until(self, t: float) -> None:
        """Fire everything due at or before t, in time order (callbacks may schedule more)."""
        while self._heap and self._heap[0][0] <= t:
            ts, _, fn = heapq.heappop(self._heap)
            self.clock.advance_to(ts)
            _safe(fn)

    def __len__(self) -> int:
        return len(self._heap)

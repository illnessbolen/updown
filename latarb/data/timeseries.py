"""Append-only time series with age-based pruning and O(log n) time lookups."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import List, Optional, Tuple


class TimeSeries:
    __slots__ = ("max_age_s", "_ts", "_v", "_start")

    def __init__(self, max_age_s: float) -> None:
        self.max_age_s = max_age_s
        self._ts: List[float] = []
        self._v: List[float] = []
        self._start = 0

    def __len__(self) -> int:
        return len(self._ts) - self._start

    def append(self, ts: float, value: float) -> None:
        if self._ts and ts < self._ts[-1]:
            # keep the index sorted; a late sample is stamped at the last time
            ts = self._ts[-1]
        self._ts.append(ts)
        self._v.append(value)
        cutoff = ts - self.max_age_s
        start, tss = self._start, self._ts
        while start < len(tss) - 1 and tss[start] < cutoff:
            start += 1
        self._start = start
        if start > 4096 and start * 2 > len(tss):
            del self._ts[:start]
            del self._v[:start]
            self._start = 0

    def last(self) -> Optional[Tuple[float, float]]:
        if len(self) == 0:
            return None
        return self._ts[-1], self._v[-1]

    def at(self, ts: float) -> Optional[Tuple[float, float]]:
        """Previous-tick value: the last sample with time <= ts."""
        i = bisect_right(self._ts, ts, lo=self._start) - 1
        if i < self._start:
            return None
        return self._ts[i], self._v[i]

    def nearest(self, ts: float, tolerance_s: float) -> Optional[Tuple[float, float]]:
        """Sample closest to ts within tolerance; on a tie the later sample wins."""
        i = bisect_left(self._ts, ts, lo=self._start)
        best = None
        for j in (i - 1, i):
            if self._start <= j < len(self._ts):
                dt = abs(self._ts[j] - ts)
                if dt <= tolerance_s and (best is None or dt <= best[0]):
                    best = (dt, j)
        if best is None:
            return None
        j = best[1]
        return self._ts[j], self._v[j]

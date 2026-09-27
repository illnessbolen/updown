"""Log-basis between two price series: b = ln(A) - ln(B).

Used twice:
  * Coinbase (USD) vs Binance (USDT): the "normal" basis includes the USDT/USD
    premium, so a fixed bps threshold would either be too loose or fire all the
    time. Instead the check is on the DEVIATION of the current basis from its
    EWMA mean, scaled by its EWMA std (plus a floor and the current spreads).
  * Chainlink oracle vs Binance: the mean converts the fast Binance price into
    resolution-source units, the std is the oracle noise `eta` in the pricing
    model (the resolution price will not equal our estimate exactly).

EWMA weights are time-based (half-life in seconds) with a 1/(n+1) floor so the
first samples produce a plain average instead of being dominated by sample #1.
"""
from __future__ import annotations

import math
from typing import Optional


class BasisTracker:
    __slots__ = ("halflife_s", "min_samples", "mean", "var", "n", "last_ts", "last")

    def __init__(self, halflife_s: float, min_samples: int) -> None:
        self.halflife_s = halflife_s
        self.min_samples = min_samples
        self.mean = 0.0
        self.var = 0.0
        self.n = 0
        self.last_ts: Optional[float] = None
        self.last: Optional[float] = None

    def update(self, ts: float, a: float, b: float) -> None:
        if not (a > 0.0 and b > 0.0):
            return
        x = math.log(a) - math.log(b)
        self.last = x
        if self.n == 0:
            self.mean, self.var, self.n, self.last_ts = x, 0.0, 1, ts
            return
        dt = max(ts - (self.last_ts if self.last_ts is not None else ts), 0.0)
        alpha = 1.0 - 0.5 ** (dt / self.halflife_s)
        alpha = max(alpha, 1.0 / (self.n + 1))
        diff = x - self.mean
        incr = alpha * diff
        self.mean += incr
        self.var = (1.0 - alpha) * (self.var + diff * incr)
        self.n += 1
        self.last_ts = ts

    @property
    def ready(self) -> bool:
        return self.n >= self.min_samples

    @property
    def std(self) -> float:
        return math.sqrt(self.var) if self.var > 0.0 else 0.0

    def deviation(self, current: float) -> float:
        return current - self.mean

    def tolerance(self, k_std: float, floor: float) -> float:
        return max(floor, k_std * self.std)

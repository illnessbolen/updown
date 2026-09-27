"""Realized volatility from the live tick stream (no hard-coded constants).

Ticks arrive irregularly (tens per second for BTC, sparse for DOGE), so the
price is first sampled on a fixed grid (VOL_SAMPLE_S, previous-tick rule) and
EWMAs of squared grid log-returns are kept at two half-lives:

    fast (~2 min)   reacts to a volatility burst quickly
    slow (~30 min)  stable baseline

The EWMAs are bias-corrected during warm-up (divide by the sum of weights),
and `ready` stays False until VOL_MIN_SAMPLES grid returns were seen — the
signal layer refuses to price a market before that.

Horizon matching: the variance used for a window with tau seconds left is
    w * var_fast + (1 - w) * var_slow,   w = exp(-tau / fast_horizon)
so the last seconds of a 5m window are priced with the current regime while a
daily window mostly uses the long baseline.

A feed gap longer than max_gap_s restarts the grid instead of producing one
giant return (the outage would otherwise be booked as a volatility spike).

MultiScaleVol (below) adds the same estimator at longer sampling scales
(10 s, 60 s) and prices each window with the scale closest to its horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


class EwmaVariance:
    """Bias-corrected EWMA of squared returns, expressed as variance per second."""

    __slots__ = ("lam", "sample_s", "_acc", "_wsum", "n")

    def __init__(self, halflife_s: float, sample_s: float) -> None:
        self.lam = 0.5 ** (sample_s / halflife_s)
        self.sample_s = sample_s
        self._acc = 0.0
        self._wsum = 0.0
        self.n = 0

    def add(self, r: float) -> None:
        self._acc = self.lam * self._acc + (1.0 - self.lam) * (r * r) / self.sample_s
        self._wsum = self.lam * self._wsum + (1.0 - self.lam)
        self.n += 1

    def variance(self) -> Optional[float]:
        return self._acc / self._wsum if self._wsum > 0.0 else None


class RealizedVol:
    def __init__(self, sample_s: float = 1.0, fast_halflife_s: float = 120.0,
                 slow_halflife_s: float = 1800.0, min_samples: int = 300,
                 max_gap_s: float = 5.0) -> None:
        self.sample_s = sample_s
        self.min_samples = min_samples
        self.max_gap_s = max_gap_s
        self.fast = EwmaVariance(fast_halflife_s, sample_s)
        self.slow = EwmaVariance(slow_halflife_s, sample_s)
        self._last_ts: Optional[float] = None
        self._last_px: float = 0.0
        self._grid_ts: float = 0.0
        self._grid_px: float = 0.0
        self.gaps = 0

    # ------------------------------------------------------------------
    def _anchor(self, ts: float, px: float) -> None:
        self._grid_ts = math.floor(ts / self.sample_s) * self.sample_s
        self._grid_px = px
        self._last_ts = ts
        self._last_px = px

    def update(self, ts: float, price: float) -> None:
        if not price > 0.0:
            return
        if self._last_ts is None:
            self._anchor(ts, price)
            return
        if ts < self._last_ts:
            return                       # out-of-order tick; ignore
        if ts - self._last_ts > self.max_gap_s:
            self.gaps += 1
            self._anchor(ts, price)
            return
        nxt = self._grid_ts + self.sample_s
        while nxt <= ts:
            # price prevailing at grid point nxt = last tick before this one
            r = math.log(self._last_px / self._grid_px)
            self.fast.add(r)
            self.slow.add(r)
            self._grid_px = self._last_px
            self._grid_ts = nxt
            nxt += self.sample_s
        self._last_ts = ts
        self._last_px = price

    # ------------------------------------------------------------------
    @property
    def samples(self) -> int:
        return self.fast.n

    @property
    def ready(self) -> bool:
        return self.fast.n >= self.min_samples

    def sigmas(self) -> Optional[Tuple[float, float]]:
        """(sigma_fast, sigma_slow) per sqrt(second), or None before any return."""
        vf, vs = self.fast.variance(), self.slow.variance()
        if vf is None or vs is None:
            return None
        return math.sqrt(vf), math.sqrt(vs)

    def sigma_for_horizon(self, tau_s: float, fast_horizon_s: float) -> Optional[float]:
        vf, vs = self.fast.variance(), self.slow.variance()
        if vf is None or vs is None:
            return None
        w = math.exp(-max(tau_s, 0.0) / fast_horizon_s)
        return math.sqrt(w * vf + (1.0 - w) * vs)

    def sigma_band(self, tau_s: float, fast_horizon_s: float,
                   rel_uncertainty: float) -> Optional[Tuple[float, float, float]]:
        """(lo, point, hi): the fast/slow range widened by rel_uncertainty."""
        sg = self.sigmas()
        mid = self.sigma_for_horizon(tau_s, fast_horizon_s)
        if sg is None or mid is None:
            return None
        lo = min(sg) * (1.0 - rel_uncertainty)
        hi = max(sg) * (1.0 + rel_uncertainty)
        return lo, mid, hi


class ScaleVariance:
    """EWMA variance per second from log-returns sampled every `scale_s` seconds (previous tick)."""

    def __init__(self, scale_s: float, halflife_s: float, max_gap_s: float) -> None:
        self.scale_s = scale_s
        self.max_gap_s = max_gap_s
        self.ewma = EwmaVariance(halflife_s, scale_s)
        self._last_ts: Optional[float] = None
        self._last_px = 0.0
        self._grid_ts = 0.0
        self._grid_px = 0.0

    @property
    def n(self) -> int:
        return self.ewma.n

    def update(self, ts: float, price: float) -> None:
        if not price > 0.0:
            return
        if self._last_ts is None or ts - self._last_ts > self.max_gap_s:
            self._grid_ts = math.floor(ts / self.scale_s) * self.scale_s
            self._grid_px = price
            self._last_ts, self._last_px = ts, price
            return
        if ts < self._last_ts:
            return
        nxt = self._grid_ts + self.scale_s
        while nxt <= ts:
            self.ewma.add(math.log(self._last_px / self._grid_px))
            self._grid_px = self._last_px
            self._grid_ts = nxt
            nxt += self.scale_s
        self._last_ts, self._last_px = ts, price

    def sigma(self) -> Optional[float]:
        v = self.ewma.variance()
        return math.sqrt(v) if v is not None else None


@dataclass(frozen=True)
class VolEstimate:
    lo: float
    point: float
    hi: float
    scale_s: float          # sampling scale the point estimate comes from


class MultiScaleVol:
    """Realized vol at several sampling scales (1 s fast/slow + longer scales).

    Why: variance measured on 1-second returns does not have to scale linearly to
    a 4-minute horizon (short-horizon momentum / mean reversion, sticky quotes in
    calm periods). A window with tau seconds left is priced with the variance
    measured on the LONGEST scale <= tau (closest to the horizon), and the band
    used for the conservative edge spans every scale's estimate. A scale is used
    only after `min_obs` returns at that scale; a window whose horizon needs a
    scale that is not warmed up yet is not priced.
    """

    def __init__(self, base: RealizedVol, scales_s: Tuple[float, ...], halflife_s: float,
                 min_obs: int, max_gap_s: float) -> None:
        self.base = base
        self.min_obs = min_obs
        self.scales = {s: ScaleVariance(s, max(halflife_s, 30.0 * s), max_gap_s) for s in scales_s if s > base.sample_s}

    # compatibility with the 1 s estimator
    @property
    def ready(self) -> bool:
        return self.base.ready

    @property
    def samples(self) -> int:
        return self.base.samples

    @property
    def min_samples(self) -> int:
        return self.base.min_samples

    def sigmas(self) -> Optional[Tuple[float, float]]:
        return self.base.sigmas()

    def update(self, ts: float, price: float) -> None:
        self.base.update(ts, price)
        for s in self.scales.values():
            s.update(ts, price)

    def scale_sigmas(self) -> dict:
        """{scale_s: sigma or None if not warmed up}, including the 1 s slow estimate."""
        out = {}
        sg = self.base.sigmas()
        out[self.base.sample_s] = sg[1] if sg and self.base.ready else None
        for s, est in self.scales.items():
            out[s] = est.sigma() if est.n >= self.min_obs else None
        return out

    def estimate(self, tau_s: float, fast_horizon_s: float, rel_uncertainty: float) -> Tuple[Optional[VolEstimate], str]:
        if not self.base.ready:
            return None, "vol_warmup"
        sg = self.base.sigmas()
        base_point = self.base.sigma_for_horizon(tau_s, fast_horizon_s)
        if sg is None or base_point is None:
            return None, "vol_warmup"
        cands = {self.base.sample_s: base_point}
        spread = [sg[0], sg[1], base_point]
        for s, est in sorted(self.scales.items()):
            if s > max(tau_s, self.base.sample_s):
                break
            if est.n < self.min_obs:
                if tau_s >= 2.0 * s:
                    return None, "vol_warmup_long"     # the horizon needs this scale; do not guess
                break
            sig = est.sigma()
            if sig is not None:
                cands[s] = sig
                spread.append(sig)
        scale = max(cands)
        return VolEstimate(lo=min(spread) * (1.0 - rel_uncertainty), point=cands[scale],
                           hi=max(spread) * (1.0 + rel_uncertainty), scale_s=scale), ""

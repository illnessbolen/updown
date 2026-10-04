"""Reference price ("price to beat") for each window, with provenance.

Priority:
  1. Gamma publishes a price-to-beat for the event          -> gamma_price_to_beat
                         (about a minute after the window start, so it rarely helps a live bot)
  2. chainlink-resolved, settled on a TWAP (since 2026-08-07; markets.settlement_twap_s):
                         the TWAP of the Chainlink ticks over the lookback before the window start
                         (previous tick each second, RTDS history)  -> chainlink_twap
     chainlink-resolved, settled on the spot price (before):
                         Chainlink tick at the window start  -> chainlink_rtds (exact timestamp)
                         (RTDS history, 5h deep)                chainlink_rtds_nearest (within tolerance)
  3. binance-resolved:   open of the Binance kline that starts at the window start
                         REST /api/v3/klines                 -> binance_kline (authoritative)
                         first trade of the minute on the WS -> binance_trade_stream (provisional)
  4. otherwise           no reference -> the window is never priced.

A window the bot joined after its start without a stored tick at the start
stays unpriced for its whole life ("reference_missing"): guessing K from a
later price would be pricing a different contract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from ..config import Settings
from .gamma import make_session
from .hub import MarketDataHub
from .markets import BINANCE, CHAINLINK, MarketWindow, settlement_twap_s

log = logging.getLogger("latarb.reference")

KLINE_INTERVALS = {60: "1m", 300: "5m", 900: "15m", 3600: "1h", 14400: "4h", 86400: "1d"}


@dataclass(frozen=True)
class Reference:
    price: float
    source: str
    detail: str = ""
    final: bool = True


class BinanceRest:
    def __init__(self, cfg: Settings, session: Optional[requests.Session] = None) -> None:
        self.base = cfg.BINANCE_REST_URL.rstrip("/")
        self.timeout = cfg.HTTP_TIMEOUT_S
        self.session = session or make_session(pool=2)

    def kline_open(self, symbol: str, interval: str, start_ts: float) -> Optional[float]:
        start_ms = int(round(start_ts * 1000))
        r = self.session.get(self.base + "/api/v3/klines", timeout=self.timeout,
                             params={"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1})
        r.raise_for_status()
        rows = r.json()
        if rows and int(rows[0][0]) == start_ms:
            return float(rows[0][1])
        return None

    def server_time(self) -> float:
        r = self.session.get(self.base + "/api/v3/time", timeout=self.timeout)
        r.raise_for_status()
        return float(r.json()["serverTime"]) / 1000.0


def kline_interval(w: MarketWindow) -> Optional[str]:
    iv = KLINE_INTERVALS.get(w.duration_s)
    if iv is None or int(w.start_ts) % w.duration_s != 0:
        return None          # Binance candles are UTC-epoch aligned; e.g. a noon-ET daily window is not one candle
    return iv


class ReferenceResolver:
    REST_RETRY_S = 5.0

    def __init__(self, cfg: Settings, hub: MarketDataHub) -> None:
        self.cfg = cfg
        self.hub = hub
        self._cache: Dict[str, Reference] = {}
        self._rest_wanted: Dict[str, MarketWindow] = {}
        self._rest_next_try: Dict[str, float] = {}

    def get(self, w: MarketWindow, now: float) -> Tuple[Optional[Reference], str]:
        ref = self._cache.get(w.slug)
        if ref is not None:
            return ref, ""
        if now < w.start_ts:
            return None, "window_not_started"
        if w.price_to_beat:
            ref = Reference(w.price_to_beat, "gamma_price_to_beat")
            self._cache[w.slug] = ref
            return ref, ""
        st = self.hub.assets.get(w.asset)
        if st is None:
            return None, "asset_not_tracked"
        if w.resolution == CHAINLINK:
            lookback = settlement_twap_s(w)
            if lookback > 0:
                return self._twap_reference(w, st, now, int(lookback))
            tol = self.cfg.ORACLE_REF_TOLERANCE_S
            hit = st.oracle_hist.nearest(w.start_ts, tol)
            waiting = now <= w.start_ts + tol + self.cfg.ORACLE_STALE_S
            if hit is None:
                return None, ("reference_pending" if waiting else "reference_missing")
            dt = hit[0] - w.start_ts
            exact = abs(dt) < 5e-4
            if not exact and now <= w.start_ts + tol:
                return None, "reference_pending"        # a tick closer to the start may still arrive
            ref = Reference(hit[1], "chainlink_rtds" if exact else "chainlink_rtds_nearest",
                            "" if exact else f"dt={dt:+.3f}s")
            self._cache[w.slug] = ref
            return ref, ""
        if w.resolution == BINANCE:
            if kline_interval(w) is None:
                return None, "reference_rule_unknown"
            self._rest_wanted.setdefault(w.slug, w)
            cap = st.first_trades.get(int(w.start_ts))
            if cap is not None and cap[1]:
                ref = Reference(cap[0], "binance_trade_stream", final=False)
                self._cache[w.slug] = ref
                return ref, ""
            return None, "reference_pending"
        return None, "resolution_unknown"

    def _twap_reference(self, w: MarketWindow, st, now: float, lookback: int) -> Tuple[Optional[Reference], str]:
        """TWAP-settled window: the price to beat is the TWAP stream's value at the start, i.e. the mean of the
        oracle over [start - lookback, start). Rebuilt from the RTDS ticks, previous tick each second: on a 2 h
        recording (2026-10-02, 31 BTC 5m/15m windows priced) it was 0.09 bps from Gamma's priceToBeat in median
        (max 0.45), where the tick at the start was 2.1 bps off (max 13.4)."""
        start = int(w.start_ts)
        last = st.oracle_hist.last()
        if last is None or last[0] < start - 1:          # the last second before the start is not in yet
            waiting = now <= w.start_ts + self.cfg.ORACLE_REF_TOLERANCE_S + self.cfg.ORACLE_STALE_S
            return None, ("reference_pending" if waiting else "reference_missing")
        secs = range(start - lookback, start)
        pts = [st.oracle_hist.at(t) for t in secs]
        if any(p is None or t - p[0] > self.cfg.ORACLE_STALE_S for t, p in zip(secs, pts)):
            return None, "reference_missing"              # the history does not cover the averaging interval
        ref = Reference(sum(p[1] for p in pts) / lookback, "chainlink_twap", f"{lookback}s")
        self._cache[w.slug] = ref
        return ref, ""

    # ------------------------------------------------------------------ REST confirmation
    def rest_jobs(self, now: float) -> List[MarketWindow]:
        jobs = []
        for slug, w in list(self._rest_wanted.items()):
            if now > w.end_ts:
                del self._rest_wanted[slug]
                continue
            if now >= w.start_ts + 1.0 and self._rest_next_try.get(slug, 0.0) <= now:
                self._rest_next_try[slug] = now + self.REST_RETRY_S
                jobs.append(w)
        return jobs

    def fetch_rest(self, rest: BinanceRest, w: MarketWindow) -> Optional[float]:
        """Blocking; run in a worker thread."""
        symbol = self.cfg.BINANCE_SYMBOLS.get(w.asset)
        iv = kline_interval(w)
        if symbol is None or iv is None:
            return None
        try:
            return rest.kline_open(symbol, iv, w.start_ts)
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
            log.warning("kline open for %s failed: %s", w.slug, e)
            return None

    def set_rest_result(self, w: MarketWindow, price: Optional[float]) -> None:
        if price is None or price <= 0:
            return
        prev = self._cache.get(w.slug)
        if prev is not None and prev.final:
            return
        if prev is not None and abs(prev.price - price) > 1e-9 * price:
            log.warning("reference %s: trade-stream open %.8g != kline open %.8g, using kline",
                        w.slug, prev.price, price)
        self._cache[w.slug] = Reference(price, "binance_kline")
        self._rest_wanted.pop(w.slug, None)

    def prune(self, live_slugs) -> None:
        for slug in [s for s in self._cache if s not in live_slugs]:
            del self._cache[slug]

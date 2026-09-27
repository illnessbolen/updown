"""MarketDataHub: the single in-memory view of all feeds.

Transports feed parsed events in; the signal engine reads state out and is
notified synchronously (no queue in between: every hop costs latency).

Per asset it keeps:
    fast       latest Binance bookTicker (the fastest price; drives the model)
    sanity     latest Coinbase ticker (independent cross-check)
    oracle     latest Chainlink tick from Polymarket RTDS (resolution source)
    vol        realized volatility from the Binance mid
    sanity_basis   ln(coinbase) - ln(binance)
    oracle_basis   ln(chainlink) - ln(binance at the oracle timestamp)
    first_trades   first Binance trade of each minute (candle opens)
Plus one OrderBook per subscribed Polymarket token and per-feed liveness.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from ..clock import Clock
from ..config import Settings
from ..model.basis import BasisTracker
from ..model.volatility import RealizedVol
from .events import BookLevel, BookSnapshot, LastTrade, OracleTick, SpotQuote, SpotTrade, TickSizeChange
from .markets import MarketWindow
from .orderbook import OrderBook
from .timeseries import TimeSeries

log = logging.getLogger("latarb.hub")

SpotListener = Callable[[str, float], None]            # (asset, trigger recv_ts)
BookListener = Callable[[MarketWindow, float], None]   # (window, trigger recv_ts)
TradeListener = Callable[[LastTrade], None]            # Polymarket prints (paper maker fills)


class AssetState:
    FIRST_TRADES_KEEP = 26 * 60     # minutes

    def __init__(self, asset: str, cfg: Settings) -> None:
        self.asset = asset
        self.fast: Optional[SpotQuote] = None
        self.sanity: Optional[SpotQuote] = None
        self.oracle: Optional[OracleTick] = None
        self.fast_hist = TimeSeries(cfg.FAST_HISTORY_S)
        self.oracle_hist = TimeSeries(cfg.ORACLE_HISTORY_S)
        self.vol = RealizedVol(cfg.VOL_SAMPLE_S, cfg.VOL_FAST_HALFLIFE_S, cfg.VOL_SLOW_HALFLIFE_S,
                               cfg.VOL_MIN_SAMPLES, cfg.VOL_MAX_GAP_S)
        self.sanity_basis = BasisTracker(cfg.BASIS_HALFLIFE_S, cfg.BASIS_MIN_SAMPLES)
        self.oracle_basis = BasisTracker(cfg.ORACLE_BASIS_HALFLIFE_S, cfg.BASIS_MIN_SAMPLES)
        self.first_trades: Dict[int, Tuple[float, bool]] = {}   # minute start -> (price, certain)
        self.last_trade_exch_ts: Optional[float] = None
        self.feed_latency_ms: Optional[float] = None            # EWMA of recv - exchange time


class MarketDataHub:
    HIST_HEARTBEAT_S = 0.5

    def __init__(self, cfg: Settings, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self.assets: Dict[str, AssetState] = {a: AssetState(a, cfg) for a in cfg.ASSETS}
        self.books: Dict[str, OrderBook] = {}
        self.markets: Dict[str, MarketWindow] = {}
        self.by_asset: Dict[str, List[MarketWindow]] = {a: [] for a in cfg.ASSETS}
        self.by_token: Dict[str, MarketWindow] = {}
        self.feed_last_msg: Dict[str, float] = {}
        self.feed_connected: Dict[str, bool] = {}
        self.clock_skew_ms: Optional[float] = None
        self.spot_listeners: List[SpotListener] = []
        self.book_listeners: List[BookListener] = []
        self.trade_listeners: List[TradeListener] = []
        self.counters: Counter = Counter()

    # ------------------------------------------------------------------ markets
    def set_markets(self, windows: Iterable[MarketWindow]) -> Tuple[Set[str], Set[str]]:
        """Replace the tracked window set; returns (added_tokens, removed_tokens)."""
        windows = [w for w in windows if w.asset in self.assets]
        before = set(self.books)
        self.markets = {w.slug: w for w in windows}
        self.by_asset = {a: [] for a in self.assets}
        self.by_token = {}
        for w in windows:
            self.by_asset[w.asset].append(w)
            for t in w.tokens():
                self.by_token[t] = w
                if t not in self.books:
                    self.books[t] = OrderBook(t, w.tick_size)
        after = set(self.by_token)
        for t in before - after:
            del self.books[t]
        return after - before, before - after

    # ------------------------------------------------------------------ feed liveness
    def note_message(self, src: str, ts: float) -> None:
        self.feed_last_msg[src] = ts

    def on_feed_open(self, src: str, ts: float) -> None:
        self.feed_connected[src] = True
        self.feed_last_msg[src] = ts
        if src == "polymarket":
            for b in self.books.values():
                b.reset()                        # wait for fresh snapshots on the new connection
        elif src == "binance":
            for st in self.assets.values():
                st.last_trade_exch_ts = None     # trade continuity broken

    def on_feed_close(self, src: str, ts: float) -> None:
        self.feed_connected[src] = False
        if src == "polymarket":
            for b in self.books.values():
                b.synced = False

    def feed_age(self, src: str, now: float) -> float:
        t = self.feed_last_msg.get(src)
        return math.inf if t is None else now - t

    def carry_forward(self, now: float) -> None:
        """Called on the engine timer. bookTicker only pushes changes, so while the
        Binance connection is demonstrably alive, silence for one symbol means its price
        did not change: feed that to the vol grid so a quiet asset is not mistaken for an outage."""
        if self.feed_age("binance", now) > self.cfg.SPOT_STALE_S:
            return
        for st in self.assets.values():
            if st.fast is not None:
                st.vol.update(now, st.fast.mid)

    # ------------------------------------------------------------------ events
    def apply_many(self, events: list) -> None:
        """Apply a frame's events, then notify listeners once per touched asset / market."""
        spot_touched: Dict[str, float] = {}
        books_touched: Dict[str, Tuple[MarketWindow, float]] = {}
        for ev in events:
            kind = type(ev)
            if kind is SpotQuote:
                if self._on_quote(ev):
                    spot_touched[ev.asset] = ev.recv_ts
            elif kind is BookLevel or kind is BookSnapshot:
                w = self._on_book(ev)
                if w is not None:
                    books_touched[w.slug] = (w, ev.recv_ts)
            elif kind is SpotTrade:
                self._on_trade(ev)
            elif kind is OracleTick:
                self._on_oracle(ev)
            elif kind is TickSizeChange:
                b = self.books.get(ev.token)
                if b is not None:
                    b.tick_size = ev.tick_size
                    log.info("tick size change %s… -> %s", ev.token[:10], ev.tick_size)
            elif kind is LastTrade:
                self.counters["poly_trades"] += 1
                if ev.token in self.by_token:
                    for cb in self.trade_listeners:
                        cb(ev)
        for asset, ts in spot_touched.items():
            for cb in self.spot_listeners:
                cb(asset, ts)
        for w, ts in books_touched.values():
            for cb in self.book_listeners:
                cb(w, ts)

    def _on_quote(self, ev: SpotQuote) -> bool:
        st = self.assets.get(ev.asset)
        if st is None:
            return False
        if ev.src == "binance":
            prev = st.fast
            st.fast = ev
            mid = ev.mid
            # every quote advances the vol grid (an unchanged quote is information: zero return,
            # and it keeps a quiet-but-alive feed from looking like an outage)
            st.vol.update(ev.recv_ts, mid)
            if prev is not None and prev.bid == ev.bid and prev.ask == ev.ask:
                # size-only update: nothing to re-price, but keep the history dense enough that
                # "what was the mid at time t" lookups (oracle basis) see a fresh sample
                last = st.fast_hist.last()
                if last is None or ev.recv_ts - last[0] >= self.HIST_HEARTBEAT_S:
                    st.fast_hist.append(ev.recv_ts, mid)
                return False
            st.fast_hist.append(ev.recv_ts, mid)
            self.counters["binance_quotes"] += 1
            return True
        if ev.src == "coinbase":
            st.sanity = ev
            fq = st.fast
            if fq is not None and ev.recv_ts - fq.recv_ts <= self.cfg.SPOT_STALE_S:
                st.sanity_basis.update(ev.recv_ts, ev.mid, fq.mid)
            self.counters["coinbase_quotes"] += 1
        return False

    def _on_trade(self, ev: SpotTrade) -> None:
        if ev.src != "binance" or ev.exch_ts is None:
            return
        st = self.assets.get(ev.asset)
        if st is None:
            return
        lat = (ev.recv_ts - ev.exch_ts) * 1000.0
        st.feed_latency_ms = lat if st.feed_latency_ms is None else 0.99 * st.feed_latency_ms + 0.01 * lat
        minute = int(ev.exch_ts // 60) * 60
        if minute not in st.first_trades:
            # "certain" only if we saw an earlier trade on this same connection
            certain = st.last_trade_exch_ts is not None and st.last_trade_exch_ts < minute
            st.first_trades[minute] = (ev.price, certain)
            if len(st.first_trades) > AssetState.FIRST_TRADES_KEEP:
                for k in sorted(st.first_trades)[:len(st.first_trades) - AssetState.FIRST_TRADES_KEEP]:
                    del st.first_trades[k]
        if st.last_trade_exch_ts is None or ev.exch_ts > st.last_trade_exch_ts:
            st.last_trade_exch_ts = ev.exch_ts

    def _on_oracle(self, ev: OracleTick) -> None:
        st = self.assets.get(ev.asset)
        if st is None:
            return
        last = st.oracle_hist.last()
        if last is not None and ev.ts <= last[0]:
            return                               # duplicate / backfill older than what we hold
        st.oracle_hist.append(ev.ts, ev.price)
        st.oracle = ev
        self.counters["oracle_ticks"] += 1
        hit = st.fast_hist.at(ev.ts)
        if hit is not None and ev.ts - hit[0] <= self.cfg.SPOT_STALE_S:
            st.oracle_basis.update(ev.ts, ev.price, hit[1])

    def _on_book(self, ev) -> Optional[MarketWindow]:
        book = self.books.get(ev.token)
        if book is None:
            return None
        if type(ev) is BookSnapshot:
            book.apply_snapshot(ev.bids, ev.asks, ev.recv_ts)
            self.counters["poly_snapshots"] += 1
        else:
            if not book.synced:
                return None                      # deltas before a snapshot are meaningless
            book.apply_level(ev.side, ev.price, ev.size, ev.recv_ts)
            self.counters["poly_deltas"] += 1
        return self.by_token.get(ev.token)

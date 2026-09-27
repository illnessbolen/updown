"""Concrete feeds: Binance, Coinbase, Polymarket RTDS (Chainlink) and the CLOB market channel.

Each feed = ResilientWS transport + subscribe payload + pure parser. The frame
handler is the same for all of them: mark liveness, optionally record the raw
frame, parse, apply to the hub (which notifies the signal engine).
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

from ..clock import Clock
from ..config import Settings
from ..fastjson import dumps
from .hub import MarketDataHub
from .parsers import build_parsers, parse_safely
from .recorder import TickRecorder
from .ws import ResilientWS

log = logging.getLogger("latarb.feeds")


class FeedSet:
    def __init__(self, cfg: Settings, clock: Clock, hub: MarketDataHub,
                 recorder: Optional[TickRecorder] = None) -> None:
        self.cfg = cfg
        self.clock = clock
        self.hub = hub
        self.recorder = recorder
        self.parsers = build_parsers(cfg.BINANCE_SYMBOLS, cfg.COINBASE_PRODUCTS, cfg.CHAINLINK_SYMBOLS)
        self.parse_errors: Counter = Counter()
        self._last_parse_error: Dict[str, str] = {}

        self.binance_symbols = [cfg.BINANCE_SYMBOLS[a] for a in cfg.ASSETS if a in cfg.BINANCE_SYMBOLS]
        self.coinbase_products = [cfg.COINBASE_PRODUCTS[a] for a in cfg.ASSETS if a in cfg.COINBASE_PRODUCTS]
        self.chainlink_symbols = [cfg.CHAINLINK_SYMBOLS[a] for a in cfg.ASSETS if a in cfg.CHAINLINK_SYMBOLS]

        common = dict(protocol_ping_s=cfg.WS_PROTOCOL_PING_S, silence_timeout_s=cfg.WS_SILENCE_RECONNECT_S,
                      open_timeout_s=cfg.WS_OPEN_TIMEOUT_S, backoff_min_s=cfg.WS_BACKOFF_MIN_S,
                      backoff_max_s=cfg.WS_BACKOFF_MAX_S, proxy=cfg.WS_PROXY)
        self.binance = ResilientWS(
            "binance", self._binance_url, clock, on_message=self._handler("binance"),
            on_open=self._opener("binance", None), on_close=self._closer("binance"),
            should_connect=lambda: bool(self.binance_symbols), **common)
        self.coinbase = ResilientWS(
            "coinbase", lambda: cfg.COINBASE_WS_URL, clock, on_message=self._handler("coinbase"),
            on_open=self._opener("coinbase", self._coinbase_sub), on_close=self._closer("coinbase"),
            should_connect=lambda: bool(self.coinbase_products), **common)
        self.rtds = ResilientWS(
            "rtds", lambda: cfg.RTDS_WS_URL, clock, on_message=self._handler("rtds"),
            on_open=self._opener("rtds", self._rtds_sub), on_close=self._closer("rtds"),
            should_connect=lambda: bool(self.chainlink_symbols),
            app_ping_text="PING", app_ping_interval_s=cfg.RTDS_APP_PING_S, **common)
        self.poly = PolymarketFeed(cfg, clock, hub, self._handler("polymarket"),
                                   self._opener("polymarket", None), self._closer("polymarket"), common)

    def all(self) -> List[ResilientWS]:
        return [self.binance, self.coinbase, self.rtds, self.poly.ws]

    # ------------------------------------------------------------------ plumbing
    def _record(self, ts: float, src: str, raw: str) -> None:
        if self.recorder is not None:
            self.recorder.write(ts, src, raw)

    def _handler(self, src: str):
        parser = self.parsers[src]
        hub = self.hub

        def handle(raw: str, recv_ts: float) -> None:
            hub.note_message(src, recv_ts)
            if self.recorder is not None:
                self.recorder.write(recv_ts, src, raw)
            errs: List[str] = []
            events = parse_safely(parser, raw, recv_ts, errs)
            if errs:
                self.parse_errors[src] += 1
                self._last_parse_error[src] = errs[-1]
                if self.parse_errors[src] <= 5:
                    log.warning("%s: unparseable frame (%s): %.200s", src, errs[-1], raw)
            if events:
                hub.apply_many(events)
        return handle

    def _opener(self, src: str, subscribe):
        async def on_open(ws, ts: float) -> None:
            self._record(ts, "@open", src)
            self.hub.on_feed_open(src, ts)
            if subscribe is not None:
                for payload in subscribe():
                    await ws.send(payload)
        return on_open

    def _closer(self, src: str):
        def on_close(ts: float) -> None:
            self._record(ts, "@close", src)
            self.hub.on_feed_close(src, ts)
        return on_close

    # ------------------------------------------------------------------ subscriptions
    def _binance_url(self) -> str:
        streams = []
        for sym in self.binance_symbols:
            s = sym.lower()
            streams += [f"{s}@bookTicker", f"{s}@trade"]
        return f"{self.cfg.BINANCE_WS_URL}?streams={'/'.join(streams)}"

    def _coinbase_sub(self) -> Iterable[str]:
        yield dumps({"type": "subscribe", "product_ids": self.coinbase_products,
                     "channels": ["ticker", "heartbeat"]})

    def _rtds_sub(self) -> Iterable[str]:
        subs = [{"topic": "crypto_prices_chainlink", "type": "*", "filters": dumps({"symbol": s})}
                for s in self.chainlink_symbols]
        yield dumps({"action": "subscribe", "subscriptions": subs})


class PolymarketFeed:
    """CLOB market channel with a changing token set (windows roll every 5 minutes)."""

    MAX_DYNAMIC_FAILURES = 3

    def __init__(self, cfg: Settings, clock: Clock, hub: MarketDataHub, handler, on_open, on_close,
                 common: dict) -> None:
        self.cfg = cfg
        self.clock = clock
        self.hub = hub
        self._on_open_cb = on_open
        self.desired: Set[str] = set()
        self.subscribed: Set[str] = set()
        self.dynamic = cfg.POLY_DYNAMIC_SUBSCRIBE
        self._dynamic_failures = 0
        self._pending: Dict[str, float] = {}       # token -> snapshot deadline (dynamic adds only)
        self.ws = ResilientWS(
            "polymarket", lambda: cfg.POLY_WS_URL, clock, on_message=handler,
            on_open=self._open, on_close=on_close, should_connect=lambda: bool(self.desired),
            app_ping_text="PING", app_ping_interval_s=cfg.POLY_APP_PING_S, **common)

    async def _open(self, ws, ts: float) -> None:
        await self._on_open_cb(ws, ts)            # records @open, resets books
        tokens = sorted(self.desired)
        self.subscribed = set(tokens)
        self._pending.clear()
        await ws.send(dumps({"assets_ids": tokens, "type": "market"}))

    async def set_tokens(self, tokens: Iterable[str]) -> None:
        new = set(tokens)
        self.desired = new
        if not self.ws.connected:
            self.ws.poke()
            return
        add, remove = new - self.subscribed, self.subscribed - new
        if not add and not remove:
            return
        if not new:
            self.ws.reconnect("no tokens left")
            return
        if not self.dynamic:
            self.ws.reconnect(f"token set changed (+{len(add)} -{len(remove)})")
            return
        ok = True
        if add:
            ok = await self.ws.send_json(dumps({"assets_ids": sorted(add), "operation": "subscribe"}))
            deadline = self.clock.now() + self.cfg.POLY_SNAPSHOT_TIMEOUT_S
            for t in add:
                self._pending[t] = deadline
        if remove and ok:
            ok = await self.ws.send_json(dumps({"assets_ids": sorted(remove), "operation": "unsubscribe"}))
        if ok:
            self.subscribed = new
        else:
            self.ws.reconnect("dynamic subscribe send failed")

    def check_snapshots(self, now: float) -> None:
        """Dynamic subscribe must yield a book snapshot for the new tokens; if none of
        them got one in time, fall back to a full reconnect (and after repeated
        failures stop using dynamic subscribe at all)."""
        if not self._pending:
            return
        due = {t: d for t, d in self._pending.items() if d <= now}
        if not due:
            return
        got_any = False
        for t in due:
            b = self.hub.books.get(t)
            if b is not None and b.synced:
                got_any = True
        for t in list(self._pending):
            b = self.hub.books.get(t)
            if b is None or b.synced or t in due:
                self._pending.pop(t, None)
        if got_any:
            self._dynamic_failures = 0
            return
        self._dynamic_failures += 1
        if self._dynamic_failures >= self.MAX_DYNAMIC_FAILURES and self.dynamic:
            self.dynamic = False
            log.warning("polymarket: dynamic subscribe produced no snapshots %d times - "
                        "switching to reconnect-on-change", self._dynamic_failures)
        self.ws.reconnect("no book snapshot for newly added tokens")

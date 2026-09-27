"""Pre-trade book refresh: never send an order priced off a book seen more than BOOK_MAX_AGE_MS ago.

    RestBookRefresher   GET /book for both outcome tokens right before sending (live paper,
                        and the live exchange later); its latency counts toward the budget.
    WsBookRefresher     the local WebSocket books after a simulated refresh delay (replay,
                        or PRE_TRADE_REFRESH=ws); the price's age is the time since the last
                        frame on the synced CLOB connection (incremental feed: silence on a live,
                        synced connection means "unchanged"; PONGs alone rarely keep it < 1.5 s).

Both call done(FreshBooks | None, error | None) later, on the event loop / scheduler.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import requests

from ..clock import Clock
from ..config import Settings
from ..data.gamma import make_session
from ..data.hub import MarketDataHub
from ..data.markets import MarketWindow
from ..data.orderbook import OrderBook
from ..data.parsers import _levels, _ms_to_s
from ..scheduler import Scheduler

log = logging.getLogger("latarb.books")


@dataclass
class FreshBooks:
    up: OrderBook
    down: OrderBook
    source: str                 # rest | ws
    requested_ts: float
    received_ts: float
    exch_ts: Optional[float] = None
    feed_ts: Optional[float] = None   # ws: last frame on the (synced) CLOB connection

    @property
    def latency_ms(self) -> float:
        return (self.received_ts - self.requested_ts) * 1000.0

    def age_ms(self, token: str, now: float) -> float:
        """How old the information behind a price on `token` is."""
        if self.source == "rest":
            return (now - self.requested_ts) * 1000.0      # the book is at least as new as the request
        # The market channel pushes every change in order, so a synced book is current as of the
        # last frame received on its connection (or its own last change, if that is newer).
        book = self.up if self.up.token == token else self.down
        seen = max(book.updated_ts, self.feed_ts or 0.0)
        return (now - seen) * 1000.0


Done = Callable[[Optional[FreshBooks], Optional[str]], None]


def book_from_rest(token: str, data: dict, tick_size: float) -> OrderBook:
    b = OrderBook(token, tick_size)
    b.apply_snapshot(_levels(data.get("bids")), _levels(data.get("asks")), 0.0)
    return b


class RestBookRefresher:
    counts_as_error = True          # a failed REST call is an execution error (circuit breaker)

    def __init__(self, cfg: Settings, hub: MarketDataHub, clock: Clock,
                 session: Optional[requests.Session] = None) -> None:
        self.base = cfg.CLOB_URL.rstrip("/")
        self.timeout = min(cfg.HTTP_TIMEOUT_S, 2.0)      # a slow book is a stale book
        self.hub = hub
        self.clock = clock
        self.session = session or make_session(pool=4)

    def _get(self, token: str) -> dict:
        r = self.session.get(self.base + "/book", params={"token_id": token}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def refresh(self, w: MarketWindow, done: Done) -> None:
        asyncio.get_running_loop().create_task(self._run(w, done))

    async def _run(self, w: MarketWindow, done: Done) -> None:
        t0 = self.clock.now()
        try:
            up, down = await asyncio.gather(asyncio.to_thread(self._get, w.up_token),
                                            asyncio.to_thread(self._get, w.down_token))
        except (requests.RequestException, ValueError) as e:
            done(None, f"{type(e).__name__}: {e}")
            return
        t1 = self.clock.now()
        tick = lambda t: self.hub.books[t].tick_size if t in self.hub.books else w.tick_size  # noqa: E731
        ts = [x for x in (_ms_to_s(up.get("timestamp")), _ms_to_s(down.get("timestamp"))) if x]
        done(FreshBooks(book_from_rest(w.up_token, up, tick(w.up_token)),
                        book_from_rest(w.down_token, down, tick(w.down_token)),
                        "rest", t0, t1, min(ts) if ts else None), None)


class WsBookRefresher:
    counts_as_error = False         # "no synced book" is a data condition, not an error

    def __init__(self, hub: MarketDataHub, clock: Clock, scheduler: Scheduler, latency_ms: float) -> None:
        self.hub = hub
        self.clock = clock
        self.scheduler = scheduler
        self.latency_s = latency_ms / 1000.0

    def refresh(self, w: MarketWindow, done: Done) -> None:
        t0 = self.clock.now()

        def fire() -> None:
            up, down = self.hub.books.get(w.up_token), self.hub.books.get(w.down_token)
            if up is None or down is None or not (up.synced and down.synced):
                done(None, "no synced book")
                return
            done(FreshBooks(up, down, "ws", t0, self.clock.now(),
                            feed_ts=self.hub.feed_last_msg.get("polymarket")), None)

        self.scheduler.call_at(t0 + self.latency_s, fire)

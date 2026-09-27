"""Paper exchange: simulated order handling against the LIVE Polymarket book.

The point of paper trading a latency strategy is to find out whether the stale
quote is still there when the order arrives. So an order is matched against
the local book as it looks PAPER_FILL_DELAY_MS after it was sent (order travel
time + our own feed delay), not against the price that triggered the signal:
if faster traders took the quote meanwhile, the paper order misses it too.

Taker (FAK): walks the asks up to the limit — the token's own asks and the
complement (1 - bid of the other outcome, CTF mint matching) — fills what is
there, cancels the rest. Taker fee per level: rate * (p(1-p))^exp.

Maker (GTC bid at `limit`, fee 0):
  * whatever crosses on arrival executes immediately as taker (as on a real CLOB);
  * the rest rests; it is filled only by evidence that a seller would have
    hit it:
      - an ask (or complement bid) appears at or through our price,
      - a print on our token BELOW our price (trade-through),
      - prints AT our price beyond the size that was queued ahead of us;
  * cancelled at MAKER_TIMEOUT_MS, when the pipeline says the edge is gone,
    or at the window end.
Prints on the other outcome are ignored (conservative: fewer maker fills).

Paper fills never remove liquidity from the real book, so each simulated fill
is remembered for PAPER_LIQUIDITY_MEMORY_S and subtracted from that level;
otherwise the same quote could be "bought" again by the next order.

Known optimism that remains: no queue competition at the ask we lift beyond
what the book already shows, and no rejects. Treat paper P&L as an upper bound.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set, Tuple

from ..clock import Clock
from ..config import Settings
from ..data.events import LastTrade
from ..data.hub import MarketDataHub
from ..data.markets import MarketWindow
from ..model.fees import taker_fee_per_share
from ..scheduler import Scheduler

log = logging.getLogger("latarb.paper")

EPS = 1e-9


@dataclass
class PaperOrder:
    window: MarketWindow
    side: str
    token: str
    other_token: str
    kind: str                       # maker | taker
    limit: float
    shares: float
    t_send: float
    fee_rate: float
    fee_exp: float
    on_done: Callable[["PaperOrder"], None]
    order_id: str = ""
    arrival_ts: float = 0.0
    status: str = "pending"         # pending | resting | done
    end_reason: str = ""
    filled: float = 0.0
    notional: float = 0.0
    fee: float = 0.0
    maker_filled: float = 0.0
    taker_filled: float = 0.0
    queue_ahead: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def remaining(self) -> float:
        return max(0.0, self.shares - self.filled)

    @property
    def avg_price(self) -> float:
        return self.notional / self.filled if self.filled > 0 else 0.0


class PaperExchange:
    def __init__(self, cfg: Settings, hub: MarketDataHub, clock: Clock, scheduler: Scheduler) -> None:
        self.cfg = cfg
        self.hub = hub
        self.clock = clock
        self.scheduler = scheduler
        self.resting: Dict[str, PaperOrder] = {}
        self._by_slug: Dict[str, Set[str]] = {}
        self._consumed: Dict[Tuple[str, float], Tuple[float, float]] = {}
        self._ids = itertools.count(1)
        hub.book_listeners.append(self._on_book)
        hub.trade_listeners.append(self._on_trade)

    # ------------------------------------------------------------------ API
    def submit(self, o: PaperOrder) -> PaperOrder:
        o.order_id = f"P{next(self._ids)}"
        o.arrival_ts = o.t_send + self.cfg.PAPER_FILL_DELAY_MS / 1000.0
        self.scheduler.call_at(o.arrival_ts, lambda: self._arrive(o))
        return o

    def cancel(self, order_id: str, reason: str) -> None:
        o = self.resting.get(order_id)
        if o is not None:
            self._finish(o, reason)

    def cancel_all(self, reason: str) -> None:
        for oid in list(self.resting):
            self.cancel(oid, reason)

    def resting_for(self, slug: str) -> List[PaperOrder]:
        return [self.resting[i] for i in self._by_slug.get(slug, ()) if i in self.resting]

    # ------------------------------------------------------------------ liquidity
    def _available(self, token: str, price: float, displayed: float) -> float:
        c = self._consumed.get((token, price))
        if c is not None and c[1] > self.clock.now():
            return max(0.0, displayed - c[0])
        return displayed

    def _consume(self, token: str, price: float, qty: float) -> None:
        now = self.clock.now()
        c = self._consumed.get((token, price))
        prev = c[0] if c is not None and c[1] > now else 0.0
        self._consumed[(token, price)] = (prev + qty, now + self.cfg.PAPER_LIQUIDITY_MEMORY_S)
        if len(self._consumed) > 5000:
            self._consumed = {k: v for k, v in self._consumed.items() if v[1] > now}

    def _sell_levels(self, o: PaperOrder) -> List[Tuple[float, float, str, float]]:
        """(effective price, available size, token of the level, level price), best first, <= limit."""
        own, other = self.hub.books.get(o.token), self.hub.books.get(o.other_token)
        out = []
        if own is not None:
            for p, s in own.asks.items():
                if p <= o.limit + EPS:
                    out.append((p, self._available(o.token, p, s), o.token, p))
        if other is not None and self.cfg.USE_COMPLEMENT_BOOK:
            for p, s in other.bids.items():
                eff = round(1.0 - p, 6)
                if eff <= o.limit + EPS:
                    out.append((eff, self._available(o.other_token, p, s), o.other_token, p))
        out.sort(key=lambda x: x[0])
        return out

    def _fill(self, o: PaperOrder, qty: float, price: float, maker: bool) -> None:
        o.filled += qty
        o.notional += qty * price
        if maker:
            o.maker_filled += qty
        else:
            o.taker_filled += qty
            o.fee += qty * taker_fee_per_share(price, o.fee_rate, o.fee_exp)

    def _lift(self, o: PaperOrder, as_maker: bool) -> None:
        for price, size, tok, lvl in self._sell_levels(o):
            if o.remaining <= EPS:
                break
            q = min(o.remaining, size)
            if q <= EPS:
                continue
            self._fill(o, q, o.limit if as_maker else price, as_maker)
            self._consume(tok, lvl, q)

    # ------------------------------------------------------------------ lifecycle
    def _arrive(self, o: PaperOrder) -> None:
        w = o.window
        now = self.clock.now()
        if now >= w.end_ts:
            return self._finish(o, "window_closed")
        own, other = self.hub.books.get(o.token), self.hub.books.get(o.other_token)
        if own is None or other is None or not (own.synced and other.synced):
            return self._finish(o, "no_book")
        self._lift(o, as_maker=False)             # anything marketable executes as taker
        if o.remaining <= EPS:
            return self._finish(o, "filled")
        if o.kind == "taker":
            return self._finish(o, "partial" if o.filled > 0 else "no_liquidity")
        o.status = "resting"
        o.queue_ahead = own.bids.get(round(o.limit, 6), 0.0)
        self.resting[o.order_id] = o
        self._by_slug.setdefault(w.slug, set()).add(o.order_id)
        expire = min(now + self.cfg.MAKER_TIMEOUT_MS / 1000.0, w.end_ts - 0.001)
        self.scheduler.call_at(expire, lambda: self._expire(o))

    def _expire(self, o: PaperOrder) -> None:
        if o.status == "resting":
            self._finish(o, "maker_timeout" if self.clock.now() < o.window.end_ts - 0.01 else "window_closed")

    def _on_book(self, w: MarketWindow, ts: float) -> None:
        for o in self.resting_for(w.slug):
            self._lift(o, as_maker=True)          # a seller showed up at or through our bid
            if o.remaining <= EPS:
                self._finish(o, "filled")

    def _on_trade(self, t: LastTrade) -> None:
        w = self.hub.by_token.get(t.token)
        if w is None:
            return
        for o in self.resting_for(w.slug):
            if o.token != t.token or o.remaining <= EPS:
                continue
            if t.price < o.limit - EPS:
                q = min(o.remaining, t.size)          # traded through our price
            elif abs(t.price - o.limit) <= EPS:
                o.queue_ahead -= t.size
                q = min(o.remaining, -o.queue_ahead) if o.queue_ahead < 0 else 0.0
                o.queue_ahead = max(o.queue_ahead, 0.0)
            else:
                continue
            if q > EPS:
                self._fill(o, q, o.limit, maker=True)
            if o.remaining <= EPS:
                self._finish(o, "filled")

    def _finish(self, o: PaperOrder, reason: str) -> None:
        if o.status == "done":
            return
        o.status = "done"
        o.end_reason = reason
        self.resting.pop(o.order_id, None)
        ids = self._by_slug.get(o.window.slug)
        if ids is not None:
            ids.discard(o.order_id)
            if not ids:
                del self._by_slug[o.window.slug]
        o.on_done(o)

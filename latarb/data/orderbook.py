"""Local L2 book for one Polymarket outcome token, maintained from the CLOB market channel."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from .events import ASK, BID


class OrderBook:
    __slots__ = ("token", "bids", "asks", "synced", "updated_ts", "snapshot_ts", "tick_size",
                 "_best_bid", "_best_ask", "_dirty")

    def __init__(self, token: str, tick_size: float = 0.01) -> None:
        self.token = token
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.synced = False            # True only after a full snapshot on the current connection
        self.updated_ts = 0.0
        self.snapshot_ts = 0.0
        self.tick_size = tick_size
        self._best_bid: Optional[Tuple[float, float]] = None
        self._best_ask: Optional[Tuple[float, float]] = None
        self._dirty = True

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.synced = False
        self._dirty = True

    def apply_snapshot(self, bids: Iterable[Tuple[float, float]],
                       asks: Iterable[Tuple[float, float]], ts: float) -> None:
        self.bids = {p: s for p, s in bids if s > 0}
        self.asks = {p: s for p, s in asks if s > 0}
        self.synced = True
        self.snapshot_ts = ts
        self.updated_ts = ts
        self._dirty = True

    def apply_level(self, side: str, price: float, size: float, ts: float) -> None:
        levels = self.bids if side == BID else self.asks
        if size > 0:
            levels[price] = size
        else:
            levels.pop(price, None)
        self.updated_ts = ts
        self._dirty = True

    def _refresh(self) -> None:
        if self.bids:
            p = max(self.bids)
            self._best_bid = (p, self.bids[p])
        else:
            self._best_bid = None
        if self.asks:
            p = min(self.asks)
            self._best_ask = (p, self.asks[p])
        else:
            self._best_ask = None
        self._dirty = False

    def best_bid(self) -> Optional[Tuple[float, float]]:
        if self._dirty:
            self._refresh()
        return self._best_bid

    def best_ask(self) -> Optional[Tuple[float, float]]:
        if self._dirty:
            self._refresh()
        return self._best_ask

    @property
    def crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        return bb is not None and ba is not None and bb[0] >= ba[0]

    def vwap_buy(self, shares: float) -> Optional[float]:
        """Average price to lift `shares` from the asks, None if the book is too thin."""
        if shares <= 0:
            return None
        left, cost = shares, 0.0
        for p in sorted(self.asks):
            take = min(left, self.asks[p])
            cost += take * p
            left -= take
            if left <= 1e-12:
                return cost / shares
        return None

"""Normalized market-data events produced by the parsers and consumed by the hub.

recv_ts is always the LOCAL receive time (Clock.now() when the frame arrived);
exch_ts is the venue's own timestamp when the message carries one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

BID = "bid"
ASK = "ask"


@dataclass(slots=True)
class SpotQuote:
    src: str                 # "binance" | "coinbase"
    asset: str
    bid: float
    ask: float
    recv_ts: float
    exch_ts: Optional[float] = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass(slots=True)
class SpotTrade:
    src: str
    asset: str
    price: float
    size: float
    recv_ts: float
    exch_ts: Optional[float] = None


@dataclass(slots=True)
class OracleTick:
    src: str                 # "chainlink"
    asset: str
    price: float
    ts: float                # oracle observation time (seconds)
    recv_ts: float


@dataclass(slots=True)
class BookSnapshot:
    token: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    recv_ts: float
    exch_ts: Optional[float] = None


@dataclass(slots=True)
class BookLevel:
    token: str
    side: str                # BID | ASK
    price: float
    size: float              # new aggregate size at the level; 0 removes it
    recv_ts: float
    exch_ts: Optional[float] = None


@dataclass(slots=True)
class TickSizeChange:
    token: str
    tick_size: float
    recv_ts: float


@dataclass(slots=True)
class LastTrade:
    token: str
    price: float
    size: float
    side: str
    recv_ts: float
    exch_ts: Optional[float] = None

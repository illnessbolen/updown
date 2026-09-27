"""Paper ledger: cash, open positions, settlements, per-day P&L.

Accounting rules (carried over from the previous bot's FIX 1/2/3):
  * a position is booked only from ACTUAL (simulated) fills: shares, average
    price and fee as filled, not as intended;
  * the money at risk (cost + fees) stays locked until the market resolves;
  * realized P&L comes only from settlements (winner from Gamma), never from
    the model's expectation.

bankroll = cash + money locked in open positions (at cost). Every risk limit is
a fraction of it, so limits scale with the actual balance.

The state is persisted after every change (atomic replace). Restarting the bot
therefore does not reset the day's losses — a restart cannot be used to
escape the daily stop.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..data.markets import MarketWindow

log = logging.getLogger("latarb.portfolio")


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


@dataclass
class Position:
    slug: str
    asset: str
    label: str
    side: str
    token: str
    window: dict
    shares: float = 0.0
    cost: float = 0.0              # sum of price * shares
    fees: float = 0.0
    maker_shares: float = 0.0
    taker_shares: float = 0.0
    fills: int = 0
    opened_ts: float = 0.0
    end_ts: float = 0.0
    provisional_loss: bool = False  # our own end-of-window data says this lost; Gamma not final yet
    ref_price: float = 0.0          # the reference (price to beat) the position was priced against

    @property
    def key(self) -> str:
        return f"{self.slug}:{self.side}"

    @property
    def at_risk(self) -> float:
        return self.cost + self.fees

    @property
    def avg_price(self) -> float:
        return self.cost / self.shares if self.shares else 0.0


@dataclass
class Settlement:
    position: Position
    winner: str
    payout: float
    pnl: float
    ts: float

    @property
    def won(self) -> bool:
        return self.position.side == self.winner


@dataclass
class Portfolio:
    start_cash: float
    cash: Optional[float] = None   # None -> start_cash (a real 0.0 balance must survive a reload)
    positions: Dict[str, Position] = field(default_factory=dict)
    realized: float = 0.0
    wins: int = 0
    losses: int = 0
    day: str = ""
    day_start_bankroll: float = 0.0
    day_realized: float = 0.0
    cooldowns: Dict[str, float] = field(default_factory=dict)   # asset -> no new orders until ts
    path: Optional[str] = None
    reserved: float = 0.0          # held by in-flight orders; never persisted (no orders survive a restart)

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = self.start_cash

    # ------------------------------------------------------------------ views
    @property
    def open_at_risk(self) -> float:
        return sum(p.at_risk for p in self.positions.values())

    @property
    def bankroll(self) -> float:
        return self.cash + self.open_at_risk

    def provisional_losses(self) -> float:
        return sum(p.at_risk for p in self.positions.values() if p.provisional_loss)

    def day_pnl(self) -> float:
        """Realized today plus losses our own data already shows (conservative: no provisional wins)."""
        return self.day_realized - self.provisional_losses()

    def fills_on(self, slug: str, side: str) -> int:
        p = self.positions.get(f"{slug}:{side}")
        return p.fills if p else 0

    def holds(self, slug: str, side: str) -> bool:
        return f"{slug}:{side}" in self.positions

    # ------------------------------------------------------------------ mutations
    def roll_day(self, now: float) -> bool:
        d = utc_day(now)
        if d == self.day:
            return False
        self.day = d
        self.day_start_bankroll = self.bankroll
        self.day_realized = 0.0
        self.save()
        return True

    def apply_fill(self, w: MarketWindow, side: str, token: str, shares: float, notional: float, fee: float,
                   maker_shares: float, taker_shares: float, now: float, count_fill: bool = True,
                   ref_price: float = 0.0) -> Position:
        if shares <= 0:
            raise ValueError("fill with no shares")
        key = f"{w.slug}:{side}"
        pos = self.positions.get(key)
        if pos is None:
            pos = Position(slug=w.slug, asset=w.asset, label=w.label, side=side, token=token, window=w.to_dict(),
                           opened_ts=now, end_ts=w.end_ts, ref_price=ref_price)
            self.positions[key] = pos
        pos.shares += shares
        pos.cost += notional
        pos.fees += fee
        pos.maker_shares += maker_shares
        pos.taker_shares += taker_shares
        if count_fill:
            pos.fills += 1
        self.cash -= notional + fee
        self.save()
        return pos

    def mark_provisional_loss(self, slug: str, side: str) -> Optional[Position]:
        pos = self.positions.get(f"{slug}:{side}")
        if pos is not None and not pos.provisional_loss:
            pos.provisional_loss = True
            self.save()
            return pos
        return None

    def settle(self, slug: str, winner: str, now: float) -> List[Settlement]:
        out = []
        for key in [k for k, p in self.positions.items() if p.slug == slug]:
            pos = self.positions.pop(key)
            payout = pos.shares if pos.side == winner else 0.0
            pnl = payout - pos.at_risk
            self.cash += payout
            self.realized += pnl
            self.roll_day(now)
            self.day_realized += pnl
            if pnl >= 0:
                self.wins += 1
            else:
                self.losses += 1
            out.append(Settlement(pos, winner, payout, pnl, now))
        if out:
            self.save()
        return out

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"version": 1, "start_cash": self.start_cash, "cash": self.cash, "realized": self.realized,
                "wins": self.wins, "losses": self.losses, "day": self.day,
                "day_start_bankroll": self.day_start_bankroll, "day_realized": self.day_realized,
                "cooldowns": self.cooldowns, "positions": [asdict(p) for p in self.positions.values()]}

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, self.path)

    @classmethod
    def load_or_new(cls, path: Optional[str], start_cash: float) -> "Portfolio":
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            pf = cls(start_cash=d["start_cash"], cash=d["cash"], realized=d["realized"], wins=d["wins"],
                     losses=d["losses"], day=d["day"], day_start_bankroll=d["day_start_bankroll"],
                     day_realized=d["day_realized"], cooldowns=dict(d.get("cooldowns", {})), path=path)
            for pd in d.get("positions", []):
                p = Position(**pd)
                pf.positions[p.key] = p
            log.info("paper state loaded from %s: cash %.2f, %d open position(s), realized %+.2f",
                     path, pf.cash, len(pf.positions), pf.realized)
            return pf
        pf = cls(start_cash=start_cash, path=path)
        pf.save()
        return pf

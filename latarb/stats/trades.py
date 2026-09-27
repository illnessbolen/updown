"""Trades as the statistics module sees them, from any of the bot's logs.

    paper    settlements.csv — realized positions (actual simulated fills, fees, P&L);
             joined with orders.csv (model probability at send, maker/taker) and
             signals.csv (time left at the signal) by slug/side and signal_id.
    shadow   signals.csv SIGNAL rows x outcomes.csv — the first SIGNAL per window/side
             as a hypothetical 1-share taker fill at the logged ask + fee + slippage.

A trade is a binary bet: `shares` contracts bought for `cost` (all-in, fees
included) that pay `shares` if `won`. Its break-even probability is cost/shares:
under "no edge" the win probability equals exactly that, which is what the
hypothesis tests use.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Trade:
    ts: float                 # settlement time (paper) / window end (shadow)
    entry_ts: float
    slug: str
    asset: str
    label: str
    side: str
    shares: float
    cost: float               # money at risk incl. fees (and expected slippage for shadow)
    pnl: float
    won: bool
    p_model: Optional[float]  # model's (conservative) win probability at entry
    price: float              # average entry price, fees excluded
    tau_s: Optional[float]    # seconds left in the window at the signal
    kind: str                 # maker | taker | mixed | shadow
    source: str               # paper | shadow

    @property
    def breakeven(self) -> float:
        return self.cost / self.shares

    @property
    def ret(self) -> float:
        return self.pnl / self.cost if self.cost > 0 else 0.0


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _outcomes(data_dir: str) -> Dict[str, str]:
    return {r["slug"]: r["winner"] for r in _rows(os.path.join(data_dir, "outcomes.csv"))
            if r.get("winner") in ("up", "down")}


def load_paper_trades(data_dir: str) -> List[Trade]:
    signals = {r["signal_id"]: r for r in _rows(os.path.join(data_dir, "signals.csv"))}
    fills: Dict[tuple, List[dict]] = {}
    for o in _rows(os.path.join(data_dir, "orders.csv")):
        if (_num(o.get("shares_filled")) or 0.0) > 0:
            fills.setdefault((o["slug"], o["side"]), []).append(o)
    out = []
    for r in _rows(os.path.join(data_dir, "settlements.csv")):
        shares = _num(r["shares"]) or 0.0
        cost = (_num(r["cost"]) or 0.0) + (_num(r["fees"]) or 0.0)
        if shares <= 0 or cost <= 0:
            continue
        os_ = fills.get((r["slug"], r["side"]), [])
        filled = sum(_num(o["shares_filled"]) or 0.0 for o in os_)
        p_model = None
        tau = None
        if filled > 0:
            ps = [(_num(o["p_fair_cons_at_send"]), _num(o["shares_filled"]) or 0.0) for o in os_]
            if all(p is not None for p, _ in ps):
                p_model = sum(p * q for p, q in ps) / filled
            sig = signals.get(os_[0].get("signal_id", ""))
            tau = _num(sig.get("tau_s")) if sig else None
        maker = _num(r.get("maker_shares")) or 0.0
        kind = "maker" if maker >= shares - 1e-9 else ("taker" if maker <= 1e-9 else "mixed")
        out.append(Trade(ts=_num(r["ts"]) or 0.0, entry_ts=_num(r.get("opened_ts")) or 0.0, slug=r["slug"],
                         asset=r["asset"], label=r["label"], side=r["side"], shares=shares, cost=cost,
                         pnl=_num(r["pnl"]) or 0.0, won=r.get("won") == "1", p_model=p_model,
                         price=_num(r.get("avg_price")) or 0.0, tau_s=tau, kind=kind, source="paper"))
    out.sort(key=lambda t: t.ts)
    return out


def load_shadow_trades(data_dir: str) -> List[Trade]:
    outcomes = _outcomes(data_dir)
    seen = set()
    out = []
    for r in _rows(os.path.join(data_dir, "signals.csv")):
        if r.get("decision") != "SIGNAL":
            continue
        key = (r["slug"], r["side"])
        if key in seen:
            continue
        seen.add(key)
        winner = outcomes.get(r["slug"])
        cost = _num(r.get("cost"))
        if winner is None or cost is None or not 0 < cost < 1:
            continue
        won = r["side"] == winner
        out.append(Trade(ts=_num(r.get("window_end")) or _num(r["ts"]) or 0.0, entry_ts=_num(r["ts"]) or 0.0,
                         slug=r["slug"], asset=r["asset"], label=r["label"], side=r["side"], shares=1.0, cost=cost,
                         pnl=(1.0 if won else 0.0) - cost, won=won, p_model=_num(r.get("p_fair_cons")),
                         price=_num(r.get("p_market")) or 0.0, tau_s=_num(r.get("tau_s")), kind="shadow",
                         source="shadow"))
    out.sort(key=lambda t: t.ts)
    return out


def load_trades(data_dir: str, source: str = "auto") -> List[Trade]:
    """source: paper | shadow | auto (paper if any settlement exists, else shadow)."""
    if source == "paper":
        return load_paper_trades(data_dir)
    if source == "shadow":
        return load_shadow_trades(data_dir)
    paper = load_paper_trades(data_dir)
    return paper if paper else load_shadow_trades(data_dir)


def filter_period(trades: Iterable[Trade], since: Optional[float] = None,
                  until: Optional[float] = None) -> List[Trade]:
    return [t for t in trades if (since is None or t.ts >= since) and (until is None or t.ts < until)]


def paper_start_equity(data_dir: str) -> Optional[float]:
    import json
    path = os.path.join(data_dir, "paper_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return float(json.load(f)["start_cash"])
    except (OSError, ValueError, KeyError):
        return None

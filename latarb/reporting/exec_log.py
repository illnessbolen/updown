"""Execution logs of paper mode.

    orders.csv       one row per execution attempt: sent orders with their fills, and attempts
                     dropped before sending (reason: latency_budget_exceeded, stale_book,
                     signal_gone_after_refresh, ...). Timings: trigger -> send latency, book
                     refresh latency and age of the price used. Joined to signals.csv by signal_id.
    settlements.csv  one row per resolved position with the realized P&L.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from .signal_log import CsvAppender, _f

ORDER_FIELDS = [
    "ts_iso", "ts", "attempt_id", "order_id", "signal_id", "slug", "asset", "label", "side", "plan", "kind",
    "status", "reason", "detail", "shares_req", "shares_filled", "maker_shares", "taker_shares", "limit",
    "avg_price", "notional", "fee", "ask_at_signal", "ask_at_send", "p_fair_cons_at_signal",
    "p_fair_cons_at_send", "edge_cons_at_send", "kelly_full", "kelly_stake", "cap", "sized_by", "reserve",
    "t_trigger", "t_send", "latency_ms", "budget_ms", "refresh_source", "refresh_ms", "book_age_ms",
    "bankroll", "cash_after", "open_after",
]
SETTLEMENT_FIELDS = [
    "ts_iso", "ts", "slug", "asset", "label", "side", "shares", "avg_price", "cost", "fees", "maker_shares",
    "taker_shares", "fills", "ref_price", "winner", "won", "payout", "pnl", "provisional_loss", "opened_ts",
    "end_ts", "cash_after", "realized_total", "day_pnl",
]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


class ExecutionLog:
    def __init__(self, directory: str, clock, budget_ms: float) -> None:
        os.makedirs(directory, exist_ok=True)
        self.clock = clock
        self.budget_ms = budget_ms
        self.orders = CsvAppender(os.path.join(directory, "orders.csv"), ORDER_FIELDS)
        self.settlements = CsvAppender(os.path.join(directory, "settlements.csv"), SETTLEMENT_FIELDS)

    def order(self, att, o: Optional[object], status: str, pf, reason: str = "", detail: str = "") -> None:
        now = self.clock.now()
        w, sz, ev = att.window, att.size, att.ev_send
        fb = att.fresh
        row = {
            "ts_iso": _iso(now), "ts": round(now, 3), "attempt_id": att.attempt_id,
            "order_id": getattr(o, "order_id", ""), "signal_id": att.signal_id, "slug": w.slug,
            "asset": w.asset, "label": w.label, "side": att.side, "plan": att.plan, "kind": att.kind,
            "status": status, "reason": reason or getattr(o, "end_reason", ""), "detail": detail,
            "shares_req": sz.shares if sz else "", "shares_filled": _f(getattr(o, "filled", 0.0), 4),
            "maker_shares": _f(getattr(o, "maker_filled", 0.0), 4),
            "taker_shares": _f(getattr(o, "taker_filled", 0.0), 4),
            "limit": _f(att.limit) if att.limit else "", "avg_price": _f(o.avg_price) if o is not None else "",
            "notional": _f(getattr(o, "notional", 0.0)), "fee": _f(getattr(o, "fee", 0.0)),
            "ask_at_signal": _f(att.ask_at_signal), "ask_at_send": _f(ev.best.ask) if ev is not None else "",
            "p_fair_cons_at_signal": _f(att.p_cons_at_signal),
            "p_fair_cons_at_send": _f(ev.best.p_fair_cons) if ev is not None else "",
            "edge_cons_at_send": _f(ev.best.edge_cons) if ev is not None else "",
            "kelly_full": _f(sz.kelly_full) if sz else "", "kelly_stake": _f(sz.kelly_stake, 4) if sz else "",
            "cap": _f(sz.cap, 4) if sz else "", "sized_by": sz.sized_by if sz else "",
            "reserve": _f(sz.reserve, 4) if sz else "", "t_trigger": round(att.t_trigger, 4),
            "t_send": round(att.t_send, 4) if att.t_send else "",
            "latency_ms": _f(att.latency_ms, 2), "budget_ms": self.budget_ms,
            "refresh_source": fb.source if fb else "", "refresh_ms": _f(fb.latency_ms, 2) if fb else "",
            "book_age_ms": _f(att.book_age_ms, 2), "bankroll": _f(pf.bankroll, 4), "cash_after": _f(pf.cash, 4),
            "open_after": _f(pf.open_at_risk, 4),
        }
        self.orders.write(row)
        self.orders.flush()

    def settlement(self, st, pf) -> None:
        p = st.position
        self.settlements.write({
            "ts_iso": _iso(st.ts), "ts": round(st.ts, 3), "slug": p.slug, "asset": p.asset, "label": p.label,
            "side": p.side, "shares": _f(p.shares, 4), "avg_price": _f(p.avg_price), "cost": _f(p.cost, 4),
            "fees": _f(p.fees, 4), "maker_shares": _f(p.maker_shares, 4), "taker_shares": _f(p.taker_shares, 4),
            "fills": p.fills, "ref_price": _f(p.ref_price, 10), "winner": st.winner, "won": int(st.won),
            "payout": _f(st.payout, 4), "pnl": _f(st.pnl, 4), "provisional_loss": int(p.provisional_loss),
            "opened_ts": round(p.opened_ts, 3), "end_ts": round(p.end_ts, 3), "cash_after": _f(pf.cash, 4),
            "realized_total": _f(pf.realized, 4), "day_pnl": _f(pf.day_pnl(), 4),
        })
        self.settlements.flush()

    def flush(self) -> None:
        self.orders.flush()
        self.settlements.flush()

    def close(self) -> None:
        self.orders.close()
        self.settlements.close()

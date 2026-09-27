"""Execution pipeline (paper): signal -> risk -> size -> fresh book -> re-check -> budget -> order.

    on_signal (sync, from the engine, on every SIGNAL evaluation)
      1. one attempt per window/side at a time, EXEC_RETRY_S between attempts
      2. RiskManager.pre_trade (kill switch, breaker, daily stop, cooldown, fills, exposure)
      3. limit price: the highest tick price where P_cons - price - fee >= MIN_EDGE_AT_FILL,
         and never above signal ask + MAX_SLIPPAGE
      4. fractional-Kelly size inside the caps; the worst-case cost is reserved against
         the exposure cap until the order ends
      5. pre-trade book refresh (REST GET /book live, simulated delay in replay)
    after the refresh
      6. the model is re-evaluated on the FRESH book; the signal must still be a SIGNAL on the
         same side, the price must be younger than BOOK_MAX_AGE_MS, and size is recomputed
      7. latency budget: frame receipt -> order hand-off must be <= LATENCY_BUDGET_MS,
         otherwise the signal is DROPPED (never sent late at a stale price); actual ms logged
      8. maker_first: GTC bid just under the ask (<= P_cons - MIN_EDGE_AT_FILL, fee 0);
         taker: FAK at the limit
    when the order ends
      9. actual fills are booked; a maker order that timed out may fall back to a taker
         order ONLY through a brand-new signal evaluation (with its own budget)

Every attempt, sent or dropped, is one row in orders.csv with its reason and timings.
"""
from __future__ import annotations

import itertools
import logging
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from ..clock import Clock
from ..config import Settings
from ..data.hub import MarketDataHub
from ..data.markets import BINANCE, CHAINLINK, MarketWindow
from ..model.fees import taker_fee_per_share
from ..risk.limits import RiskLimits
from ..risk.manager import RiskManager
from ..risk.portfolio import Portfolio
from ..risk.sizing import SizeDecision, size_order
from ..signal.engine import SIGNAL, Evaluation
from .books import FreshBooks
from .paper import PaperExchange, PaperOrder

log = logging.getLogger("latarb.exec")

PROVISIONAL_MARGIN = 2e-4          # our end price must be > 2 bp from K to call a provisional loss


def floor_tick(p: float, tick: float) -> float:
    return round(math.floor(p / tick + 1e-9) * tick, 6)


def max_taker_price(p_cons: float, min_edge: float, rate: float, exp: float, tick: float,
                    hi: float) -> Optional[float]:
    """Highest tick price <= hi with p_cons - p - fee(p) >= min_edge (p + fee(p) is increasing)."""
    p = floor_tick(min(hi, 1.0 - tick), tick)
    while p >= tick - 1e-12:
        if p_cons - p - taker_fee_per_share(p, rate, exp) >= min_edge - 1e-12:
            return p
        p = round(p - tick, 6)
    return None


@dataclass
class Plan:
    taker_limit: float
    size: SizeDecision
    tick: float
    rate: float
    exp: float


@dataclass
class Attempt:
    attempt_id: str
    signal_id: str
    window: MarketWindow
    side: str
    plan: str                       # maker_first | taker | taker_fallback
    t_trigger: float
    ask_at_signal: float
    p_cons_at_signal: float
    ref_price: float
    counts_fill: bool = True
    reserve: float = 0.0
    size: Optional[SizeDecision] = None
    fresh: Optional[FreshBooks] = None
    ev_send: Optional[Evaluation] = None
    t_send: Optional[float] = None
    latency_ms: Optional[float] = None
    book_age_ms: Optional[float] = None
    kind: str = ""
    limit: float = 0.0
    done: bool = False


class ExecutionPipeline:
    def __init__(self, cfg: Settings, hub: MarketDataHub, engine, risk: RiskManager, portfolio: Portfolio,
                 limits: RiskLimits, exchange: PaperExchange, refresher, clock: Clock, sink) -> None:
        self.cfg = cfg
        self.hub = hub
        self.engine = engine
        self.risk = risk
        self.pf = portfolio
        self.limits = limits
        self.exchange = exchange
        self.refresher = refresher
        self.clock = clock
        self.sink = sink
        self.active: Dict[Tuple[str, str], Attempt] = {}
        self.next_try: Dict[Tuple[str, str], float] = {}
        self.stats: Counter = Counter()
        self.latencies: Deque[float] = deque(maxlen=5000)
        self._ids = itertools.count(1)
        self._provisional_checked: Dict[str, float] = {}
        engine.executor = self

    # ================================================================== planning
    def _tick(self, w: MarketWindow, side: str) -> float:
        token = w.up_token if side == "up" else w.down_token
        b = self.hub.books.get(token)
        return b.tick_size if b is not None and b.tick_size > 0 else w.tick_size

    def _plan(self, ev: Evaluation, room_extra: float) -> Tuple[Optional[Plan], str]:
        w, b = ev.window, ev.best
        tick = self._tick(w, b.side)
        rate = w.taker_fee_rate if w.taker_fee_rate is not None else self.cfg.TAKER_FEE_RATE
        exp = w.fee_exponent if w.fee_exponent is not None else self.cfg.FEE_EXPONENT
        limit = max_taker_price(b.p_fair_cons, self.cfg.MIN_EDGE_AT_FILL, rate, exp, tick,
                                b.ask + self.cfg.MAX_SLIPPAGE)
        if limit is None or limit < b.ask - 1e-9:
            return None, "no_price_with_min_edge"
        unit_cash = limit + taker_fee_per_share(limit, rate, exp)
        size, why = size_order(b.p_fair_cons, b.cost, unit_cash, self.pf.bankroll, self.limits,
                               self.risk.exposure_room() + room_extra, w.min_order_size, self.cfg.MIN_ORDER_USDC)
        if size is None:
            return None, why
        return Plan(limit, size, tick, rate, exp), ""

    def _order_type(self, plan_name: str, ev: Evaluation, plan: Plan) -> Tuple[str, float]:
        if plan_name == "maker_first":
            b = ev.best
            hi = min(b.ask - plan.tick, b.p_fair_cons - max(self.cfg.MIN_EDGE_AT_FILL, self.cfg.MAKER_CANCEL_EDGE))
            bid = floor_tick(hi, plan.tick)
            if bid >= plan.tick:
                return "maker", bid
        return "taker", plan.taker_limit

    # ================================================================== engine hooks
    def on_signal(self, ev: Evaluation, plan: Optional[str] = None) -> str:
        w, b = ev.window, ev.best
        key = (w.slug, b.side)
        now = self.clock.now()
        fallback = plan == "taker_fallback"
        if key in self.active:
            return "in_flight"
        if not self.cfg.ALLOW_BOTH_SIDES and (w.slug, "down" if b.side == "up" else "up") in self.active:
            return "opposite_in_flight"
        if not fallback and self.next_try.get(key, 0.0) > now:
            return "retry_wait"
        why = self.risk.pre_trade(w, b.side, fallback=fallback)
        if why:
            self.stats["refused:" + why] += 1
            return why
        p, why = self._plan(ev, room_extra=0.0)
        if p is None:
            self.stats["refused:" + why] += 1
            return why
        att = Attempt(attempt_id=f"A{next(self._ids)}", signal_id=ev.signal_id, window=w, side=b.side,
                      plan=plan or self.cfg.EXECUTION_STYLE,
                      t_trigger=ev.trigger_ts if ev.trigger_ts is not None else ev.ts,
                      ask_at_signal=b.ask, p_cons_at_signal=b.p_fair_cons, ref_price=ev.reference,
                      counts_fill=not (fallback and self.pf.fills_on(w.slug, b.side) > 0))
        att.size = p.size
        att.reserve = p.size.reserve
        self.pf.reserved += att.reserve
        self.active[key] = att
        self.stats["queued"] += 1
        self.refresher.refresh(w, lambda fb, err: self._guard(att, self._after_refresh, fb, err))
        return "queued"

    def on_evaluation(self, ev: Evaluation) -> None:
        """Resting maker bids are withdrawn as soon as the model or a gate says so."""
        for o in self.exchange.resting_for(ev.window.slug):
            if ev.reasons:
                self.exchange.cancel(o.order_id, "gated:" + ev.reasons[0])
                continue
            p_cons = ev.p_up_lo if o.side == "up" else 1.0 - ev.p_up_hi
            if p_cons - o.limit < self.cfg.MAKER_CANCEL_EDGE:
                self.exchange.cancel(o.order_id, "maker_edge_lost")

    def on_timer(self, now: float) -> None:
        halt = self.risk.halted()
        if halt and self.exchange.resting:
            self.exchange.cancel_all("halt:" + halt)
        self._check_provisional(now)

    def on_outcome(self, slug: str, winner: str, now: float) -> None:
        for st in self.pf.settle(slug, winner, now):
            self.risk.on_settlement(st)
            self.sink.settlement(st, self.pf)
            p = st.position
            log.info("SETTLED %s %s %s %s: %.0f sh avg %.3f -> %s %+.2f | realized %+.2f | cash %.2f | W/L %d/%d",
                     p.asset.upper(), p.label, p.slug, p.side.upper(), p.shares, p.avg_price,
                     "WIN" if st.won else "LOSS", st.pnl, self.pf.realized, self.pf.cash,
                     self.pf.wins, self.pf.losses)

    # ================================================================== steps
    def _guard(self, att: Attempt, fn, *args) -> None:
        try:
            fn(att, *args)
        except Exception as e:  # noqa: BLE001 - never leak a reservation or an active slot
            log.exception("execution step failed for %s", att.window.slug)
            self.risk.record_error(f"{type(e).__name__}: {e}")
            if not att.done:
                self._drop(att, "internal_error", str(e))

    def _after_refresh(self, att: Attempt, fb: Optional[FreshBooks], err: Optional[str]) -> None:
        w = att.window
        now = self.clock.now()
        if fb is None:
            if getattr(self.refresher, "counts_as_error", False):
                self.risk.record_error(f"book refresh {w.slug}: {err}")
            return self._drop(att, "refresh_failed", err or "")
        att.fresh = fb
        halt = self.risk.halted()
        if halt:
            return self._drop(att, halt)
        ev = self.engine.evaluate(w, None, books=(fb.up, fb.down))
        if ev is None:
            return self._drop(att, "unpriced_after_refresh")
        if ev.decision != SIGNAL or ev.best.side != att.side:
            detail = ",".join(ev.reasons) or (f"edge_cons={ev.best.edge_cons:+.4f}" if ev.best else "no_ask")
            return self._drop(att, "signal_gone_after_refresh", detail)
        b = ev.best
        price_token = (w.up_token if b.side == "up" else w.down_token) if b.ask_src == "book" \
            else (w.down_token if b.side == "up" else w.up_token)
        att.book_age_ms = fb.age_ms(price_token, now)
        if att.book_age_ms > self.cfg.BOOK_MAX_AGE_MS:
            return self._drop(att, "stale_book", f"{att.book_age_ms:.0f}ms")
        p, why = self._plan(ev, room_extra=att.reserve)
        if p is None:
            return self._drop(att, why)
        t_send = now + self.cfg.PAPER_SUBMIT_OVERHEAD_MS / 1000.0
        att.latency_ms = (t_send - att.t_trigger) * 1000.0
        att.ev_send = ev
        if att.latency_ms > self.cfg.LATENCY_BUDGET_MS:
            return self._drop(att, "latency_budget_exceeded",
                              f"{att.latency_ms:.0f}ms > {self.cfg.LATENCY_BUDGET_MS:.0f}ms")
        att.kind, att.limit = self._order_type(att.plan, ev, p)
        self.pf.reserved += p.size.reserve - att.reserve
        att.reserve = p.size.reserve
        att.size = p.size
        att.t_send = t_send
        self.latencies.append(att.latency_ms)
        other = w.down_token if att.side == "up" else w.up_token
        token = w.up_token if att.side == "up" else w.down_token
        order = PaperOrder(window=w, side=att.side, token=token, other_token=other, kind=att.kind, limit=att.limit,
                           shares=p.size.shares, t_send=t_send, fee_rate=p.rate, fee_exp=p.exp,
                           on_done=lambda o: self._guard(att, self._on_done, o))
        self.exchange.submit(order)
        self.stats["sent:" + att.kind] += 1
        log.info("PAPER SEND %s %s %s %s %s %d @ %.3f | ask %.3f P_cons %.4f | kelly %.4f -> %s | "
                 "latency %.0fms (budget %.0f, refresh %s %.0fms) | reserve %.2f",
                 att.attempt_id, w.asset.upper(), w.label, att.side.upper(), att.kind, p.size.shares, att.limit,
                 b.ask, b.p_fair_cons, p.size.kelly_full, p.size.sized_by, att.latency_ms,
                 self.cfg.LATENCY_BUDGET_MS, fb.source, fb.latency_ms, att.reserve)

    def _on_done(self, att: Attempt, o: PaperOrder) -> None:
        w = att.window
        now = self.clock.now()
        self._release(att)
        if o.filled > 0:
            self.pf.apply_fill(w, att.side, o.token, o.filled, o.notional, o.fee, o.maker_filled, o.taker_filled,
                               now, count_fill=att.counts_fill, ref_price=att.ref_price)
            self.stats["filled:" + o.kind] += 1
        else:
            self.stats["unfilled:" + o.end_reason] += 1
        self.risk.record_ok()
        status = "filled" if o.remaining <= 1e-9 else ("partial" if o.filled > 0 else "unfilled")
        self.sink.order(att, o, status, self.pf)
        log.info("PAPER %s %s %s %s %s: %.0f/%d sh avg %.3f fee %.3f (maker %.0f taker %.0f) %s | cash %.2f "
                 "exposure %.2f", status.upper(), att.attempt_id, w.asset.upper(), att.side.upper(), o.kind,
                 o.filled, int(o.shares), o.avg_price, o.fee, o.maker_filled, o.taker_filled, o.end_reason,
                 self.pf.cash, self.pf.open_at_risk)
        self._finish(att)
        if (o.kind == "maker" and o.end_reason == "maker_timeout" and o.remaining > 0
                and self.cfg.MAKER_FALLBACK_TAKER):
            status = self.engine.execute_fresh(w, att.side, plan="taker_fallback")
            self.stats["fallback:" + status] += 1

    # ================================================================== bookkeeping
    def _release(self, att: Attempt) -> None:
        self.pf.reserved = max(0.0, self.pf.reserved - att.reserve)
        att.reserve = 0.0

    def _finish(self, att: Attempt) -> None:
        att.done = True
        key = (att.window.slug, att.side)
        if self.active.get(key) is att:
            del self.active[key]
        self.next_try[key] = self.clock.now() + self.cfg.EXEC_RETRY_S

    def _drop(self, att: Attempt, reason: str, detail: str = "") -> None:
        self._release(att)
        self.stats["drop:" + reason] += 1
        self.sink.order(att, None, "dropped", self.pf, reason=reason, detail=detail)
        log.info("PAPER DROP %s %s %s %s: %s%s", att.attempt_id, att.window.asset.upper(), att.window.label,
                 att.side.upper(), reason, f" ({detail})" if detail else "")
        self._finish(att)

    # ================================================================== provisional outcome
    def _derived_winner(self, w: MarketWindow, ref: float) -> Optional[str]:
        st = self.hub.assets.get(w.asset)
        if st is None or ref <= 0:
            return None
        if w.resolution == CHAINLINK:
            hit = st.oracle_hist.nearest(w.end_ts, self.cfg.ORACLE_REF_TOLERANCE_S)
        elif w.resolution == BINANCE:
            hit = st.fast_hist.at(w.end_ts)
            if hit is not None and w.end_ts - hit[0] > self.cfg.SPOT_STALE_S:
                hit = None
        else:
            return None
        if hit is None:
            return None
        x = math.log(hit[1] / ref)
        if abs(x) < PROVISIONAL_MARGIN:
            return None
        return "up" if x > 0 else "down"

    def _check_provisional(self, now: float) -> None:
        for pos in list(self.pf.positions.values()):
            if pos.provisional_loss or now < pos.end_ts + 2.0:
                continue
            key = pos.key
            if self._provisional_checked.get(key, 0.0) > now:
                continue
            self._provisional_checked[key] = now + 5.0
            if now > pos.end_ts + 120.0:
                continue                 # no usable end price: wait for Gamma
            w = MarketWindow.from_dict(pos.window)
            winner = self._derived_winner(w, pos.ref_price)
            if winner is not None and winner != pos.side:
                self.pf.mark_provisional_loss(pos.slug, pos.side)
                self.risk.on_provisional_loss(pos.asset, pos.slug)
                log.info("provisional LOSS %s %s (our end-of-window data); counted toward the daily stop "
                         "until Gamma resolves", pos.slug, pos.side.upper())

    # ================================================================== status
    def status_line(self) -> str:
        lat = sorted(self.latencies)
        q = (lambda f: lat[min(len(lat) - 1, int(f * len(lat)))]) if lat else None
        max_bet, max_exp, stop = self.risk.caps()
        drops = ", ".join(f"{k[5:]}={v}" for k, v in self.stats.most_common() if k.startswith("drop:"))
        refused = ", ".join(f"{k[8:]}={v}" for k, v in self.stats.most_common(6) if k.startswith("refused:"))
        return (f"paper: cash {self.pf.cash:.2f} bankroll {self.pf.bankroll:.2f} | open {self.pf.open_at_risk:.2f}"
                f"/{max_exp:.2f} reserved {self.pf.reserved:.2f} | realized {self.pf.realized:+.2f} "
                f"day {self.pf.day_pnl():+.2f}/-{stop:.2f} | W/L {self.pf.wins}/{self.pf.losses} | "
                f"positions {len(self.pf.positions)} resting {len(self.exchange.resting)} | "
                f"sent m/t {self.stats['sent:maker']}/{self.stats['sent:taker']} "
                f"filled m/t {self.stats['filled:maker']}/{self.stats['filled:taker']} | "
                f"latency p50/p95 {q(0.5) if q else float('nan'):.0f}/{q(0.95) if q else float('nan'):.0f}ms | "
                f"drops: {drops or '-'} | refused: {refused or '-'} | halt: {self.risk.halted() or '-'}")

"""Pre-trade risk checks. Every order must pass `pre_trade`; nothing bypasses it.

Halts (no new orders at all, resting orders are cancelled by the pipeline):
    kill_switch       the KILL_SWITCH_FILE exists (create it to stop trading instantly)
    circuit_breaker   MAX_CONSECUTIVE_ERRORS execution errors in a row; cleared only by a restart
    daily_stop        today's P&L <= -daily_stop_pct * bankroll at the start of the UTC day,
                      where today's P&L = realized settlements + losses our own end-of-window
                      data already shows (provisional wins are NOT counted)

Per-order refusals:
    cooldown          the asset lost recently (COOLDOWN_AFTER_LOSS_S, persisted across restarts)
    max_fills         fills on this window/side reached the profile limit
    opposite_side     already holding the other outcome of this window (ALLOW_BOTH_SIDES=false)
    exposure_limit    open + reserved exposure leaves no room for a minimum order
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from ..clock import Clock
from ..config import Settings
from ..data.markets import MarketWindow
from .limits import RiskLimits
from .portfolio import Portfolio, Settlement

log = logging.getLogger("latarb.risk")


class RiskManager:
    def __init__(self, cfg: Settings, limits: RiskLimits, portfolio: Portfolio, clock: Clock) -> None:
        self.cfg = cfg
        self.limits = limits
        self.pf = portfolio
        self.clock = clock
        self.errors_in_row = 0
        self.breaker_tripped = False
        self._kill_checked = (-1.0, False)
        self._last_halt: Optional[str] = None

    # ------------------------------------------------------------------ limits in USDC
    def caps(self) -> Tuple[float, float, float]:
        """(max bet, max exposure, daily stop) in USDC for the current bankroll."""
        b = self.pf.bankroll
        return b * self.limits.bet_pct, b * self.limits.exposure_pct, \
            self.pf.day_start_bankroll * self.limits.daily_stop_pct

    def exposure_room(self) -> float:
        return self.pf.bankroll * self.limits.exposure_pct - self.pf.open_at_risk - self.pf.reserved

    # ------------------------------------------------------------------ halts
    def kill_switch_active(self) -> bool:
        now = self.clock.now()
        ts, val = self._kill_checked
        if now - ts >= 0.5 or now < ts:
            val = os.path.exists(self.cfg.KILL_SWITCH_FILE)
            self._kill_checked = (now, val)
        return val

    def daily_stop_hit(self) -> bool:
        self.pf.roll_day(self.clock.now())
        limit = self.pf.day_start_bankroll * self.limits.daily_stop_pct
        return self.pf.day_pnl() <= -limit

    def halted(self) -> Optional[str]:
        reason = None
        if self.kill_switch_active():
            reason = "kill_switch"
        elif self.breaker_tripped:
            reason = "circuit_breaker"
        elif self.daily_stop_hit():
            reason = "daily_stop"
        if reason != self._last_halt:
            if reason == "kill_switch":
                log.critical("KILL SWITCH: file %r exists - no new orders, resting orders cancelled. "
                             "Delete the file to resume.", self.cfg.KILL_SWITCH_FILE)
            elif reason == "daily_stop":
                log.critical("DAILY STOP: day P&L %.2f <= -%.2f (%.1f%% of day-start bankroll %.2f) - "
                             "no new orders until the next UTC day.", self.pf.day_pnl(),
                             self.pf.day_start_bankroll * self.limits.daily_stop_pct,
                             self.limits.daily_stop_pct * 100, self.pf.day_start_bankroll)
            elif reason is None and self._last_halt is not None:
                log.warning("trading resumed (was halted: %s)", self._last_halt)
            self._last_halt = reason
        return reason

    # ------------------------------------------------------------------ per order
    def pre_trade(self, w: MarketWindow, side: str, fallback: bool = False) -> Optional[str]:
        """fallback=True: the taker leg of a maker order that timed out; it belongs to the same
        attempt, so it does not consume another fill slot (every other check still applies)."""
        now = self.clock.now()
        halt = self.halted()
        if halt:
            return halt
        if self.pf.cooldowns.get(w.asset, 0.0) > now:
            return "cooldown"
        other = "down" if side == "up" else "up"
        if not self.cfg.ALLOW_BOTH_SIDES and self.pf.holds(w.slug, other):
            return "opposite_side"
        if not fallback and self.pf.fills_on(w.slug, side) >= self.limits.max_fills_per_side:
            return "max_fills"
        if self.exposure_room() < self.cfg.MIN_ORDER_USDC:
            return "exposure_limit"
        return None

    # ------------------------------------------------------------------ feedback
    def record_error(self, what: str) -> None:
        self.errors_in_row += 1
        log.error("execution error %d/%d: %s", self.errors_in_row, self.cfg.MAX_CONSECUTIVE_ERRORS, what)
        if self.errors_in_row >= self.cfg.MAX_CONSECUTIVE_ERRORS and not self.breaker_tripped:
            self.breaker_tripped = True
            log.critical("CIRCUIT BREAKER: %d consecutive execution errors - trading halted until restart. "
                         "Check connectivity/API before restarting.", self.errors_in_row)

    def record_ok(self) -> None:
        self.errors_in_row = 0

    def _cooldown(self, asset: str, why: str) -> None:
        until = self.clock.now() + self.cfg.COOLDOWN_AFTER_LOSS_S
        if until > self.pf.cooldowns.get(asset, 0.0):
            self.pf.cooldowns[asset] = until
            self.pf.save()
            log.info("cooldown %s for %.0fs after %s", asset.upper(), self.cfg.COOLDOWN_AFTER_LOSS_S, why)

    def on_settlement(self, st: Settlement) -> None:
        if st.pnl < 0:
            self._cooldown(st.position.asset, f"loss {st.pnl:+.2f} on {st.position.slug}")

    def on_provisional_loss(self, asset: str, slug: str) -> None:
        self._cooldown(asset, f"provisional loss on {slug}")

"""Order size proportional to signal strength: fractional Kelly inside hard caps.

Buying one share at all-in cost c (price + fee + expected slippage) that pays 1
with probability p is a bet with net odds (1 - c) / c. Full Kelly for it is

    f* = (p - c) / (1 - c)            (fraction of bankroll)

so the stake grows with the edge p - c and is 0 when there is none. We use
p = P_fair on the conservative end of the sigma band (the model's own
uncertainty), and only a FRACTION of f* (profile kelly_fraction, <= 0.5) —
full Kelly on an estimated p is far too volatile.

Order of operations (Kelly never works on top of the caps, only inside them):
    1. kelly_stake = kelly_fraction * f* * bankroll
    2. cap         = min(bet_pct * bankroll, remaining exposure room)
    3. stake       = min(kelly_stake, cap)
    4. below the exchange minimum (orderMinSize shares, MIN_ORDER_USDC): raised
       to the minimum only if the minimum itself fits under the cap, else skip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .limits import RiskLimits


@dataclass(frozen=True)
class SizeDecision:
    shares: int
    unit_cash: float        # worst-case cash per share: limit price + taker fee at the limit
    reserve: float          # shares * unit_cash, held against the exposure cap until the order ends
    kelly_full: float       # f*
    kelly_stake: float      # kelly_fraction * f* * bankroll, before caps
    cap: float
    sized_by: str           # kelly | max_bet | exposure | exchange_min


def kelly_binary(p: float, cost: float) -> float:
    if not 0.0 < cost < 1.0:
        return 0.0
    return max(0.0, (p - cost) / (1.0 - cost))


def size_order(p_win: float, cost: float, unit_cash: float, bankroll: float, limits: RiskLimits,
               exposure_room: float, min_shares: float, min_usdc: float) -> Tuple[Optional[SizeDecision], str]:
    if bankroll <= 0 or unit_cash <= 0:
        return None, "no_bankroll"
    f = kelly_binary(p_win, cost)
    if f <= 0.0:
        return None, "no_kelly_edge"
    kelly_stake = limits.kelly_fraction * f * bankroll
    max_bet = limits.bet_pct * bankroll
    room = max(0.0, exposure_room)
    cap = min(max_bet, room)
    if kelly_stake <= cap:
        stake, sized_by = kelly_stake, "kelly"
    else:
        stake, sized_by = cap, ("max_bet" if max_bet <= room else "exposure")
    shares = int(math.floor(stake / unit_cash + 1e-9))
    min_eff = max(int(math.ceil(min_shares - 1e-9)), int(math.ceil(min_usdc / unit_cash - 1e-9)), 1)
    if shares < min_eff:
        if min_eff * unit_cash <= cap + 1e-9:
            shares, sized_by = min_eff, "exchange_min"
        else:
            return None, ("exposure_limit" if room < max_bet else "below_exchange_min")
    return SizeDecision(shares, unit_cash, shares * unit_cash, f, kelly_stake, cap, sized_by), ""

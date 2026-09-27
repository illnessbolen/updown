"""Polymarket taker fee model: fee per share = rate * (p * (1 - p)) ** exponent.

exponent = 1 is the schedule the previous bot used (crypto taker rate 0.07).
Per-market rate/exponent from Gamma `feeSchedule` override the defaults when
present (see data/markets.py). Makers pay 0.
"""
from __future__ import annotations


def taker_fee_per_share(price: float, rate: float, exponent: float = 1.0) -> float:
    if not 0.0 <= price <= 1.0:
        raise ValueError(f"price must be within [0, 1], got {price}")
    return rate * (price * (1.0 - price)) ** exponent

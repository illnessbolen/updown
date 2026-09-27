"""Risk profiles and the hard bounds nothing may cross.

Profiles set the per-trade cap, the open-exposure cap, the daily stop and the
Kelly fraction, all as FRACTIONS OF THE CURRENT BANKROLL (never fixed dollar
amounts), plus the number of fills allowed per window/side.

Every value — from a profile or from an RISK_* override — must lie inside
HARD_BOUNDS. The bounds are constants in code on purpose: they are not
settings, and a value outside them stops the bot at startup instead of being
clamped silently. Tightening is always allowed within the bounds; loosening
beyond them requires changing this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..config import ConfigError, Settings

# name -> (min, max), inclusive
HARD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "bet_pct": (0.001, 0.05),          # at most 5% of bankroll in a single order
    "exposure_pct": (0.01, 0.30),      # at most 30% of bankroll at risk at any time
    "daily_stop_pct": (0.005, 0.10),   # trading halts after losing 10% of the day-start bankroll, at most
    "kelly_fraction": (0.01, 0.50),    # never more than half Kelly (ТЗ: 25-50% of full Kelly at most)
    "max_fills_per_side": (1, 5),
}

PROFILES: Dict[str, Dict[str, float]] = {
    "conservative": {"bet_pct": 0.01, "exposure_pct": 0.10, "daily_stop_pct": 0.03,
                     "kelly_fraction": 0.10, "max_fills_per_side": 1},
    "moderate":     {"bet_pct": 0.02, "exposure_pct": 0.15, "daily_stop_pct": 0.05,
                     "kelly_fraction": 0.25, "max_fills_per_side": 2},
    "aggressive":   {"bet_pct": 0.04, "exposure_pct": 0.25, "daily_stop_pct": 0.08,
                     "kelly_fraction": 0.40, "max_fills_per_side": 3},
}

_OVERRIDES = {
    "bet_pct": "RISK_BET_PCT",
    "exposure_pct": "RISK_EXPOSURE_PCT",
    "daily_stop_pct": "RISK_DAILY_STOP_PCT",
    "kelly_fraction": "RISK_KELLY_FRACTION",
    "max_fills_per_side": "RISK_MAX_FILLS_PER_SIDE",
}


def _check(name: str, value: float, where: str) -> None:
    lo, hi = HARD_BOUNDS[name]
    if not lo <= value <= hi:
        raise ConfigError(f"{where}: {name}={value} is outside the hard bound [{lo}, {hi}]")


for _pname, _p in PROFILES.items():          # the shipped profiles must respect the bounds too
    for _k, _v in _p.items():
        _check(_k, _v, f"profile {_pname}")


@dataclass(frozen=True)
class RiskLimits:
    profile: str
    bet_pct: float
    exposure_pct: float
    daily_stop_pct: float
    kelly_fraction: float
    max_fills_per_side: int
    overridden: Tuple[str, ...] = ()

    def describe(self, bankroll: float) -> str:
        return (f"profile={self.profile}{' (overrides: ' + ','.join(self.overridden) + ')' if self.overridden else ''} | "
                f"max bet {self.bet_pct:.1%} = {bankroll * self.bet_pct:.2f} | "
                f"max exposure {self.exposure_pct:.1%} = {bankroll * self.exposure_pct:.2f} | "
                f"daily stop {self.daily_stop_pct:.1%} = {bankroll * self.daily_stop_pct:.2f} | "
                f"Kelly x{self.kelly_fraction:.2f} | fills/side {self.max_fills_per_side}")


def resolve_limits(cfg: Settings) -> RiskLimits:
    name = cfg.RISK_PROFILE.strip().lower()
    if name not in PROFILES:
        raise ConfigError(f"RISK_PROFILE={cfg.RISK_PROFILE!r}; expected one of {tuple(PROFILES)}")
    values = dict(PROFILES[name])
    overridden = []
    for key, env_name in _OVERRIDES.items():
        v = getattr(cfg, env_name)
        if v:                                  # 0 = not set
            _check(key, v, env_name)
            values[key] = v
            overridden.append(key)
    for k, v in values.items():
        _check(k, v, f"profile {name}")
    return RiskLimits(profile=name, bet_pct=float(values["bet_pct"]), exposure_pct=float(values["exposure_pct"]),
                      daily_stop_pct=float(values["daily_stop_pct"]),
                      kelly_fraction=float(values["kelly_fraction"]),
                      max_fills_per_side=int(values["max_fills_per_side"]), overridden=tuple(overridden))

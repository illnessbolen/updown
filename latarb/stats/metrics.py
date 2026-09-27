"""Performance statistics over realized (or hypothetical) trades.

Only facts about trades that actually settled are reported. Every number
comes with its sample size; below MIN_SAMPLE trades the output carries an
explicit warning that no statistically meaningful conclusion is possible.

Win rate is always printed next to the break-even win rate (all-in cost per
share): buying at 0.97 needs 97% wins to break even, so a high win rate alone
says nothing about profitability.

"Sharpe-like" numbers are descriptive only:
    per trade   mean(return) / std(return), return = pnl / money at risk
    daily       mean / std of daily P&L (calendar days, days without trades = 0),
                x sqrt(365); relative to the start bankroll when it is known
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .trades import Trade

MIN_SAMPLE = 100


def sample_warning(n: int, what: str = "сделок") -> Optional[str]:
    if n < MIN_SAMPLE:
        return (f"ВНИМАНИЕ: n={n} {what} < {MIN_SAMPLE} — недостаточно данных для статистически "
                f"значимого вывода о прибыльности.")
    return None


@dataclass
class Metrics:
    n: int
    wins: int = 0
    win_rate: float = float("nan")
    breakeven_win_rate: float = float("nan")
    total_pnl: float = 0.0
    total_cost: float = 0.0
    roi: float = float("nan")
    mean_pnl: float = float("nan")
    median_pnl: float = float("nan")
    std_pnl: float = float("nan")
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    payoff_ratio: Optional[float] = None
    sharpe_per_trade: Optional[float] = None
    days: int = 0
    sharpe_daily_ann: Optional[float] = None
    max_drawdown: float = 0.0
    max_drawdown_pct: Optional[float] = None
    max_losing_streak: int = 0
    model_expected_pnl: Optional[float] = None
    model_realized_pnl: Optional[float] = None
    model_covered: int = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None


def _day(ts: float) -> int:
    return int(ts // 86400)


def equity_drawdown(pnls: List[float], start_equity: Optional[float]) -> tuple:
    """(max drawdown in currency, max drawdown as a fraction of the peak or None)."""
    base = start_equity or 0.0
    equity = peak = base
    mdd = 0.0
    mdd_pct = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        mdd = max(mdd, dd)
        if start_equity and peak > 0:
            mdd_pct = max(mdd_pct, dd / peak)
    return mdd, (mdd_pct if start_equity else None)


def compute(trades: List[Trade], start_equity: Optional[float] = None) -> Metrics:
    n = len(trades)
    m = Metrics(n=n)
    if n == 0:
        return m
    trades = sorted(trades, key=lambda t: t.ts)
    pnls = [t.pnl for t in trades]
    rets = [t.ret for t in trades]
    wins = [t for t in trades if t.won]
    m.wins = len(wins)
    m.win_rate = len(wins) / n
    shares = sum(t.shares for t in trades)
    m.total_cost = sum(t.cost for t in trades)
    m.breakeven_win_rate = m.total_cost / shares if shares else float("nan")
    m.total_pnl = sum(pnls)
    m.roi = m.total_pnl / m.total_cost if m.total_cost else float("nan")
    m.mean_pnl = statistics.fmean(pnls)
    m.median_pnl = statistics.median(pnls)
    m.std_pnl = statistics.stdev(pnls) if n > 1 else float("nan")
    gains = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    m.avg_win = statistics.fmean(gains) if gains else None
    m.avg_loss = statistics.fmean(losses) if losses else None
    if losses:
        m.profit_factor = sum(gains) / -sum(losses)
    elif gains:
        m.profit_factor = math.inf
    if m.avg_win is not None and m.avg_loss is not None:
        m.payoff_ratio = m.avg_win / -m.avg_loss
    if n > 1:
        sd = statistics.stdev(rets)
        m.sharpe_per_trade = statistics.fmean(rets) / sd if sd > 0 else None
    # calendar-day series, zero-filled
    first, last = _day(trades[0].ts), _day(trades[-1].ts)
    daily: Dict[int, float] = {d: 0.0 for d in range(first, last + 1)}
    for t in trades:
        daily[_day(t.ts)] += t.pnl
    m.days = len(daily)
    if m.days >= 2:
        series = [v / start_equity for v in daily.values()] if start_equity else list(daily.values())
        sd = statistics.stdev(series)
        m.sharpe_daily_ann = statistics.fmean(series) / sd * math.sqrt(365.0) if sd > 0 else None
    m.max_drawdown, m.max_drawdown_pct = equity_drawdown(pnls, start_equity)
    streak = 0
    for t in trades:
        streak = streak + 1 if not t.won else 0
        m.max_losing_streak = max(m.max_losing_streak, streak)
    covered = [t for t in trades if t.p_model is not None]
    m.model_covered = len(covered)
    if covered:
        m.model_expected_pnl = sum(t.shares * t.p_model - t.cost for t in covered)
        m.model_realized_pnl = sum(t.pnl for t in covered)
    m.first_ts, m.last_ts = trades[0].ts, trades[-1].ts
    return m


def breakdown(trades: List[Trade], key: Callable[[Trade], str]) -> Dict[str, Metrics]:
    groups: Dict[str, List[Trade]] = {}
    for t in trades:
        groups.setdefault(key(t), []).append(t)
    return {k: compute(v) for k, v in sorted(groups.items())}


def price_bucket(t: Trade) -> str:
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0]
    for lo, hi in zip(edges, edges[1:]):
        if t.price < hi or hi == 1.0:
            return f"{lo:.2f}-{hi:.2f}"
    return "?"


def tau_bucket(t: Trade) -> str:
    if t.tau_s is None:
        return "?"
    for hi, name in ((10, "<10s"), (30, "10-30s"), (60, "30-60s"), (120, "1-2m"), (300, "2-5m"),
                     (900, "5-15m"), (3600, "15-60m")):
        if t.tau_s < hi:
            return name
    return ">1h"


BREAKDOWNS: Dict[str, Callable[[Trade], str]] = {
    "asset": lambda t: t.asset,
    "label": lambda t: t.label,
    "side": lambda t: t.side,
    "kind": lambda t: t.kind,
    "price": price_bucket,
    "tau": tau_bucket,
}


# ---------------------------------------------------------------- formatting
def _f(x: Optional[float], fmt: str) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return format(x, fmt)


def _iso(ts: Optional[float]) -> str:
    return "-" if ts is None else datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


def format_metrics(m: Metrics, unit: str = "USDC") -> List[str]:
    if m.n == 0:
        return ["сделок нет"]
    lines = [
        f"период {_iso(m.first_ts)} .. {_iso(m.last_ts)} UTC, дней {m.days}, сделок {m.n}",
        f"винрейт {m.win_rate:.3f} ({m.wins}/{m.n}) | безубыточный винрейт {m.breakeven_win_rate:.3f} "
        f"| разница {m.win_rate - m.breakeven_win_rate:+.3f}",
        f"P&L всего {m.total_pnl:+.2f} {unit} | средний {m.mean_pnl:+.4f} | медианный {m.median_pnl:+.4f} "
        f"| σ {_f(m.std_pnl, '.4f')} | ROI {_f(m.roi * 100 if m.roi == m.roi else None, '+.2f')}%",
        f"средний выигрыш {_f(m.avg_win, '+.4f')} | средний проигрыш {_f(m.avg_loss, '+.4f')} "
        f"| payoff {_f(m.payoff_ratio, '.3f')} | profit factor {_f(m.profit_factor, '.3f')}",
        f"макс. просадка {m.max_drawdown:.2f} {unit}"
        + (f" ({m.max_drawdown_pct:.2%} от пика)" if m.max_drawdown_pct is not None else "")
        + f" | макс. серия убытков {m.max_losing_streak}",
        f"Sharpe-подобная: на сделку {_f(m.sharpe_per_trade, '.3f')} | по дням (годовая) "
        f"{_f(m.sharpe_daily_ann, '.2f')}" + (" (дней < 30: неустойчиво)" if 0 < m.days < 30 else ""),
    ]
    if m.model_expected_pnl is not None:
        lines.append(f"модель ожидала {m.model_expected_pnl:+.2f} {unit} на {m.model_covered} сделках, "
                     f"получено {m.model_realized_pnl:+.2f}")
    w = sample_warning(m.n)
    if w:
        lines.append(w)
    return lines


def format_breakdown(name: str, groups: Dict[str, Metrics]) -> List[str]:
    lines = [f"по {name}:", f"  {'группа':<12} {'n':>6} {'винрейт':>8} {'безубыт.':>9} {'P&L':>10} {'ROI':>8}"]
    for k, m in groups.items():
        lines.append(f"  {k:<12} {m.n:>6} {m.win_rate:>8.3f} {m.breakeven_win_rate:>9.3f} "
                     f"{m.total_pnl:>+10.2f} {_f(m.roi * 100, '+7.2f'):>7}%")
    return lines

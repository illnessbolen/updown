"""Offline analysis of the shadow run: is P(fair) any good, and would the signals have paid?

Two independent questions:

1. Calibration (snapshots x outcomes): Brier score and log-loss of P(fair)
   versus the market's own mid price, on the same rows, by time-to-close
   bucket, plus a reliability table. If the model is not better calibrated
   than the market mid, any "edge" it reports is noise.
   Rows within one window are strongly correlated, so the effective sample
   size is closer to the number of WINDOWS than the number of rows.

2. Shadow signals (signals.csv x outcomes): the first SIGNAL per (window,
   side) is treated as a hypothetical taker fill at the logged ask plus fee
   and expected slippage. Win rate is shown next to the break-even win rate
   (= average cost per share): buying at 0.97 needs > 97% wins just to break even,
   so a high win rate alone says nothing about profitability.

The full statistics module (Sharpe, drawdown, binomial / payoff-aware
hypothesis test) comes with the execution phase; this is the model check.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import statistics
from typing import Dict, Iterable, List, Optional, Tuple

from .signal_log import read_outcomes

MIN_SAMPLE = 100
TAU_BUCKETS = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 300), (300, 900), (900, 3600), (3600, math.inf)]


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _rows(paths: Iterable[str]) -> Iterable[dict]:
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)


def _logloss(p: float, y: int) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -math.log(p if y else 1 - p)


def sample_warning(n: int, what: str, about: str = "о прибыльности") -> Optional[str]:
    if n < MIN_SAMPLE:
        return (f"ВНИМАНИЕ: {what}: n={n} < {MIN_SAMPLE} — недостаточно данных для статистически "
                f"значимого вывода {about}.")
    return None


def calibration(snapshot_paths: List[str], outcomes: Dict[str, str]) -> dict:
    rows = []
    for r in _rows(snapshot_paths):
        y = outcomes.get(r["slug"])
        p = _num(r.get("p_fair_up"))
        tau = _num(r.get("tau_s"))
        if y is None or p is None or tau is None or r.get("reasons"):
            continue                    # only rows whose model inputs passed every gate
        ub, ua = _num(r.get("up_bid")), _num(r.get("up_ask"))
        mid = 0.5 * (ub + ua) if ub is not None and ua is not None else None
        rows.append((r["slug"], tau, p, mid, 1 if y == "up" else 0))
    res = {"rows": len(rows), "windows": len({r[0] for r in rows}), "buckets": [], "reliability": []}
    paired = [r for r in rows if r[3] is not None]
    if paired:
        res["brier_model"] = statistics.fmean((p - y) ** 2 for _, _, p, _, y in paired)
        res["brier_market"] = statistics.fmean((m - y) ** 2 for _, _, _, m, y in paired)
        res["logloss_model"] = statistics.fmean(_logloss(p, y) for _, _, p, _, y in paired)
        res["logloss_market"] = statistics.fmean(_logloss(m, y) for _, _, _, m, y in paired)
        res["paired_rows"] = len(paired)
    for lo, hi in TAU_BUCKETS:
        b = [r for r in paired if lo <= r[1] < hi]
        if b:
            res["buckets"].append({
                "tau": f"{lo}-{hi if math.isfinite(hi) else 'inf'}s", "rows": len(b),
                "windows": len({r[0] for r in b}),
                "brier_model": statistics.fmean((p - y) ** 2 for _, _, p, _, y in b),
                "brier_market": statistics.fmean((m - y) ** 2 for _, _, _, m, y in b)})
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        b = [r for r in rows if lo <= r[2] < hi or (k == 9 and r[2] == 1.0)]
        if b:
            res["reliability"].append({"bin": f"{lo:.1f}-{hi:.1f}", "rows": len(b),
                                       "mean_p": statistics.fmean(r[2] for r in b),
                                       "freq_up": statistics.fmean(r[4] for r in b)})
    return res


def shadow_signals(signal_paths: List[str], outcomes: Dict[str, str],
                   decision: str = "SIGNAL") -> Tuple[dict, List[dict]]:
    first: Dict[Tuple[str, str], dict] = {}
    for r in _rows(signal_paths):
        if r.get("decision") != decision:
            continue
        key = (r["slug"], r["side"])
        if key not in first:
            first[key] = r
    resolved = []
    for (slug, side), r in first.items():
        y = outcomes.get(slug)
        cost = _num(r.get("cost"))
        if y is None or cost is None:
            continue
        won = side == y
        resolved.append(dict(r, outcome=y, won=int(won), pnl_per_share=round((1.0 if won else 0.0) - cost, 6)))
    n = len(resolved)
    res = {"decision": decision, "signals": len(first), "resolved": n}
    if n:
        pnl = [x["pnl_per_share"] for x in resolved]
        costs = [_num(x["cost"]) for x in resolved]
        res.update({
            "wins": sum(x["won"] for x in resolved),
            "win_rate": statistics.fmean(x["won"] for x in resolved),
            "breakeven_win_rate": statistics.fmean(costs),
            "mean_pnl_per_share": statistics.fmean(pnl),
            "median_pnl_per_share": statistics.median(pnl),
            "total_pnl_1_share_each": sum(pnl),
            "roi": sum(pnl) / sum(costs),
            "mean_model_edge_cons": statistics.fmean(_num(x["edge_cons"]) or 0.0 for x in resolved),
        })
    return res, resolved


def run(data_dir: str, out=print) -> dict:
    outcomes = read_outcomes(os.path.join(data_dir, "outcomes.csv"))
    snaps = sorted(glob.glob(os.path.join(data_dir, "snapshots-*.csv")))
    sig_path = os.path.join(data_dir, "signals.csv")
    sigs = [sig_path] if os.path.exists(sig_path) else []
    cal = calibration(snaps, outcomes)
    shadow, resolved = shadow_signals(sigs, outcomes, "SIGNAL")
    gated, _ = shadow_signals(sigs, outcomes, "GATED")

    out(f"=== model check: {data_dir} | outcomes known for {len(outcomes)} windows ===")
    out(f"\n-- calibration of P(fair) vs market mid (rows without gate reasons) --")
    out(f"rows={cal['rows']} windows={cal['windows']}")
    if "brier_model" in cal:
        out(f"Brier   model {cal['brier_model']:.5f} | market {cal['brier_market']:.5f}  (lower is better)")
        out(f"LogLoss model {cal['logloss_model']:.5f} | market {cal['logloss_market']:.5f}")
        for b in cal["buckets"]:
            out(f"  tau {b['tau']:>11}: rows={b['rows']:6d} windows={b['windows']:5d} "
                f"brier model {b['brier_model']:.4f} market {b['brier_market']:.4f}")
        out("  reliability (P_fair bin -> observed Up frequency):")
        for r in cal["reliability"]:
            out(f"    {r['bin']}: rows={r['rows']:6d} mean P={r['mean_p']:.3f} freq={r['freq_up']:.3f}")
    w = sample_warning(cal["windows"], "калибровка (независимых окон)", "о качестве модели")
    if w:
        out(w)

    for res in (shadow, gated):
        out(f"\n-- shadow {res['decision']} (first per window/side, taker at logged ask+fee+slippage) --")
        out(f"logged={res['signals']} resolved={res['resolved']}")
        if res["resolved"]:
            out(f"win rate {res['win_rate']:.3f} vs break-even {res['breakeven_win_rate']:.3f} | "
                f"mean PnL/share {res['mean_pnl_per_share']:+.4f} median {res['median_pnl_per_share']:+.4f} | "
                f"sum {res['total_pnl_1_share_each']:+.3f} | ROI {res['roi']:+.2%} | "
                f"mean model edge {res['mean_model_edge_cons']:+.4f}")
        w = sample_warning(res["resolved"], f"{res['decision']}")
        if w:
            out(w)

    if resolved:
        path = os.path.join(data_dir, "signals_resolved.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(resolved[0].keys()))
            wr.writeheader()
            wr.writerows(resolved)
        out(f"\nper-signal outcomes -> {path}")
    return {"calibration": cal, "signals": shadow, "gated": gated}

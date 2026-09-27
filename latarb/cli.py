"""Command line.

    python bot.py discover                 list currently active Up/Down windows (Gamma)
    python bot.py shadow [--record]        live feeds -> P(fair) -> signal log (no orders at all)
    python bot.py paper  [--record] [--reset]
                                           shadow + simulated execution with risk management and a
                                           virtual balance against the live Polymarket books
    python bot.py replay data/ticks [--paper]
                                           backtest the signal layer (and paper execution) on recorded frames
    python bot.py resolve                  fetch winners for logged windows
    python bot.py analyze [--data DIR]     model calibration, shadow signals, paper results
    python bot.py stats [--source paper|shadow] [--days N] [--by asset,label,...]
                                           performance: win rate vs break-even, P&L, drawdown, Sharpe-like
    python bot.py hypothesis [--alpha 0.05] [--cluster hour]
                                           is the edge statistically distinguishable from zero?
    python bot.py report [--days 7]        write the weekly report now (the running bot does it weekly)
    python bot.py price --spot ... --ref ... --tau ... --sigma-bps ...   one-off P(fair)

There is no live-trading command in this version: nothing can send a real order.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import math
import os
import sys
from datetime import datetime, timezone

from .config import ConfigError, load_settings


def _setup_logging(level: str, log_dir: str | None) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(level.upper())
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    root.addHandler(h)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.TimedRotatingFileHandler(os.path.join(log_dir, "latarb.log"), when="midnight",
                                                       backupCount=14, encoding="utf-8", utc=True)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _ts(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bot.py", description="Polymarket Up/Down latency-arbitrage research bot "
                                                            "(data, P(fair), signals, paper execution; no live orders)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover", help="list active Up/Down windows")
    sp = sub.add_parser("shadow", help="live data, model and signal log; never places orders")
    sp.add_argument("--record", action="store_true", help="also record raw frames to DATA_DIR/ticks")
    sp.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    pa = sub.add_parser("paper", help="live data + simulated execution with a virtual balance")
    pa.add_argument("--record", action="store_true", help="also record raw frames to DATA_DIR/ticks")
    pa.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    pa.add_argument("--reset", action="store_true",
                    help="start a new paper account (the old state file is kept as a .bak)")
    rp = sub.add_parser("replay", help="run the signal layer over recorded frames")
    rp.add_argument("paths", nargs="+", help="tick files or directories")
    rp.add_argument("--out", default=None, help="output dir (default DATA_DIR/replay-<utc stamp>)")
    rp.add_argument("--paper", action="store_true", help="also simulate execution (fresh paper account)")
    sub.add_parser("resolve", help="fetch outcomes for windows seen in the logs")
    an = sub.add_parser("analyze", help="calibration + shadow signal outcomes")
    an.add_argument("--data", default=None, help="directory with signals/snapshots/outcomes (default DATA_DIR)")
    def add_common(p):
        p.add_argument("--data", default=None, help="directory with the logs (default DATA_DIR)")
        p.add_argument("--source", default="auto", choices=("auto", "paper", "shadow"),
                       help="paper = settled paper positions, shadow = first SIGNAL per window (1 share)")
        p.add_argument("--days", type=float, default=None, help="only trades settled in the last N days")
    st = sub.add_parser("stats", help="performance statistics of settled trades")
    add_common(st)
    st.add_argument("--by", default="asset,label,kind,price,tau", help="breakdowns: asset,label,side,kind,price,tau")
    hy = sub.add_parser("hypothesis", help="hypothesis test: is the edge distinguishable from zero?")
    add_common(hy)
    hy.add_argument("--alpha", type=float, default=0.05)
    hy.add_argument("--sims", type=int, default=20000, help="Monte-Carlo simulations under H0")
    hy.add_argument("--cluster", default="hour", choices=("trade", "window", "hour", "day"),
                    help="bootstrap cluster (correlated trades are resampled together)")
    hy.add_argument("--json", action="store_true", help="print the result as JSON")
    rp2 = sub.add_parser("report", help="write the weekly report now")
    rp2.add_argument("--data", default=None)
    rp2.add_argument("--days", type=float, default=7.0)
    pp = sub.add_parser("price", help="evaluate P(fair) for given inputs")
    pp.add_argument("--spot", type=float, required=True)
    pp.add_argument("--ref", type=float, required=True)
    pp.add_argument("--tau", type=float, required=True, help="seconds to close")
    pp.add_argument("--sigma-bps", type=float, required=True, help="vol in bp per sqrt(second)")
    pp.add_argument("--noise-bps", type=float, default=0.0)
    pp.add_argument("--dof", type=float, default=None, help="Student-t dof (default TAIL_DOF), inf = Gaussian")
    args = ap.parse_args(argv)

    try:
        cfg = load_settings()
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "price":
        from .model.pricing import fair_value
        dof = cfg.TAIL_DOF if args.dof is None else args.dof
        fv = fair_value(args.spot, args.ref, args.tau, args.sigma_bps * 1e-4, args.noise_bps * 1e-4, dof)
        print(f"x = {fv.x * 1e4:+.3f} bp | sd = {fv.sd * 1e4:.3f} bp | d = {fv.d:+.4f} | "
              f"P(up) = {fv.p_up:.6f} | P(down) = {fv.p_down:.6f} | tails: "
              f"{'gaussian' if math.isinf(dof) else f't({dof:g})'}")
        return 0

    _setup_logging(cfg.LOG_LEVEL, os.path.join(cfg.DATA_DIR, "logs") if args.cmd in ("shadow", "paper") else None)
    log = logging.getLogger("latarb")

    if args.cmd == "discover":
        from .app import discover_once
        ws = discover_once(cfg)
        print(f"{len(ws)} window(s)")
        for w in ws:
            print(f"{w.asset:5s} {w.label:4s} {_ts(w.start_ts)} -> {_ts(w.end_ts)} UTC  "
                  f"res={w.resolution:9s} tick={w.tick_size:<6g} fee={w.taker_fee_rate if w.taker_fee_rate is not None else cfg.TAKER_FEE_RATE:<5g} "
                  f"{'CLOSED ' if w.closed else ''}{w.slug}")
        return 0
    if args.cmd in ("shadow", "paper"):
        from .app import reset_paper_state, run_live
        if args.cmd == "paper":
            from .risk.limits import resolve_limits
            try:
                resolve_limits(cfg)                  # refuse to start on out-of-bound risk settings
            except ConfigError as e:
                print(f"configuration error: {e}", file=sys.stderr)
                return 2
            if args.reset:
                os.makedirs(cfg.DATA_DIR, exist_ok=True)
                backup = reset_paper_state(cfg.DATA_DIR)
                log.warning("paper account reset%s", f"; previous state kept in {backup}" if backup else "")
        try:
            import uvloop  # type: ignore
            uvloop.install()
        except ImportError:
            pass
        asyncio.run(run_live(cfg, args.cmd, record=args.record or cfg.RECORD_TICKS, duration_s=args.duration))
        return 0
    if args.cmd == "replay":
        from .app import run_replay
        out = args.out or os.path.join(cfg.DATA_DIR, "replay-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        run_replay(cfg, args.paths, out, paper=args.paper)
        log.info("replay outputs in %s (python bot.py analyze --data %s)", out, out)
        return 0
    if args.cmd == "resolve":
        from .app import resolve_outcomes
        log.info("resolved %d window(s)", resolve_outcomes(cfg))
        return 0
    if args.cmd == "analyze":
        from .reporting.analyze import run
        run(args.data or cfg.DATA_DIR)
        return 0
    if args.cmd in ("stats", "hypothesis"):
        import time as _time

        from .stats.trades import filter_period, load_trades, paper_start_equity
        data = args.data or cfg.DATA_DIR
        trades = load_trades(data, args.source)
        if args.days:
            trades = filter_period(trades, since=_time.time() - args.days * 86400)
        source = trades[0].source if trades else args.source
        if args.cmd == "stats":
            from .stats.metrics import BREAKDOWNS, breakdown, compute, format_breakdown, format_metrics
            unit = "USDC" if source == "paper" else "ед. (1 акция на сигнал)"
            print(f"=== статистика [{source}] {data} ===")
            eq = paper_start_equity(data) if source == "paper" else None
            for line in format_metrics(compute(trades, eq), unit):
                print(line)
            for dim in [d.strip() for d in args.by.split(",") if d.strip()]:
                if dim not in BREAKDOWNS:
                    print(f"unknown breakdown {dim!r}; choose from {sorted(BREAKDOWNS)}", file=sys.stderr)
                    return 2
                if trades:
                    for line in format_breakdown(dim, breakdown(trades, BREAKDOWNS[dim])):
                        print(line)
            return 0
        from .stats import hypothesis
        if not 0 < args.alpha < 0.5:
            print("--alpha must be in (0, 0.5)", file=sys.stderr)
            return 2
        res = hypothesis.run(trades, alpha=args.alpha, sims=args.sims, cluster=args.cluster, source=source)
        if args.json:
            import json as _json
            print(_json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
        else:
            for line in res.lines():
                print(line)
        return 0
    if args.cmd == "report":
        import time as _time

        from .stats.report import write_report
        print(write_report(args.data or cfg.DATA_DIR, _time.time(), args.days, profile=cfg.RISK_PROFILE))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

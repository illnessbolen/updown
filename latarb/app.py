"""Run modes. None of them can send a real order: execution in this version is paper only.

shadow   live WebSocket feeds + Gamma discovery -> P(fair) -> signals.csv / snapshots / outcomes;
         optional raw tick recording for later replays.
paper    shadow + simulated execution against the live Polymarket books: risk checks, Kelly
         sizing, pre-trade book refresh, latency budget, maker-first / taker orders, virtual
         balance, settlement from Gamma outcomes.
replay   the same hub + engine (+ paper execution with --paper) driven by recorded frames,
         a ReplayClock and a ReplayScheduler; outputs go to a separate directory.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import signal as _signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .clock import Clock, ReplayClock, WallClock
from .config import LABEL_SECONDS, Settings
from .data.feeds import FeedSet
from .data.gamma import Discovery, GammaClient, fetch_winner
from .data.hub import MarketDataHub
from .data.markets import MarketWindow, parse_slug
from .data.parsers import build_parsers, parse_safely
from .data.recorder import TickRecorder, iter_recording
from .data.reference import BinanceRest, ReferenceResolver
from .execution.books import RestBookRefresher, WsBookRefresher
from .execution.paper import PaperExchange
from .execution.pipeline import ExecutionPipeline
from .fastjson import dumps, loads
from .reporting.exec_log import ExecutionLog
from .reporting.signal_log import OutcomeTracker, SignalSink
from .risk.limits import RiskLimits, resolve_limits
from .risk.manager import RiskManager
from .risk.portfolio import Portfolio
from .scheduler import LoopScheduler, ReplayScheduler, Scheduler
from .signal.engine import SignalEngine
from .stats.report import maybe_write_weekly

log = logging.getLogger("latarb.app")


def _bp(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 1e4:.2f}"


def status_lines(hub: MarketDataHub, engine: SignalEngine, now: float) -> List[str]:
    lines = []
    feeds = " ".join(f"{src}:{'up' if hub.feed_connected.get(src) else 'DOWN'}/{hub.feed_age(src, now):.1f}s"
                     for src in ("binance", "coinbase", "rtds", "polymarket"))
    skew = "?" if hub.clock_skew_ms is None else f"{hub.clock_skew_ms:+.0f}ms"
    lines.append(f"feeds {feeds} | clock skew {skew} | windows {len(hub.markets)}")
    for a, st in hub.assets.items():
        if st.fast is None:
            lines.append(f"  {a}: no Binance quote yet")
            continue
        sg = st.vol.sigmas()
        lines.append(
            f"  {a}: mid {st.fast.mid:.8g} age {now - st.fast.recv_ts:.1f}s | sigma fast/slow "
            f"{_bp(sg[0]) if sg else '-'}/{_bp(sg[1]) if sg else '-'} bp/s^0.5 "
            f"({st.vol.samples}/{st.vol.min_samples}) | oracle basis {_bp(st.oracle_basis.mean)}"
            f"±{_bp(st.oracle_basis.std)}bp n={st.oracle_basis.n} | cb basis {_bp(st.sanity_basis.mean)}"
            f"±{_bp(st.sanity_basis.std)}bp | trade lag "
            f"{'-' if st.feed_latency_ms is None else f'{st.feed_latency_ms:.0f}ms'}")
    s = engine.stats
    top_unpriced = ", ".join(f"{k}={v}" for k, v in s.unpriced.most_common(4)) or "-"
    top_gates = ", ".join(f"{k}={v}" for k, v in s.gates.most_common(5)) or "-"
    lines.append(f"  engine: evals {s.evaluations} signals {s.signals} gated {s.gated} | "
                 f"unpriced: {top_unpriced} | gates: {top_gates}")
    return lines


# ====================================================================== paper stack
@dataclass
class PaperStack:
    limits: RiskLimits
    portfolio: Portfolio
    risk: RiskManager
    exchange: PaperExchange
    pipeline: ExecutionPipeline
    log: ExecutionLog


def state_path(directory: str) -> str:
    return os.path.join(directory, "paper_state.json")


def reset_paper_state(directory: str) -> Optional[str]:
    """Move the current paper state aside (never deleted); returns the backup path."""
    path = state_path(directory)
    if not os.path.exists(path):
        return None
    backup = f"{path}.{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.bak"
    os.replace(path, backup)
    return backup


def build_paper(cfg: Settings, hub: MarketDataHub, engine: SignalEngine, clock: Clock, scheduler: Scheduler,
                refresher, out_dir: str, outcomes: OutcomeTracker) -> PaperStack:
    limits = resolve_limits(cfg)
    pf = Portfolio.load_or_new(state_path(out_dir), cfg.PAPER_START_USDC)
    risk = RiskManager(cfg, limits, pf, clock)
    exchange = PaperExchange(cfg, hub, clock, scheduler)
    xlog = ExecutionLog(out_dir, clock, cfg.LATENCY_BUDGET_MS)
    pipe = ExecutionPipeline(cfg, hub, engine, risk, pf, limits, exchange, refresher, clock, xlog)
    outcomes.listeners.append(pipe.on_outcome)
    for p in list(pf.positions.values()):
        winner = outcomes.known_winner(p.slug)
        if winner:                                   # resolved while we were down
            pipe.on_outcome(p.slug, winner, clock.now())
        else:
            outcomes.track([MarketWindow.from_dict(p.window)])
    return PaperStack(limits, pf, risk, exchange, pipe, xlog)


def paper_preflight(cfg: Settings, st: PaperStack) -> bool:
    """Mandatory checks before paper trading starts. Returns False to refuse starting."""
    if os.path.exists(cfg.KILL_SWITCH_FILE):
        log.critical("kill-switch file %r exists - refusing to start. Delete it to trade.", cfg.KILL_SWITCH_FILE)
        return False
    pf = st.portfolio
    if pf.bankroll <= 0:
        log.critical("paper bankroll is %.2f - nothing to trade with (use --reset for a new paper account).",
                     pf.bankroll)
        return False
    log.warning("PAPER mode: orders are simulated against live Polymarket books with a virtual balance. "
                "No real order can be sent by this version.")
    log.info("paper account: cash %.2f | bankroll %.2f | realized %+.2f | open positions %d | W/L %d/%d",
             pf.cash, pf.bankroll, pf.realized, len(pf.positions), pf.wins, pf.losses)
    log.info("risk: %s", st.limits.describe(pf.bankroll))
    log.info("execution: style=%s budget=%.0fms refresh=%s book_max_age=%.0fms maker_timeout=%.0fms "
             "fallback=%s min_edge_at_fill=%.3f max_slippage=%.3f fill_delay=%.0fms cooldown=%.0fs "
             "breaker=%d kill_switch=%s", cfg.EXECUTION_STYLE, cfg.LATENCY_BUDGET_MS, cfg.PRE_TRADE_REFRESH,
             cfg.BOOK_MAX_AGE_MS, cfg.MAKER_TIMEOUT_MS, cfg.MAKER_FALLBACK_TAKER, cfg.MIN_EDGE_AT_FILL,
             cfg.MAX_SLIPPAGE, cfg.PAPER_FILL_DELAY_MS, cfg.COOLDOWN_AFTER_LOSS_S, cfg.MAX_CONSECUTIVE_ERRORS,
             os.path.abspath(cfg.KILL_SWITCH_FILE))
    if st.risk.daily_stop_hit():
        log.warning("daily stop is already hit for today (%s): no new orders until the next UTC day", pf.day)
    return True


# ====================================================================== live (shadow / paper)
async def run_live(cfg: Settings, mode: str = "shadow", record: bool = False,
                   duration_s: Optional[float] = None) -> None:
    assert mode in ("shadow", "paper")
    clock = WallClock()
    hub = MarketDataHub(cfg, clock)
    recorder = TickRecorder(os.path.join(cfg.DATA_DIR, "ticks"), cfg.RECORD_ROTATE_S) if record else None
    sink = SignalSink(cfg.DATA_DIR)
    refs = ReferenceResolver(cfg, hub)
    engine = SignalEngine(cfg, hub, refs, clock, sink)
    gamma = GammaClient(cfg)
    rest = BinanceRest(cfg)
    discovery = Discovery(cfg, gamma, clock)
    outcomes = OutcomeTracker(cfg, sink)
    feeds = FeedSet(cfg, clock, hub, recorder)
    stop = asyncio.Event()
    paper: Optional[PaperStack] = None
    if mode == "paper":
        scheduler = LoopScheduler(clock)
        refresher = (RestBookRefresher(cfg, hub, clock) if cfg.PRE_TRADE_REFRESH == "rest"
                     else WsBookRefresher(hub, clock, scheduler, cfg.PAPER_REFRESH_LATENCY_MS))
        paper = build_paper(cfg, hub, engine, clock, scheduler, refresher, cfg.DATA_DIR, outcomes)
        if not paper_preflight(cfg, paper):
            sink.close()
            paper.log.close()
            return
    else:
        log.info("SHADOW mode: live market data, model and signal log only - no orders of any kind.")
    log.info("assets=%s deterministic=%s edge_threshold=%.3f tail_dof=%s data_dir=%s record=%s",
             ",".join(cfg.ASSETS), ",".join(cfg.DETERMINISTIC_LABELS), cfg.EDGE_THRESHOLD, cfg.TAIL_DOF,
             cfg.DATA_DIR, record)

    async def every(period: float, fn, name: str) -> None:
        while True:
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - a periodic job must survive transient failures
                log.exception("%s failed: %s", name, e)
            await asyncio.sleep(period)

    last_market_set: set = set()

    async def discovery_job() -> None:
        nonlocal last_market_set
        await asyncio.to_thread(discovery.refresh)
        now = clock.now()
        relevant = discovery.relevant(now)
        hub.set_markets(relevant)
        live = set(hub.markets)
        refs.prune(live)
        engine.forget(live)
        await feeds.poly.set_tokens(t for w in relevant for t in w.tokens())
        outcomes.track(discovery.windows.values())
        if recorder is not None and live != last_market_set:
            recorder.write(now, "@markets", dumps([w.to_dict() for w in relevant]))
        last_market_set = live
        feeds.poly.check_snapshots(now)

    async def reference_job() -> None:
        for w in refs.rest_jobs(clock.now()):
            refs.set_rest_result(w, await asyncio.to_thread(refs.fetch_rest, rest, w))

    async def outcome_job() -> None:
        for w in outcomes.due(clock.now()):
            try:
                winner = await asyncio.to_thread(fetch_winner, gamma, w.slug)
            except Exception as e:  # noqa: BLE001
                log.warning("outcome fetch %s failed: %s", w.slug, e)
                continue
            if winner:
                now = clock.now()
                outcomes.record(w, winner, now)
                if recorder is not None:
                    recorder.write(now, "@outcome", dumps({"slug": w.slug, "winner": winner,
                                                           "window": w.to_dict()}))

    async def timer_job() -> None:
        engine.on_timer()

    async def clock_job() -> None:
        t0 = time.time()
        server = await asyncio.to_thread(rest.server_time)
        t1 = time.time()
        rtt = t1 - t0
        if rtt > 1.0:
            log.warning("clock check skipped: REST round trip %.0fms too slow to measure skew", rtt * 1000)
            return
        hub.clock_skew_ms = (server - 0.5 * (t0 + t1)) * 1000.0
        lvl = logging.WARNING if abs(hub.clock_skew_ms) > cfg.MAX_CLOCK_SKEW_MS else logging.INFO
        log.log(lvl, "clock vs Binance: %+.1fms (rtt %.0fms, limit %.0fms)", hub.clock_skew_ms, rtt * 1000,
                cfg.MAX_CLOCK_SKEW_MS)

    async def weekly_job() -> None:
        path = await asyncio.to_thread(maybe_write_weekly, cfg.DATA_DIR, clock.now(), profile=cfg.RISK_PROFILE)
        if path:
            log.info("weekly report written: %s", path)

    async def status_job() -> None:
        for line in status_lines(hub, engine, clock.now()):
            log.info(line)
        if paper is not None:
            log.info("  %s", paper.pipeline.status_line())
            paper.log.flush()
        sink.flush()
        if recorder is not None:
            recorder.flush()

    tasks = [asyncio.create_task(f.run(), name=f.name) for f in feeds.all()]
    tasks += [
        asyncio.create_task(every(cfg.DISCOVERY_TICK_S, discovery_job, "discovery")),
        asyncio.create_task(every(1.0, reference_job, "reference")),
        asyncio.create_task(every(cfg.RESOLVE_INTERVAL_S, outcome_job, "outcomes")),
        asyncio.create_task(every(cfg.EVAL_TIMER_S, timer_job, "timer")),
        asyncio.create_task(every(cfg.CLOCK_CHECK_INTERVAL_S, clock_job, "clock")),
        asyncio.create_task(every(cfg.STATUS_INTERVAL_S, status_job, "status")),
        asyncio.create_task(every(3600.0, weekly_job, "weekly report")),
    ]
    loop = asyncio.get_running_loop()
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):   # Windows
            pass
    try:
        if duration_s:
            try:
                await asyncio.wait_for(stop.wait(), timeout=duration_s)
            except asyncio.TimeoutError:
                pass
        else:
            await stop.wait()
    finally:
        log.info("stopping...")
        discovery.stop()
        for f in feeds.all():
            await f.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if paper is not None:
            paper.exchange.cancel_all("shutdown")
            log.info("  %s", paper.pipeline.status_line())
            paper.log.close()
        for line in status_lines(hub, engine, clock.now()):
            log.info(line)
        sink.close()
        if recorder is not None:
            recorder.close()


async def run_shadow(cfg: Settings, record: bool = False, duration_s: Optional[float] = None) -> None:
    await run_live(cfg, "shadow", record, duration_s)


# ====================================================================== replay
def run_replay(cfg: Settings, paths: List[str], out_dir: str, paper: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    clock = ReplayClock()
    scheduler = ReplayScheduler(clock)
    hub = MarketDataHub(cfg, clock)
    sink = SignalSink(out_dir)
    refs = ReferenceResolver(cfg, hub)
    engine = SignalEngine(cfg, hub, refs, clock, sink)
    outcomes = OutcomeTracker(cfg, sink)
    stack: Optional[PaperStack] = None
    if paper:
        # a backtest has its own fresh account and must not react to the live bot's kill switch
        cfg = dataclasses.replace(cfg, KILL_SWITCH_FILE=os.path.join(out_dir, "STOP"))
        reset_paper_state(out_dir)
        refresher = WsBookRefresher(hub, clock, scheduler, cfg.PAPER_REFRESH_LATENCY_MS)
        stack = build_paper(cfg, hub, engine, clock, scheduler, refresher, out_dir, outcomes)
    parsers = build_parsers(cfg.BINANCE_SYMBOLS, cfg.COINBASE_PRODUCTS, cfg.CHAINLINK_SYMBOLS)
    frames = bad = 0
    first_ts = last_ts = None
    for ts, src, raw in iter_recording(paths):
        if first_ts is None:
            first_ts = ts
            scheduler.every(cfg.EVAL_TIMER_S, engine.on_timer, ts)
        scheduler.run_until(ts)
        clock.advance_to(ts)
        last_ts = ts
        frames += 1
        if src == "@markets":
            hub.set_markets(MarketWindow.from_dict(d) for d in loads(raw))
            refs.prune(set(hub.markets))
            engine.forget(set(hub.markets))
        elif src == "@open":
            hub.on_feed_open(raw, ts)
        elif src == "@close":
            hub.on_feed_close(raw, ts)
        elif src == "@outcome":
            d = loads(raw)
            outcomes.record(MarketWindow.from_dict(d["window"]), d["winner"], ts, source="recorded")
        else:
            parser = parsers.get(src)
            if parser is None:
                continue
            hub.note_message(src, ts)
            errs: List[str] = []
            events = parse_safely(parser, raw, ts, errs)
            bad += bool(errs)
            if events:
                hub.apply_many(events)
    if last_ts is not None:
        scheduler.run_until(last_ts + 30.0)           # let in-flight orders finish
    for line in status_lines(hub, engine, clock.now()):
        log.info(line)
    sink.close()
    s = engine.stats
    summary = {"frames": frames, "unparseable": bad, "from": first_ts, "to": last_ts,
               "evaluations": s.evaluations, "signals": s.signals, "gated": s.gated,
               "unpriced": dict(s.unpriced), "gates": dict(s.gates), "out_dir": out_dir}
    if stack is not None:
        log.info("  %s", stack.pipeline.status_line())
        stack.log.close()
        pf = stack.portfolio
        summary["paper"] = {"cash": pf.cash, "bankroll": pf.bankroll, "realized": pf.realized, "wins": pf.wins,
                            "losses": pf.losses, "open_positions": len(pf.positions),
                            "stats": dict(stack.pipeline.stats)}
    log.info("replay done: %s", {k: v for k, v in summary.items() if k not in ("unpriced", "gates")})
    return summary


# ====================================================================== one-shot helpers
def discover_once(cfg: Settings) -> List[MarketWindow]:
    clock = WallClock()
    d = Discovery(cfg, GammaClient(cfg), clock)
    d.refresh(force_scan=True)
    return sorted(d.windows.values(), key=lambda w: (w.asset, w.duration_s, w.start_ts))


def resolve_outcomes(cfg: Settings) -> int:
    """Fetch winners for every window that appears in signals/snapshots but not in outcomes.csv."""
    import csv
    import glob

    sink = SignalSink(cfg.DATA_DIR, echo=False)
    tracker = OutcomeTracker(cfg, sink)
    gamma = GammaClient(cfg)
    windows = {}
    for p in [os.path.join(cfg.DATA_DIR, "signals.csv")] + glob.glob(os.path.join(cfg.DATA_DIR, "snapshots-*.csv")):
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                slug = r.get("slug")
                if slug and slug not in tracker.known and slug not in windows:
                    windows[slug] = r
    now = time.time()
    n = 0
    for slug, r in windows.items():
        try:
            winner = fetch_winner(gamma, slug)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s", slug, e)
            continue
        if winner:
            dur = LABEL_SECONDS.get(r.get("label", ""), 0)
            info = parse_slug(slug, float(r.get("ts") or now))
            start = info.start_ts if info is not None and info.start_ts is not None else None
            if start is None:
                end = float(r.get("window_end") or 0) or float(r.get("ts") or now) + float(r.get("tau_s") or 0)
                start = end - dur
            w = MarketWindow(slug=slug, asset=r.get("asset", ""), label=r.get("label", ""), duration_s=dur,
                             start_ts=start, end_ts=start + dur, up_token="", down_token="")
            tracker.record(w, winner, now)
            n += 1
    sink.close()
    return n

"""Run modes of this phase. Neither of them can place an order.

shadow   live WebSocket feeds + Gamma discovery -> P(fair) -> signals.csv / snapshots / outcomes;
         optional raw tick recording for later replays.
replay   the same hub + engine driven by recorded frames and a ReplayClock (backtest of the
         signal layer; outputs go to a separate directory).
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import signal as _signal
import time
from typing import List, Optional

from .clock import ReplayClock, WallClock
from .config import LABEL_SECONDS, Settings
from .data.feeds import FeedSet
from .data.gamma import Discovery, GammaClient, fetch_winner
from .data.hub import MarketDataHub
from .data.markets import MarketWindow, parse_slug
from .data.parsers import build_parsers, parse_safely
from .data.recorder import TickRecorder, iter_recording
from .data.reference import BinanceRest, ReferenceResolver
from .fastjson import dumps, loads
from .reporting.signal_log import OutcomeTracker, SignalSink
from .signal.engine import SignalEngine

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


# ====================================================================== shadow
async def run_shadow(cfg: Settings, record: bool = False, duration_s: Optional[float] = None) -> None:
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

    log.info("SHADOW mode: live market data, model and signal log only - no orders can be placed.")
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

    async def status_job() -> None:
        for line in status_lines(hub, engine, clock.now()):
            log.info(line)
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
        for line in status_lines(hub, engine, clock.now()):
            log.info(line)
        sink.close()
        if recorder is not None:
            recorder.close()


# ====================================================================== replay
def run_replay(cfg: Settings, paths: List[str], out_dir: str) -> dict:
    clock = ReplayClock()
    hub = MarketDataHub(cfg, clock)
    sink = SignalSink(out_dir)
    refs = ReferenceResolver(cfg, hub)
    engine = SignalEngine(cfg, hub, refs, clock, sink)
    outcomes = OutcomeTracker(cfg, sink)
    parsers = build_parsers(cfg.BINANCE_SYMBOLS, cfg.COINBASE_PRODUCTS, cfg.CHAINLINK_SYMBOLS)
    frames = bad = 0
    first_ts = last_ts = None
    next_timer = math.inf
    for ts, src, raw in iter_recording(paths):
        if first_ts is None:
            first_ts = next_timer = ts
        while next_timer <= ts:
            clock.advance_to(next_timer)
            engine.on_timer()
            next_timer += cfg.EVAL_TIMER_S
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
    for line in status_lines(hub, engine, clock.now()):
        log.info(line)
    sink.close()
    s = engine.stats
    summary = {"frames": frames, "unparseable": bad, "from": first_ts, "to": last_ts,
               "evaluations": s.evaluations, "signals": s.signals, "gated": s.gated,
               "unpriced": dict(s.unpriced), "gates": dict(s.gates), "out_dir": out_dir}
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

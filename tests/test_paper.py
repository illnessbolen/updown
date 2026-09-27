"""Paper exchange matching rules and the execution pipeline (budget, refresh, maker/taker, settlement)."""
import csv
from types import SimpleNamespace

import pytest

from latarb.clock import ReplayClock
from latarb.config import load_settings
from latarb.data.events import ASK, BID, BookLevel, BookSnapshot, LastTrade, OracleTick
from latarb.data.hub import MarketDataHub
from latarb.data.markets import CHAINLINK, MarketWindow
from latarb.execution.books import WsBookRefresher
from latarb.execution.paper import PaperExchange, PaperOrder
from latarb.execution.pipeline import ExecutionPipeline, max_taker_price
from latarb.reporting.exec_log import ExecutionLog
from latarb.risk.limits import resolve_limits
from latarb.risk.manager import RiskManager
from latarb.risk.portfolio import Portfolio
from latarb.scheduler import ReplayScheduler

from .test_pipeline import _primed

T = 1_790_000_100.0


def cfg(**kw):
    base = dict(ASSETS=("btc",))
    base.update(kw)
    return load_settings(env={}, dotenv=False, **base)


# ================================================================ exchange
def _exchange(**kw):
    c = cfg(**kw)
    clock = ReplayClock(T)
    sched = ReplayScheduler(clock)
    hub = MarketDataHub(c, clock)
    w = MarketWindow(slug="w", asset="btc", label="5m", duration_s=300, start_ts=T - 100, end_ts=T + 200,
                     up_token="U", down_token="D", resolution=CHAINLINK)
    hub.set_markets([w])
    hub.apply_many([BookSnapshot("U", [(0.48, 20), (0.50, 20)], [(0.52, 10), (0.54, 10), (0.56, 50)], T),
                    BookSnapshot("D", [(0.47, 5)], [(0.60, 10)], T)])
    return c, clock, sched, hub, w, PaperExchange(c, hub, clock, sched)


def _order(w, kind, limit, shares, clock, done):
    return PaperOrder(window=w, side="up", token="U", other_token="D", kind=kind, limit=limit, shares=shares,
                      t_send=clock.now(), fee_rate=0.07, fee_exp=1.0, on_done=done.append)


def test_taker_walks_own_asks_and_complement_up_to_the_limit():
    c, clock, sched, hub, w, ex = _exchange()
    done = []
    ex.submit(_order(w, "taker", 0.55, 30, clock, done))
    sched.run_until(T + 1)
    o = done[0]
    # 10 @ 0.52, 5 @ 0.53 (= 1 - 0.47 Down bid), 10 @ 0.54; 0.56 is above the limit
    assert o.end_reason == "partial" and o.filled == 25
    assert o.notional == pytest.approx(10 * 0.52 + 5 * 0.53 + 10 * 0.54)
    fee = sum(q * 0.07 * p * (1 - p) for q, p in ((10, 0.52), (5, 0.53), (10, 0.54)))
    assert o.fee == pytest.approx(fee) and o.maker_filled == 0


def test_order_meets_the_book_as_it_is_on_arrival_not_at_send():
    c, clock, sched, hub, w, ex = _exchange(PAPER_FILL_DELAY_MS=100)
    done = []
    ex.submit(_order(w, "taker", 0.52, 10, clock, done))
    clock.advance_to(T + 0.05)                      # someone faster lifts the 0.52 ask meanwhile
    hub.apply_many([BookLevel("U", ASK, 0.52, 0, T + 0.05)])
    sched.run_until(T + 1)
    assert done[0].filled == 0 and done[0].end_reason == "no_liquidity"


def test_paper_fills_do_not_reuse_the_same_quote():
    c, clock, sched, hub, w, ex = _exchange()
    done = []
    ex.submit(_order(w, "taker", 0.52, 10, clock, done))
    sched.run_until(T + 1)
    ex.submit(_order(w, "taker", 0.52, 10, clock, done))
    sched.run_until(T + 2)
    assert [o.filled for o in done] == [10, 0]


def test_maker_queue_trade_through_and_crossing_fills_at_own_price_without_fee():
    c, clock, sched, hub, w, ex = _exchange(MAKER_TIMEOUT_MS=5000)
    done = []
    o = ex.submit(_order(w, "maker", 0.50, 30, clock, done))
    sched.run_until(T + 0.2)
    assert o.status == "resting" and o.queue_ahead == 20        # 20 already bid at 0.50
    hub.apply_many([LastTrade("U", 0.50, 15, "SELL", T + 0.3)])
    assert o.filled == 0 and o.queue_ahead == 5
    hub.apply_many([LastTrade("U", 0.50, 10, "SELL", T + 0.4)])
    assert o.filled == 5                                        # queue exhausted, 5 reach us
    hub.apply_many([LastTrade("U", 0.49, 12, "SELL", T + 0.5)])  # printed below our bid
    assert o.filled == 17
    hub.apply_many([BookLevel("U", ASK, 0.50, 8, T + 0.6)])       # a seller now offers at our price
    assert o.filled == 25
    hub.apply_many([BookLevel("D", BID, 0.51, 9, T + 0.7)])       # Down bid 0.51 = Up at 0.49: mint
    assert o.filled == 30 and o.status == "done" and o.end_reason == "filled"
    assert o.avg_price == pytest.approx(0.50) and o.fee == 0 and o.maker_filled == 30


def test_maker_times_out_and_crossing_part_is_taker():
    c, clock, sched, hub, w, ex = _exchange(MAKER_TIMEOUT_MS=1000)
    done = []
    o = ex.submit(_order(w, "maker", 0.52, 15, clock, done))     # 10 available at 0.52 already
    sched.run_until(T + 0.2)
    assert o.taker_filled == 10 and o.status == "resting"
    sched.run_until(T + 5)
    assert done[0].end_reason == "maker_timeout" and done[0].filled == 10 and done[0].fee > 0


def test_orders_arriving_after_the_close_do_nothing():
    c, clock, sched, hub, w, ex = _exchange(PAPER_FILL_DELAY_MS=500)
    clock.advance_to(w.end_ts - 0.2)
    done = []
    ex.submit(_order(w, "taker", 0.60, 10, clock, done))
    sched.run_until(w.end_ts + 1)
    assert done[0].end_reason == "window_closed" and done[0].filled == 0


def test_max_taker_price_respects_min_edge_after_fee():
    p = max_taker_price(0.60, 0.01, 0.07, 1.0, 0.01, 0.70)
    assert p is not None and 0.60 - p - 0.07 * p * (1 - p) >= 0.01
    nxt = round(p + 0.01, 2)
    assert 0.60 - nxt - 0.07 * nxt * (1 - nxt) < 0.01
    assert max_taker_price(0.30, 0.01, 0.07, 1.0, 0.01, 0.50) == 0.27
    assert max_taker_price(0.005, 0.01, 0.07, 1.0, 0.01, 0.5) is None


# ================================================================ pipeline
def stack(tmp_path, **kw):
    base = dict(VOL_MIN_SAMPLES=100, BASIS_MIN_SAMPLES=10, KILL_SWITCH_FILE=str(tmp_path / "STOP"),
                EXECUTION_STYLE="taker", PRE_TRADE_REFRESH="ws")
    base.update(kw)
    c = cfg(**base)
    w, hub, eng, _sink = _primed(c)
    clock = hub.clock
    sched = ReplayScheduler(clock)
    pf = Portfolio.load_or_new(str(tmp_path / "state.json"), 1000.0)
    pf.roll_day(clock.now())
    limits = resolve_limits(c)
    risk = RiskManager(c, limits, pf, clock)
    ex = PaperExchange(c, hub, clock, sched)
    refresher = WsBookRefresher(hub, clock, sched, c.PAPER_REFRESH_LATENCY_MS)
    log = ExecutionLog(str(tmp_path), clock, c.LATENCY_BUDGET_MS)
    pipe = ExecutionPipeline(c, hub, eng, risk, pf, limits, ex, refresher, clock, log)
    return SimpleNamespace(c=c, w=w, hub=hub, eng=eng, clock=clock, sched=sched, pf=pf, risk=risk, ex=ex,
                           pipe=pipe, log=log, dir=tmp_path)


def orders(s):
    s.log.flush()
    with open(s.dir / "orders.csv", newline="") as f:
        return list(csv.DictReader(f))


def fire(s):
    s.eng._run(s.w, s.clock.now())


def test_taker_signal_is_refreshed_budgeted_sized_and_filled(tmp_path):
    s = stack(tmp_path)
    fire(s)
    assert ("btc-updown-5m-1790000000", "up") in s.pipe.active and s.pf.reserved > 0
    s.sched.run_until(s.clock.now() + 1)
    [row] = orders(s)
    assert row["status"] == "filled" and row["kind"] == "taker" and row["side"] == "up"
    assert float(row["latency_ms"]) == pytest.approx(60 + 10)         # refresh + submit overhead
    assert row["refresh_source"] == "ws" and float(row["book_age_ms"]) <= s.c.BOOK_MAX_AGE_MS
    pos = s.pf.positions["btc-updown-5m-1790000000:up"]
    # conservative: 1% of 1000 max bet, taker limit 0.53 -> unit cash 0.5474 -> 18 shares filled at 0.51
    assert pos.shares == 18 and pos.avg_price == pytest.approx(0.51) and row["sized_by"] == "max_bet"
    assert s.pf.reserved == 0 and not s.pipe.active
    assert s.pf.cash == pytest.approx(1000 - 18 * 0.51 - 18 * 0.07 * 0.51 * 0.49)


def test_latency_budget_breach_drops_the_signal(tmp_path):
    s = stack(tmp_path, PAPER_REFRESH_LATENCY_MS=400, LATENCY_BUDGET_MS=300)
    fire(s)
    s.sched.run_until(s.clock.now() + 1)
    [row] = orders(s)
    assert row["status"] == "dropped" and row["reason"] == "latency_budget_exceeded"
    assert float(row["latency_ms"]) == pytest.approx(410)
    assert not s.pf.positions and s.pf.reserved == 0


def test_signal_gone_after_refresh_is_not_sent(tmp_path):
    s = stack(tmp_path)
    fire(s)
    t = s.clock.now() + 0.03
    s.clock.advance_to(t)                                                # the book reprices before we send
    s.hub.apply_many([BookSnapshot("UP", [(0.97, 100)], [(0.99, 100)], t),
                      BookSnapshot("DOWN", [(0.01, 100)], [(0.03, 100)], t)])
    s.sched.run_until(t + 1)
    [row] = orders(s)
    assert row["status"] == "dropped" and row["reason"] == "signal_gone_after_refresh"
    assert not s.pf.positions


def test_stale_book_is_never_traded(tmp_path):
    s = stack(tmp_path, BOOK_MAX_AGE_MS=100, PAPER_REFRESH_LATENCY_MS=150, LATENCY_BUDGET_MS=500)
    fire(s)                                                              # book last changed 150 ms before send
    s.sched.run_until(s.clock.now() + 1)
    [row] = orders(s)
    assert row["reason"] == "stale_book" and not s.pf.positions


def test_maker_first_times_out_then_falls_back_to_taker_on_a_fresh_signal(tmp_path):
    s = stack(tmp_path, EXECUTION_STYLE="maker_first", MAKER_TIMEOUT_MS=1000)
    fire(s)
    s.sched.run_until(s.clock.now() + 3)
    rows = orders(s)
    assert [(r["kind"], r["status"], r["reason"]) for r in rows] == [
        ("maker", "unfilled", "maker_timeout"), ("taker", "filled", "filled")]
    assert rows[0]["limit"] == "0.5"                                      # 1 tick under the 0.51 ask
    assert rows[1]["plan"] == "taker_fallback" and rows[1]["signal_id"] != rows[0]["signal_id"]
    assert s.pf.positions["btc-updown-5m-1790000000:up"].fills == 1


def test_kill_switch_refuses_and_retry_wait_throttles(tmp_path):
    s = stack(tmp_path)
    fire(s)
    s.sched.run_until(s.clock.now() + 1)
    fire(s)
    ev = s.eng.evaluate(s.w, s.clock.now())
    assert s.pipe.on_signal(ev) == "retry_wait"
    (tmp_path / "STOP").write_text("")
    s.clock.advance_to(s.clock.now() + 3)
    ev = s.eng.evaluate(s.w, s.clock.now())
    assert s.pipe.on_signal(ev) == "kill_switch"


def test_settlement_books_realized_pnl_and_cooldown(tmp_path):
    s = stack(tmp_path)
    fire(s)
    s.sched.run_until(s.clock.now() + 1)
    s.clock.advance_to(s.w.end_ts + 30)
    s.pipe.on_outcome(s.w.slug, "down", s.clock.now())
    s.log.flush()
    [st] = list(csv.DictReader(open(tmp_path / "settlements.csv", newline="")))
    assert st["won"] == "0" and float(st["pnl"]) < 0 and not s.pf.positions
    assert s.pf.losses == 1 and s.pf.cooldowns["btc"] > s.clock.now()


def test_provisional_loss_counts_before_gamma_resolves(tmp_path):
    s = stack(tmp_path)
    fire(s)
    s.sched.run_until(s.clock.now() + 1)
    pos = s.pf.positions["btc-updown-5m-1790000000:up"]
    end = s.w.end_ts
    s.clock.advance_to(end + 0.5)
    s.hub.apply_many([OracleTick("chainlink", "btc", pos.ref_price * 0.995, end, end + 0.5)])   # closed 50 bp lower
    s.clock.advance_to(end + 3)
    s.pipe.on_timer(s.clock.now())
    assert pos.provisional_loss and s.pf.day_pnl() == pytest.approx(-pos.at_risk)


# ================================================================ REST refresh, breaker, replay --paper
def test_rest_refresher_reads_both_books_and_measures_latency():
    import asyncio
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import requests

    from latarb.clock import WallClock
    from latarb.execution.books import RestBookRefresher

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            token = self.path.split("token_id=")[1]
            body = {"asset_id": token, "timestamp": "1790000000000",
                    "bids": [{"price": "0.40", "size": "5"}, {"price": "0.45", "size": "7"}],   # not best-first
                    "asks": [{"price": "0.60", "size": "3"}, {"price": "0.55", "size": "4"}]}
            data = _json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = cfg(CLOB_URL=f"http://127.0.0.1:{srv.server_port}")
        hub = MarketDataHub(c, WallClock())
        w = MarketWindow(slug="w", asset="btc", label="5m", duration_s=300, start_ts=0, end_ts=1e12,
                         up_token="U", down_token="D", resolution=CHAINLINK)
        plain = requests.Session()
        plain.trust_env = False
        ref = RestBookRefresher(c, hub, WallClock(), session=plain)
        got = []

        async def go():
            ref.refresh(w, lambda fb, err: got.append((fb, err)))
            for _ in range(200):
                if got:
                    return
                await asyncio.sleep(0.01)
        asyncio.run(go())
        fb, err = got[0]
        assert err is None and fb.source == "rest"
        assert fb.up.best_ask() == (0.55, 4.0) and fb.down.best_bid() == (0.45, 7.0)
        assert 0 <= fb.latency_ms < 1000 and fb.age_ms("U", fb.received_ts) == pytest.approx(fb.latency_ms)
    finally:
        srv.shutdown()


def test_repeated_refresh_failures_trip_the_circuit_breaker(tmp_path):
    s = stack(tmp_path, MAX_CONSECUTIVE_ERRORS=2, EXEC_RETRY_S=0.1)

    class Failing:
        counts_as_error = True

        def refresh(self, w, done):
            done(None, "HTTP 503")

    s.pipe.refresher = Failing()
    for _ in range(2):
        fire(s)
        s.clock.advance_to(s.clock.now() + 0.2)
    assert s.risk.breaker_tripped
    assert s.pipe.on_signal(s.eng.evaluate(s.w, s.clock.now())) == "circuit_breaker"
    assert [r["reason"] for r in orders(s)] == ["refresh_failed", "refresh_failed"]
    assert s.pf.reserved == 0


def test_replay_paper_end_to_end_accounts_consistently(tmp_path):
    from latarb.app import run_replay
    from latarb.reporting import analyze

    from . import synthetic
    synthetic.write_session(tmp_path / "ticks")
    out = tmp_path / "out"
    live_stop = tmp_path / "STOP"
    live_stop.write_text("")               # the LIVE bot is kill-switched; a backtest must not care
    res = run_replay(cfg(EXECUTION_STYLE="taker", KILL_SWITCH_FILE=str(live_stop)),
                     [str(tmp_path / "ticks")], str(out), paper=True)
    p = res["paper"]
    assert p["open_positions"] == 0 and p["wins"] + p["losses"] >= 1
    assert p["cash"] == pytest.approx(1000.0 + p["realized"])          # everything settled back into cash
    sets = list(csv.DictReader(open(out / "settlements.csv", newline="")))
    assert sum(float(r["pnl"]) for r in sets) == pytest.approx(p["realized"], abs=1e-3)
    rep = analyze.run(str(out), out=lambda *a: None)
    assert rep["paper"]["settled"] == len(sets) and rep["paper"]["sent"] >= 1


def test_no_order_on_the_other_side_while_one_is_in_flight(tmp_path):
    s = stack(tmp_path)
    fire(s)                                                       # Up attempt now waiting for its refresh
    ev = s.eng.evaluate(s.w, s.clock.now())
    ev.best = ev.down                                             # pretend the model flipped to Down
    assert s.pipe.on_signal(ev) == "opposite_in_flight"

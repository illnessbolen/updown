"""End-to-end: recorded frames -> parsers -> hub -> engine -> CSV -> analysis."""
import csv

import pytest

from latarb.app import run_replay
from latarb.clock import ReplayClock
from latarb.config import load_settings
from latarb.data.events import BookSnapshot, OracleTick, SpotQuote
from latarb.data.hub import MarketDataHub
from latarb.data.reference import ReferenceResolver
from latarb.reporting import analyze
from latarb.signal.engine import GATED, NONE, SIGNAL, SignalEngine

from . import synthetic


def cfg(**kw):
    base = dict(ASSETS=("btc",), VOL_MIN_SAMPLES=300)
    base.update(kw)
    return load_settings(env={}, dotenv=False, **base)


def _signals(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_replay_finds_the_stale_book_after_a_spot_jump(tmp_path):
    synthetic.write_session(tmp_path / "ticks")
    out = tmp_path / "out"
    summary = run_replay(cfg(), [str(tmp_path / "ticks")], str(out))
    assert summary["unparseable"] == 0
    rows = _signals(out / "signals.csv")
    jump = synthetic.T0 + synthetic.JUMP_AT
    # an efficient market (book tracks fair value) produces no signal before the jump
    assert not [r for r in rows if float(r["ts"]) < jump]
    sig = [r for r in rows if r["decision"] == SIGNAL]
    assert sig, f"no signal; gates={summary['gates']} unpriced={summary['unpriced']}"
    assert all(r["side"] == "up" for r in sig)
    first = sig[0]
    assert first["ref_source"] == "chainlink_rtds" and first["oracle_proxy"] == "chainlink"
    assert float(first["p_fair"]) > float(first["p_market"]) + 0.03 and float(first["x_bps"]) > 10
    assert first["exec_status"] == ""
    # the market never catches up: once the move is older than the longest lookback, the book no
    # longer looks like a lagging copy of our model -> gated, not traded
    assert all(float(r["ts"]) <= jump + 5.5 for r in sig)
    late = [r for r in rows if float(r["ts"]) > jump + 6 and r["decision"] == GATED]
    assert late and all("edge_not_latency" in r["reasons"] for r in late)
    # at the jump Coinbase has not ticked yet: gated as unconfirmed, then signalled once it confirms
    at_jump = [r for r in rows if jump <= float(r["ts"]) < jump + 1]
    assert at_jump[0]["decision"] == GATED and at_jump[0]["reasons"] == "sanity_unconfirmed"
    assert 0 < float(first["ts"]) - float(at_jump[0]["ts"]) <= 0.5
    assert not any("sanity_divergence" in r["reasons"] for r in rows)
    # sigma is estimated, not assumed (true 1 bp/sqrt(s)); the market (Gaussian, true sigma) implies
    # about the same vol, and it is logged next to ours
    snaps = _signals(next(out.glob("snapshots-*.csv")))
    assert all(r["vol_scale_s"] for r in snaps)
    res = analyze.run(str(out), out=lambda *a: None)
    assert res["signals"]["resolved"] >= 1 and res["signals"]["wins"] >= 1


def test_efficient_market_gives_no_signal_and_its_implied_vol_matches_ours(tmp_path):
    # no jump: the book tracks the true fair value for the whole window
    synthetic.write_session(tmp_path / "ticks", jump_bp=0.0)
    out = tmp_path / "out"
    run_replay(cfg(), [str(tmp_path / "ticks")], str(out))
    assert not [r for r in _signals(out / "signals.csv") if r["decision"] == SIGNAL]
    snaps = _signals(next(out.glob("snapshots-*.csv")))
    assert all(0.8 < float(r["sigma_bps"]) < 1.3 for r in snaps)             # truth: 1 bp/sqrt(s)
    # (the synthetic market clips at 0.03/0.97 and rounds to 1c: near the clip the implied vol is biased)
    implied = sorted(float(r["sigma_implied_bps"]) for r in snaps
                     if r["sigma_implied_bps"] and 0.1 < float(r["mkt_mid_up"]) < 0.9)
    assert len(implied) >= 5 and 0.7 < implied[len(implied) // 2] < 1.4
    rep = analyze.run(str(out), out=lambda *a: None)["model_vs_market"]
    assert rep["x_gap_abs_median_bps"] < 2.0


def test_coinbase_divergence_gates_the_whole_window(tmp_path):
    synthetic.write_session(tmp_path / "ticks", coinbase_glitch_bp=80.0)
    out = tmp_path / "out"
    summary = run_replay(cfg(), [str(tmp_path / "ticks")], str(out))
    rows = _signals(out / "signals.csv")
    glitch_from = synthetic.T0 + synthetic.JUMP_AT - 5   # Coinbase prints +80 bp away from here on
    after = [r for r in rows if float(r["ts"]) >= glitch_from + 0.5]
    assert after and not [r for r in after if r["decision"] == SIGNAL]
    assert all("sanity_divergence" in r["reasons"] for r in after)
    assert summary["gates"].get("sanity_divergence", 0) > 0


def test_silent_polymarket_connection_is_stale(tmp_path):
    synthetic.write_session(tmp_path / "ticks", poly_pongs=False, poly_active=False)
    out = tmp_path / "out"
    summary = run_replay(cfg(), [str(tmp_path / "ticks")], str(out))
    assert not [r for r in _signals(out / "signals.csv") if r["decision"] == SIGNAL]
    assert summary["gates"].get("book_stale", 0) > 0


# ---------------------------------------------------------------- engine unit tests
class ListSink:
    def __init__(self):
        self.signals, self.snaps = [], []

    def signal(self, ev, sid):
        self.signals.append(ev)

    def snapshot(self, ev):
        self.snaps.append(ev)


def _primed(c, *, up_book=(0.49, 0.51), down_book=(0.49, 0.51), spot=100_000.0, oracle_mult=1.0, jump=0.002):
    """Hub with a warmed-up vol estimator, oracle basis and a synced book, at T0 + 250.
    Spot jumps +20 bp at T0 + 247: recent enough for the latency check (5 s lookback) to see
    that the book still reflects the pre-jump price."""
    w = synthetic.window()
    clock = ReplayClock(w.start_ts - 400)
    hub = MarketDataHub(c, clock)
    hub.set_markets([w])
    refs = ReferenceResolver(c, hub)
    sink = ListSink()
    eng = SignalEngine(c, hub, refs, clock, sink)
    t = w.start_ts - 400
    k = 0
    while t < w.start_ts + 250:
        t += 0.5
        k += 1
        clock.advance_to(t)
        px = spot * (1 + (1e-5 if k % 2 else -1e-5)) if t < w.start_ts + 247 else spot * (1 + jump)
        hub.apply_many([SpotQuote("binance", "btc", px - 0.5, px + 0.5, t),
                        SpotQuote("coinbase", "btc", px - 0.5, px + 0.5, t)])
        if abs(t - round(t)) < 1e-9:
            hub.apply_many([OracleTick("chainlink", "btc", px * oracle_mult, t, t)])
            hub.note_message("polymarket", t)
    hub.on_feed_open("polymarket", t)
    hub.apply_many([BookSnapshot("UP", [(up_book[0], 100)], [(up_book[1], 100)], t),
                    BookSnapshot("DOWN", [(down_book[0], 100)], [(down_book[1], 100)], t)])
    return w, hub, eng, sink


def test_engine_signals_on_a_mispriced_book_and_respects_fees():
    c = cfg(VOL_MIN_SAMPLES=100, BASIS_MIN_SAMPLES=10)
    w, hub, eng, sink = _primed(c)
    ev = eng.evaluate(w)
    assert ev is not None and ev.decision == SIGNAL and ev.best.side == "up"
    b = ev.best
    assert b.fee == pytest.approx(0.07 * b.ask * (1 - b.ask))
    assert b.edge == pytest.approx(b.p_fair - b.ask - b.fee - c.EXPECTED_SLIPPAGE)
    assert b.edge_cons <= b.edge
    assert ev.p_up_lo <= ev.p_up <= ev.p_up_hi


def test_engine_uses_the_complement_book_when_it_is_cheaper():
    c = cfg(VOL_MIN_SAMPLES=100, BASIS_MIN_SAMPLES=10)
    # Up asks 0.60 on its own book, but someone bids 0.55 for Down -> Up is buyable at 0.45
    w, hub, eng, sink = _primed(c, up_book=(0.40, 0.60), down_book=(0.55, 0.62))
    ev = eng.evaluate(w)
    assert ev.up.ask == pytest.approx(0.45) and ev.up.ask_src == "complement"


def test_engine_does_not_signal_when_the_book_already_repriced():
    c = cfg(VOL_MIN_SAMPLES=100, BASIS_MIN_SAMPLES=10)
    w, hub, eng, sink = _primed(c, up_book=(0.97, 0.99), down_book=(0.01, 0.03))
    ev = eng.evaluate(w)
    assert ev.decision == NONE


def test_engine_gates_near_the_close_and_on_clock_skew():
    c = cfg(VOL_MIN_SAMPLES=100, BASIS_MIN_SAMPLES=10, MIN_TIME_LEFT_S=60)
    w, hub, eng, sink = _primed(c)
    hub.clock_skew_ms = 900.0
    ev = eng.evaluate(w)
    assert ev.decision == GATED and "min_time_left" in ev.reasons and "clock_skew" in ev.reasons


def test_engine_refuses_to_price_without_reference():
    c = cfg(VOL_MIN_SAMPLES=100, BASIS_MIN_SAMPLES=10, ORACLE_HISTORY_S=60)
    w, hub, _, _ = _primed(c)             # oracle history too short to still hold the start tick
    late = SignalEngine(c, hub, ReferenceResolver(c, hub), hub.clock, ListSink())   # a bot started late
    assert late.evaluate(w) is None
    assert late.stats.unpriced["reference_missing"] == 1


def test_market_ahead_of_our_feed_is_not_a_latency_signal():
    # the screenshot case: our spot is flat at K (P ~ 0.5) but the market already prices Up at 0.36,
    # i.e. it knows about a drop our feed has not shown yet. Our model "sees" cheap Up; the market
    # matches no recent version of the model, so this must not trade.
    c = cfg(VOL_MIN_SAMPLES=100, BASIS_MIN_SAMPLES=10)
    w, hub, eng, sink = _primed(c, up_book=(0.35, 0.37), down_book=(0.63, 0.65), jump=0.0)
    ev = eng.evaluate(w)
    assert ev.best.side == "up" and ev.best.edge_cons > c.EDGE_THRESHOLD
    assert ev.decision == GATED and "edge_not_latency" in ev.reasons
    assert ev.lag_err > c.LATENCY_MATCH_TOL
    ok = eng.evaluate(w)                     # the gate can be switched off, and then it is a SIGNAL
    eng.cfg = cfg(VOL_MIN_SAMPLES=100, BASIS_MIN_SAMPLES=10, REQUIRE_LATENCY_EXPLANATION=False)
    assert eng.evaluate(w).decision == SIGNAL and ok.decision == GATED


def test_market_mid_uses_the_complement_book():
    from latarb.data.orderbook import OrderBook
    from latarb.signal.engine import market_mid_up
    up, dn = OrderBook("U"), OrderBook("D")
    up.apply_snapshot([(0.40, 1)], [(0.60, 1)], 0)
    dn.apply_snapshot([(0.55, 1)], [(0.62, 1)], 0)
    mid, half = market_mid_up(up, dn)       # bid max(0.40, 1-0.62) = 0.40, ask min(0.60, 1-0.55) = 0.45
    assert mid == pytest.approx(0.425) and half == pytest.approx(0.025)

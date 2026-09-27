"""Parsers, order book, slug grammar, Gamma parsing, hub state, references, recorder."""
import json
import math
import random
from datetime import datetime, timezone

import pytest

from latarb.clock import ReplayClock
from latarb.config import load_settings
from latarb.data.events import ASK, BID, BookLevel, BookSnapshot, OracleTick, SpotQuote, SpotTrade
from latarb.data.gamma import Discovery
from latarb.data.hub import MarketDataHub
from latarb.data.markets import (BINANCE, CHAINLINK, UNKNOWN, MarketWindow, parse_slug, windows_from_event,
                                 winner_from_market)
from latarb.data.orderbook import OrderBook
from latarb.data.parsers import (BinanceParser, CoinbaseParser, RtdsChainlinkParser, parse_iso_ts,
                                 parse_polymarket, parse_safely)
from latarb.data.recorder import TickRecorder, iter_recording
from latarb.data.reference import ReferenceResolver
from latarb.data.timeseries import TimeSeries
from latarb.model.volatility import RealizedVol


def cfg(**kw):
    return load_settings(env={}, dotenv=False, **kw)


# ---------------------------------------------------------------- parsers
def test_binance_book_ticker_and_trade():
    p = BinanceParser({"btc": "BTCUSDT"})
    q = p(json.dumps({"stream": "btcusdt@bookTicker",
                      "data": {"u": 1, "s": "BTCUSDT", "b": "100.10", "B": "1", "a": "100.20", "A": "2"}}), 5.0)
    assert len(q) == 1 and isinstance(q[0], SpotQuote)
    assert q[0].asset == "btc" and q[0].mid == pytest.approx(100.15) and q[0].recv_ts == 5.0
    t = p(json.dumps({"stream": "btcusdt@trade", "data": {"e": "trade", "E": 1700000000123, "s": "BTCUSDT",
                                                          "p": "100.15", "q": "0.5", "T": 1700000000120}}), 6.0)
    assert isinstance(t[0], SpotTrade) and t[0].exch_ts == pytest.approx(1700000000.120)
    assert p(json.dumps({"stream": "ethusdt@bookTicker", "data": {"s": "ETHUSDT", "b": "1", "a": "2"}}), 0) == []
    # crossed quote is rejected
    assert p(json.dumps({"data": {"s": "BTCUSDT", "b": "101", "a": "100", "u": 1}}), 0) == []


def test_coinbase_ticker():
    p = CoinbaseParser({"btc": "BTC-USD"})
    out = p(json.dumps({"type": "ticker", "product_id": "BTC-USD", "price": "100.5", "best_bid": "100.4",
                        "best_ask": "100.6", "time": "2025-01-02T03:04:05.123456789Z", "last_size": "0.1"}), 1.0)
    quote = [e for e in out if isinstance(e, SpotQuote)][0]
    assert quote.mid == pytest.approx(100.5)
    assert quote.exch_ts == pytest.approx(datetime(2025, 1, 2, 3, 4, 5, 123456, timezone.utc).timestamp())
    assert p(json.dumps({"type": "heartbeat", "product_id": "BTC-USD"}), 1.0) == []


def test_rtds_chainlink_update_and_backfill():
    p = RtdsChainlinkParser({"btc": "btc/usd"})
    one = p(json.dumps({"topic": "crypto_prices_chainlink", "type": "update", "timestamp": 1700000001000,
                        "payload": {"symbol": "btc/usd", "timestamp": 1700000000000, "value": 67000.5}}), 9.0)
    assert one[0].price == 67000.5 and one[0].ts == 1700000000.0 and one[0].recv_ts == 9.0
    many = p(json.dumps({"topic": "crypto_prices_chainlink", "payload": {
        "symbol": "btc/usd", "data": [{"timestamp": 1700000000000, "value": 1.0},
                                      {"timestamp": 1700000001000, "value": 2.0}]}}), 9.0)
    assert [e.price for e in many] == [1.0, 2.0]
    assert p("PONG", 1.0) == [] and p(json.dumps({"topic": "crypto_prices", "payload": {}}), 1.0) == []


def test_polymarket_book_and_both_price_change_layouts():
    snap = parse_polymarket(json.dumps([{"event_type": "book", "asset_id": "T1", "timestamp": "1700000000000",
                                         "bids": [{"price": ".48", "size": "30"}],
                                         "asks": [{"price": ".52", "size": "25"}]}]), 1.0)
    assert isinstance(snap[0], BookSnapshot) and snap[0].bids == [(0.48, 30.0)] and snap[0].asks == [(0.52, 25.0)]
    new = parse_polymarket(json.dumps({"event_type": "price_change", "market": "0x1", "price_changes": [
        {"asset_id": "T1", "price": "0.5", "size": "10", "side": "BUY"},
        {"asset_id": "T2", "price": "0.51", "size": "0", "side": "SELL"}]}), 2.0)
    assert [(e.token, e.side, e.price, e.size) for e in new] == [("T1", BID, 0.5, 10.0), ("T2", ASK, 0.51, 0.0)]
    old = parse_polymarket(json.dumps({"event_type": "price_change", "asset_id": "T1",
                                       "changes": [{"price": "0.53", "side": "SELL", "size": "7"}]}), 3.0)
    assert [(e.token, e.side, e.price) for e in old] == [("T1", ASK, 0.53)]
    assert parse_polymarket("PONG", 1.0) == []


def test_parse_safely_swallows_malformed_frames():
    errs = []
    assert parse_safely(BinanceParser({"btc": "BTCUSDT"}), '{"data": {"s": "BTCUSDT", "b": "x", "a": "1"}}', 0, errs) == []
    assert errs and errs[0].startswith("ValueError")


def test_iso_parsing():
    assert parse_iso_ts("2026-09-27T14:00:00Z") == datetime(2026, 9, 27, 14, tzinfo=timezone.utc).timestamp()
    assert parse_iso_ts("2026-09-27T14:00:00") is None       # naive timestamps are refused
    assert parse_iso_ts("") is None


# ---------------------------------------------------------------- order book
def test_order_book_levels_and_vwap():
    b = OrderBook("T")
    b.apply_snapshot([(0.40, 10), (0.45, 5)], [(0.50, 3), (0.55, 10), (0.60, 0)], 1.0)
    assert b.best_bid() == (0.45, 5) and b.best_ask() == (0.50, 3)
    b.apply_level(ASK, 0.50, 0, 2.0)           # level removed
    assert b.best_ask() == (0.55, 10)
    b.apply_level(ASK, 0.49, 4, 3.0)
    assert b.best_ask() == (0.49, 4)
    assert b.vwap_buy(8) == pytest.approx((4 * 0.49 + 4 * 0.55) / 8)
    assert b.vwap_buy(100) is None
    assert not b.crossed
    b.apply_level(BID, 0.60, 1, 4.0)
    assert b.crossed


# ---------------------------------------------------------------- slugs & gamma
def test_parse_unix_slugs():
    i = parse_slug("btc-updown-5m-1790000100")
    assert (i.asset, i.label, i.duration_s, i.start_ts, i.kind) == ("btc", "5m", 300, 1790000100.0, "unix")
    assert parse_slug("eth-updown-4h-1790006400").duration_s == 14400
    assert parse_slug("eth-updown-15m-1790000100").duration_s == 900
    assert parse_slug("foo-updown-5m-1790000100") is None


def test_parse_hourly_et_slug_is_dst_aware():
    summer = parse_slug("bitcoin-up-or-down-september-27-2026-10am-et")
    assert summer.start_ts == datetime(2026, 9, 27, 14, tzinfo=timezone.utc).timestamp()   # EDT = UTC-4
    winter = parse_slug("ethereum-up-or-down-january-15-2026-3pm-et")
    assert winter.start_ts == datetime(2026, 1, 15, 20, tzinfo=timezone.utc).timestamp()   # EST = UTC-5
    midnight = parse_slug("solana-up-or-down-march-2-2026-12am-et")
    assert midnight.start_ts == datetime(2026, 3, 2, 5, tzinfo=timezone.utc).timestamp()
    # year omitted -> inferred from the end-date hint, including across new year
    hint = datetime(2027, 1, 1, 1, tzinfo=timezone.utc).timestamp()
    nye = parse_slug("bitcoin-up-or-down-december-31-7pm-et", hint)
    assert nye.start_ts == datetime(2027, 1, 1, 0, tzinfo=timezone.utc).timestamp()
    assert parse_slug("bitcoin-up-or-down-december-31-7pm-et") is None      # no year, no hint


def test_parse_daily_slug():
    i = parse_slug("bitcoin-up-or-down-on-september-27")
    assert (i.asset, i.label, i.duration_s, i.start_ts, i.kind) == ("btc", "1d", 86400, None, "daily")


def _event(slug, end_iso, outcomes=("Up", "Down"), tokens=("111", "222"), **market):
    m = {"slug": slug, "endDate": end_iso, "outcomes": json.dumps(list(outcomes)),
         "clobTokenIds": json.dumps(list(tokens)), "conditionId": "0xc", "orderPriceMinTickSize": 0.01,
         "orderMinSize": 5, "closed": False, "acceptingOrders": True}
    m.update(market)
    return {"slug": slug, "title": "Bitcoin Up or Down", "markets": [m]}


def test_windows_from_event_maps_tokens_by_outcome_name():
    ev = _event("btc-updown-5m-1790000100", "2026-09-21T14:20:00Z", outcomes=("Down", "Up"),
                resolutionSource="https://data.chain.link/streams/btc-usd")
    ws, rej = windows_from_event(ev, ["btc"])
    assert rej == [] and len(ws) == 1
    w = ws[0]
    assert (w.up_token, w.down_token) == ("222", "111")
    assert w.resolution == CHAINLINK and w.start_ts == 1790000100 and w.end_ts == 1790000400


def test_windows_from_event_rejects_inconsistent_end_and_filters_assets():
    ev = _event("btc-updown-5m-1790000100", "2026-09-21T15:00:00Z")
    ws, rej = windows_from_event(ev, ["btc"])
    assert ws == [] and rej[0][1].startswith("end_mismatch")
    assert windows_from_event(_event("btc-updown-5m-1790000100", "2026-09-21T14:20:00Z"), ["eth"]) == ([], [])


def test_hourly_event_resolution_and_fee_schedule():
    ev = _event("bitcoin-up-or-down-september-27-2026-10am-et", "2026-09-27T15:00:00Z",
                description="...resolution source is Binance, BTC/USDT 1 hour candle...",
                feeSchedule={"rate": 0.05, "exponent": 1})
    w = windows_from_event(ev, ["btc"])[0][0]
    assert w.resolution == BINANCE and w.duration_s == 3600 and w.taker_fee_rate == 0.05
    assert w.start_ts == datetime(2026, 9, 27, 14, tzinfo=timezone.utc).timestamp()


def test_daily_event_uses_gamma_end_date():
    ev = _event("bitcoin-up-or-down-on-september-27", "2026-09-27T16:00:00Z")
    w = windows_from_event(ev, ["btc"])[0][0]
    assert w.end_ts - w.start_ts == 86400 and w.resolution == UNKNOWN


def test_winner_detection():
    closed = {"closed": True, "outcomes": '["Up","Down"]', "outcomePrices": '["0","1"]'}
    assert winner_from_market(closed) == "down"
    assert winner_from_market(dict(closed, outcomePrices='["0.6","0.4"]')) is None
    assert winner_from_market(dict(closed, closed=False)) is None


class FakeGamma:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def event_by_slug(self, slug):
        self.calls.append(("slug", slug))
        return next((e for e in self.events if e["slug"] == slug), None)

    def get_events(self, **params):
        self.calls.append(("scan", params["offset"]))
        return [e for e in self.events if "updown" not in e["slug"]]


def test_discovery_probes_scans_and_caches():
    now = 1790000130.0
    iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = [_event("btc-updown-5m-1790000100", iso(1790000400)),
              _event("bitcoin-up-or-down-september-21-2026-10am-et", iso(1789999200 + 3600),
                     description="Binance 1 hour candle")]
    fake = FakeGamma(events)
    d = Discovery(cfg(ASSETS=("btc",), DETERMINISTIC_LABELS=("5m",)), fake, ReplayClock(now))
    got = d.refresh()
    assert set(got) == {"btc-updown-5m-1790000100", "bitcoin-up-or-down-september-21-2026-10am-et"}
    n_calls = len(fake.calls)
    d.refresh()                                  # cached: no new probe, no new scan
    assert len(fake.calls) == n_calls
    d.clock.advance_to(1789999200 + 3600 + 200)  # both windows over -> pruned
    d.refresh()
    assert "btc-updown-5m-1790000100" not in d.windows
    assert "bitcoin-up-or-down-september-21-2026-10am-et" not in d.windows


# ---------------------------------------------------------------- time series & vol
def test_timeseries_lookup_and_pruning():
    ts = TimeSeries(max_age_s=10)
    for t in range(100):
        ts.append(float(t), float(t) * 2)
    assert ts.at(95.5) == (95.0, 190.0)
    assert ts.at(50.0) is None                   # pruned
    assert ts.nearest(97.4, 0.5) == (97.0, 194.0)
    assert ts.nearest(97.5, 0.5) == (98.0, 196.0)   # tie -> later sample
    assert ts.nearest(120.0, 1.0) is None


def test_realized_vol_recovers_known_sigma():
    rng = random.Random(7)
    sigma = 2e-4                                 # per sqrt(second)
    v = RealizedVol(sample_s=1.0, fast_halflife_s=300, slow_halflife_s=3000, min_samples=300, max_gap_s=5)
    px, t = 100.0, 0.0
    for _ in range(20000):                       # 10 ticks per second
        t += 0.1
        px *= math.exp(rng.gauss(0.0, sigma * math.sqrt(0.1)))
        v.update(t, px)
    assert v.ready
    fast, slow = v.sigmas()
    assert fast == pytest.approx(sigma, rel=0.12)
    assert slow == pytest.approx(sigma, rel=0.12)
    lo, mid, hi = v.sigma_band(30.0, 120.0, 0.2)
    assert lo < mid < hi


def test_realized_vol_does_not_book_an_outage_as_a_jump():
    v = RealizedVol(sample_s=1.0, fast_halflife_s=60, slow_halflife_s=600, min_samples=10, max_gap_s=5)
    for t in range(100):
        v.update(float(t), 100.0)
    v.update(200.0, 150.0)                       # 100 s outage, then a very different price
    assert v.gaps == 1 and v.sigmas()[0] == 0.0


# ---------------------------------------------------------------- hub & references
def _window(**kw):
    base = dict(slug="btc-updown-5m-1000200", asset="btc", label="5m", duration_s=300,
                start_ts=1000200.0, end_ts=1000500.0, up_token="U", down_token="D", resolution=CHAINLINK)
    base.update(kw)
    return MarketWindow(**base)


def test_hub_tracks_oracle_basis_and_books():
    c = cfg(ASSETS=("btc",), BASIS_MIN_SAMPLES=5)
    clock = ReplayClock(1000000.0)
    hub = MarketDataHub(c, clock)
    hub.set_markets([_window()])
    touched = []
    hub.spot_listeners.append(lambda a, ts: touched.append(a))
    for i in range(20):
        t = 1000000.0 + i
        hub.apply_many([SpotQuote("binance", "btc", 99.99, 100.01, t)])
        hub.apply_many([OracleTick("chainlink", "btc", 100.05, t, t + 0.3)])
    st = hub.assets["btc"]
    assert st.oracle_basis.ready and st.oracle_basis.mean == pytest.approx(math.log(100.05 / 100.0), abs=1e-9)
    assert touched == ["btc"]                    # identical quotes after the first do not re-trigger pricing
    hub.apply_many([BookLevel("U", ASK, 0.5, 10, 1.0)])
    assert hub.books["U"].best_ask() is None     # delta before snapshot ignored
    hub.apply_many([BookSnapshot("U", [(0.4, 1)], [(0.6, 1)], 2.0)])
    assert hub.books["U"].best_ask() == (0.6, 1)
    hub.on_feed_open("polymarket", 3.0)
    assert not hub.books["U"].synced


def test_chainlink_reference_waits_for_start_tick_then_caches():
    c = cfg(ASSETS=("btc",))
    hub = MarketDataHub(c, ReplayClock())
    w = _window()
    refs = ReferenceResolver(c, hub)
    assert refs.get(w, w.start_ts - 1)[1] == "window_not_started"
    assert refs.get(w, w.start_ts + 0.5)[1] == "reference_pending"
    hub.apply_many([OracleTick("chainlink", "btc", 101.0, w.start_ts - 1, w.start_ts - 0.5),
                    OracleTick("chainlink", "btc", 102.0, w.start_ts, w.start_ts + 0.8)])
    ref, why = refs.get(w, w.start_ts + 1.0)
    assert why == "" and ref.price == 102.0 and ref.source == "chainlink_rtds"


def test_chainlink_reference_missing_when_joined_late():
    c = cfg(ASSETS=("btc",))
    hub = MarketDataHub(c, ReplayClock())
    w = _window()
    hub.apply_many([OracleTick("chainlink", "btc", 101.0, w.start_ts + 60, w.start_ts + 60)])
    assert ReferenceResolver(c, hub).get(w, w.start_ts + 61)[1] == "reference_missing"


def test_binance_reference_from_trade_stream_then_kline():
    c = cfg(ASSETS=("btc",))
    hub = MarketDataHub(c, ReplayClock())
    w = _window(slug="bitcoin-up-or-down-x", label="1h", duration_s=3600, start_ts=3600.0 * 500,
                end_ts=3600.0 * 501, resolution=BINANCE)
    refs = ReferenceResolver(c, hub)
    s = w.start_ts
    hub.apply_many([SpotTrade("binance", "btc", 99.0, 1, s - 2, s - 2.5),     # previous minute
                    SpotTrade("binance", "btc", 100.0, 1, s + 0.2, s + 0.1),  # first trade of the candle
                    SpotTrade("binance", "btc", 101.0, 1, s + 0.4, s + 0.3)])
    ref, _ = refs.get(w, s + 1)
    assert ref.price == 100.0 and ref.source == "binance_trade_stream" and not ref.final
    assert [j.slug for j in refs.rest_jobs(s + 1.5)] == [w.slug]
    refs.set_rest_result(w, 100.0)
    assert refs.get(w, s + 2)[0].source == "binance_kline"


def test_binance_reference_unknown_rule_for_misaligned_daily():
    c = cfg(ASSETS=("btc",))
    hub = MarketDataHub(c, ReplayClock())
    w = _window(label="1d", duration_s=86400, start_ts=86400.0 * 100 + 16 * 3600,
                end_ts=86400.0 * 101 + 16 * 3600, resolution=BINANCE)
    assert ReferenceResolver(c, hub).get(w, w.start_ts + 5)[1] == "reference_rule_unknown"


# ---------------------------------------------------------------- recorder
def test_recorder_roundtrip(tmp_path):
    rec = TickRecorder(str(tmp_path), rotate_s=3600)
    rec.write(100.0, "binance", '{"a":\n1}')
    rec.write(3700.5, "@open", "polymarket")      # next hour -> new file
    rec.close()
    rows = list(iter_recording([str(tmp_path)]))
    assert rows == [(100.0, "binance", '{"a": 1}'), (3700.5, "@open", "polymarket")]
    assert len(list(tmp_path.iterdir())) == 2


def test_discovery_aborts_the_round_when_gamma_is_unreachable():
    import requests

    class DownGamma:
        calls = 0

        def event_by_slug(self, slug):
            DownGamma.calls += 1
            raise requests.ConnectionError("proxy said no")

        def get_events(self, **params):
            DownGamma.calls += 1
            raise requests.ConnectionError("proxy said no")

    d = Discovery(cfg(ASSETS=("btc", "eth"), DETERMINISTIC_LABELS=("5m", "15m")), DownGamma(), ReplayClock(1790000130.0))
    assert d.refresh() == {}
    assert DownGamma.calls == 1          # one failed probe, no pile-up of retries, no scan


# ---------------------------------------------------------------- multi-scale volatility
def _msv(min_obs=20):
    from latarb.model.volatility import MultiScaleVol
    return MultiScaleVol(RealizedVol(1.0, 120, 1800, 100, 5.0), (10.0, 60.0), 3600.0, min_obs, 5.0)


def test_multiscale_picks_the_scale_closest_to_the_horizon_and_waits_for_it():
    rng = random.Random(1)
    v = _msv()
    px, t = 100.0, 0.0
    while t < 700:                           # 700 s: 1 s and 10 s ready, 60 s has only 11 returns
        t += 0.25
        px *= math.exp(rng.gauss(0, 1e-4 * math.sqrt(0.25)))
        v.update(t, px)
    est, why = v.estimate(5.0, 120.0, 0.2)
    assert why == "" and est.scale_s == 1.0
    est, _ = v.estimate(45.0, 120.0, 0.2)
    assert est.scale_s == 10.0 and est.lo < est.point < est.hi
    assert v.estimate(90.0, 120.0, 0.2)[0].scale_s == 10.0          # 60 s not needed below 2 x 60 s
    assert v.estimate(240.0, 120.0, 0.2) == (None, "vol_warmup_long")  # a 4-minute horizon needs it
    while t < 1500:
        t += 0.25
        px *= math.exp(rng.gauss(0, 1e-4 * math.sqrt(0.25)))
        v.update(t, px)
    est, _ = v.estimate(240.0, 120.0, 0.2)
    assert est.scale_s == 60.0 and 0.6e-4 < est.point < 1.5e-4


def test_multiscale_sees_trending_moves_that_1s_returns_understate():
    # a staircase: flat for 9 s, then a 3 bp step, always in the same direction within each minute.
    # 1 s returns see rare jumps; 60 s returns see the full minute-scale drift of the path.
    v = _msv(min_obs=10)
    px, t, sign = 100.0, 0.0, 1
    while t < 3000:
        t += 1.0
        if int(t) % 60 == 0:
            sign = -sign
        if int(t) % 10 == 0:
            px *= math.exp(sign * 3e-4)
        v.update(t, px)
    sig = v.scale_sigmas()
    assert sig[60.0] > 1.5 * sig[1.0]
    est, _ = v.estimate(240.0, 120.0, 0.0)
    assert est.point == pytest.approx(sig[60.0]) and est.hi >= sig[60.0] and est.lo <= sig[1.0] * 1.01

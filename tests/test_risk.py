"""Risk profiles, hard bounds, Kelly sizing, paper ledger and pre-trade checks."""
import json

import pytest

from latarb.clock import ReplayClock
from latarb.config import ConfigError, load_settings
from latarb.data.markets import CHAINLINK, MarketWindow
from latarb.risk.limits import HARD_BOUNDS, PROFILES, RiskLimits, resolve_limits
from latarb.risk.manager import RiskManager
from latarb.risk.portfolio import Portfolio
from latarb.risk.sizing import kelly_binary, size_order


def cfg(**kw):
    return load_settings(env={}, dotenv=False, **kw)


def win(slug="btc-updown-5m-1790000000", asset="btc", start=1790000000.0):
    return MarketWindow(slug=slug, asset=asset, label="5m", duration_s=300, start_ts=start, end_ts=start + 300,
                        up_token="U", down_token="D", resolution=CHAINLINK)


LIM = RiskLimits("test", bet_pct=0.02, exposure_pct=0.15, daily_stop_pct=0.05, kelly_fraction=0.25,
                 max_fills_per_side=2)


# ---------------------------------------------------------------- profiles
def test_profiles_are_ordered_and_inside_hard_bounds():
    for name, p in PROFILES.items():
        for k, v in p.items():
            lo, hi = HARD_BOUNDS[k]
            assert lo <= v <= hi, (name, k)
    c, m, a = (PROFILES[n] for n in ("conservative", "moderate", "aggressive"))
    for k in ("bet_pct", "exposure_pct", "daily_stop_pct", "kelly_fraction"):
        assert c[k] < m[k] < a[k]
    assert a["kelly_fraction"] <= 0.5


def test_overrides_inside_bounds_apply_outside_bounds_abort():
    lim = resolve_limits(cfg(RISK_PROFILE="moderate", RISK_BET_PCT=0.01))
    assert lim.bet_pct == 0.01 and lim.exposure_pct == PROFILES["moderate"]["exposure_pct"]
    assert lim.overridden == ("bet_pct",)
    for kw in ({"RISK_BET_PCT": 0.2}, {"RISK_DAILY_STOP_PCT": 0.5}, {"RISK_KELLY_FRACTION": 1.0},
               {"RISK_EXPOSURE_PCT": 0.9}, {"RISK_MAX_FILLS_PER_SIDE": 50}):
        with pytest.raises(ConfigError):
            resolve_limits(cfg(**kw))
    with pytest.raises(ConfigError):
        resolve_limits(cfg(RISK_PROFILE="yolo"))


# ---------------------------------------------------------------- Kelly
def test_kelly_formula():
    assert kelly_binary(0.6, 0.5) == pytest.approx(0.2)            # (0.6-0.5)/(1-0.5)
    assert kelly_binary(0.99, 0.95) == pytest.approx(0.8)
    assert kelly_binary(0.5, 0.55) == 0.0
    assert kelly_binary(0.7, 1.0) == 0.0


def test_stake_grows_with_edge_inside_the_caps():
    stakes = []
    for p in (0.54, 0.55, 0.56):          # Kelly stakes 5.3 / 10.6 / 16.0 USDC, all under the 20 cap
        d, why = size_order(p, 0.53, 0.53, 1000.0, LIM, exposure_room=150.0, min_shares=5, min_usdc=1.0)
        stakes.append(d.reserve)
        assert d.sized_by == "kelly"
        assert d.kelly_stake == pytest.approx(0.25 * (p - 0.53) / 0.47 * 1000.0)
    assert stakes[0] < stakes[1] < stakes[2]


def test_kelly_never_exceeds_max_bet_or_exposure_room():
    d, _ = size_order(0.95, 0.50, 0.52, 1000.0, LIM, exposure_room=150.0, min_shares=5, min_usdc=1.0)
    assert d.sized_by == "max_bet" and d.reserve <= 0.02 * 1000.0 + 1e-9
    d, _ = size_order(0.95, 0.50, 0.52, 1000.0, LIM, exposure_room=12.0, min_shares=5, min_usdc=1.0)
    assert d.sized_by == "exposure" and d.reserve <= 12.0


def test_exchange_minimum_is_used_only_if_it_fits_under_the_cap():
    # tiny edge -> Kelly stake 1.04 USDC = 2 shares; the minimum (5 * 0.51) fits under the 20 USDC cap
    d, _ = size_order(0.522, 0.52, 0.51, 1000.0, LIM, exposure_room=150.0, min_shares=5, min_usdc=1.0)
    assert d.shares == 5 and d.sized_by == "exchange_min"
    # small bankroll: 5 shares at 0.51 = 2.55 > 2% of 100 -> skip instead of breaking the cap
    d, why = size_order(0.53, 0.52, 0.51, 100.0, LIM, exposure_room=15.0, min_shares=5, min_usdc=1.0)
    assert d is None and why == "below_exchange_min"
    d, why = size_order(0.50, 0.52, 0.51, 1000.0, LIM, exposure_room=150.0, min_shares=5, min_usdc=1.0)
    assert d is None and why == "no_kelly_edge"


# ---------------------------------------------------------------- portfolio
def test_portfolio_fill_settle_and_persistence(tmp_path):
    path = str(tmp_path / "state.json")
    pf = Portfolio.load_or_new(path, 1000.0)
    pf.roll_day(1790000000.0)
    w = win()
    pf.apply_fill(w, "up", "U", 20, 10.0, 0.35, maker_shares=0, taker_shares=20, now=1790000100.0)
    assert pf.cash == pytest.approx(1000 - 10.35) and pf.bankroll == pytest.approx(1000.0)
    again = Portfolio.load_or_new(path, 1000.0)
    assert again.cash == pytest.approx(pf.cash) and again.positions["%s:up" % w.slug].shares == 20
    st = again.settle(w.slug, "up", 1790000400.0)[0]
    assert st.won and st.pnl == pytest.approx(20 - 10.35)
    assert again.cash == pytest.approx(1000 - 10.35 + 20) and again.wins == 1 and not again.positions
    assert json.load(open(path))["realized"] == pytest.approx(9.65)


def test_zero_cash_survives_reload(tmp_path):
    path = str(tmp_path / "state.json")
    pf = Portfolio.load_or_new(path, 10.0)
    pf.apply_fill(win(), "up", "U", 20, 10.0, 0.0, 0, 20, 1.0)
    assert Portfolio.load_or_new(path, 10.0).cash == 0.0


# ---------------------------------------------------------------- manager
def _mgr(tmp_path, **kw):
    c = cfg(KILL_SWITCH_FILE=str(tmp_path / "STOP"), **kw)
    clock = ReplayClock(1790000000.0)
    pf = Portfolio.load_or_new(str(tmp_path / "state.json"), 1000.0)
    return RiskManager(c, LIM, pf, clock), pf, clock


def test_kill_switch_blocks_and_releases(tmp_path):
    rm, pf, clock = _mgr(tmp_path)
    assert rm.pre_trade(win(), "up") is None
    (tmp_path / "STOP").write_text("")
    clock.advance_to(clock.now() + 1)
    assert rm.pre_trade(win(), "up") == "kill_switch"
    (tmp_path / "STOP").unlink()
    clock.advance_to(clock.now() + 1)
    assert rm.pre_trade(win(), "up") is None


def test_circuit_breaker_needs_consecutive_errors(tmp_path):
    rm, pf, clock = _mgr(tmp_path, MAX_CONSECUTIVE_ERRORS=3)
    rm.record_error("a"); rm.record_error("b"); rm.record_ok(); rm.record_error("c"); rm.record_error("d")
    assert rm.pre_trade(win(), "up") is None
    rm.record_error("e")
    assert rm.pre_trade(win(), "up") == "circuit_breaker"
    rm.record_ok()
    assert rm.pre_trade(win(), "up") == "circuit_breaker"      # only a restart clears it


def test_daily_stop_counts_realized_and_provisional_losses(tmp_path):
    rm, pf, clock = _mgr(tmp_path)
    pf.roll_day(clock.now())                                     # day-start bankroll 1000 -> stop at -50
    w1, w2 = win("a"), win("b")
    pf.apply_fill(w1, "up", "U", 60, 30.0, 0.0, 0, 60, clock.now())
    pf.apply_fill(w2, "up", "U", 60, 30.0, 0.0, 0, 60, clock.now())
    pf.settle("a", "down", clock.now())                          # -30 realized
    assert rm.pre_trade(win("c"), "up") is None
    pf.mark_provisional_loss("b", "up")                          # our data says -30 more
    assert rm.pre_trade(win("c"), "up") == "daily_stop"
    clock.advance_to(clock.now() + 86400)                        # next UTC day
    assert rm.pre_trade(win("c"), "up") is None


def test_cooldown_after_loss_is_per_asset_and_persisted(tmp_path):
    rm, pf, clock = _mgr(tmp_path, COOLDOWN_AFTER_LOSS_S=300)
    pf.apply_fill(win("a"), "up", "U", 10, 5.0, 0.0, 0, 10, clock.now())
    for st in pf.settle("a", "down", clock.now()):
        rm.on_settlement(st)
    assert rm.pre_trade(win("b", asset="btc"), "up") == "cooldown"
    assert rm.pre_trade(win("e", asset="eth"), "up") is None
    reloaded = Portfolio.load_or_new(str(tmp_path / "state.json"), 1000.0)
    assert reloaded.cooldowns["btc"] == pytest.approx(clock.now() + 300)
    clock.advance_to(clock.now() + 301)
    assert rm.pre_trade(win("b"), "up") is None


def test_fill_limits_opposite_side_and_exposure(tmp_path):
    rm, pf, clock = _mgr(tmp_path)
    w = win()
    pf.apply_fill(w, "up", "U", 10, 5.0, 0.0, 0, 10, clock.now())
    assert rm.pre_trade(w, "down") == "opposite_side"
    pf.apply_fill(w, "up", "U", 10, 5.0, 0.0, 0, 10, clock.now())
    assert rm.pre_trade(w, "up") == "max_fills"
    pf.reserved = 1000.0 * 0.15 - 10.0                          # exposure cap nearly used by in-flight orders
    assert rm.pre_trade(win("other"), "up") == "exposure_limit"

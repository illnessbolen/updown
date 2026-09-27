"""Statistics and hypothesis tests: exact distributions, error rates, the 98%-win-rate trap."""
import math
import random

import pytest

from latarb.stats import hypothesis as H
from latarb.stats.metrics import compute, equity_drawdown, sample_warning
from latarb.stats.trades import Trade

DAY = 86400.0


def mk(n, price, true_p, seed=0, shares=10.0, fee_rate=0.0, start=1_790_000_000.0, spacing=600.0,
       asset=None, label="5m"):
    """n independent binary bets bought at `price` (+ fee) that win with probability true_p."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = price + fee_rate * price * (1 - price)
        won = rng.random() < true_p
        cost = shares * c
        out.append(Trade(ts=start + i * spacing, entry_ts=start + i * spacing - 60, slug=f"w{i}",
                         asset=asset or ("btc" if i % 2 else "eth"), label=label, side="up", shares=shares,
                         cost=cost, pnl=(shares if won else 0.0) - cost, won=won, p_model=true_p, price=price,
                         tau_s=30.0, kind="taker", source="paper"))
    return out


# ---------------------------------------------------------------- distributions
def test_poisson_binomial_reduces_to_binomial_and_sums_to_one():
    pmf = H.poisson_binomial_pmf([0.3] * 10)
    for k in range(11):
        assert pmf[k] == pytest.approx(math.comb(10, k) * 0.3 ** k * 0.7 ** (10 - k), abs=1e-14)
    ps = [0.1, 0.5, 0.97, 0.6, 0.02]
    assert sum(H.poisson_binomial_pmf(ps)) == pytest.approx(1.0, abs=1e-14)
    assert H.poisson_binomial_pmf([0.2, 0.7]) == pytest.approx([0.24, 0.62, 0.14])


def test_refined_normal_approximation_is_close_to_exact():
    rng = random.Random(3)
    ps = [rng.uniform(0.5, 0.99) for _ in range(1500)]
    pmf = H.poisson_binomial_pmf(ps)
    mu = sum(ps)
    for k in (int(mu) - 20, int(mu), int(mu) + 15):
        assert H._rna_cdf(ps, k) == pytest.approx(sum(pmf[:k + 1]), abs=5e-3)


def test_inv_norm_and_holm():
    assert H.inv_norm(0.95) == pytest.approx(1.6448536, abs=1e-6)
    assert H.inv_norm(0.5) == pytest.approx(0.0, abs=1e-9)
    assert H.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


# ---------------------------------------------------------------- error rates
def test_no_edge_is_rarely_called_significant():
    # H0 true (win prob == price): the verdict "edge > 0" must fire at about alpha or less
    false_pos = 0
    runs = 60
    for seed in range(runs):
        prices = [0.3, 0.55, 0.8, 0.95]
        trades = []
        for j, pr in enumerate(prices):
            trades += mk(60, pr, pr, seed=seed * 10 + j, start=1_790_000_000.0 + j * 1e6)
        res = H.run(trades, alpha=0.05, sims=2000, n_boot=500, seed=seed)
        false_pos += res.positive
    assert false_pos / runs <= 0.10


def test_real_edge_is_detected():
    trades = mk(1500, 0.50, 0.56, seed=1)            # +6 points of edge at 0.50
    res = H.run(trades, alpha=0.05, sims=3000, n_boot=1000)
    assert res.positive and not res.negative
    assert res.pnl["p_pos_mc"] < 0.01 and res.wins["p_pos"] < 0.01
    assert res.power["required_n"] < 1500


def test_negative_edge_is_detected():
    trades = mk(1500, 0.50, 0.44, seed=2)
    res = H.run(trades, alpha=0.05, sims=3000, n_boot=1000)
    assert res.negative and not res.positive and "ОТРИЦАТЕЛЕН" in res.verdict


def test_the_98_percent_win_rate_trap():
    # buying favourites at 0.985 (+ fee) and winning 98.2% of the time looks great and loses money
    trades = mk(2000, 0.985, 0.982, seed=5, fee_rate=0.07)
    res = H.run(trades, alpha=0.05, sims=2000, n_boot=1000)
    m = compute(trades)
    assert m.win_rate > 0.97 and m.total_pnl < 0 and m.breakeven_win_rate > m.win_rate
    assert res.naive["p"] < 1e-100                    # "98% vs 50%" is overwhelmingly "significant"...
    assert not res.positive                           # ...and says nothing: there is no edge
    assert "нет статистически значимых" in res.verdict or res.negative


def test_overconfident_model_is_flagged():
    trades = [t.__class__(**{**t.__dict__, "p_model": 0.70}) for t in mk(800, 0.50, 0.50, seed=9)]
    res = H.run(trades, alpha=0.05, sims=1000, n_boot=300)
    assert res.calibration["overconfident"] and res.calibration["p_two_sided"] < 1e-6
    assert any("переоценивает" in w for w in res.warnings)


def test_small_samples_always_carry_the_warning():
    res = H.run(mk(40, 0.5, 0.9, seed=1), sims=500, n_boot=200)
    assert res.warnings[0].startswith("ВНИМАНИЕ: n=40") and "недостаточно данных" in res.warnings[0]
    assert sample_warning(100) is None and sample_warning(99) is not None
    assert H.run([]).lines() == ["[-] закрытых сделок нет — проверять нечего."]


def test_cluster_bootstrap_is_wider_for_correlated_trades():
    # same outcome for all trades in an hour (perfect correlation) -> far fewer effective samples
    rng = random.Random(4)
    trades = []
    for h in range(60):
        won = rng.random() < 0.5
        for i in range(10):
            ts = 1_790_000_000.0 + h * 3600 + i
            trades.append(Trade(ts, ts, f"w{h}-{i}", "btc", "5m", "up", 1.0, 0.5, (1.0 if won else 0.0) - 0.5,
                                won, None, 0.5, None, "taker", "paper"))
    per_trade = H.cluster_bootstrap(trades, "trade", 1000)
    per_hour = H.cluster_bootstrap(trades, "hour", 1000)
    width = lambda b: b["mean_ci95"][1] - b["mean_ci95"][0]  # noqa: E731
    assert per_hour["clusters"] == 60 and width(per_hour) > 2 * width(per_trade)


# ---------------------------------------------------------------- metrics
def test_drawdown_and_streaks():
    assert equity_drawdown([1, -2, -1, 3, -4, 1], None) == (4.0, None)
    mdd, pct = equity_drawdown([10, -30, 5], 100.0)
    assert mdd == 30.0 and pct == pytest.approx(30 / 110)
    t = mk(6, 0.5, 0.0, seed=0)
    m = compute(t)
    assert m.max_losing_streak == 6 and m.win_rate == 0 and m.profit_factor == 0.0


def test_metrics_values():
    trades = mk(400, 0.6, 0.6, seed=11, spacing=DAY / 10)
    m = compute(trades, start_equity=1000.0)
    wins = sum(t.won for t in trades)
    assert m.n == 400 and m.wins == wins and m.breakeven_win_rate == pytest.approx(0.6)
    assert m.total_pnl == pytest.approx(wins * 10 - 400 * 6)
    assert m.roi == pytest.approx(m.total_pnl / 2400)
    assert m.days == int(trades[-1].ts // DAY) - int(trades[0].ts // DAY) + 1   # calendar days, zero-filled
    assert m.sharpe_daily_ann is not None
    assert m.model_expected_pnl == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- logs -> trades, CLI, weekly report
def _write(path, fields, rows):
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def make_logs(d, n=150, p_true=0.62, price=0.55, seed=3):
    """A data dir in the bot's own log formats: settlements + orders + signals + outcomes + state."""
    import json as _json

    from latarb.reporting.exec_log import ORDER_FIELDS, SETTLEMENT_FIELDS
    from latarb.reporting.signal_log import OUTCOME_FIELDS, SIGNAL_FIELDS
    rng = random.Random(seed)
    sets, orders, sigs, outs = [], [], [], []
    t0 = 1_790_000_000.0
    fee = 0.07 * price * (1 - price)
    for i in range(n):
        slug, ts = f"btc-updown-5m-{int(t0 + i * 300)}", t0 + i * 300
        won = rng.random() < p_true
        winner = "up" if won else "down"
        maker = 10.0 if i % 3 == 0 else 0.0
        cost, fees = 10 * price, (10 - maker) * fee
        sid = f"s{i}"
        sigs.append({"signal_id": sid, "decision": "SIGNAL", "slug": slug, "asset": "btc", "label": "5m",
                     "side": "up", "ts": ts + 200, "window_end": ts + 300, "tau_s": 100.0, "p_fair_cons": 0.60,
                     "p_market": price, "cost": price + fee + 0.005})
        orders.append({"signal_id": sid, "slug": slug, "side": "up", "status": "filled", "kind": "taker",
                       "shares_req": 10, "shares_filled": 10, "p_fair_cons_at_send": 0.61, "ts": ts + 200.1,
                       "latency_ms": 70})
        sets.append({"ts": ts + 330, "slug": slug, "asset": "btc", "label": "5m", "side": "up", "shares": 10,
                     "avg_price": price, "cost": cost, "fees": fees, "maker_shares": maker,
                     "taker_shares": 10 - maker, "won": int(won), "pnl": (10 if won else 0) - cost - fees,
                     "opened_ts": ts + 200.2})
        outs.append({"slug": slug, "winner": winner})
    _write(d / "settlements.csv", SETTLEMENT_FIELDS, sets)
    _write(d / "orders.csv", ORDER_FIELDS, orders)
    _write(d / "signals.csv", SIGNAL_FIELDS, sigs)
    _write(d / "outcomes.csv", OUTCOME_FIELDS, outs)
    (d / "paper_state.json").write_text(_json.dumps({"start_cash": 1000.0}))


def test_paper_and_shadow_trades_are_read_from_the_logs(tmp_path):
    from latarb.stats.trades import load_paper_trades, load_shadow_trades, load_trades, paper_start_equity
    make_logs(tmp_path, n=30)
    paper = load_paper_trades(str(tmp_path))
    assert len(paper) == 30 and paper[0].p_model == pytest.approx(0.61) and paper[0].tau_s == 100.0
    assert {t.kind for t in paper} == {"maker", "taker"}
    maker = next(t for t in paper if t.kind == "maker")
    assert maker.breakeven == pytest.approx(0.55)                   # makers pay no fee
    shadow = load_shadow_trades(str(tmp_path))
    assert len(shadow) == 30 and all(t.shares == 1.0 for t in shadow)
    assert shadow[0].pnl == pytest.approx((1.0 if shadow[0].won else 0.0) - shadow[0].cost)
    assert load_trades(str(tmp_path), "auto")[0].source == "paper"
    assert paper_start_equity(str(tmp_path)) == 1000.0


def test_cli_stats_hypothesis_and_report(tmp_path, capsys):
    import json as _json

    from latarb.cli import main
    make_logs(tmp_path, n=150)
    assert main(["stats", "--data", str(tmp_path), "--by", "kind,price"]) == 0
    out = capsys.readouterr().out
    assert "безубыточный винрейт" in out and "по kind" in out and "ВНИМАНИЕ" not in out
    assert main(["hypothesis", "--data", str(tmp_path), "--sims", "500", "--json"]) == 0
    res = _json.loads(capsys.readouterr().out)
    assert res["n"] == 150 and res["source"] == "paper" and "p_pos_mc" in res["pnl"]
    assert main(["report", "--data", str(tmp_path)]) == 0
    path = capsys.readouterr().out.strip().splitlines()[-1]
    text = open(path, encoding="utf-8").read()
    assert "Проверка гипотезы" in text and "Исполнение за период" in text
    for banned in ("гарантир", "guarantee"):
        assert banned not in text.lower()


def test_weekly_schedule_starts_the_clock_then_writes_every_seven_days(tmp_path):
    from latarb.stats.report import maybe_write_weekly
    make_logs(tmp_path, n=5)
    t = 1_790_100_000.0
    assert maybe_write_weekly(str(tmp_path), t) is None                     # first call: clock starts
    assert maybe_write_weekly(str(tmp_path), t + 6 * DAY) is None
    path = maybe_write_weekly(str(tmp_path), t + 7 * DAY + 1)
    assert path and path.endswith(".md")
    assert maybe_write_weekly(str(tmp_path), t + 8 * DAY) is None

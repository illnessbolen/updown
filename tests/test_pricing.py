"""P(fair) boundary cases with known analytical answers."""
import math

import pytest

from latarb.model.distributions import betainc, norm_cdf, std_t_cdf, student_t_cdf
from latarb.model.fees import taker_fee_per_share
from latarb.model.pricing import fair_value, fair_value_band

SIGMA = 1e-4          # ~ BTC: 1 bp per sqrt(second)
TAILS = [math.inf, 5.0, 3.0]


# ---------------------------------------------------------------- limits
@pytest.mark.parametrize("dof", TAILS)
def test_at_the_money_as_time_runs_out_is_a_coin_flip(dof):
    # |P - 0.5| ~ f(0) * sigma * sqrt(tau) / 2 -> 0: must shrink monotonically to 0.5
    devs = []
    for tau in (60.0, 1.0, 1e-2, 1e-4, 1e-6, 1e-9):
        fv = fair_value(spot=100_000.0, reference=100_000.0, tau_s=tau, sigma=SIGMA, tail_dof=dof)
        assert fv.p_up + fv.p_down == pytest.approx(1.0, abs=1e-12)
        devs.append(abs(fv.p_up - 0.5))
    assert all(a > b for a, b in zip(devs, devs[1:]))
    assert devs[-1] < 1e-9
    assert devs[0] < 1e-3          # even with a minute left, ATM is ~0.5 (drift-free model)


@pytest.mark.parametrize("dof", TAILS)
@pytest.mark.parametrize("tau", [1e-2, 1e-4, 1e-8])
def test_large_deviation_as_time_runs_out_is_certain(tau, dof):
    up = fair_value(spot=100_500.0, reference=100_000.0, tau_s=tau, sigma=SIGMA, tail_dof=dof)
    down = fair_value(spot=99_500.0, reference=100_000.0, tau_s=tau, sigma=SIGMA, tail_dof=dof)
    assert up.p_up > 1 - 1e-6 and up.p_down < 1e-6
    assert down.p_up < 1e-6 and down.p_down > 1 - 1e-6


def test_zero_time_zero_noise_is_deterministic_and_ties_resolve_up():
    assert fair_value(101.0, 100.0, 0.0, SIGMA).p_up == 1.0
    assert fair_value(99.0, 100.0, 0.0, SIGMA).p_up == 0.0
    assert fair_value(100.0, 100.0, 0.0, SIGMA).p_up == 1.0      # rule is ">="
    assert fair_value(100.0, 100.0, -5.0, SIGMA).p_up == 1.0     # closed window treated as tau = 0


def test_oracle_noise_keeps_the_last_second_uncertain():
    # 0.5 bp above K at tau = 0, but the oracle price has 1 bp noise around our estimate
    fv = fair_value(100_005.0, 100_000.0, 0.0, SIGMA, oracle_noise=1e-4)
    assert fv.p_up == pytest.approx(norm_cdf(math.log(1.00005) / 1e-4), rel=1e-12)
    assert 0.6 < fv.p_up < 0.75


# ---------------------------------------------------------------- exact value
def test_matches_black_scholes_digital_at_d_equal_one():
    tau, sigma = 100.0, 1e-3
    v = sigma * sigma * tau
    x = math.sqrt(v) + v / 2           # constructed so that d = 1 exactly
    fv = fair_value(100.0 * math.exp(x), 100.0, tau, sigma)
    assert fv.d == pytest.approx(1.0, abs=1e-12)
    assert fv.p_up == pytest.approx(0.8413447460685429, abs=1e-12)   # Phi(1)


# ---------------------------------------------------------------- structure
@pytest.mark.parametrize("dof", TAILS)
def test_up_and_down_sum_to_one(dof):
    for spot in (99_000.0, 99_990.0, 100_000.0, 100_010.0, 101_000.0):
        fv = fair_value(spot, 100_000.0, 60.0, SIGMA, 2e-5, dof)
        assert fv.p_up + fv.p_down == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("dof", TAILS)
def test_reflection_identity(dof):
    # d(x) = -d(x') for x' = -x + sigma^2 tau  =>  P_up(x) = 1 - P_up(x')
    tau, sigma, k = 200.0, 3e-4, 50_000.0
    v = sigma * sigma * tau
    for x in (-0.004, -0.001, 0.0, 0.0007, 0.003):
        p1 = fair_value(k * math.exp(x), k, tau, sigma, tail_dof=dof).p_up
        p2 = fair_value(k * math.exp(-x + v), k, tau, sigma, tail_dof=dof).p_up
        assert p1 == pytest.approx(1.0 - p2, abs=1e-12)


def test_monotone_in_spot():
    ps = [fair_value(s, 100.0, 120.0, SIGMA).p_up for s in (99.9, 99.95, 100.0, 100.05, 100.1)]
    assert ps == sorted(ps) and len(set(ps)) == len(ps)


def test_more_time_left_dilutes_the_same_move():
    # same +10 bp move: nearly decided with 5 s left, much less so with 15 min / 1 day left
    ps = [fair_value(100_100.0, 100_000.0, tau, SIGMA).p_up for tau in (5, 60, 300, 900, 3600, 86400)]
    assert all(a > b for a, b in zip(ps, ps[1:]))
    assert ps[0] > 0.99 and ps[-1] < 0.55


def test_fat_tails_are_less_confident_far_from_the_money():
    # 3.5 sd away: Gaussian ~0.99977, t(5) noticeably less sure -> fewer 0.99 "snipes"
    args = (100_000.0 * math.exp(3.5e-4), 100_000.0, 1.0, SIGMA)
    g = fair_value(*args, tail_dof=math.inf).p_up
    t = fair_value(*args, tail_dof=5.0).p_up
    assert g > 0.9997 and t < g - 1e-3


def test_band_brackets_point_estimate():
    lo, hi = fair_value_band(100_050.0, 100_000.0, 60.0, (0.8e-4, 1e-4, 1.2e-4), 0.0, 5.0)
    p = fair_value(100_050.0, 100_000.0, 60.0, 1e-4, 0.0, 5.0).p_up
    assert lo < p < hi


def test_rejects_garbage_inputs():
    for bad in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (math.inf, 1.0)):
        with pytest.raises(ValueError):
            fair_value(bad[0], bad[1], 10.0, SIGMA)
    with pytest.raises(ValueError):
        fair_value(1.0, 1.0, 10.0, -1.0)
    with pytest.raises(ValueError):
        fair_value(1.0, 1.0, 10.0, math.nan)


# ---------------------------------------------------------------- distributions
@pytest.mark.parametrize("x", [-30.0, -3.0, -0.7, 0.0, 0.4, 2.5, 12.0])
def test_student_t_matches_closed_forms(x):
    assert student_t_cdf(x, 1.0) == pytest.approx(0.5 + math.atan(x) / math.pi, abs=1e-12)
    assert student_t_cdf(x, 2.0) == pytest.approx(0.5 + x / (2 * math.sqrt(2 + x * x)), abs=1e-12)


def test_unit_variance_t_converges_to_normal():
    for z in (-2.0, -0.5, 1.0, 3.0):
        assert std_t_cdf(z, 1e6) == pytest.approx(norm_cdf(z), abs=1e-5)


def test_betainc_symmetry():
    for a, b, x in ((2.0, 3.0, 0.3), (0.5, 0.5, 0.9), (7.5, 0.5, 0.2)):
        assert betainc(a, b, x) == pytest.approx(1.0 - betainc(b, a, 1.0 - x), abs=1e-13)


def test_fee_formula():
    assert taker_fee_per_share(0.5, 0.07) == pytest.approx(0.0175)
    assert taker_fee_per_share(0.98, 0.07) == pytest.approx(0.07 * 0.98 * 0.02)
    assert taker_fee_per_share(1.0, 0.07) == 0.0
    with pytest.raises(ValueError):
        taker_fee_per_share(1.2, 0.07)

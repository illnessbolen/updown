"""Fair probability of an "Up or Down" window: a cash-or-nothing digital option.

The market pays 1 if S_T >= K, where
    K   = resolution-source price at the window START (the "price to beat"),
    S_T = resolution-source price at the window END.

At time t inside the window the path is split in two ("broken" at t):
    [start, t]  already realized  -> fully summarized by the current price S_t
    [t, end]    still random      -> Brownian increment with variance sigma^2 * tau

so what matters is the DISTANCE of S_t from K measured in units of the
remaining uncertainty, not the fact that price recently moved:

    x   = ln(S_t / K)
    v   = sigma^2 * tau                      (diffusion variance of ln S until close)
    eta = oracle noise (std of ln(resolution price) - ln(our estimate) at close)
    d   = (x - v/2) / sqrt(v + eta^2)        (-v/2: S is a martingale, zero drift)
    P_up = F(d),  P_down = F(-d)

F is N(0,1) or a unit-variance Student-t (fat tails, TAIL_DOF). With F = Phi
and eta = 0 this is exactly the Black-Scholes digital N(d2) at r = 0.

Limits (checked in tests/test_pricing.py):
    S = K,       tau -> 0   :  P_up -> 0.5
    S >> K,      tau -> 0   :  P_up -> 1
    S << K,      tau -> 0   :  P_up -> 0
    fixed S > K, tau grows  :  P_up decreases toward 0.5 (a move is "diluted" by the
                               remaining time; longer windows react less to spot)
    tau == 0, eta == 0      :  deterministic, ties resolve Up (rule is ">=")
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .distributions import std_t_cdf


@dataclass(frozen=True, slots=True)
class FairValue:
    p_up: float
    p_down: float
    x: float        # ln(S/K)
    sd: float       # total std of ln(S_T/K) conditional on now
    d: float        # standardized distance; +inf/-inf when sd == 0


def fair_value(spot: float, reference: float, tau_s: float, sigma: float,
               oracle_noise: float = 0.0, tail_dof: float = math.inf) -> FairValue:
    """P(S_T >= K).

    spot          current price estimate in resolution-source units (> 0)
    reference     K, resolution-source price at the window start (> 0)
    tau_s         seconds until the window closes (negative is treated as 0)
    sigma         volatility of ln(price) per sqrt(second) (>= 0)
    oracle_noise  std (in log units) of the resolution price around our estimate
    tail_dof      Student-t degrees of freedom (> 2) or math.inf for Gaussian
    """
    if not (spot > 0.0 and reference > 0.0) or math.isinf(spot) or math.isinf(reference):
        raise ValueError(f"spot and reference must be positive and finite (got {spot}, {reference})")
    if not sigma >= 0.0 or not oracle_noise >= 0.0:
        raise ValueError("sigma and oracle_noise must be >= 0")
    tau = tau_s if tau_s > 0.0 else 0.0
    x = math.log(spot / reference)
    diff_var = sigma * sigma * tau
    var = diff_var + oracle_noise * oracle_noise
    if var <= 1e-30:
        up = 1.0 if x >= 0.0 else 0.0
        return FairValue(up, 1.0 - up, x, 0.0, math.inf if x >= 0.0 else -math.inf)
    sd = math.sqrt(var)
    d = (x - 0.5 * diff_var) / sd
    return FairValue(std_t_cdf(d, tail_dof), std_t_cdf(-d, tail_dof), x, sd, d)


def fair_value_band(spot: float, reference: float, tau_s: float, sigmas: Iterable[float],
                    oracle_noise: float = 0.0, tail_dof: float = math.inf) -> Tuple[float, float]:
    """(min, max) of P_up over a set of plausible sigmas.

    P_up is monotone in sigma over every realistic regime (sigma^2 * tau << 1),
    so evaluating at the ends plus the point estimate brackets it. The signal
    layer uses the side-unfavourable end: an edge must survive sigma being wrong.
    """
    ps = [fair_value(spot, reference, tau_s, s, oracle_noise, tail_dof).p_up for s in sigmas]
    if not ps:
        raise ValueError("need at least one sigma")
    return min(ps), max(ps)


# ---------------------------------------------------------------- market-implied inputs
def implied_sigma(spot: float, reference: float, tau_s: float, p_up_market: float, oracle_noise: float = 0.0,
                  tail_dof: float = math.inf, sigma_max: float = 1e-2) -> Optional[float]:
    """Volatility (per sqrt second) at which the model reproduces the market's P(up), given OUR spot
    and reference. None when no sigma can (|x| ~ 0, or the market is on the other side of 0.5 from
    x: then the market disagrees about where the price is, not about vol)."""
    if tau_s <= 0 or not 0.0 < p_up_market < 1.0:
        return None
    x = math.log(spot / reference)
    if abs(x) < 1e-7:
        return None
    f = lambda s: fair_value(spot, reference, tau_s, s, oracle_noise, tail_dof).p_up - p_up_market  # noqa: E731
    lo, hi = 1e-9, sigma_max
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if flo * fhi > 0:
        return None
    for _ in range(50):
        mid = math.sqrt(lo * hi)                      # geometric bisection: sigma spans decades
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return math.sqrt(lo * hi)


def implied_log_moneyness(tau_s: float, sigma: float, p_up_market: float, oracle_noise: float = 0.0,
                          tail_dof: float = math.inf) -> Optional[float]:
    """ln(S/K) at which the model (with OUR sigma) reproduces the market's P(up)."""
    if not 0.0 < p_up_market < 1.0 or (sigma <= 0 and oracle_noise <= 0) or tau_s < 0:
        return None
    f = lambda x: fair_value(math.exp(x), 1.0, tau_s, sigma, oracle_noise, tail_dof).p_up - p_up_market  # noqa: E731
    lo, hi = -0.5, 0.5
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

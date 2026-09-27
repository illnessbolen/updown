"""CDFs used by the pricing model (pure Python, no scipy on the hot path).

norm_cdf uses erfc, which stays accurate deep in both tails (1 - Phi(8) is
~6e-16, not rounded to 0), so P_up and P_down can each be computed directly
without 1 - x cancellation.

The Student-t CDF goes through the regularized incomplete beta function
(continued fraction, Numerical Recipes 6.4). It is verified in the tests
against the closed forms for nu = 1 (Cauchy) and nu = 2.
"""
from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _betacf(a: float, b: float, x: float, max_iter: int = 500, eps: float = 1e-15) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(x: float, nu: float) -> float:
    """CDF of the (non-standardized) Student-t with nu degrees of freedom."""
    if nu <= 0:
        raise ValueError("nu must be positive")
    if math.isinf(nu):
        return norm_cdf(x)
    if x == 0.0:
        return 0.5
    t = nu / (nu + x * x)
    tail = 0.5 * betainc(0.5 * nu, 0.5, t)      # P(T > |x|)
    return 1.0 - tail if x > 0 else tail


def std_t_cdf(z: float, nu: float) -> float:
    """CDF of a Student-t rescaled to unit variance (requires nu > 2).

    Z = T * sqrt((nu - 2) / nu)  ->  P(Z <= z) = F_nu(z * sqrt(nu / (nu - 2))).
    Same variance as N(0,1), heavier tails: this is what lets the model admit
    that a 4-sigma move within a few seconds is not a 1-in-30000 event in crypto.
    """
    if math.isinf(nu):
        return norm_cdf(z)
    if nu <= 2:
        raise ValueError("unit-variance t needs nu > 2")
    return student_t_cdf(z * math.sqrt(nu / (nu - 2.0)), nu)

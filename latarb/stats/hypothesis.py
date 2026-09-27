"""Hypothesis test mode: is the observed edge statistically distinguishable from zero?

Null hypothesis with the REAL payoff asymmetry
---------------------------------------------
A trade buys `shares` binary contracts for an all-in cost c per share (fees and,
for shadow trades, expected slippage included). It pays shares*(1-c) if it wins
and loses shares*c otherwise. "No edge" means the market price was fair:

    H0:  P(win_i) = c_i          (so E[pnl_i] = 0 for every trade)

Buying favourites at 0.97 and winning 97% of the time is exactly H0 — which is
why a win rate compared to 50% (printed below only as a contrast) proves nothing.

Tests (one-sided for "edge > 0", mirrored for "edge < 0"):
  1. wins      number of wins W vs its exact H0 distribution: Poisson-binomial with
               p_i = c_i (every trade has its own break-even probability). Exact up to
               EXACT_MAX trades, refined normal approximation above that.
  2. pnl       total P&L S; under H0 E[S] = 0, Var[S] = sum shares_i^2 c_i (1 - c_i).
               z-test plus Monte-Carlo p-value from simulating H0 outcomes (the payoff
               distribution of 0.98-favourites is very skewed; the normal approximation
               alone is not trusted for small n).
  3. bootstrap cluster bootstrap CI of mean P&L per trade and ROI. Trades in the same
               hour (BTC 5m + 15m, BTC + ETH ...) are correlated, which makes tests 1-2
               (independence) optimistic; resampling whole clusters does not assume it.
  4. model     are wins consistent with the model's own probabilities? W vs
               Poisson-binomial(p_model), two-sided. Overconfidence shows up here first.
  5. power     trades needed to detect the observed mean edge (alpha, 80% power) and
               the smallest edge detectable with the current sample.

Verdict "edge > 0" requires tests 1-2 AND the cluster-bootstrap lower bound to agree.
Segment tests (by asset / duration) are exploratory and Holm-adjusted.
Nothing here is a forecast: it describes the trades that already settled.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..model.distributions import norm_cdf, norm_pdf
from .metrics import BREAKDOWNS, MIN_SAMPLE, sample_warning
from .trades import Trade

EXACT_MAX = 2000


# ================================================================ distributions
def poisson_binomial_pmf(ps: Sequence[float]) -> List[float]:
    pmf = [1.0]
    for p in ps:
        q = 1.0 - p
        new = [0.0] * (len(pmf) + 1)
        for k, v in enumerate(pmf):
            new[k] += v * q
            new[k + 1] += v * p
        pmf = new
    return pmf


def _rna_cdf(ps: Sequence[float], k: float) -> float:
    """Refined normal approximation of P(W <= k) (Volkova 1996), skewness-corrected."""
    mu = sum(ps)
    var = sum(p * (1 - p) for p in ps)
    if var <= 0:
        return 1.0 if k >= mu else 0.0
    sd = math.sqrt(var)
    gamma = sum(p * (1 - p) * (1 - 2 * p) for p in ps) / sd ** 3
    x = (k + 0.5 - mu) / sd
    return min(1.0, max(0.0, norm_cdf(x) + gamma * (1 - x * x) * norm_pdf(x) / 6.0))


def pb_tails(ps: Sequence[float], w: int) -> Tuple[float, float, str]:
    """(P(W >= w), P(W <= w), method) under Poisson-binomial(ps)."""
    if len(ps) <= EXACT_MAX:
        pmf = poisson_binomial_pmf(ps)
        return min(1.0, sum(pmf[w:])), min(1.0, sum(pmf[:w + 1])), "exact"
    return 1.0 - _rna_cdf(ps, w - 1), _rna_cdf(ps, w), "refined normal"


def inv_norm(p: float) -> float:
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ================================================================ tests
def wins_test(trades: List[Trade]) -> dict:
    ps = [t.breakeven for t in trades]
    w = sum(t.won for t in trades)
    p_ge, p_le, method = pb_tails(ps, w)
    return {"wins": w, "expected_wins_h0": sum(ps), "p_pos": p_ge, "p_neg": p_le, "method": method}


def pnl_test(trades: List[Trade], sims: int = 20000, seed: int = 7, max_draws: int = 3_000_000) -> dict:
    n = len(trades)
    s_obs = sum(t.pnl for t in trades)
    legs = [(t.shares * (1 - t.breakeven), -t.shares * t.breakeven, t.breakeven) for t in trades]
    var0 = sum(t.shares ** 2 * t.breakeven * (1 - t.breakeven) for t in trades)
    z = s_obs / math.sqrt(var0) if var0 > 0 else 0.0
    n_sims = max(200, min(sims, max_draws // max(n, 1)))
    rng = random.Random(seed)
    ge = le = 0
    tol = 1e-9 * max(1.0, abs(s_obs))
    for _ in range(n_sims):
        s = 0.0
        for win, lose, c in legs:
            s += win if rng.random() < c else lose
        if s >= s_obs - tol:
            ge += 1
        if s <= s_obs + tol:
            le += 1
    return {"pnl": s_obs, "sd_h0": math.sqrt(var0), "z": z,
            "p_pos_normal": 1.0 - norm_cdf(z), "p_neg_normal": norm_cdf(z),
            "p_pos_mc": (1 + ge) / (1 + n_sims), "p_neg_mc": (1 + le) / (1 + n_sims), "sims": n_sims}


CLUSTERS: Dict[str, Callable[[Trade], str]] = {
    "trade": lambda t: f"{t.slug}:{t.side}:{t.entry_ts}",
    "window": lambda t: t.slug,
    "hour": lambda t: str(int(t.ts // 3600)),
    "day": lambda t: str(int(t.ts // 86400)),
}


def cluster_bootstrap(trades: List[Trade], cluster: str = "hour", n_boot: int = 5000, seed: int = 11,
                      alpha: float = 0.05) -> dict:
    groups: Dict[str, List[float]] = {}
    for t in trades:
        g = groups.setdefault(CLUSTERS[cluster](t), [0.0, 0.0, 0.0])
        g[0] += t.pnl
        g[1] += t.cost
        g[2] += 1
    cl = list(groups.values())
    k = len(cl)
    rng = random.Random(seed)
    means, rois = [], []
    for _ in range(n_boot):
        p = c = n = 0.0
        for _ in range(k):
            g = cl[rng.randrange(k)]
            p += g[0]
            c += g[1]
            n += g[2]
        means.append(p / n)
        rois.append(p / c if c > 0 else 0.0)
    means.sort()
    rois.sort()
    q = lambda xs, a: xs[min(len(xs) - 1, max(0, int(a * len(xs))))]  # noqa: E731
    return {"cluster": cluster, "clusters": k, "n_boot": n_boot,
            "mean_ci95": (q(means, 0.025), q(means, 0.975)), "roi_ci95": (q(rois, 0.025), q(rois, 0.975)),
            "mean_lower": q(means, alpha), "mean_upper": q(means, 1 - alpha)}


def calibration_test(trades: List[Trade]) -> Optional[dict]:
    cov = [t for t in trades if t.p_model is not None]
    if not cov:
        return None
    ps = [min(max(t.p_model, 0.0), 1.0) for t in cov]
    w = sum(t.won for t in cov)
    p_ge, p_le, method = pb_tails(ps, w)
    return {"n": len(cov), "wins": w, "expected_wins_model": sum(ps),
            "p_two_sided": min(1.0, 2.0 * min(p_ge, p_le)), "overconfident": w < sum(ps),
            "brier": statistics.fmean((p - t.won) ** 2 for p, t in zip(ps, cov)),
            "brier_breakeven": statistics.fmean((t.breakeven - t.won) ** 2 for t in cov), "method": method}


def power_analysis(trades: List[Trade], alpha: float, power: float = 0.8) -> dict:
    n = len(trades)
    pnls = [t.pnl for t in trades]
    if n < 2:
        return {"required_n": None, "mde": None}
    mu, sd = statistics.fmean(pnls), statistics.stdev(pnls)
    za, zb = inv_norm(1 - alpha), inv_norm(power)
    req = math.ceil(((za + zb) * sd / mu) ** 2) if mu > 0 and sd > 0 else None
    return {"mean": mu, "sd": sd, "required_n": req, "mde": (za + zb) * sd / math.sqrt(n), "power": power}


def naive_winrate_test(trades: List[Trade]) -> dict:
    """Win rate vs 50% — shown ONLY as a contrast: it ignores prices and proves nothing about edge."""
    n = len(trades)
    w = sum(t.won for t in trades)
    p_ge, _, method = pb_tails([0.5] * n, w)
    return {"wins": w, "n": n, "p": p_ge, "method": method}


def segment_tests(trades: List[Trade], dims=("asset", "label"), min_n: int = 20) -> Dict[str, dict]:
    raw: Dict[str, dict] = {}
    for dim in dims:
        groups: Dict[str, List[Trade]] = {}
        for t in trades:
            groups.setdefault(BREAKDOWNS[dim](t), []).append(t)
        for k, ts in groups.items():
            if len(ts) < min_n:
                continue
            var0 = sum(t.shares ** 2 * t.breakeven * (1 - t.breakeven) for t in ts)
            s = sum(t.pnl for t in ts)
            z = s / math.sqrt(var0) if var0 > 0 else 0.0
            raw[f"{dim}={k}"] = {"n": len(ts), "pnl": s, "z": z, "p": 1.0 - norm_cdf(z)}
    adj = holm({k: v["p"] for k, v in raw.items()})
    for k in raw:
        raw[k]["p_holm"] = adj[k]
    return raw


# ================================================================ driver
@dataclass
class HypothesisResult:
    n: int
    alpha: float
    source: str
    wins: dict = field(default_factory=dict)
    pnl: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)
    calibration: Optional[dict] = None
    power: dict = field(default_factory=dict)
    naive: dict = field(default_factory=dict)
    segments: Dict[str, dict] = field(default_factory=dict)
    verdict: str = ""
    positive: bool = False
    negative: bool = False
    warnings: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        if self.n == 0:
            return [f"[{self.source}] закрытых сделок нет — проверять нечего."]
        a = self.alpha
        w, p, b = self.wins, self.pnl, self.bootstrap
        out = [f"=== проверка гипотезы [{self.source}] n={self.n}, α={a} ===",
               "H0: сделка выигрывает с вероятностью, равной её полной цене (edge = 0, асимметрия выплат учтена)",
               f"1) число побед: {w['wins']} при ожидаемых под H0 {w['expected_wins_h0']:.1f} | "
               f"p(edge>0)={w['p_pos']:.4g} p(edge<0)={w['p_neg']:.4g} ({w['method']})",
               f"2) P&L: {p['pnl']:+.4f} при σ под H0 {p['sd_h0']:.4f}, z={p['z']:+.2f} | "
               f"p(edge>0) норм. {p['p_pos_normal']:.4g} / Монте-Карло {p['p_pos_mc']:.4g} | "
               f"p(edge<0) норм. {p['p_neg_normal']:.4g} / МК {p['p_neg_mc']:.4g} ({p['sims']} симуляций)",
               f"3) кластерный бутстрэп по '{b['cluster']}' ({b['clusters']} кластеров): средний P&L на сделку "
               f"95% ДИ [{b['mean_ci95'][0]:+.4f}, {b['mean_ci95'][1]:+.4f}], ROI 95% ДИ "
               f"[{b['roi_ci95'][0]:+.2%}, {b['roi_ci95'][1]:+.2%}]"]
        c = self.calibration
        if c is not None:
            out.append(f"4) калибровка модели на {c['n']} сделках: побед {c['wins']}, модель ожидала "
                       f"{c['expected_wins_model']:.1f} | p (двуст.)={c['p_two_sided']:.4g} | Brier модели "
                       f"{c['brier']:.4f} vs цены {c['brier_breakeven']:.4f}")
        pw = self.power
        if pw.get("mde") is not None:
            req = pw["required_n"]
            out.append(f"5) мощность: минимальный обнаружимый средний P&L на сделку при n={self.n}: "
                       f"{pw['mde']:.4f}; для текущего среднего {pw['mean']:+.4f} нужно "
                       + (f"~{req} сделок (α={a}, мощность {pw['power']:.0%})" if req else "— (среднее ≤ 0)"))
        nv = self.naive
        out.append(f"   контраст: винрейт против 50% — p={nv['p']:.4g}. Этот тест цены не учитывает и "
                   f"ничего не говорит о прибыльности; приведён, чтобы было видно разницу.")
        if self.segments:
            out.append("   сегменты (разведочно, поправка Холма):")
            for k, v in sorted(self.segments.items()):
                out.append(f"     {k:<14} n={v['n']:<5} P&L {v['pnl']:+.3f} z={v['z']:+.2f} "
                           f"p={v['p']:.4g} p_holm={v['p_holm']:.4g}")
        out += [f"ВЫВОД: {self.verdict}"] + self.warnings
        return out

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("n", "alpha", "source", "wins", "pnl", "bootstrap", "calibration",
                                              "power", "naive", "segments", "verdict", "positive", "negative",
                                              "warnings")}


def run(trades: List[Trade], alpha: float = 0.05, sims: int = 20000, cluster: str = "hour",
        n_boot: int = 5000, seed: int = 7, source: str = "") -> HypothesisResult:
    res = HypothesisResult(n=len(trades), alpha=alpha, source=source or (trades[0].source if trades else "-"))
    if not trades:
        res.verdict = "закрытых сделок нет"
        return res
    res.wins = wins_test(trades)
    res.pnl = pnl_test(trades, sims=sims, seed=seed)
    res.bootstrap = cluster_bootstrap(trades, cluster, n_boot, seed + 1, alpha)
    res.calibration = calibration_test(trades)
    res.power = power_analysis(trades, alpha)
    res.naive = naive_winrate_test(trades)
    res.segments = segment_tests(trades)
    w, p, b = res.wins, res.pnl, res.bootstrap
    res.positive = (w["p_pos"] < alpha and p["p_pos_mc"] < alpha and p["p_pos_normal"] < alpha
                    and b["mean_lower"] > 0)
    res.negative = (p["p_neg_mc"] < alpha and p["p_neg_normal"] < alpha and b["mean_upper"] < 0)
    roi = sum(t.pnl for t in trades) / sum(t.cost for t in trades)
    if res.positive:
        res.verdict = (f"положительный edge статистически отличим от нуля (α={alpha}, односторонние тесты числа "
                       f"побед и P&L под H0, нижняя граница кластерного бутстрэпа > 0). ROI {roi:+.2%}, 95% ДИ "
                       f"[{b['roi_ci95'][0]:+.2%}, {b['roi_ci95'][1]:+.2%}]. Это описание прошедших сделок, "
                       f"а не обещание будущих результатов.")
    elif res.negative:
        res.verdict = (f"edge статистически значимо ОТРИЦАТЕЛЕН (α={alpha}): в этом виде стратегия теряет "
                       f"деньги. ROI {roi:+.2%}.")
    else:
        res.verdict = (f"нет статистически значимых свидетельств ненулевого edge (α={alpha}). ROI {roi:+.2%}, "
                       f"95% ДИ [{b['roi_ci95'][0]:+.2%}, {b['roi_ci95'][1]:+.2%}].")
    c = res.calibration
    if c is not None and c["p_two_sided"] < alpha:
        res.warnings.append(
            f"Модель {'переоценивает' if c['overconfident'] else 'недооценивает'} вероятность выигрыша: "
            f"ожидала {c['expected_wins_model']:.1f} побед, получено {c['wins']} (p={c['p_two_sided']:.3g}) — "
            f"edge, который она показывает, завышен/занижен.")
    if b["clusters"] < 30:
        res.warnings.append(f"кластеров всего {b['clusters']} — бутстрэп-интервал ненадёжен.")
    res.warnings.append("Тесты 1-2 предполагают независимость сделок; одновременные окна коррелированы, поэтому "
                        "вывод о положительном edge дополнительно требует кластерного бутстрэпа.")
    w_ = sample_warning(res.n)
    if w_:
        res.warnings.insert(0, w_)
    return res


__all__ = ["run", "HypothesisResult", "poisson_binomial_pmf", "pb_tails", "pnl_test", "wins_test",
           "cluster_bootstrap", "calibration_test", "power_analysis", "holm", "inv_norm", "MIN_SAMPLE"]

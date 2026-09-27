"""Signal engine: P(fair) vs P(market) after fees and slippage, with explicit gates.

Evaluated on every Binance mid change for the asset, on every book change of
the window, and on a 1 s timer (time decay alone moves P(fair)).

For each open window:

  1. core inputs — spot, reference, realized vol. Missing any of them means
     the window cannot be priced at all (counted, not logged per tick).
  2. spot in resolution units:
        chainlink windows: S = binance_mid * exp(mean ln(chainlink/binance)),
                           oracle noise = std of that basis
        binance windows:   S = binance_mid, noise = half spread
  3. P_up with the point sigma, plus the [min, max] of P_up over the sigma band.
  4. per side: ask (own book, or 1 - bid of the other outcome when cheaper),
     fee = rate * (p(1-p))^exp, cost = ask + fee + expected slippage,
        edge      = P_side            - cost
        edge_cons = P_side(worst sigma) - cost      <- the decision uses this
  5. gates: any failed gate turns a would-be SIGNAL into GATED (still logged:
     that is the material for false-positive / missed-opportunity analysis).

The engine never places orders itself. In paper mode an ExecutionPipeline is
attached as `executor`: every SIGNAL is offered to it (its answer is logged as
exec_status), every evaluation lets it manage resting orders, and it can ask
for a fresh re-evaluation against a just-refreshed book (`books=`).
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..clock import Clock
from ..config import Settings
from ..data.hub import MarketDataHub
from ..data.markets import BINANCE, CHAINLINK, MarketWindow
from ..data.orderbook import OrderBook
from ..data.reference import ReferenceResolver
from ..model.fees import taker_fee_per_share
from ..model.pricing import fair_value, fair_value_band, implied_log_moneyness, implied_sigma

log = logging.getLogger("latarb.signal")

SIGNAL = "SIGNAL"
GATED = "GATED"
NONE = "NONE"


@dataclass(slots=True)
class SideQuote:
    side: str
    ask: float
    ask_size: float
    ask_src: str               # "book" | "complement"
    bid: Optional[float]
    p_fair: float
    p_fair_cons: float         # side-unfavourable end of the sigma band
    fee: float
    cost: float
    edge: float
    edge_cons: float


@dataclass(slots=True)
class Evaluation:
    ts: float
    window: MarketWindow
    tau_s: float
    spot: float                # Binance mid
    spot_eff: float            # in resolution-source units
    oracle_proxy: str
    reference: float
    ref_source: str
    sigma: float
    sigma_lo: float
    sigma_hi: float
    oracle_noise: float
    x: float
    d: float
    p_up: float
    p_up_lo: float
    p_up_hi: float
    up: Optional[SideQuote]
    down: Optional[SideQuote]
    best: Optional[SideQuote]
    decision: str
    reasons: Tuple[str, ...]
    threshold: float
    sanity_div_bps: Optional[float] = None
    spot_age_ms: float = 0.0
    oracle_age_ms: Optional[float] = None
    book_age_ms: Optional[float] = None
    detect_latency_ms: Optional[float] = None
    trigger_ts: Optional[float] = None   # receive time of the frame that caused this evaluation
    signal_id: str = ""
    exec_status: str = ""                # what the executor did with a SIGNAL (queued / refusal reason)
    vol_scale_s: float = 1.0             # sampling scale the point sigma comes from
    conv: float = 1.0                    # spot_eff / Binance mid (resolution-unit conversion)
    mkt_mid_up: Optional[float] = None   # market's P(up): mid of the Up book incl. the complement
    mkt_half_spread: Optional[float] = None
    lag_err: Optional[float] = None      # |P_model(t - lag) - market mid|, best over the lookbacks
    lag_s: Optional[float] = None        # the lookback that matched best (0 = now)
    sigma_implied: Optional[float] = None  # sigma at which our model reproduces the market mid
    x_implied: Optional[float] = None      # ln(S/K) the market mid implies under our sigma


def market_mid_up(ub: OrderBook, db: OrderBook) -> Optional[Tuple[float, float]]:
    """(mid, half spread) of the market's P(up), taking the complement book into account:
    bid_up = max(bid Up, 1 - ask Down), ask_up = min(ask Up, 1 - bid Down)."""
    ub_b, ub_a, db_b, db_a = ub.best_bid(), ub.best_ask(), db.best_bid(), db.best_ask()
    bids = [p for p in ((ub_b[0] if ub_b else None), (1.0 - db_a[0] if db_a else None)) if p is not None]
    asks = [p for p in ((ub_a[0] if ub_a else None), (1.0 - db_b[0] if db_b else None)) if p is not None]
    if not bids or not asks:
        return None
    bid, ask = max(bids), min(asks)
    if not 0.0 < bid <= ask < 1.0:
        return None
    return 0.5 * (bid + ask), 0.5 * (ask - bid)


@dataclass
class EngineStats:
    evaluations: int = 0
    unpriced: Counter = field(default_factory=Counter)    # why a window could not be priced
    gates: Counter = field(default_factory=Counter)       # why a priced window could not trade
    signals: int = 0
    gated: int = 0


class SignalEngine:
    def __init__(self, cfg: Settings, hub: MarketDataHub, refs: ReferenceResolver, clock: Clock, sink) -> None:
        self.cfg = cfg
        self.hub = hub
        self.refs = refs
        self.clock = clock
        self.sink = sink
        self.stats = EngineStats()
        self._last_logged: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
        self._last_snapshot: Dict[str, float] = {}
        self._blocked_until: Dict[str, float] = {}       # slug -> ts (sanity divergence)
        self._seq = 0
        self.executor = None                               # set by ExecutionPipeline in paper mode
        self._lookbacks = tuple(sorted(float(x) for x in cfg.LATENCY_LOOKBACKS_S))
        hub.spot_listeners.append(self.on_spot)
        hub.book_listeners.append(self.on_book)

    # ------------------------------------------------------------------ triggers
    def on_spot(self, asset: str, trigger_ts: float) -> None:
        for w in self.hub.by_asset.get(asset, ()):
            self._run(w, trigger_ts)

    def on_book(self, w: MarketWindow, trigger_ts: float) -> None:
        self._run(w, trigger_ts)

    def on_timer(self) -> None:
        now = self.clock.now()
        self.hub.carry_forward(now)
        for w in list(self.hub.markets.values()):
            self._run(w, None)
        for slug in [s for s, t in self._blocked_until.items() if t < now]:
            del self._blocked_until[slug]
        if self.executor is not None:
            self.executor.on_timer(now)

    # ------------------------------------------------------------------ core
    def _new_signal_id(self, ev: Evaluation) -> str:
        self._seq += 1
        return f"{int(ev.ts * 1000)}-{self._seq}"

    def _run(self, w: MarketWindow, trigger_ts: Optional[float]) -> None:
        ev = self.evaluate(w, trigger_ts)
        if ev is None:
            return
        self._maybe_snapshot(ev)
        if self.executor is not None:
            self.executor.on_evaluation(ev)
            if ev.decision == SIGNAL:
                ev.signal_id = self._new_signal_id(ev)
                ev.exec_status = self.executor.on_signal(ev)
        if ev.decision != NONE:
            self._maybe_log_signal(ev, force=ev.exec_status == "queued")

    def execute_fresh(self, w: MarketWindow, side: str, plan: str) -> str:
        """Re-detect from scratch (e.g. taker fallback after a maker timeout): a new evaluation on
        the current book must still say SIGNAL on this side before anything is sent."""
        ev = self.evaluate(w, self.clock.now())
        if ev is None or ev.decision != SIGNAL or ev.best.side != side:
            return "signal_gone"
        ev.signal_id = self._new_signal_id(ev)
        ev.exec_status = self.executor.on_signal(ev, plan=plan)
        self._maybe_log_signal(ev, force=True)
        return ev.exec_status

    def _side(self, name: str, p: float, p_cons: float, book: OrderBook, other: OrderBook,
              rate: float, exponent: float) -> Optional[SideQuote]:
        ask = book.best_ask()
        px, size, src = (ask[0], ask[1], "book") if ask else (None, 0.0, "")
        if self.cfg.USE_COMPLEMENT_BOOK:
            ob = other.best_bid()
            if ob is not None:
                comp = round(1.0 - ob[0], 6)
                if px is None or comp < px:
                    px, size, src = comp, ob[1], "complement"
        if px is None:
            return None
        bid = book.best_bid()
        fee = taker_fee_per_share(px, rate, exponent)
        cost = px + fee + self.cfg.EXPECTED_SLIPPAGE
        return SideQuote(name, px, size, src, bid[0] if bid else None, p, p_cons, fee, cost,
                         p - cost, p_cons - cost)

    def evaluate(self, w: MarketWindow, trigger_ts: Optional[float] = None,
                 books: Optional[Tuple[OrderBook, OrderBook]] = None) -> Optional[Evaluation]:
        """books=(up, down) prices against those books (a fresh REST snapshot before an order)
        instead of the local WebSocket books."""
        cfg, hub = self.cfg, self.hub
        now = self.clock.now()
        if now < w.start_ts or now >= w.end_ts:
            return None
        st = hub.assets.get(w.asset)
        if st is None:
            return None
        tau = w.end_ts - now
        q = st.fast
        if q is None:
            self.stats.unpriced["spot_missing"] += 1
            return None
        ref, why = self.refs.get(w, now)
        if ref is None:
            self.stats.unpriced[why] += 1
            return None
        vol, why = st.vol.estimate(tau, cfg.VOL_FAST_HORIZON_S, cfg.SIGMA_UNCERTAINTY)
        if vol is None:
            self.stats.unpriced[why] += 1
            return None
        self.stats.evaluations += 1
        reasons = []

        # -------- spot in resolution units
        mid = q.mid
        spot_age = now - q.recv_ts
        if spot_age > cfg.SPOT_STALE_S:
            reasons.append("spot_stale")
        oracle_age = None
        if w.resolution == CHAINLINK:
            ob = st.oracle_basis
            oracle_age = (now - st.oracle.recv_ts) if st.oracle is not None else math.inf
            if ob.ready and oracle_age <= cfg.ORACLE_STALE_S:
                spot_eff, noise, proxy = mid * math.exp(ob.mean), ob.std, "chainlink"
            elif not cfg.REQUIRE_ORACLE_FEED and st.sanity_basis.ready:
                sb = st.sanity_basis
                spot_eff, noise, proxy = mid * math.exp(sb.mean), sb.std, "coinbase"
            else:
                spot_eff = mid * math.exp(ob.mean if ob.n else 0.0)
                noise, proxy = ob.std, "none"
                if st.oracle is None:
                    reasons.append("oracle_missing")
                elif oracle_age > cfg.ORACLE_STALE_S:
                    reasons.append("oracle_stale")
                else:
                    reasons.append("oracle_basis_warmup")
        elif w.resolution == BINANCE:
            spot_eff, noise, proxy = mid, 0.5 * (q.ask - q.bid) / mid, "binance"
        else:
            spot_eff, noise, proxy = mid, 0.0, "none"
            reasons.append("resolution_unknown")

        # -------- independent cross-venue sanity check
        div_bps = None
        sq = st.sanity
        if sq is None:
            if cfg.REQUIRE_SANITY_SOURCE:
                reasons.append("sanity_missing" if w.asset in cfg.COINBASE_PRODUCTS else "sanity_unavailable")
        elif now - sq.recv_ts > cfg.SANITY_STALE_S:
            if cfg.REQUIRE_SANITY_SOURCE:
                reasons.append("sanity_stale")
        else:
            # Two different questions, because Coinbase ticks are sparser than Binance's:
            #  * at MATCHED times (Binance mid when the Coinbase quote arrived) the venues must agree;
            #    if not, one feed is wrong -> the window is skipped (ТЗ), capped for long windows.
            #  * NOW, right after a fast Binance move, Coinbase may simply not have ticked yet: that
            #    is exactly when latency signals appear, so it is a transient "unconfirmed" gate
            #    that clears when Coinbase catches up, not a reason to drop the window.
            sb = st.sanity_basis
            mean = sb.mean if sb.ready else 0.0
            spreads = 0.5 * ((sq.ask - sq.bid) / sq.mid + (q.ask - q.bid) / mid)
            tol = max(cfg.SANITY_MIN_DIVERGENCE_BPS * 1e-4, cfg.SANITY_K_STD * sb.std, spreads)
            hit = st.fast_hist.at(sq.recv_ts)
            matched = math.log(sq.mid / (hit[1] if hit is not None else mid))
            dev_now = math.log(sq.mid / mid) - mean
            div_bps = dev_now * 1e4
            if abs(matched) > cfg.SANITY_MAX_ABS_BASIS_BPS * 1e-4 or (sb.ready and abs(matched - mean) > tol):
                self._blocked_until[w.slug] = max(self._blocked_until.get(w.slug, 0.0),
                                                  min(w.end_ts, now + cfg.SANITY_BLOCK_MAX_S))
            elif not sb.ready:
                if cfg.REQUIRE_SANITY_SOURCE:
                    reasons.append("sanity_warmup")
            elif abs(dev_now) > tol:
                reasons.append("sanity_unconfirmed")
        if self._blocked_until.get(w.slug, 0.0) > now:
            reasons.append("sanity_divergence")

        # -------- environment / timing gates
        skew = hub.clock_skew_ms
        if skew is not None and abs(skew) > cfg.MAX_CLOCK_SKEW_MS:
            reasons.append("clock_skew")
        if tau < cfg.MIN_TIME_LEFT_S:
            reasons.append("min_time_left")
        if cfg.MAX_TIME_LEFT_S > 0 and tau > cfg.MAX_TIME_LEFT_S:
            reasons.append("max_time_left")
        if w.closed or not w.accepting_orders:
            reasons.append("market_closed")

        # -------- fair value (sigma from the sampling scale closest to the horizon)
        lo, sig, hi = vol.lo, vol.point, vol.hi
        fv = fair_value(spot_eff, ref.price, tau, sig, noise, cfg.TAIL_DOF)
        p_lo, p_hi = fair_value_band(spot_eff, ref.price, tau, (lo, sig, hi), noise, cfg.TAIL_DOF)

        # -------- market side
        up = down = best = None
        book_age = None
        mkt = None
        lag_err = lag_s = None
        ub, db = books if books is not None else (hub.books.get(w.up_token), hub.books.get(w.down_token))
        if ub is None or db is None or not (ub.synced and db.synced):
            reasons.append("book_missing")
        else:
            book_age = (now - max(ub.updated_ts, db.updated_ts)) * 1000.0
            if hub.feed_age("polymarket", now) > cfg.POLY_STALE_S:
                reasons.append("book_stale")
            if ub.crossed or db.crossed:
                reasons.append("book_crossed")
            rate = w.taker_fee_rate if w.taker_fee_rate is not None else cfg.TAKER_FEE_RATE
            exp = w.fee_exponent if w.fee_exponent is not None else cfg.FEE_EXPONENT
            up = self._side("up", fv.p_up, p_lo, ub, db, rate, exp)
            down = self._side("down", fv.p_down, 1.0 - p_hi, db, ub, rate, exp)
            eligible = [s for s in (up, down) if s is not None and cfg.MIN_ASK <= s.ask <= cfg.MAX_ASK]
            if eligible:
                best = max(eligible, key=lambda s: s.edge_cons)
            elif up is None and down is None:
                reasons.append("no_ask")
            else:
                reasons.append("ask_out_of_range")
            mkt = market_mid_up(ub, db)
            # Latency thesis check: the market must look like OUR model a moment ago. If it matches no
            # recent version of the model, it disagrees with the model itself (vol, reference, or it is
            # ahead of our feed) and the "edge" is not a latency effect.
            if (cfg.REQUIRE_LATENCY_EXPLANATION and mkt is not None and best is not None
                    and best.edge_cons >= cfg.EDGE_THRESHOLD):
                lag_err, lag_s = self._lag_match(st, now, spot_eff / mid, ref.price, tau, sig, noise, fv.p_up, mkt[0])
                if lag_err is not None and lag_err > max(cfg.LATENCY_MATCH_TOL, mkt[1]):
                    reasons.append("edge_not_latency")

        if best is not None and best.edge_cons >= cfg.EDGE_THRESHOLD:
            decision = GATED if reasons else SIGNAL
        else:
            decision = NONE
        for r in reasons:
            self.stats.gates[r] += 1
        latency = None
        if trigger_ts is not None:
            latency = (self.clock.now() - trigger_ts) * 1000.0
        return Evaluation(
            ts=now, window=w, tau_s=tau, spot=mid, spot_eff=spot_eff, oracle_proxy=proxy,
            reference=ref.price, ref_source=ref.source, sigma=sig, sigma_lo=lo, sigma_hi=hi,
            oracle_noise=noise, x=fv.x, d=fv.d, p_up=fv.p_up, p_up_lo=p_lo, p_up_hi=p_hi,
            up=up, down=down, best=best, decision=decision, reasons=tuple(reasons),
            threshold=cfg.EDGE_THRESHOLD, sanity_div_bps=div_bps, spot_age_ms=spot_age * 1000.0,
            oracle_age_ms=None if oracle_age is None or math.isinf(oracle_age) else oracle_age * 1000.0,
            book_age_ms=book_age, detect_latency_ms=latency, trigger_ts=trigger_ts, vol_scale_s=vol.scale_s,
            conv=spot_eff / mid, mkt_mid_up=mkt[0] if mkt else None, mkt_half_spread=mkt[1] if mkt else None,
            lag_err=lag_err, lag_s=lag_s)

    def _lag_match(self, st, now: float, conv: float, reference: float, tau: float, sigma: float, noise: float,
                   p_now: float, mkt_up: float) -> Tuple[Optional[float], Optional[float]]:
        """Smallest |P_model(t - lag) - market mid| over lag in {0} + LATENCY_LOOKBACKS_S."""
        best_err, best_lag = abs(p_now - mkt_up), 0.0
        for lag in self._lookbacks:
            hit = st.fast_hist.at(now - lag)
            if hit is None:
                continue
            p = fair_value(hit[1] * conv, reference, tau + lag, sigma, noise, self.cfg.TAIL_DOF).p_up
            err = abs(p - mkt_up)
            if err < best_err:
                best_err, best_lag = err, lag
        return best_err, best_lag

    def _fill_diagnostics(self, ev: Evaluation) -> None:
        """Market-implied sigma / moneyness and the lag match — computed only for rows that get logged."""
        if ev.mkt_mid_up is None or ev.sigma_implied is not None or ev.x_implied is not None:
            return
        p = min(max(ev.mkt_mid_up, 1e-4), 1 - 1e-4)
        ev.sigma_implied = implied_sigma(ev.spot_eff, ev.reference, ev.tau_s, p, ev.oracle_noise, self.cfg.TAIL_DOF)
        ev.x_implied = implied_log_moneyness(ev.tau_s, ev.sigma, p, ev.oracle_noise, self.cfg.TAIL_DOF)
        if ev.lag_err is None:
            st = self.hub.assets.get(ev.window.asset)
            if st is not None:
                ev.lag_err, ev.lag_s = self._lag_match(st, ev.ts, ev.conv, ev.reference, ev.tau_s, ev.sigma,
                                                       ev.oracle_noise, ev.p_up, ev.mkt_mid_up)

    # ------------------------------------------------------------------ logging policy
    def _maybe_snapshot(self, ev: Evaluation) -> None:
        # ~20 samples per window regardless of its length, denser near the close
        interval = min(30.0, max(1.0, 0.05 * ev.tau_s))
        last = self._last_snapshot.get(ev.window.slug)
        if last is None or ev.ts - last >= interval:
            self._last_snapshot[ev.window.slug] = ev.ts
            self._fill_diagnostics(ev)
            self.sink.snapshot(ev)

    def _maybe_log_signal(self, ev: Evaluation, force: bool = False) -> None:
        key = (ev.window.slug, ev.best.side, ev.decision)
        prev = self._last_logged.get(key)
        if prev is not None and not force:
            last_ts, last_edge = prev
            if (ev.ts - last_ts < self.cfg.SIGNAL_RELOG_S
                    and abs(ev.best.edge_cons - last_edge) < self.cfg.SIGNAL_RELOG_EDGE_DELTA):
                return
        self._last_logged[key] = (ev.ts, ev.best.edge_cons)
        if not ev.signal_id:
            ev.signal_id = self._new_signal_id(ev)
        self._fill_diagnostics(ev)
        if ev.decision == SIGNAL:
            self.stats.signals += 1
        else:
            self.stats.gated += 1
        self.sink.signal(ev, ev.signal_id)

    def forget(self, live_slugs) -> None:
        for k in [k for k in self._last_logged if k[0] not in live_slugs]:
            del self._last_logged[k]
        for s in [s for s in self._last_snapshot if s not in live_slugs]:
            del self._last_snapshot[s]

"""CSV sinks for everything the analysis needs later.

    signals.csv            every SIGNAL / GATED evaluation (deduplicated by the engine),
                           with all model inputs, the market price, fees, edge and gate reasons
    snapshots-YYYYMMDD.csv periodic model-vs-market samples for EVERY priced window, whether or
                           not it signalled (calibration + missed-opportunity analysis)
    outcomes.csv           resolved winner per window (from Gamma)

Files are append-only; the realized PnL of a signal is obtained by joining
signals.csv with outcomes.csv (bot.py analyze), so nothing is ever rewritten.
Log lines report measured numbers only.
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger("latarb.signals")

SIGNAL_FIELDS = [
    "ts_iso", "ts", "signal_id", "decision", "slug", "asset", "label", "resolution", "window_start",
    "window_end", "tau_s", "spot", "spot_eff", "oracle_proxy", "reference", "ref_source", "x_bps",
    "sigma_bps", "sigma_lo_bps", "sigma_hi_bps", "oracle_noise_bps", "d", "p_fair_up", "p_fair_up_lo",
    "p_fair_up_hi", "side", "p_fair", "p_fair_cons", "p_market", "ask_src", "ask_size", "bid", "fee",
    "slippage", "cost", "edge", "edge_cons", "threshold", "up_ask", "down_ask", "reasons",
    "sanity_div_bps", "spot_age_ms", "oracle_age_ms", "book_age_ms", "detect_latency_ms", "executed",
]
SNAPSHOT_FIELDS = [
    "ts", "slug", "asset", "label", "resolution", "tau_s", "spot_eff", "reference", "x_bps",
    "sigma_bps", "oracle_noise_bps", "d", "p_fair_up", "p_fair_up_lo", "p_fair_up_hi",
    "up_bid", "up_ask", "down_bid", "down_ask", "best_side", "best_edge_cons", "reasons",
]
OUTCOME_FIELDS = ["slug", "asset", "label", "window_start", "window_end", "winner", "source", "resolved_ts"]


def _f(v, nd: int = 6):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) >= 1e6 else round(v, nd)
    return v


class CsvAppender:
    def __init__(self, path: str, fields: List[str]) -> None:
        self.path = path
        self.fields = fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != fields:
                # schema changed between versions: keep the old file, start a new one
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                os.replace(path, f"{path}.old-{stamp}")
                log.warning("%s had a different header; moved aside", path)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=fields)
        if new:
            self._w.writeheader()
        self.rows = 0

    def write(self, row: Dict) -> None:
        self._w.writerow(row)
        self.rows += 1

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_outcomes(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("winner") in ("up", "down"):
                    out[r["slug"]] = r["winner"]
    return out


class SignalSink:
    def __init__(self, directory: str, echo: bool = True) -> None:
        self.directory = directory
        self.echo = echo
        os.makedirs(directory, exist_ok=True)
        self.signals = CsvAppender(os.path.join(directory, "signals.csv"), SIGNAL_FIELDS)
        self.outcomes = CsvAppender(os.path.join(directory, "outcomes.csv"), OUTCOME_FIELDS)
        self._snap: Optional[CsvAppender] = None
        self._snap_day = ""

    # ------------------------------------------------------------------ writers
    def signal(self, ev, signal_id: str) -> None:
        w, b = ev.window, ev.best
        row = {
            "ts_iso": datetime.fromtimestamp(ev.ts, timezone.utc).isoformat(timespec="milliseconds"),
            "ts": round(ev.ts, 3), "signal_id": signal_id, "decision": ev.decision,
            "slug": w.slug, "asset": w.asset, "label": w.label, "resolution": w.resolution,
            "window_start": int(w.start_ts), "window_end": int(w.end_ts), "tau_s": round(ev.tau_s, 3),
            "spot": _f(ev.spot, 10), "spot_eff": _f(ev.spot_eff, 10), "oracle_proxy": ev.oracle_proxy,
            "reference": _f(ev.reference, 10), "ref_source": ev.ref_source,
            "x_bps": _f(ev.x * 1e4, 4), "sigma_bps": _f(ev.sigma * 1e4, 4),
            "sigma_lo_bps": _f(ev.sigma_lo * 1e4, 4), "sigma_hi_bps": _f(ev.sigma_hi * 1e4, 4),
            "oracle_noise_bps": _f(ev.oracle_noise * 1e4, 4), "d": _f(ev.d, 4),
            "p_fair_up": _f(ev.p_up), "p_fair_up_lo": _f(ev.p_up_lo), "p_fair_up_hi": _f(ev.p_up_hi),
            "side": b.side, "p_fair": _f(b.p_fair), "p_fair_cons": _f(b.p_fair_cons),
            "p_market": _f(b.ask), "ask_src": b.ask_src, "ask_size": _f(b.ask_size, 2), "bid": _f(b.bid),
            "fee": _f(b.fee), "slippage": _f(b.cost - b.ask - b.fee), "cost": _f(b.cost),
            "edge": _f(b.edge), "edge_cons": _f(b.edge_cons), "threshold": ev.threshold,
            "up_ask": _f(ev.up.ask) if ev.up else "", "down_ask": _f(ev.down.ask) if ev.down else "",
            "reasons": "|".join(ev.reasons), "sanity_div_bps": _f(ev.sanity_div_bps, 3),
            "spot_age_ms": _f(ev.spot_age_ms, 1), "oracle_age_ms": _f(ev.oracle_age_ms, 1),
            "book_age_ms": _f(ev.book_age_ms, 1), "detect_latency_ms": _f(ev.detect_latency_ms, 3),
            "executed": 0,
        }
        self.signals.write(row)
        self.signals.flush()
        if self.echo:
            log.info("%s %s %s %s buy %s | %.1fs left | S=%.8g K=%.8g (%s) x=%+.2fbp sigma=%.3fbp/s^0.5 | "
                     "P_fair=%.4f [cons %.4f] vs ask %.3f%s fee %.4f | edge %+.4f (cons %+.4f)%s%s",
                     ev.decision, w.asset.upper(), w.label, w.slug, b.side.upper(), ev.tau_s, ev.spot_eff,
                     ev.reference, ev.ref_source, ev.x * 1e4, ev.sigma * 1e4, b.p_fair, b.p_fair_cons, b.ask,
                     "*" if b.ask_src == "complement" else "", b.fee, b.edge, b.edge_cons,
                     f" | detect {ev.detect_latency_ms:.2f}ms" if ev.detect_latency_ms is not None else "",
                     f" | gates: {','.join(ev.reasons)}" if ev.reasons else "")

    def snapshot(self, ev) -> None:
        day = datetime.fromtimestamp(ev.ts, timezone.utc).strftime("%Y%m%d")
        if day != self._snap_day:
            if self._snap is not None:
                self._snap.close()
            self._snap = CsvAppender(os.path.join(self.directory, f"snapshots-{day}.csv"), SNAPSHOT_FIELDS)
            self._snap_day = day
        w = ev.window
        self._snap.write({
            "ts": round(ev.ts, 3), "slug": w.slug, "asset": w.asset, "label": w.label,
            "resolution": w.resolution, "tau_s": round(ev.tau_s, 3), "spot_eff": _f(ev.spot_eff, 10),
            "reference": _f(ev.reference, 10), "x_bps": _f(ev.x * 1e4, 4), "sigma_bps": _f(ev.sigma * 1e4, 4),
            "oracle_noise_bps": _f(ev.oracle_noise * 1e4, 4), "d": _f(ev.d, 4),
            "p_fair_up": _f(ev.p_up), "p_fair_up_lo": _f(ev.p_up_lo), "p_fair_up_hi": _f(ev.p_up_hi),
            "up_bid": _f(ev.up.bid) if ev.up else "", "up_ask": _f(ev.up.ask) if ev.up else "",
            "down_bid": _f(ev.down.bid) if ev.down else "", "down_ask": _f(ev.down.ask) if ev.down else "",
            "best_side": ev.best.side if ev.best else "",
            "best_edge_cons": _f(ev.best.edge_cons) if ev.best else "", "reasons": "|".join(ev.reasons),
        })

    def outcome(self, w, winner: str, source: str, ts: float) -> None:
        self.outcomes.write({"slug": w.slug, "asset": w.asset, "label": w.label,
                             "window_start": int(w.start_ts), "window_end": int(w.end_ts),
                             "winner": winner, "source": source, "resolved_ts": round(ts, 3)})
        self.outcomes.flush()

    def flush(self) -> None:
        self.signals.flush()
        self.outcomes.flush()
        if self._snap is not None:
            self._snap.flush()

    def close(self) -> None:
        self.signals.close()
        self.outcomes.close()
        if self._snap is not None:
            self._snap.close()


class OutcomeTracker:
    """Which finished windows still need their winner fetched from Gamma."""

    def __init__(self, cfg, sink: SignalSink) -> None:
        self.cfg = cfg
        self.sink = sink
        self.known: Dict[str, str] = read_outcomes(sink.outcomes.path)
        self.pending: Dict[str, object] = {}
        self._next_try: Dict[str, float] = {}

    def track(self, windows) -> None:
        for w in windows:
            if w.slug not in self.known:
                self.pending.setdefault(w.slug, w)

    def due(self, now: float) -> list:
        out = []
        for slug, w in list(self.pending.items()):
            if now > w.end_ts + self.cfg.RESOLVE_GIVE_UP_S:
                log.warning("outcome for %s not available after %.0fh - giving up", slug,
                            self.cfg.RESOLVE_GIVE_UP_S / 3600)
                del self.pending[slug]
                continue
            if now >= w.end_ts + self.cfg.RESOLVE_GRACE_S and self._next_try.get(slug, 0.0) <= now:
                self._next_try[slug] = now + self.cfg.RESOLVE_INTERVAL_S
                out.append(w)
        return out

    def record(self, w, winner: str, now: float, source: str = "gamma") -> None:
        if w.slug in self.known:
            return
        self.known[w.slug] = winner
        self.pending.pop(w.slug, None)
        self._next_try.pop(w.slug, None)
        self.sink.outcome(w, winner, source, now)

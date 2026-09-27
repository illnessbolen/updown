"""Weekly report (as in the previous bot), written as Markdown to DATA_DIR/reports/.

The running bot checks once an hour; a report is written when 7 days have
passed since the previous one (report_state.json). The first run only starts
the clock. `bot.py report` writes one immediately.

Contents: the period's paper trades (metrics + breakdowns), cumulative
metrics, execution quality (sent / dropped by reason / latency / fill ratio),
signal counts with the shadow outcome of signals, and the hypothesis test on
all settled trades. Only measured numbers; small samples are flagged.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional

from . import hypothesis
from .metrics import BREAKDOWNS, breakdown, compute, format_breakdown, format_metrics
from .trades import filter_period, load_paper_trades, load_shadow_trades, paper_start_equity

log = logging.getLogger("latarb.report")

STATE_FILE = "report_state.json"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _block(lines: List[str]) -> List[str]:
    return ["```"] + lines + ["```", ""]


def execution_summary(data_dir: str, since: float, until: float) -> List[str]:
    orders = [o for o in _rows(os.path.join(data_dir, "orders.csv"))
              if since <= (_num(o.get("ts")) or 0.0) < until]
    if not orders:
        return ["попыток исполнения за период нет"]
    sent = [o for o in orders if o["status"] != "dropped"]
    lat = sorted(_num(o["latency_ms"]) for o in sent if _num(o.get("latency_ms")) is not None)
    req = sum(_num(o["shares_req"]) or 0.0 for o in sent)
    got = sum(_num(o["shares_filled"]) or 0.0 for o in sent)
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]  # noqa: E731
    out = [f"попыток {len(orders)}, отправлено {len(sent)} "
           f"({dict(Counter(o['kind'] for o in sent))}), результаты {dict(Counter(o['kind'] + ':' + o['status'] for o in sent))}",
           f"сброшено до отправки: {dict(Counter(o['reason'] for o in orders if o['status'] == 'dropped')) or '-'}"]
    if lat:
        out.append(f"задержка сигнал→отправка: p50 {pct(0.5):.0f} мс, p95 {pct(0.95):.0f} мс, макс {lat[-1]:.0f} мс; "
                   f"fill ratio {got / req:.1%}" if req else "")
    return out


def signal_summary(data_dir: str, since: float, until: float) -> List[str]:
    rows = [r for r in _rows(os.path.join(data_dir, "signals.csv"))
            if since <= (_num(r.get("ts")) or 0.0) < until]
    c = Counter(r["decision"] for r in rows)
    windows = len({(r["slug"], r["side"]) for r in rows if r["decision"] == "SIGNAL"})
    gates = Counter(g for r in rows if r["decision"] == "GATED" for g in r["reasons"].split("|") if g)
    return [f"записей SIGNAL {c.get('SIGNAL', 0)} (окон/сторон {windows}), GATED {c.get('GATED', 0)}",
            f"частые гейты: {dict(gates.most_common(6)) or '-'}"]


def build_report(data_dir: str, now: float, days: float = 7.0, alpha: float = 0.05, sims: int = 5000,
                 profile: str = "") -> str:
    since = now - days * 86400.0
    start_equity = paper_start_equity(data_dir)
    paper_all = load_paper_trades(data_dir)
    paper = filter_period(paper_all, since, now)
    shadow_all = load_shadow_trades(data_dir)
    shadow = filter_period(shadow_all, since, now)
    md = [f"# Еженедельный отчёт: {_iso(since)} — {_iso(now)}", "",
          f"Сформирован {_iso(now)} | данные: `{os.path.abspath(data_dir)}`"
          + (f" | риск-профиль: {profile}" if profile else ""), "",
          "Все цифры — фактические метрики по закрытым сделкам (paper = симуляция против живой книги, "
          "оценка сверху для реальной торговли).", ""]

    md += ["## Paper-сделки за период", ""] + _block(format_metrics(compute(paper, start_equity)))
    if paper:
        for dim in ("asset", "label", "kind"):
            md += _block(format_breakdown(dim, breakdown(paper, BREAKDOWNS[dim])))
    md += ["## Paper-сделки с начала", ""] + _block(format_metrics(compute(paper_all, start_equity)))

    md += ["## Исполнение за период", ""] + _block(execution_summary(data_dir, since, now))
    md += ["## Сигналы за период", ""] + _block(signal_summary(data_dir, since, now))
    md += ["### Теневые сигналы за период (гипотетический fill 1 акции по ask + fee + slippage)", ""]
    md += _block(format_metrics(compute(shadow), unit="ед."))

    trades = paper_all if paper_all else shadow_all
    src = "paper" if paper_all else "shadow"
    res = hypothesis.run(trades, alpha=alpha, sims=sims, n_boot=2000, source=src)
    md += [f"## Проверка гипотезы: все закрытые {src}-сделки", ""] + _block(res.lines())
    return "\n".join(md) + "\n"


def write_report(data_dir: str, now: float, days: float = 7.0, **kw) -> str:
    out_dir = os.path.join(data_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "weekly_" + datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d_%H%M") + ".md")
    text = build_report(data_dir, now, days, **kw)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    _save_state(data_dir, now)
    log.info("weekly report -> %s", path)
    return path


def _state_path(data_dir: str) -> str:
    return os.path.join(data_dir, STATE_FILE)


def _save_state(data_dir: str, ts: float) -> None:
    tmp = _state_path(data_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"last_report_ts": ts}, f)
    os.replace(tmp, _state_path(data_dir))


def maybe_write_weekly(data_dir: str, now: float, days: float = 7.0, **kw) -> Optional[str]:
    """Write a report if `days` passed since the last one. The very first call only starts the clock."""
    path = _state_path(data_dir)
    try:
        with open(path, encoding="utf-8") as f:
            last = float(json.load(f)["last_report_ts"])
    except (OSError, ValueError, KeyError):
        os.makedirs(data_dir, exist_ok=True)
        _save_state(data_dir, now)
        return None
    if now - last < days * 86400.0:
        return None
    return write_report(data_dir, now, days, **kw)

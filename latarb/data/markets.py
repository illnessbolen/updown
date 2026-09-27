"""Up/Down market windows: slug grammar, Gamma event parsing, resolution source.

Slug families (all observed on Polymarket; the parser is strict and anything it
does not understand is reported as rejected instead of guessed):

    {asset}-updown-{N}{m|h|d}-{unix_start}           btc-updown-5m-1790000000, eth-updown-4h-...
    {name}-up-or-down-{month}-{day}[-{year}]-{h}{am|pm}-et
                                                     bitcoin-up-or-down-september-27-10am-et (1h, ET, DST-aware)
    {name}-up-or-down-on-{month}-{day}[-{year}]      bitcoin-up-or-down-on-september-27 (daily; start = end - 24h)

Token ids are mapped by OUTCOME NAME ("Up"/"Down"), never by list position.

Resolution source (decides how the reference price is obtained):
    chainlink   "price to beat" = Chainlink price at window start (5m/15m/4h families)
    binance     Binance candle open -> close (hourly family: 1h BTC/USDT candle)
    unknown     discovered and logged, but never priced for trading
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .parsers import parse_iso_ts

ET = ZoneInfo("America/New_York")

CHAINLINK = "chainlink"
BINANCE = "binance"
UNKNOWN = "unknown"

ASSET_ALIASES: Dict[str, str] = {
    "btc": "btc", "bitcoin": "btc",
    "eth": "eth", "ethereum": "eth",
    "sol": "sol", "solana": "sol",
    "xrp": "xrp", "ripple": "xrp",
    "doge": "doge", "dogecoin": "doge",
    "bnb": "bnb", "binance-coin": "bnb",
}

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_UNIT_S = {"m": 60, "h": 3600, "d": 86400}

RE_UNIX = re.compile(r"^(?P<asset>[a-z]+)-updown-(?P<n>\d+)(?P<unit>[mhd])-(?P<start>\d{9,11})$")
RE_HOURLY = re.compile(
    r"^(?P<asset>[a-z]+(?:-[a-z]+)?)-up-or-down-(?P<month>[a-z]+)-(?P<day>\d{1,2})"
    r"(?:-(?P<year>\d{4}))?-(?P<hour>\d{1,2})(?P<ampm>am|pm)-et$")
RE_DAILY = re.compile(
    r"^(?P<asset>[a-z]+(?:-[a-z]+)?)-up-or-down-on-(?P<month>[a-z]+)-(?P<day>\d{1,2})(?:-(?P<year>\d{4}))?$")
RE_UPDOWN_TEXT = re.compile(r"updown|up-or-down|up or down", re.I)


@dataclass(frozen=True)
class SlugInfo:
    asset: str
    label: str
    duration_s: int
    start_ts: Optional[float]     # None when the slug does not encode it (daily)
    kind: str                     # "unix" | "hourly_et" | "daily"


@dataclass(frozen=True)
class MarketWindow:
    slug: str
    asset: str
    label: str
    duration_s: int
    start_ts: float
    end_ts: float
    up_token: str
    down_token: str
    condition_id: str = ""
    resolution: str = UNKNOWN
    tick_size: float = 0.01
    min_order_size: float = 5.0
    taker_fee_rate: Optional[float] = None     # None -> Settings.TAKER_FEE_RATE
    fee_exponent: Optional[float] = None       # None -> Settings.FEE_EXPONENT
    price_to_beat: Optional[float] = None      # only if Gamma publishes it
    closed: bool = False
    accepting_orders: bool = True
    title: str = ""
    slug_kind: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MarketWindow":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def tokens(self) -> Tuple[str, str]:
        return self.up_token, self.down_token


# ---------------------------------------------------------------- slugs
def _label(n: int, unit: str) -> str:
    return f"{n}{unit}"


def _infer_year(month: int, day: int, hint_ts: float) -> int:
    hint = datetime.fromtimestamp(hint_ts, ET)
    year = hint.year
    candidate = datetime(year, month, day, tzinfo=ET).timestamp()
    if candidate > hint_ts + 45 * 86400:     # Jan hint, Dec slug
        year -= 1
    elif candidate < hint_ts - 320 * 86400:  # Dec hint, Jan slug
        year += 1
    return year


def parse_slug(slug: str, hint_ts: Optional[float] = None) -> Optional[SlugInfo]:
    """hint_ts (the market's end time or 'now') is only used when the year is omitted."""
    slug = (slug or "").strip().lower()
    m = RE_UNIX.match(slug)
    if m:
        asset = ASSET_ALIASES.get(m["asset"])
        if asset is None:
            return None
        n, unit = int(m["n"]), m["unit"]
        return SlugInfo(asset, _label(n, unit), n * _UNIT_S[unit], float(m["start"]), "unix")
    m = RE_HOURLY.match(slug)
    if m:
        asset = ASSET_ALIASES.get(m["asset"])
        month = _MONTHS.get(m["month"])
        if asset is None or month is None:
            return None
        hour12 = int(m["hour"])
        if not 1 <= hour12 <= 12:
            return None
        hour = hour12 % 12 + (12 if m["ampm"] == "pm" else 0)
        day = int(m["day"])
        if m["year"]:
            year = int(m["year"])
        elif hint_ts is not None:
            year = _infer_year(month, day, hint_ts)
        else:
            return None
        try:
            start = datetime(year, month, day, hour, tzinfo=ET).timestamp()
        except ValueError:
            return None
        return SlugInfo(asset, "1h", 3600, start, "hourly_et")
    m = RE_DAILY.match(slug)
    if m:
        asset = ASSET_ALIASES.get(m["asset"])
        if asset is None or _MONTHS.get(m["month"]) is None:
            return None
        return SlugInfo(asset, "1d", 86400, None, "daily")
    return None


def deterministic_slug(asset: str, label: str, start_ts: int) -> str:
    return f"{asset}-updown-{label}-{start_ts}"


def looks_like_updown(slug: str, title: str = "") -> bool:
    return bool(RE_UPDOWN_TEXT.search(slug or "") or RE_UPDOWN_TEXT.search(title or ""))


# ---------------------------------------------------------------- Gamma fields
def _json_field(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return v


def _pos_float(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def detect_resolution(market: dict, event: dict) -> str:
    sources = " ".join(str(x) for x in (market.get("resolutionSource"), event.get("resolutionSource")) if x)
    descs = " ".join(str(x) for x in (market.get("description"), event.get("description")) if x)
    for text in (sources.lower(), descs.lower()):
        if "chain.link" in text or "chainlink" in text:
            return CHAINLINK
        if "binance" in text:
            return BINANCE
    return UNKNOWN


def fee_schedule(market: dict) -> Tuple[Optional[float], Optional[float]]:
    fs = _json_field(market.get("feeSchedule"))
    if isinstance(fs, dict):
        rate = fs.get("rate", fs.get("takerRate", fs.get("feeRate")))
        exp = fs.get("exponent")
        try:
            rate = float(rate) if rate is not None else None
            exp = float(exp) if exp is not None else None
        except (TypeError, ValueError):
            return None, None
        if rate is not None and rate > 1:          # given in bps
            rate /= 10_000.0
        return rate, exp
    return None, None


def price_to_beat(market: dict, event: dict) -> Optional[float]:
    for obj in (market, event):
        for key in ("eventMetadata", "metadata"):
            meta = _json_field(obj.get(key))
            if isinstance(meta, dict):
                v = _pos_float(meta.get("priceToBeat", meta.get("price_to_beat")))
                if v:
                    return v
        v = _pos_float(obj.get("priceToBeat"))
        if v:
            return v
    return None


def windows_from_event(event: dict, assets: Iterable[str],
                       now: Optional[float] = None) -> Tuple[List[MarketWindow], List[Tuple[str, str]]]:
    """Gamma event -> (windows, [(slug, reject_reason), ...])."""
    assets = set(assets)
    out: List[MarketWindow] = []
    rejected: List[Tuple[str, str]] = []
    ev_slug = str(event.get("slug") or "")
    for m in event.get("markets") or []:
        slug = str(m.get("slug") or ev_slug)
        end_ts = parse_iso_ts(str(m.get("endDate") or event.get("endDate") or ""))
        info = parse_slug(ev_slug, end_ts or now) or parse_slug(slug, end_ts or now)
        if info is None:
            if looks_like_updown(slug, str(event.get("title") or "")):
                rejected.append((slug, "unparsed_slug"))
            continue
        if info.asset not in assets:
            continue
        start = info.start_ts
        if start is None:
            if end_ts is None:
                rejected.append((slug, "no_end_date"))
                continue
            start = end_ts - info.duration_s
        expected_end = start + info.duration_s
        if end_ts is None:
            end_ts = expected_end
        elif abs(end_ts - expected_end) > 90:
            rejected.append((slug, f"end_mismatch(slug={expected_end:.0f},gamma={end_ts:.0f})"))
            continue
        outcomes = _json_field(m.get("outcomes")) or []
        tokens = _json_field(m.get("clobTokenIds")) or []
        if not isinstance(outcomes, list) or not isinstance(tokens, list) or len(outcomes) != len(tokens):
            rejected.append((slug, "bad_outcomes"))
            continue
        by_name = {str(o).strip().lower(): str(t) for o, t in zip(outcomes, tokens)}
        if "up" not in by_name or "down" not in by_name:
            rejected.append((slug, f"outcomes={outcomes}"))
            continue
        rate, exp = fee_schedule(m)
        out.append(MarketWindow(
            slug=slug, asset=info.asset, label=info.label, duration_s=info.duration_s,
            start_ts=float(start), end_ts=float(expected_end),
            up_token=by_name["up"], down_token=by_name["down"],
            condition_id=str(m.get("conditionId") or ""),
            resolution=detect_resolution(m, event),
            tick_size=_pos_float(m.get("orderPriceMinTickSize")) or 0.01,
            min_order_size=_pos_float(m.get("orderMinSize")) or 5.0,
            taker_fee_rate=rate, fee_exponent=exp,
            price_to_beat=price_to_beat(m, event),
            closed=bool(m.get("closed", False)),
            accepting_orders=bool(m.get("acceptingOrders", True)),
            title=str(event.get("title") or m.get("question") or ""),
            slug_kind=info.kind,
        ))
    return out, rejected


def winner_from_market(market: dict) -> Optional[str]:
    """'up' / 'down' once Gamma shows a closed market with a 1/0 price vector, else None."""
    if not market.get("closed"):
        return None
    outcomes = _json_field(market.get("outcomes")) or []
    prices = _json_field(market.get("outcomePrices")) or []
    if not isinstance(outcomes, list) or not isinstance(prices, list) or len(outcomes) != len(prices):
        return None
    try:
        pairs = [(str(o).strip().lower(), float(p)) for o, p in zip(outcomes, prices)]
    except (TypeError, ValueError):
        return None
    winners = [o for o, p in pairs if p >= 0.99]
    losers = [o for o, p in pairs if p <= 0.01]
    if len(winners) == 1 and len(losers) == len(pairs) - 1 and winners[0] in ("up", "down"):
        return winners[0]
    return None

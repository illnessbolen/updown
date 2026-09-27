"""Pure parsers: raw WebSocket frame -> list of normalized events.

Parsers never touch state. The same functions run on live frames and on
recorded frames during replay, so a parser fix can be re-validated against old
recordings. Unknown / control messages return [] (they still count as
liveness for the transport).

Formats handled (as documented by the venues):

Binance combined stream  {"stream": "btcusdt@bookTicker", "data": {"u":..,"s":"BTCUSDT","b":"..","B":"..","a":"..","A":".."}}
                         {"stream": "btcusdt@trade", "data": {"e":"trade","E":ms,"s":"BTCUSDT","p":"..","q":"..","T":ms,...}}
Coinbase Exchange feed   {"type":"ticker","product_id":"BTC-USD","price":"..","best_bid":"..","best_ask":"..","time":"...Z"}
Polymarket RTDS          {"topic":"crypto_prices_chainlink","payload":{"symbol":"btc/usd","timestamp":ms,"value":x}}
                         (payload may instead carry "data": [{"timestamp":ms,"value":x}, ...] as a backfill)
Polymarket CLOB market   event_type = book | price_change (both the "price_changes" and the older
                         "changes" layout) | tick_size_change | last_trade_price; a frame may be a list.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List

from ..fastjson import loads
from .events import ASK, BID, BookLevel, BookSnapshot, LastTrade, OracleTick, SpotQuote, SpotTrade, TickSizeChange

Parser = Callable[[str, float], list]


def _is_json(raw) -> bool:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    s = raw.lstrip()
    return bool(s) and s[0] in "[{"


def _ms_to_s(v) -> float | None:
    if v in (None, ""):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x / 1000.0 if x > 1e11 else x


def parse_iso_ts(s: str) -> float | None:
    """ISO-8601 with 'Z' and up to 9 fractional digits -> epoch seconds."""
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if "." in s:
        head, tail = s.split(".", 1)
        frac, tz = tail, ""
        for sep in ("+", "-"):
            if sep in tail:
                frac, tz = tail.split(sep, 1)
                tz = sep + tz
                break
        s = f"{head}.{frac[:6].ljust(6, '0')}{tz}"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.timestamp()


class BinanceParser:
    def __init__(self, symbols: Dict[str, str]) -> None:
        self.by_symbol = {v.upper(): k for k, v in symbols.items()}

    def __call__(self, raw: str, recv_ts: float) -> list:
        if not _is_json(raw):
            return []
        msg = loads(raw)
        data = msg.get("data", msg) if isinstance(msg, dict) else None
        if not isinstance(data, dict):
            return []
        asset = self.by_symbol.get(str(data.get("s", "")).upper())
        if asset is None:
            return []
        if data.get("e") == "trade":
            return [SpotTrade("binance", asset, float(data["p"]), float(data["q"]), recv_ts,
                              _ms_to_s(data.get("T") or data.get("E")))]
        if "b" in data and "a" in data:
            bid, ask = float(data["b"]), float(data["a"])
            if bid > 0 and ask > 0 and ask >= bid:
                return [SpotQuote("binance", asset, bid, ask, recv_ts, _ms_to_s(data.get("E") or data.get("T")))]
        return []


class CoinbaseParser:
    def __init__(self, products: Dict[str, str]) -> None:
        self.by_product = {v.upper(): k for k, v in products.items()}

    def __call__(self, raw: str, recv_ts: float) -> list:
        if not _is_json(raw):
            return []
        msg = loads(raw)
        if not isinstance(msg, dict) or msg.get("type") != "ticker":
            return []
        asset = self.by_product.get(str(msg.get("product_id", "")).upper())
        if asset is None:
            return []
        exch_ts = parse_iso_ts(msg.get("time", ""))
        try:
            bid, ask = float(msg["best_bid"]), float(msg["best_ask"])
        except (KeyError, TypeError, ValueError):
            return []
        out: list = []
        if bid > 0 and ask > 0 and ask >= bid:
            out.append(SpotQuote("coinbase", asset, bid, ask, recv_ts, exch_ts))
        if msg.get("price") not in (None, ""):
            out.append(SpotTrade("coinbase", asset, float(msg["price"]),
                                 float(msg.get("last_size") or 0.0), recv_ts, exch_ts))
        return out


class RtdsChainlinkParser:
    TOPIC = "crypto_prices_chainlink"

    def __init__(self, symbols: Dict[str, str]) -> None:
        self.by_symbol = {v.lower(): k for k, v in symbols.items()}

    def __call__(self, raw: str, recv_ts: float) -> list:
        if not _is_json(raw):
            return []
        msg = loads(raw)
        if not isinstance(msg, dict) or msg.get("topic") != self.TOPIC:
            return []
        payload = msg.get("payload") or {}
        asset = self.by_symbol.get(str(payload.get("symbol", "")).lower())
        if asset is None:
            return []
        out: list = []
        if isinstance(payload.get("data"), list):
            for row in payload["data"]:
                ts, v = _ms_to_s(row.get("timestamp")), row.get("value")
                if ts is not None and v is not None and float(v) > 0:
                    out.append(OracleTick("chainlink", asset, float(v), ts, recv_ts))
        elif payload.get("value") is not None:
            ts = _ms_to_s(payload.get("timestamp")) or _ms_to_s(msg.get("timestamp")) or recv_ts
            if float(payload["value"]) > 0:
                out.append(OracleTick("chainlink", asset, float(payload["value"]), ts, recv_ts))
        return out


def _levels(rows) -> list:
    out = []
    for r in rows or []:
        try:
            out.append((float(r["price"]), float(r["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_polymarket(raw: str, recv_ts: float) -> list:
    if not _is_json(raw):
        return []                                   # "PONG" and other text control frames
    msg = loads(raw)
    items = msg if isinstance(msg, list) else [msg]
    out: list = []
    for it in items:
        if not isinstance(it, dict):
            continue
        et = it.get("event_type") or it.get("type")
        ts = _ms_to_s(it.get("timestamp"))
        if et == "book":
            token = it.get("asset_id")
            if token:
                out.append(BookSnapshot(str(token), _levels(it.get("bids") or it.get("buys")),
                                        _levels(it.get("asks") or it.get("sells")), recv_ts, ts))
        elif et == "price_change":
            if isinstance(it.get("price_changes"), list):
                changes = it["price_changes"]
            else:
                changes = [dict(c, asset_id=it.get("asset_id")) for c in it.get("changes") or []]
            for ch in changes:
                try:
                    side = BID if str(ch["side"]).upper() == "BUY" else ASK
                    out.append(BookLevel(str(ch["asset_id"]), side, float(ch["price"]), float(ch["size"]),
                                         recv_ts, ts))
                except (KeyError, TypeError, ValueError):
                    continue
        elif et == "tick_size_change":
            try:
                out.append(TickSizeChange(str(it["asset_id"]), float(it["new_tick_size"]), recv_ts))
            except (KeyError, TypeError, ValueError):
                pass
        elif et == "last_trade_price":
            try:
                out.append(LastTrade(str(it["asset_id"]), float(it["price"]), float(it.get("size") or 0),
                                     str(it.get("side", "")), recv_ts, ts))
            except (KeyError, TypeError, ValueError):
                pass
    return out


def build_parsers(binance_symbols: Dict[str, str], coinbase_products: Dict[str, str],
                  chainlink_symbols: Dict[str, str]) -> Dict[str, Parser]:
    return {
        "binance": BinanceParser(binance_symbols),
        "coinbase": CoinbaseParser(coinbase_products),
        "rtds": RtdsChainlinkParser(chainlink_symbols),
        "polymarket": parse_polymarket,
    }


def parse_safely(parser: Parser, raw: str, recv_ts: float, errors: List[str] | None = None) -> list:
    """Parsers are strict about types; a malformed frame must never kill a feed."""
    try:
        return parser(raw, recv_ts)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        if errors is not None:
            errors.append(f"{type(e).__name__}: {e}")
        return []

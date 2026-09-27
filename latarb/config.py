"""Settings.

Every field can be overridden by an environment variable with the same name in
UPPER_CASE (a .env file next to bot.py is loaded too). Values outside the hard
bounds in BOUNDS are rejected at startup instead of being silently accepted:
a mistyped EDGE_THRESHOLD=0.0003 must stop the bot, not make it trade noise.

Network endpoints are parameters so the bot can be moved to a VPS near the
exchanges (or behind a regional mirror such as data-stream.binance.vision)
without code changes.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from typing import Dict, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


class ConfigError(ValueError):
    pass


ALL_ASSETS: Tuple[str, ...] = ("btc", "eth", "sol", "xrp", "doge", "bnb")


@dataclass
class Settings:
    # ---------------- universe ----------------
    ASSETS: Tuple[str, ...] = ALL_ASSETS
    # durations whose slug is {asset}-updown-{label}-{unix_start} with UTC-aligned
    # starts; they are probed directly every window. Other families (1h, 4h,
    # daily) are found by the periodic Gamma scan.
    DETERMINISTIC_LABELS: Tuple[str, ...] = ("5m", "15m")

    # ---------------- network ----------------
    BINANCE_WS_URL: str = "wss://stream.binance.com:9443/stream"
    BINANCE_REST_URL: str = "https://api.binance.com"
    COINBASE_WS_URL: str = "wss://ws-feed.exchange.coinbase.com"
    POLY_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    RTDS_WS_URL: str = "wss://ws-live-data.polymarket.com"
    GAMMA_URL: str = "https://gamma-api.polymarket.com"
    CLOB_URL: str = "https://clob.polymarket.com"
    HTTP_TIMEOUT_S: float = 5.0
    # WS_PROXY: "" = honour HTTPS_PROXY/NO_PROXY from the environment, "none" = always direct,
    # or an explicit proxy URL (http://..., socks5://...)
    WS_PROXY: str = ""
    WS_OPEN_TIMEOUT_S: float = 10.0
    WS_BACKOFF_MIN_S: float = 0.5
    WS_BACKOFF_MAX_S: float = 30.0
    WS_PROTOCOL_PING_S: float = 20.0
    POLY_APP_PING_S: float = 10.0          # CLOB market channel expects text "PING"
    RTDS_APP_PING_S: float = 5.0
    # a connection that delivers nothing (not even PONG/heartbeat) for this long
    # is torn down and re-established
    WS_SILENCE_RECONNECT_S: float = 30.0
    # POLY_DYNAMIC_SUBSCRIBE: add/remove tokens on a live connection with
    # {"operation": "subscribe"|"unsubscribe"}; if a newly added token gets no
    # book snapshot within POLY_SNAPSHOT_TIMEOUT_S the feed reconnects with the
    # full token set instead.
    POLY_DYNAMIC_SUBSCRIBE: bool = True
    POLY_SNAPSHOT_TIMEOUT_S: float = 8.0

    # ---------------- symbol maps (asset -> venue symbol) ----------------
    BINANCE_SYMBOLS: Dict[str, str] = field(default_factory=lambda: {
        "btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT",
        "xrp": "XRPUSDT", "doge": "DOGEUSDT", "bnb": "BNBUSDT"})
    COINBASE_PRODUCTS: Dict[str, str] = field(default_factory=lambda: {
        "btc": "BTC-USD", "eth": "ETH-USD", "sol": "SOL-USD",
        "xrp": "XRP-USD", "doge": "DOGE-USD"})
    CHAINLINK_SYMBOLS: Dict[str, str] = field(default_factory=lambda: {
        "btc": "btc/usd", "eth": "eth/usd", "sol": "sol/usd",
        "xrp": "xrp/usd", "doge": "doge/usd", "bnb": "bnb/usd"})

    # ---------------- staleness (per source) ----------------
    SPOT_STALE_S: float = 2.0              # Binance bookTicker
    SANITY_STALE_S: float = 10.0           # Coinbase ticker (trade-driven, sparser)
    ORACLE_STALE_S: float = 5.0            # Chainlink via Polymarket RTDS
    POLY_STALE_S: float = 25.0             # CLOB connection silence incl. PONGs

    # ---------------- cross-venue sanity (Binance vs Coinbase) ----------------
    REQUIRE_SANITY_SOURCE: bool = True
    SANITY_MIN_DIVERGENCE_BPS: float = 10.0
    SANITY_K_STD: float = 5.0
    SANITY_MAX_ABS_BASIS_BPS: float = 150.0
    SANITY_BLOCK_MAX_S: float = 900.0      # a divergence blocks the window to its end, at most this long
    BASIS_HALFLIFE_S: float = 300.0
    BASIS_MIN_SAMPLES: int = 30

    # ---------------- resolution oracle ----------------
    # Chainlink-resolved markets need the oracle feed both for the reference
    # price and for the Binance->oracle basis. With REQUIRE_ORACLE_FEED=false
    # Coinbase USD is used as a proxy for the oracle (and flagged in the logs).
    REQUIRE_ORACLE_FEED: bool = True
    ORACLE_BASIS_HALFLIFE_S: float = 120.0
    ORACLE_REF_TOLERANCE_S: float = 2.0
    ORACLE_HISTORY_S: float = 5 * 3600.0
    FAST_HISTORY_S: float = 120.0

    # ---------------- volatility ----------------
    VOL_SAMPLE_S: float = 1.0
    VOL_FAST_HALFLIFE_S: float = 120.0
    VOL_SLOW_HALFLIFE_S: float = 1800.0
    VOL_MIN_SAMPLES: int = 300             # >= 5 minutes of 1s returns before trusting sigma
    VOL_MAX_GAP_S: float = 5.0             # feed gap longer than this restarts the sampling grid
    VOL_FAST_HORIZON_S: float = 120.0      # weight of fast vol = exp(-tau / horizon)
    SIGMA_UNCERTAINTY: float = 0.20        # +-20% band around the sigma range for conservative edge
    TAIL_DOF: float = 5.0                  # Student-t tails; "inf" = Gaussian

    # ---------------- signal ----------------
    EDGE_THRESHOLD: float = 0.03           # min conservative edge (prob. points) after fee+slippage
    EXPECTED_SLIPPAGE: float = 0.005
    TAKER_FEE_RATE: float = 0.07           # fee/share = rate * (p*(1-p))**exponent
    FEE_EXPONENT: float = 1.0
    MIN_TIME_LEFT_S: float = 3.0
    MAX_TIME_LEFT_S: float = 0.0           # 0 = no upper limit
    MIN_ASK: float = 0.01
    MAX_ASK: float = 0.99
    USE_COMPLEMENT_BOOK: bool = True       # buy Down ~ sell Up: ask_down_eff = min(ask_down, 1 - bid_up)
    MAX_CLOCK_SKEW_MS: float = 250.0
    EVAL_TIMER_S: float = 1.0
    SIGNAL_RELOG_S: float = 5.0
    SIGNAL_RELOG_EDGE_DELTA: float = 0.01

    # ---------------- discovery ----------------
    DISCOVERY_TICK_S: float = 5.0          # deterministic 5m/15m probing cadence
    DISCOVERY_SCAN_INTERVAL_S: float = 120.0
    DISCOVERY_PREFETCH_S: float = 90.0     # subscribe books this long before a window opens
    DISCOVERY_NEGATIVE_TTL_S: float = 15.0
    GAMMA_SCAN_TAG: str = "crypto"
    GAMMA_SCAN_PAGE_SIZE: int = 200
    GAMMA_SCAN_MAX_PAGES: int = 10
    GAMMA_SCAN_HORIZON_H: float = 26.0

    # ---------------- outcomes ----------------
    RESOLVE_INTERVAL_S: float = 30.0
    RESOLVE_GRACE_S: float = 20.0
    RESOLVE_GIVE_UP_S: float = 48 * 3600.0

    # ---------------- io ----------------
    DATA_DIR: str = "data"
    RECORD_TICKS: bool = False
    RECORD_ROTATE_S: float = 3600.0
    CLOCK_CHECK_INTERVAL_S: float = 600.0
    STATUS_INTERVAL_S: float = 30.0
    LOG_LEVEL: str = "INFO"


LABEL_SECONDS: Dict[str, int] = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# (min, max) inclusive. Anything outside is a configuration error.
BOUNDS: Dict[str, Tuple[float, float]] = {
    "HTTP_TIMEOUT_S": (0.5, 60),
    "WS_OPEN_TIMEOUT_S": (1, 60),
    "WS_BACKOFF_MIN_S": (0.05, 10),
    "WS_BACKOFF_MAX_S": (1, 600),
    "WS_PROTOCOL_PING_S": (1, 120),
    "POLY_APP_PING_S": (1, 60),
    "RTDS_APP_PING_S": (1, 60),
    "WS_SILENCE_RECONNECT_S": (2, 600),
    "POLY_SNAPSHOT_TIMEOUT_S": (1, 120),
    "SPOT_STALE_S": (0.1, 30),
    "SANITY_STALE_S": (0.5, 120),
    "ORACLE_STALE_S": (0.5, 120),
    "POLY_STALE_S": (1, 300),
    "SANITY_MIN_DIVERGENCE_BPS": (0.5, 500),
    "SANITY_K_STD": (1, 50),
    "SANITY_MAX_ABS_BASIS_BPS": (1, 2000),
    "SANITY_BLOCK_MAX_S": (0, 86400),
    "BASIS_HALFLIFE_S": (5, 86400),
    "BASIS_MIN_SAMPLES": (1, 100000),
    "ORACLE_BASIS_HALFLIFE_S": (5, 86400),
    "ORACLE_REF_TOLERANCE_S": (0, 30),
    "ORACLE_HISTORY_S": (60, 7 * 86400),
    "FAST_HISTORY_S": (10, 3600),
    "VOL_SAMPLE_S": (0.1, 60),
    "VOL_FAST_HALFLIFE_S": (5, 86400),
    "VOL_SLOW_HALFLIFE_S": (5, 7 * 86400),
    "VOL_MIN_SAMPLES": (30, 1000000),
    "VOL_MAX_GAP_S": (0.5, 600),
    "VOL_FAST_HORIZON_S": (1, 86400),
    "SIGMA_UNCERTAINTY": (0.0, 0.9),
    "TAIL_DOF": (2.5, math.inf),
    "EDGE_THRESHOLD": (0.005, 0.5),
    "EXPECTED_SLIPPAGE": (0.0, 0.1),
    "TAKER_FEE_RATE": (0.0, 1.0),
    "FEE_EXPONENT": (0.5, 4.0),
    "MIN_TIME_LEFT_S": (1.0, 3600),
    "MAX_TIME_LEFT_S": (0.0, 30 * 86400),
    "MIN_ASK": (0.001, 0.5),
    "MAX_ASK": (0.5, 0.999),
    "MAX_CLOCK_SKEW_MS": (1, 10000),
    "EVAL_TIMER_S": (0.05, 60),
    "SIGNAL_RELOG_S": (0.1, 3600),
    "SIGNAL_RELOG_EDGE_DELTA": (0.0, 1.0),
    "DISCOVERY_TICK_S": (1, 300),
    "DISCOVERY_SCAN_INTERVAL_S": (60, 300),   # ТЗ: раз в 1-5 минут
    "DISCOVERY_PREFETCH_S": (0, 3600),
    "DISCOVERY_NEGATIVE_TTL_S": (1, 600),
    "GAMMA_SCAN_PAGE_SIZE": (10, 500),
    "GAMMA_SCAN_MAX_PAGES": (1, 100),
    "GAMMA_SCAN_HORIZON_H": (1, 24 * 14),
    "RESOLVE_INTERVAL_S": (5, 3600),
    "RESOLVE_GRACE_S": (0, 3600),
    "RESOLVE_GIVE_UP_S": (600, 30 * 86400),
    "RECORD_ROTATE_S": (60, 86400),
    "CLOCK_CHECK_INTERVAL_S": (30, 86400),
    "STATUS_INTERVAL_S": (1, 3600),
}


def _parse_value(name: str, raw: str, default):
    raw = raw.strip()
    if isinstance(default, bool):
        v = raw.lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{name}: expected a boolean, got {raw!r}")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError as e:
            raise ConfigError(f"{name}: expected an integer, got {raw!r}") from e
    if isinstance(default, float):
        try:
            return float(raw)          # accepts "inf"
        except ValueError as e:
            raise ConfigError(f"{name}: expected a number, got {raw!r}") from e
    if isinstance(default, tuple):
        return tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    if isinstance(default, dict):
        # "btc:BTCUSDT,eth:ETHUSDT" ; an empty value for a key removes it ("bnb:")
        out = {}
        for part in raw.split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ConfigError(f"{name}: expected key:value pairs, got {part!r}")
            k, v = part.split(":", 1)
            if v.strip():
                out[k.strip().lower()] = v.strip()
        return out
    return raw


def validate(s: Settings) -> None:
    for name, (lo, hi) in BOUNDS.items():
        v = getattr(s, name)
        if not (lo <= v <= hi) or (isinstance(v, float) and math.isnan(v)):
            raise ConfigError(f"{name}={v!r} is outside the allowed range [{lo}, {hi}]")
    unknown = [a for a in s.ASSETS if a not in ALL_ASSETS]
    if unknown:
        raise ConfigError(f"ASSETS: unsupported {unknown}; supported: {ALL_ASSETS}")
    bad = [l for l in s.DETERMINISTIC_LABELS if l not in LABEL_SECONDS]
    if bad:
        raise ConfigError(f"DETERMINISTIC_LABELS: unknown {bad}; known: {tuple(LABEL_SECONDS)}")
    if s.MIN_ASK >= s.MAX_ASK:
        raise ConfigError("MIN_ASK must be < MAX_ASK")
    if s.VOL_FAST_HALFLIFE_S > s.VOL_SLOW_HALFLIFE_S:
        raise ConfigError("VOL_FAST_HALFLIFE_S must be <= VOL_SLOW_HALFLIFE_S")
    if s.WS_BACKOFF_MIN_S > s.WS_BACKOFF_MAX_S:
        raise ConfigError("WS_BACKOFF_MIN_S must be <= WS_BACKOFF_MAX_S")
    if s.LOG_LEVEL.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ConfigError(f"LOG_LEVEL={s.LOG_LEVEL!r}")


def load_settings(env: Dict[str, str] | None = None, dotenv: bool = True, **overrides) -> Settings:
    """Defaults <- .env <- process environment (or `env`) <- explicit overrides."""
    if env is None:
        if dotenv and load_dotenv is not None:
            load_dotenv()
        env = dict(os.environ)
    s = Settings()
    for f in fields(Settings):
        if f.name in env:
            setattr(s, f.name, _parse_value(f.name, env[f.name], getattr(s, f.name)))
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise ConfigError(f"unknown setting {k}")
        setattr(s, k, v)
    validate(s)
    return s

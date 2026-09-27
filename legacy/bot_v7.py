"""
Polymarket 5-minute Up/Down bot - v6 (multi-asset)
Adapted from the v5 Up/Down bot for the 5-MINUTE markets of:
  Bitcoin (btc), Ethereum (eth), Solana (sol), BNB (bnb), XRP (xrp), Dogecoin (doge)

5M SPECIFICS (vs v5):
  - Market discovery is DETERMINISTIC - no scanning of the Gamma catalog:
      window_start = now - (now % 300)            # UTC floor, DST-free
      slug = f"{asset}-updown-5m-{window_start}"
      one call: GET https://gamma-api.polymarket.com/events?slug=<slug>
      -> market with Up/Down clobTokenIds, prices, closed flag
  - Windows rotate every 300s -> short poll (2s), short Gamma cache (5s),
    positions ALWAYS settle within <=5 min + settlement grace.
  - Snipe strategy fires only in the last SNIPE_WINDOW seconds of the
    window, when the Chainlink stream is close to final and mispricings
    vs the book are most exploitable.
  - Orders are refused if the window is already closed; nothing carries
    over to the next window.

Kept from v5 (FIX 1/2/3):
  FIX 1 exposure: open positions tracked explicitly, exposure released at
    settlement (not on entry).
  FIX 2 honest PnL: realized_pnl booked ONLY from actual settlements;
    expected_pnl tracked separately; daily stop-loss reacts to REAL losses.
  FIX 3 fills: orders re-priced against the live book right before
    submission; status polled until matched/cancelled/timeout; accounting
    uses ACTUAL filled size and price.

Fee model (2026, docs.polymarket.com/trading/fees):
  fee = shares * feeRate * p * (1 - p)      # crypto taker 0.07, maker 0.00
  Per-market rates from Gamma `feeSchedule` take precedence; global
  fallback refreshed hourly from CLOB GET /fee-rate. Makers pay nothing
  and earn daily rebates -> PREFER_MAKER default on.

Install: pip install py-clob-client python-dotenv requests
Env: PK (EOA key), SIGNATURE_TYPE (0=EOA, 1=magic/email), FUNDER (proxy),
     DRY_RUN (default true).

DISCLAIMER: educational software, NOT financial advice. 5-minute markets
are extremely noisy; keep DRY_RUN=true for a long paper-trading period.
"""
import os
import csv
import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL
import math

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("updown5m-bot")


@dataclass
class Config:
    # --- universe ---
    DIAG_LOG: bool = True
    ASSETS: Tuple[str, ...] = ("btc", "eth", "sol", "bnb", "xrp", "doge")
    # ПОД ПРОФИЛЬ bosona (87K сделок / $20.3M оборота / +$342.7K PnL = ~1.7% PnL/объём,
    # позиции сразу в нескольких длительностях 5m/15m/1h) торгуем не одним таймфреймом,
    # а несколькими одновременно - так же, как видно на его открытых позициях.
    # label -> длительность окна в секундах. Слаг детерминирован: {asset}-updown-{label}-{wstart}.
    DURATIONS: Tuple[Tuple[str, int], ...] = (("5m", 300), ("15m", 900))
    WINDOW_SECONDS: int = 300  # оставлено для обратной совместимости старого кода
    # --- fees (2026): crypto taker 0.07, makers free + rebates ---
    TAKER_FEE_RATE: float = 0.07
    MAKER_FEE_RATE: float = 0.0
    # bosona-профиль: 87K сделок при тонком среднем эдже (~$3.9/сделку) экономически
    # работает в основном на maker rebate, а не на тейкерской комиссии 7%. Поэтому
    # maker - режим по умолчанию (GTC, ждём MAKER_FILL_TIMEOUT, потом падаем в taker).
    PREFER_MAKER: bool = True
    # --- strategies ---
    MIN_ARB_EDGE: float = 0.005       # Up_ask + Down_ask <= 1 - fee - edge
    SNIPE_WINDOW: float = 45.0
    SNIPE_MIN_TIME_LEFT: float = 3.0  # never trade in the very last second
    SNIPE_PRICE: float = 0.990
    SNIPE_MIN_EDGE_CENTS: float = 0.1
    SNIPE_TICK: float = 0.001          # 0.1c tick for 5m markets (validated below)
    # --- sizing / risk ---
    # РИСК-ПРОФИЛЬ выбирается через env RISK_PROFILE=conservative|moderate|aggressive
    # (по умолчанию conservative). Размер ставки считается как % от РЕАЛЬНОГО баланса
    # (не фиксированная сумма в USDC) - так риск автоматически масштабируется с
    # депозитом и не может случайно "съесть" весь счёт одной сделкой.
    # ВАЖНО: DAILY_STOP_LOSS_PCT и MAX_OPEN_EXPOSURE_PCT — это жёсткие пределы,
    # которые НЕ снимаются ни в одном профиле, включая aggressive. Без них "больше
    # риска" means "больше шанс обнулить счёт", а не "больше ожидаемая прибыль" -
    # это математика, а не консервативность ради консервативности.
    RISK_PROFILE: str = os.environ.get("RISK_PROFILE", "conservative").lower()
    MIN_TRADE_USDC: float = 1.0
    PRICE_TICK: float = 0.01
    ORDER_MIN_SIZE: float = 5.0
    PAPER_START_USDC: float = 1000.0   # виртуальный баланс для DRY_RUN
    paper_balance: float = 0.0
    COOLDOWN_SECONDS: float = 30.0     # per-asset cooldown after a loss
    # значения ниже подставляются из RISK_PROFILES в apply_risk_profile()
    BET_PCT_OF_BANKROLL: float = 0.02      # доля депозита на ОДНУ сделку
    MAX_OPEN_EXPOSURE_PCT: float = 0.15    # доля депозита в открытых позициях одновременно
    DAILY_STOP_LOSS_PCT: float = 0.05      # дневной стоп-лосс как доля депозита
    # bosona держит несколько заходов в одну и ту же сторону/окно (видно по крупным
    # позициям вроде 498 акций BTC 5m) - разрешаем добор позиции, но с жёстким лимитом
    # числа доборов на профиль, а не бесконечно.
    MAX_FILLS_PER_SIDE: int = 1            # подставляется из RISK_PROFILES
    MAX_BET_USDC: float = 50.0             # рассчитывается динамически, см. sync_risk_limits()
    MAX_OPEN_EXPOSURE_USDC: float = 300.0
    DAILY_STOP_LOSS_USDC: float = 25.0
    MAX_CONSECUTIVE_ERRORS: int = 5        # circuit breaker: подряд идущие ошибки sweep()
    KILL_SWITCH_FILE: str = "STOP"         # создайте этот файл рядом с bot.py для мгновенной остановки
    # --- infra ---
    DRY_RUN: bool = os.environ.get("DRY_RUN", "true").lower() != "false"
    POLL_INTERVAL: float = 2.0         # 5m markets rotate fast -> short poll
    GAMMA_CACHE_TTL: float = 5.0       # market data must be FRESH in a 300s window
    SETTLE_CHECK_INTERVAL: float = 10.0
    FEE_REFRESH_INTERVAL: float = 3600.0
    BALANCE_REFRESH_INTERVAL: float = 60.0
    FILL_CONFIRM_TIMEOUT: float = 2.0  # FOK status poll budget, sec
    MAKER_FILL_TIMEOUT: float = 4.0    # GTC wait before cancel, sec
    MAX_SLIPPAGE: float = 0.005
    TRADES_CSV: str = "trades.csv"
    REPORTS_DIR: str = "reports"
    REPORT_STATE_FILE: str = "report_state.json"
    HOST: str = "https://clob.polymarket.com"
    GAMMA: str = "https://gamma-api.polymarket.com"
    CHAIN_ID: int = 137


# Три готовых профиля. aggressive даёт БОЛЬШЕ размер ставки, экспозиции и доборов
# позиции, но DAILY_STOP_LOSS_PCT и MAX_OPEN_EXPOSURE_PCT остаются обязательными
# пределами при любом профиле.
RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "conservative": {"bet_pct": 0.01, "exposure_pct": 0.10, "stop_loss_pct": 0.03, "max_fills": 1},
    "moderate":     {"bet_pct": 0.02, "exposure_pct": 0.15, "stop_loss_pct": 0.05, "max_fills": 2},
    "aggressive":   {"bet_pct": 0.04, "exposure_pct": 0.25, "stop_loss_pct": 0.08, "max_fills": 3},
}


cfg = Config()


def apply_risk_profile() -> None:
    """Подставляет bet_pct/exposure_pct/stop_loss_pct из выбранного профиля."""
    profile = RISK_PROFILES.get(cfg.RISK_PROFILE)
    if profile is None:
        log.warning("неизвестный RISK_PROFILE=%r, использую 'conservative'", cfg.RISK_PROFILE)
        profile = RISK_PROFILES["conservative"]
        cfg.RISK_PROFILE = "conservative"
    cfg.BET_PCT_OF_BANKROLL = profile["bet_pct"]
    cfg.MAX_OPEN_EXPOSURE_PCT = profile["exposure_pct"]
    cfg.DAILY_STOP_LOSS_PCT = profile["stop_loss_pct"]
    cfg.MAX_FILLS_PER_SIDE = int(profile["max_fills"])
    log.info("risk profile=%s | bet=%.1f%% of bankroll | max exposure=%.1f%% | daily stop=%.1f%% | max fills/side=%d",
              cfg.RISK_PROFILE, cfg.BET_PCT_OF_BANKROLL * 100,
              cfg.MAX_OPEN_EXPOSURE_PCT * 100, cfg.DAILY_STOP_LOSS_PCT * 100,
              cfg.MAX_FILLS_PER_SIDE)


apply_risk_profile()

# --- HTTP session with urllib3 Retry (v5 pattern) ---
session = requests.Session()
retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
              status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"],
              raise_on_status=False)
adapter = HTTPAdapter(max_retries=retry, pool_connections=6, pool_maxsize=12)
session.mount("https://", adapter)


def build_client() -> Optional[ClobClient]:
    """Build CLOB client. In DRY_RUN works even without PK."""
    pk = os.environ.get("PK")
    if not pk:
        if cfg.DRY_RUN:
            log.warning("PK is not set - continuing in DRY_RUN without a live client.")
            return None
        raise SystemExit("PK is not set. Set the PK environment variable.")
    try:
        client = ClobClient(
            cfg.HOST,
            key=pk,
            chain_id=cfg.CHAIN_ID,
            signature_type=int(os.environ.get("SIGNATURE_TYPE", "0")),
            funder=os.environ.get("FUNDER") or None,
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        return client
    except Exception as e:
        if cfg.DRY_RUN:
            log.warning("Client init failed (%s) - continuing in DRY_RUN.", e)
            return None
        raise


client = build_client()

# ---------------- live balance & dynamic risk limits ----------------
_balance_cache: Dict[str, float] = {"ts": 0.0, "usdc": 0.0}


def get_live_balance() -> Optional[float]:
    """Реальный баланс USDC на торговом счёте (collateral). None если не удалось узнать -
    это КРИТИЧНО: если баланс неизвестен, бот не должен молча использовать старое
    значение и торговать вслепую."""
    if cfg.DRY_RUN or client is None:
        return state.paper_balance
    now = time.time()
    if now - _balance_cache["ts"] < cfg.BALANCE_REFRESH_INTERVAL:
        return _balance_cache["usdc"]
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        resp = client.get_balance_allowance(params)
        raw = resp.get("balance") if isinstance(resp, dict) else None
        if raw is None:
            raise ValueError(f"unexpected balance response: {resp!r}")
        usdc = float(raw) / 1_000_000.0  # USDC has 6 decimals on Polygon
        _balance_cache["usdc"] = usdc
        _balance_cache["ts"] = now
        return usdc
    except Exception as e:
        log.error("НЕ УДАЛОСЬ получить реальный баланс кошелька (%s) - "
                  "торговля приостановлена до восстановления связи.", e)
        return None


def sync_risk_limits() -> bool:
    """Пересчитывает MAX_BET_USDC / MAX_OPEN_EXPOSURE_USDC / DAILY_STOP_LOSS_USDC
    от текущего РЕАЛЬНОГО баланса. Возвращает False, если баланс узнать не удалось -
    в этом случае вызывающий код обязан остановить торговлю на этом проходе."""
    bal = get_live_balance()
    if bal is None:
        return False
    cfg.MAX_BET_USDC = max(cfg.MIN_TRADE_USDC, bal * cfg.BET_PCT_OF_BANKROLL)
    cfg.MAX_OPEN_EXPOSURE_USDC = bal * cfg.MAX_OPEN_EXPOSURE_PCT
    cfg.DAILY_STOP_LOSS_USDC = bal * cfg.DAILY_STOP_LOSS_PCT
    return True


# ---------------- Fee model (fee = rate * p * (1 - p)) ----------------
FEE_BPS_FALLBACK = 700   # crypto taker 0.07
MAKER_RATE = 0.0         # makers never charged (rebates from the 20% fee pool)

fee_rate_cache: Dict[str, Dict[str, float]] = {}   # token_id -> {"taker": rate}
_global_fee_ts: Dict[str, float] = {"ts": 0.0}
_global_taker_rate = 0.07


def refresh_global_fee_rate() -> None:
    """Hourly refresh of the global taker fee rate from CLOB GET /fee-rate."""
    global _global_taker_rate
    if time.time() - _global_fee_ts.get("ts", 0.0) < cfg.FEE_REFRESH_INTERVAL:
        return
    try:
        r = session.get(cfg.HOST + "/fee-rate", timeout=5)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            v = data.get("taker", data.get("rate", data.get("feeRate")))
        else:
            v = data
        _global_taker_rate = float(v)
        log.info("global taker fee rate refreshed: %.4f", _global_taker_rate)
    except (requests.RequestException, ValueError, TypeError) as e:
        log.warning("fee-rate fetch failed (%s) - keeping %.4f", e, _global_taker_rate)
    _global_fee_ts["ts"] = time.time()


def market_taker_rate(gamma_market: dict) -> float:
    """Per-market `feeSchedule` takes precedence over the global rate."""
    try:
        fs = gamma_market.get("feeSchedule")
        if isinstance(fs, dict) and fs.get("rate") is not None:
            return float(fs["rate"])
        v = gamma_market.get("feeScheduleBps")
        if v not in (None, ""):
            return float(v) / 10000.0 if float(v) > 1 else float(v)
    except (TypeError, ValueError):
        pass
    return _global_taker_rate


def taker_fee(shares: float, price: float, rate: float) -> float:
    """fee = shares * feeRate * p * (1 - p)."""
    return shares * rate * price * (1.0 - price)


# ---------------- market discovery (deterministic slugs, multi-duration) ----------------
def window_start(duration_seconds: int, ts: Optional[float] = None) -> int:
    """UTC floor of the current window for a given duration. DST-free by design."""
    now = int(ts or time.time())
    return (now // duration_seconds) * duration_seconds


def build_slug(asset: str, label: str, wstart: int) -> str:
    """Deterministic slug: {asset}-updown-{label}-{unix_start}."""
    return f"{asset}-updown-{label}-{wstart}"


@dataclass
class Market5m:
    asset: str
    slug: str
    label: str                        # "5m" | "15m" | ...
    window_start: int
    window_end: int
    market: dict                      # Gamma market object
    up_token: str
    down_token: str
    outcome_prices: Tuple[float, float] = (0.5, 0.5)  # (up, down)
    closed: bool = False
    fetched_at: float = 0.0

    @property
    def seconds_left(self) -> float:
        return self.window_end - time.time()

    def gamma_url(self) -> str:
        return f"{cfg.GAMMA}/events?slug={self.slug}"


_gamma_cache: Dict[str, Market5m] = {}


def fetch_updown_market(asset: str, label: str, duration: int,
                         max_age: float = None) -> Optional[Market5m]:
    """Return the Gamma Market5m for the CURRENT window of `asset` at this duration."""
    max_age = cfg.GAMMA_CACHE_TTL if max_age is None else max_age
    wstart = window_start(duration)
    key = f"{asset}:{label}:{wstart}"
    cached = _gamma_cache.get(key)
    if cached and (time.time() - cached.fetched_at) <= max_age and not cached.closed:
        return cached
    slug = build_slug(asset, label, wstart)
    try:
        r = session.get(cfg.GAMMA + "/events", params={"slug": slug}, timeout=5)
        r.raise_for_status()
        events = r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("gamma fetch failed for %s (%s)", slug, e)
        return cached
    if not events:
        # window not yet listed, or it just rolled over
        if cached and cached.window_start == wstart:
            return cached
        log.info("no gamma event for slug=%s (window may not be listed yet)", slug)
        return None
    ev = events[0]
    markets = ev.get("markets") or []
    if not markets:
        return None
    m = markets[0]
    try:
        tokens = json.loads(m["clobTokenIds"])
        prices = json.loads(m.get("outcomePrices") or "[0.5, 0.5]")
    except (json.JSONDecodeError, TypeError, KeyError):
        log.warning("bad clobTokenIds/outcomePrices for %s", slug)
        return None
    if len(tokens) != 2:
        return None
    mk = Market5m(
        asset=asset,
        slug=slug,
        label=label,
        window_start=wstart,
        window_end=wstart + duration,
        market=m,
        up_token=tokens[0],
        down_token=tokens[1],
        outcome_prices=(float(prices[0]), float(prices[1])),
        closed=bool(m.get("closed", False)) or not m.get("active", True),
        fetched_at=time.time(),
    )
    _gamma_cache[key] = mk
    return mk


def fetch_5m_market(asset: str, max_age: float = None) -> Optional[Market5m]:
    """Backward-compat wrapper: only the 5m duration."""
    return fetch_updown_market(asset, "5m", 300, max_age)


# ---------------- live order book ----------------
def book_snapshot(token_id: str) -> Optional[dict]:
    """Top-of-book (best bid/ask) from the CLOB /book endpoint."""
    try:
        r = session.get(cfg.HOST + "/book", params={"token_id": token_id}, timeout=4)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("book fetch failed for token %s (%s)", token_id[:10], e)
        return None


def best_bid_ask(book: dict) -> Tuple[Optional[float], Optional[float]]:
    """CLOB book side lists are BEST-FIRST. Returns (best_bid, best_ask)."""
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = float(bids[0]["price"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None
    return best_bid, best_ask

_book_cache: Dict[str, Tuple[float, Tuple[Optional[float], Optional[float]]]] = {}
_BOOK_TTL = 1.5   # сек; sweep всё равно не чаще ~2 с
_diag_ts: Dict[str, float] = {}

def top_of_book(token_id: str) -> Tuple[Optional[float], Optional[float]]:
    now = time.time()
    hit = _book_cache.get(token_id)
    if hit and now - hit[0] < _BOOK_TTL:
        return hit[1]
    try:
        r = session.get(cfg.HOST + "/book", params={"token_id": token_id}, timeout=5)
        if r.status_code != 200:
            log.warning("book %s: HTTP %d", str(token_id)[:12], r.status_code)
            return None, None              # не кэшируем неудачу — повторим в след. проходе
        book = r.json()
        if isinstance(book, str):
            book = json.loads(book)
        if not isinstance(book, dict):
            return None, None
        bids = sorted((b or {} for b in book.get("bids") or []),
                      key=lambda x: float(x.get("price", 0)), reverse=True)
        asks = sorted((a or {} for a in book.get("asks") or []),
                      key=lambda x: float(x.get("price", 1)))
        res = ((float(bids[0]["price"]) if bids else None),
               (float(asks[0]["price"]) if asks else None))
        _book_cache[token_id] = (now, res)
        return res
    except (requests.RequestException, ValueError, TypeError, KeyError) as e:
        log.warning("book fetch failed for %s: %s", str(token_id)[:12], e)
        return None, None

# ---------------- positions / accounting (FIX 1) ----------------
@dataclass
class Position:
    asset: str
    slug: str
    token_id: str
    side: str                  # "up" | "down"
    shares: float = 0.0        # ACTUAL filled shares (FIX 3)
    avg_price: float = 0.0     # ACTUAL filled price
    cost: float = 0.0
    fee_paid: float = 0.0
    expected_pnl: float = 0.0  # model expectation (FIX 2) - never booked
    opened_at: float = field(default_factory=time.time)
    settled: bool = False
    realized_pnl: float = 0.0


@dataclass
class State:
    open_positions: Dict[str, List[Position]] = field(default_factory=dict)  # asset -> positions
    exposure: float = 0.0
    realized_pnl: float = 0.0
    expected_pnl: float = 0.0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    daily_loss: float = 0.0
    day_key: str = ""
    cooldown_until: Dict[str, float] = field(default_factory=dict)


state = State()


def _roll_day() -> None:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.day_key != day:
        state.day_key = day
        state.daily_loss = 0.0


def trading_halted(asset: Optional[str] = None) -> bool:
    _roll_day()
    if kill_switch_active() or circuit_breaker_tripped():
        return True
    if state.daily_loss <= -cfg.DAILY_STOP_LOSS_USDC:
        log.warning("DAILY STOP-LOSS hit (%.2f <= -%.2f) - halting.",
                    state.daily_loss, cfg.DAILY_STOP_LOSS_USDC)
        return True
    if asset and state.cooldown_until.get(asset, 0.0) > time.time():
        return True
    if state.exposure >= cfg.MAX_OPEN_EXPOSURE_USDC:
        return True
    return False


# ---------------- order execution (FIX 3) ----------------
def place_order(token_id: str, side: str, shares: float, limit_price: float,
                market: Market5m) -> Optional[dict]:
    """Re-price against the live book, submit, poll until matched. Returns fill."""
    book = book_snapshot(token_id)
    if not book:
        return None
    best_bid, best_ask = best_bid_ask(book)
    if side == "buy":
        px = best_ask if best_ask is not None else limit_price
        if px - limit_price > cfg.MAX_SLIPPAGE:
            log.info("slippage guard: live ask %.4f vs limit %.4f - skip", px, limit_price)
            return None
        price = min(px, limit_price) if cfg.PREFER_MAKER else px
    else:
        px = best_bid if best_bid is not None else limit_price
        if limit_price - px > cfg.MAX_SLIPPAGE:
            log.info("slippage guard: live bid %.4f vs limit %.4f - skip", px, limit_price)
            return None
        price = max(px, limit_price) if cfg.PREFER_MAKER else px
    price = round(price / cfg.PRICE_TICK) * cfg.PRICE_TICK
    price = round(price, 2)
    if price <= 0.005 or price >= 0.995:
        return None
    if market.seconds_left <= cfg.SNIPE_MIN_TIME_LEFT:
        return None  # never submit in the final second of the window
    if cfg.DRY_RUN or client is None:
        log.info("[DRY_RUN] order: %s %s %.0f shares @ %.3f (%s)",
                 market.asset.upper(), side, shares, price, market.slug)
        return {"status": "matched", "size": shares, "price": price, "maker": cfg.PREFER_MAKER}
    try:
        args = OrderArgs(token_id=token_id, price=price, size=shares, side=BUY if side == "buy" else SELL)
        order = client.create_order(args)
        resp = client.post_order(order, OrderType.GTC if cfg.PREFER_MAKER else OrderType.FOK)
    except Exception as e:
        log.warning("order submit failed (%s)", e)
        return None
    # poll status until matched / cancelled / timeout
    deadline = time.time() + cfg.MAKER_FILL_TIMEOUT
    order_id = (resp or {}).get("orderID") or (resp or {}).get("orderId")
    while time.time() < deadline:
        try:
            o = client.get_order(order_id)
        except Exception as e:
            log.warning("order status poll failed (%s)", e)
            o = None
        st = (o or {}).get("status", "")
        if st in ("matched", "mined", "live"):
            filled = float((o or {}).get("size_matched") or shares)
            return {"status": st, "size": filled, "price": price, "maker": cfg.PREFER_MAKER}
        if st in ("cancelled", "expired"):
            if filled := float((o or {}).get("size_matched") or 0):
                return {"status": "partial", "size": filled, "price": price, "maker": cfg.PREFER_MAKER}
            return None
        time.sleep(0.5)
    # timeout: cancel resting order, keep whatever was matched
    try:
        client.cancel(order_id)
    except Exception as e:
        log.warning("cancel failed (%s)", e)
    try:
        o = client.get_order(order_id) or {}
        filled = float(o.get("size_matched") or 0)
        if filled > 0:
            return {"status": "partial", "size": filled, "price": price, "maker": cfg.PREFER_MAKER}
    except Exception as e:
        log.warning("post-cancel status check failed (%s)", e)
    return None


def book_position(mk: Market5m, token_id: str, side_name: str, fill: dict, rate: float) -> None:
    """FIX 1: exposure grows with ACTUAL fills; FIX 2: only expected_pnl on entry."""
    shares = float(fill["size"])
    price = float(fill["price"])
    cost = shares * price
    fee = taker_fee(shares, price, rate) if not fill.get("maker") else 0.0
    pos = Position(
        asset=mk.asset, slug=mk.slug, token_id=token_id, side=side_name,
        shares=shares, avg_price=price, cost=cost, fee_paid=fee,
        expected_pnl=shares * (1.0 - price) - fee,
    )
    state.open_positions.setdefault(mk.asset, []).append(pos)
    state.exposure += cost + fee
    if cfg.DRY_RUN:
        state.paper_balance -= (cost + fee)
    state.expected_pnl += pos.expected_pnl
    state.trade_count += 1
    log.info("FILLED %s %s: %.0f @ %.3f (fee %.4f) | exposure %.2f | expPnL %.2f",
             mk.asset.upper(), side_name, shares, price, fee, state.exposure, state.expected_pnl)
    append_trade_csv(pos, fill)


def append_trade_csv(pos: Position, fill: dict) -> None:
    new = not os.path.exists(cfg.TRADES_CSV)
    with open(cfg.TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "asset", "slug", "side", "shares", "price",
                        "cost", "fee", "maker", "status", "expected_pnl"])
        w.writerow([datetime.now(timezone.utc).isoformat(), pos.asset, pos.slug,
                    pos.side, pos.shares, pos.avg_price, pos.cost,
                    pos.fee_paid, fill.get("maker", False), fill.get("status", ""),
                    round(pos.expected_pnl, 4)])


# ---------------- weekly reporting ----------------
def _load_report_state() -> dict:
    if os.path.exists(cfg.REPORT_STATE_FILE):
        try:
            with open(cfg.REPORT_STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_report_ts": 0.0}


def _save_report_state(s: dict) -> None:
    try:
        with open(cfg.REPORT_STATE_FILE, "w") as f:
            json.dump(s, f)
    except OSError as e:
        log.warning("не удалось сохранить report_state (%s)", e)


def generate_weekly_report(force: bool = False) -> None:
    """Раз в 7 дней (или по требованию) агрегирует trades.csv в отчёт: винрейт,
    суммарный/средний PnL, макс. просадка, разбивка по активам."""
    rstate = _load_report_state()
    now = time.time()
    if not force and now - rstate.get("last_report_ts", 0.0) < 7 * 86400:
        return
    if not os.path.exists(cfg.TRADES_CSV):
        rstate["last_report_ts"] = now
        _save_report_state(rstate)
        return
    rows = []
    with open(cfg.TRADES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    period_start = now - 7 * 86400
    recent = [r for r in rows if _row_ts(r) >= period_start]
    total_trades = len(recent)
    total_cost = sum(float(r.get("cost", 0) or 0) for r in recent)
    total_fees = sum(float(r.get("fee", 0) or 0) for r in recent)
    by_asset: Dict[str, int] = {}
    for r in recent:
        by_asset[r.get("asset", "?")] = by_asset.get(r.get("asset", "?"), 0) + 1
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    fname = os.path.join(cfg.REPORTS_DIR,
                          f"weekly_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.txt")
    lines = [
        f"Еженедельный отчёт — {datetime.now(timezone.utc).isoformat()}",
        f"Профиль риска: {cfg.RISK_PROFILE} | DRY_RUN={cfg.DRY_RUN}",
        f"Сделок за 7 дней: {total_trades}",
        f"Оборот (cost): {total_cost:.2f} USDC | Комиссии: {total_fees:.2f} USDC",
        f"Realized PnL (всего с запуска): {state.realized_pnl:+.2f} USDC",
        f"W/L (всего с запуска): {state.wins}/{state.losses}",
        f"По активам за неделю: {by_asset}",
        f"Текущая экспозиция: {state.exposure:.2f} USDC",
    ]
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info("weekly report saved -> %s", fname)
    rstate["last_report_ts"] = now
    _save_report_state(rstate)


def _row_ts(row: dict) -> float:
    try:
        return datetime.fromisoformat(row["ts"]).timestamp()
    except (KeyError, ValueError):
        return 0.0


# ---------------- operational safety: circuit breaker + kill switch ----------------
_consecutive_errors = {"n": 0}


def kill_switch_active() -> bool:
    if os.path.exists(cfg.KILL_SWITCH_FILE):
        log.critical("KILL-SWITCH файл '%s' обнаружен - торговля остановлена.", cfg.KILL_SWITCH_FILE)
        return True
    return False


def circuit_breaker_tripped() -> bool:
    return _consecutive_errors["n"] >= cfg.MAX_CONSECUTIVE_ERRORS


def record_sweep_error() -> None:
    _consecutive_errors["n"] += 1
    if circuit_breaker_tripped():
        log.critical("CIRCUIT BREAKER: %d ошибок подряд - торговля приостановлена. "
                     "Проверьте связь/API и удалите файл '%s' при необходимости, затем "
                     "перезапустите бота.", _consecutive_errors["n"], cfg.KILL_SWITCH_FILE)


def record_sweep_ok() -> None:
    _consecutive_errors["n"] = 0


# ---------------- settlement (FIX 1 + FIX 2) ----------------
def check_settlements() -> None:
    """Book REALIZED PnL from actual settlements; release exposure (FIX 1)."""
    for asset in list(state.open_positions.keys()):
        for pos in list(state.open_positions[asset]):
            if pos.settled:
                continue
            slug = pos.slug
            try:
                r = session.get(cfg.GAMMA + "/events", params={"slug": slug}, timeout=5)
                r.raise_for_status()
                events = r.json()
            except (requests.RequestException, ValueError) as e:
                log.warning("settlement fetch failed for %s (%s)", slug, e)
                continue
            if not events:
                continue
            markets = events[0].get("markets") or []
            if not markets:
                continue
            m = markets[0]
            if not m.get("closed"):
                continue
            try:
                tokens = json.loads(m["clobTokenIds"])
                prices = json.loads(m.get("outcomePrices") or "[]")
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            if len(tokens) != 2 or len(prices) != 2:
                continue
            win_token = tokens[0] if float(prices[0]) >= 0.99 else tokens[1]
            won = (pos.token_id == win_token)
            payout = pos.shares if won else 0.0
            pos.realized_pnl = payout - pos.cost - pos.fee_paid
            pos.settled = True
            state.realized_pnl += pos.realized_pnl
            state.exposure -= (pos.cost + pos.fee_paid)
            if cfg.DRY_RUN:
                state.paper_balance += payout
            state.daily_loss += pos.realized_pnl  # FIX 2: stop-loss reacts to REAL losses
            if pos.realized_pnl >= 0:
                state.wins += 1
            else:
                state.losses += 1
                if cfg.COOLDOWN_SECONDS > 0:
                    state.cooldown_until[pos.asset] = time.time() + cfg.COOLDOWN_SECONDS
            log.info("SETTLED %s %s (%s): %s -> realized PnL %+.2f | total %+.2f",
                     pos.asset.upper(), pos.side, pos.slug,
                     "WIN" if won else "LOSS",
                     pos.realized_pnl, state.realized_pnl)


# ---------------- strategies ----------------
def strategy_arbitrage(mk: Market5m) -> Optional[Tuple[str, float]]:
    """Buy both sides when Up_ask + Down_ask < 1 - fees - edge."""
    _, up_ask = top_of_book(mk.up_token)
    _, dn_ask = top_of_book(mk.down_token)
    if up_ask is None or dn_ask is None:
        return None
    rate = market_taker_rate(mk.market)
    total = up_ask + dn_ask
    fees = taker_fee(1.0, up_ask, rate) / up_ask + taker_fee(1.0, dn_ask, rate) / dn_ask
    if total + fees <= 1.0 - cfg.MIN_ARB_EDGE:
        side = "up" if up_ask <= dn_ask else "down"
        return side, 1.0 - total - fees
    return None

def strategy_snipe(mk: Market5m) -> Optional[Tuple[str, float]]:
    """Last SNIPE_WINDOW seconds: take the leading side if its ask is cheap."""
    if mk.seconds_left > cfg.SNIPE_WINDOW or mk.seconds_left <= cfg.SNIPE_MIN_TIME_LEFT:
        return None
    _, up_ask = top_of_book(mk.up_token)
    _, dn_ask = top_of_book(mk.down_token)
    if up_ask is None or dn_ask is None:
        return None
    lead = "up" if up_ask > dn_ask else "down"      # лидер = сторона дороже
    lead_ask = up_ask if lead == "up" else dn_ask
    rate = market_taker_rate(mk.market)
    edge = 1.0 - lead_ask - taker_fee(1.0, lead_ask, rate)
    if lead_ask <= cfg.SNIPE_PRICE and edge >= cfg.SNIPE_MIN_EDGE_CENTS / 100.0:
        return lead, edge
    return None

def execute(side: str, model_edge: float, mk: Market5m, both_sides: bool = False) -> None:
    """Single-side buy (snipe) or two-sided lock (arb: both_sides=True)."""
    if trading_halted(mk.asset):
        return
    token = mk.up_token if side == "up" else mk.down_token
    existing = [p for p in state.open_positions.get(mk.asset, [])
                if p.token_id == token and not p.settled]
    if len(existing) >= cfg.MAX_FILLS_PER_SIDE:
        return  # достигнут лимит доборов в эту сторону в этом окне (см. RISK_PROFILE)
    rate = market_taker_rate(mk.market)
    budget = min(cfg.MAX_BET_USDC, cfg.MAX_OPEN_EXPOSURE_USDC - state.exposure)
    budget = max(budget, 0.0)
    if budget < cfg.MIN_TRADE_USDC:
        log.info("budget below MIN_TRADE_USDC - skip")
        return
    _, ask = top_of_book(token)          # кэш вместо book_snapshot
    if ask is None:
        return
    shares = int(budget / ask)            # FIX: было `price` -> NameError при первом сигнале
    if shares < cfg.ORDER_MIN_SIZE:
        log.info("skip: %d shares < orderMinSize %d", shares, int(cfg.ORDER_MIN_SIZE))
        return
    fill = place_order(token, "buy", shares, ask, mk)
    if fill:
        book_position(mk, token, side, fill, rate)

# вторая сторона — ТОЛЬКО для арбитража (обе стороны дешевле 1 вместе)
    if both_sides and mk.seconds_left > 0 and state.exposure < cfg.MAX_OPEN_EXPOSURE_USDC:
        other = "down" if side == "up" else "up"
        other_token = mk.down_token if other == "down" else mk.up_token
        other_existing = [p for p in state.open_positions.get(mk.asset, [])
                          if p.token_id == other_token and not p.settled]
        if len(other_existing) < cfg.MAX_FILLS_PER_SIDE:
            _, oask = top_of_book(other_token)
            if oask is not None:
                budget2 = min(cfg.MAX_BET_USDC, cfg.MAX_OPEN_EXPOSURE_USDC - state.exposure)
                if budget2 >= cfg.MIN_TRADE_USDC:
                    shares2 = int(budget2 / oask)
                    if shares2 >= cfg.ORDER_MIN_SIZE:
                        fill2 = place_order(other_token, "buy", shares2, oask, mk)
                        if fill2:
                            book_position(mk, other_token, other, fill2, rate)

# ---------------- main loop ----------------
def sweep() -> None:
    """One pass over all assets x durations (5m, 15m, ...) currently configured."""
    refresh_global_fee_rate()
    for asset in cfg.ASSETS:
        for label, duration in cfg.DURATIONS:
            _sweep_one(asset, label, duration)


def _sweep_one(asset: str, label: str, duration: int) -> None:
    diag_key = f"{asset}:{label}"
    mk = fetch_updown_market(asset, label, duration)
    if not mk or mk.closed:
        return
    diag_full = mk.seconds_left <= 40
    if cfg.DIAG_LOG and mk and not mk.closed and mk.seconds_left > 0 \
            and (diag_full or time.time() - _diag_ts.get(diag_key, 0.0) >= 15):
        _diag_ts[diag_key] = time.time()
        ub, ua = top_of_book(mk.up_token)
        db, da = top_of_book(mk.down_token)
        if ua is not None and da is not None:
            log.info("diag %s %s: %4.0fs left | up %.3f/%.3f down %.3f/%.3f | sum_ask %.3f",
                     asset, label, mk.seconds_left, ub or -1, ua, db or -1, da, ua + da)
    # --- near-miss: реплика snipe-гейта по живому стакану ---
    if cfg.DIAG_LOG and cfg.SNIPE_MIN_TIME_LEFT < mk.seconds_left <= cfg.SNIPE_WINDOW:
        _, ua = top_of_book(mk.up_token)
        _, da = top_of_book(mk.down_token)
        why = []
        if ua is None or da is None:
            why.append("no ask on one side")
        else:
            ask = max(ua, da)                                   # лидер = сторона дороже
            fee = _global_taker_rate * ask * (1 - ask)          # fee = rate*p*(1-p)
            edge_c = (1.0 - ask - fee) * 100
            if ask > cfg.SNIPE_PRICE:
                why.append(f"ask {ask:.3f} > {cfg.SNIPE_PRICE:.2f}")
            if edge_c < cfg.SNIPE_MIN_EDGE_CENTS:
                why.append(f"edge {edge_c:.2f}c < {cfg.SNIPE_MIN_EDGE_CENTS}c")
        if why:
            log.info("snipe near-miss %s %s (%.0fs left): %s", asset, label, mk.seconds_left, "; ".join(why))
    if mk.seconds_left <= 0:
        return
    if trading_halted(asset):
        return
    try:
        sig = strategy_arbitrage(mk)
        arb = sig is not None
        if sig is None:
            sig = strategy_snipe(mk)
        if sig:
            side, edge = sig
            log.info("SIGNAL %s %s %s edge=%.3f (%.0fs left)",
                     asset.upper(), label, side, edge, max(mk.seconds_left, 0))
            execute(side, edge, mk, both_sides=arb)
    except Exception as e:
        log.exception("sweep error on %s %s (%s)", asset, label, e)



def live_preflight() -> bool:
    """Обязательные проверки перед боевой торговлей. Возвращает False, если
    запускаться нельзя."""
    if cfg.DRY_RUN:
        return True
    if client is None:
        log.critical("DRY_RUN=false, но клиент не создан (нет PK?). Остановка.")
        return False
    bal = get_live_balance()
    if bal is None:
        log.critical("Не удалось прочитать реальный баланс кошелька перед стартом. "
                     "Проверьте PK/FUNDER/SIGNATURE_TYPE и сетевой доступ к CLOB API. Остановка.")
        return False
    if bal <= 0:
        log.critical("Баланс USDC на торговом счёте = %.2f. Пополните счёт и убедитесь, "
                     "что выдан allowance (approve) для CTF Exchange контракта через "
                     "интерфейс Polymarket, иначе ордера будут отклоняться. Остановка.", bal)
        return False
    log.warning("=== БОЕВОЙ РЕЖИМ: реальные средства, баланс %.2f USDC, профиль риска '%s' ===",
                bal, cfg.RISK_PROFILE)
    log.warning("Ставка на сделку ~%.2f USDC (%.1f%% депозита), дневной стоп-лосс ~%.2f USDC (%.1f%%).",
                bal * cfg.BET_PCT_OF_BANKROLL, cfg.BET_PCT_OF_BANKROLL * 100,
                bal * cfg.DAILY_STOP_LOSS_PCT, cfg.DAILY_STOP_LOSS_PCT * 100)
    return True


def main() -> None:
    log.info("=== Up/Down bot v7 (assets: %s | durations: %s) DRY_RUN=%s RISK_PROFILE=%s PREFER_MAKER=%s ===",
             ", ".join(cfg.ASSETS), ", ".join(l for l, _ in cfg.DURATIONS),
             cfg.DRY_RUN, cfg.RISK_PROFILE, cfg.PREFER_MAKER)
    state.paper_balance = cfg.PAPER_START_USDC
    if not live_preflight():
        return
    if cfg.DRY_RUN:
        log.info("paper bankroll: %.2f USDC (DRY_RUN=%s)", state.paper_balance, cfg.DRY_RUN)
    last_settle = 0.0
    last_balance_sync = 0.0
    try:
        while True:
            now = time.time()
            if kill_switch_active():
                log.critical("Остановлено файлом kill-switch. Удалите '%s' и перезапустите для продолжения.",
                             cfg.KILL_SWITCH_FILE)
                break
            if now - last_balance_sync >= cfg.BALANCE_REFRESH_INTERVAL or last_balance_sync == 0.0:
                if not sync_risk_limits():
                    log.error("Баланс недоступен - пропускаю проход, торговля временно приостановлена.")
                    time.sleep(cfg.POLL_INTERVAL)
                    continue
                last_balance_sync = now
            if now - last_settle >= cfg.SETTLE_CHECK_INTERVAL:
                check_settlements()
                last_settle = now
            try:
                sweep()
                record_sweep_ok()
            except Exception as e:
                log.exception("sweep() упал целиком: %s", e)
                record_sweep_error()
            generate_weekly_report()
            _roll_day()
            log.info("loop: balance %.2f | exposure %.2f | realized %+.2f | expected %+.2f | W/L %d/%d | day %+.2f",
                     state.paper_balance if cfg.DRY_RUN else (get_live_balance() or float("nan")),
                     state.exposure, state.realized_pnl, state.expected_pnl,
                     state.wins, state.losses, state.daily_loss)
            time.sleep(cfg.POLL_INTERVAL)
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        generate_weekly_report(force=True)


if __name__ == "__main__":
    main()
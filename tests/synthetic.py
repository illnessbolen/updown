"""Synthetic recordings in the exact on-disk format of TickRecorder.

Only for pipeline tests: the numbers are made up and say nothing about any real edge.
"""
import json
import math
import random

from latarb.data.markets import CHAINLINK, MarketWindow
from latarb.data.recorder import TickRecorder
from latarb.model.distributions import norm_cdf

T0 = 1_790_000_000          # window start (multiple of 300)
DUR = 300
JUMP_AT = 30.0              # seconds into the window; early, while the market is still near 0.5


def window(twap_s: float = 0.0) -> MarketWindow:
    """twap_s = 0: settled on the Chainlink price at the start (the synthetic market prices against that tick);
    > 0: settled on a TWAP, the price to beat being the TWAP of the twap_s seconds before the start."""
    return MarketWindow(slug=f"btc-updown-5m-{T0}", asset="btc", label="5m", duration_s=DUR,
                        start_ts=float(T0), end_ts=float(T0 + DUR), up_token="UP", down_token="DOWN",
                        resolution=CHAINLINK, twap_s=twap_s)


def _book(token, bid, ask, ts):
    return json.dumps([{"event_type": "book", "asset_id": token, "timestamp": str(int(ts * 1000)),
                        "bids": [{"price": f"{bid:.2f}", "size": "500"}],
                        "asks": [{"price": f"{ask:.2f}", "size": "500"}]}])


def write_session(directory, *, seed=1, warmup_s=1500, jump_at=JUMP_AT, jump_bp=25.0,
                  coinbase_glitch_bp=0.0, poly_pongs=True, poly_active=True, sigma=1e-4, twap_s=0.0) -> dict:
    """BTC session: vol warm-up, one 5m window. While the window runs, the Polymarket book tracks
    the true fair value (an efficient market); at T0+jump_at spot jumps and the book FREEZES at its
    pre-jump quote — the slow repricing the strategy looks for. With poly_active=False the book
    never moves after its first snapshot (a dead market). twap_s: see window(); the returned k_ref is the price
    to beat the market is priced against."""
    rng = random.Random(seed)
    rec = TickRecorder(str(directory), rotate_s=3600)
    w = window(twap_s)
    oracle = {}
    start = T0 - warmup_s
    for src in ("binance", "coinbase", "rtds", "polymarket"):
        rec.write(start, "@open", src)
    rec.write(start + 0.001, "@markets", json.dumps([w.to_dict()]))
    rec.write(T0 - 60, "polymarket", _book("UP", 0.49, 0.51, T0 - 60))
    rec.write(T0 - 60, "polymarket", _book("DOWN", 0.49, 0.51, T0 - 60))
    px = 100_000.0
    t = start
    step = 0.2
    next_cb = next_cl = next_pong = next_poly = start
    jumped = False
    k_ref = None
    while t < T0 + DUR - 0.5:
        t = round(t + step, 6)
        px *= math.exp(rng.gauss(0.0, sigma * math.sqrt(step)))
        if not jumped and t >= T0 + jump_at:
            px *= math.exp(jump_bp * 1e-4)
            jumped = True
        bid, ask = round(px - 0.5, 2), round(px + 0.5, 2)
        rec.write(t, "binance", json.dumps({"stream": "btcusdt@bookTicker", "data": {
            "u": int(t * 10), "s": "BTCUSDT", "b": f"{bid:.2f}", "B": "1", "a": f"{ask:.2f}", "A": "1"}}))
        if t >= next_cb:
            next_cb += 0.5
            glitch = coinbase_glitch_bp if (coinbase_glitch_bp and t >= T0 + jump_at - 5) else 0.0
            cb = px * math.exp((2.0 + glitch) * 1e-4)
            rec.write(t + 0.01, "coinbase", json.dumps({
                "type": "ticker", "product_id": "BTC-USD", "price": f"{cb:.2f}",
                "best_bid": f"{cb - 0.5:.2f}", "best_ask": f"{cb + 0.5:.2f}", "time": "2026-09-21T00:00:00Z"}))
        if t >= next_cl:
            sec = math.ceil(t)
            next_cl = sec + 1.0
            oracle[sec] = px * math.exp(3e-4)
            if sec == T0:
                lb = int(twap_s)
                k_ref = sum(oracle[x] for x in range(T0 - lb, T0)) / lb if lb else oracle[sec]
            rec.write(t + 0.05, "rtds", json.dumps({
                "topic": "crypto_prices_chainlink", "type": "update", "timestamp": int((t + 0.05) * 1000),
                "payload": {"symbol": "btc/usd", "timestamp": int(sec * 1000), "value": px * math.exp(3e-4)}}))
        if poly_active and t >= max(next_poly, T0 - 60):
            # an active market: deep levels change all the time
            next_poly = t + 0.5
            if k_ref is not None and not (jumped and jump_bp):
                # efficient market until the jump: top of book around the true (Gaussian) fair value
                x = math.log(px * math.exp(3e-4) / k_ref)
                p_up = norm_cdf(x / (sigma * math.sqrt(T0 + DUR - t)))
                m = min(max(round(p_up, 2), 0.03), 0.97)
                rec.write(t, "polymarket", _book("UP", m - 0.01, m + 0.01, t))
                rec.write(t, "polymarket", _book("DOWN", 1 - m - 0.01, 1 - m + 0.01, t))
            rec.write(t, "polymarket", json.dumps({"event_type": "price_change", "market": "0x1",
                      "timestamp": str(int(t * 1000)), "price_changes": [
                          {"asset_id": "UP", "price": "0.30", "size": str(100 + int(t * 10) % 50), "side": "BUY"},
                          {"asset_id": "DOWN", "price": "0.70", "size": str(100 + int(t * 7) % 50), "side": "SELL"}]}))
        if poly_pongs and t >= next_pong:
            next_pong += 5.0
            rec.write(t, "polymarket", "PONG")
    rec.write(T0 + DUR + 30, "@outcome", json.dumps({"slug": w.slug, "winner": "up", "window": w.to_dict()}))
    rec.close()
    return {"window": w, "k_ref": k_ref}

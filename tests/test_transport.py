"""WebSocket transport against a local fake of the Polymarket CLOB market channel.

Covers: initial subscribe payload, book snapshots -> synced books, dynamic
subscribe on window roll, application-level PING/PONG, reconnect after the
server drops the connection (books reset, then re-synced), and the silence
watchdog.
"""
import asyncio
import json
import time

from websockets.asyncio.server import serve

from latarb.clock import WallClock
from latarb.config import load_settings
from latarb.data.feeds import FeedSet
from latarb.data.hub import MarketDataHub
from latarb.data.markets import CHAINLINK, MarketWindow


def cfg(**kw):
    base = dict(ASSETS=("btc",), WS_PROXY="none", WS_BACKOFF_MIN_S=0.05, WS_BACKOFF_MAX_S=1.0)
    base.update(kw)
    return load_settings(env={}, dotenv=False, **base)


def win(slug, up, down):
    now = time.time()
    return MarketWindow(slug=slug, asset="btc", label="5m", duration_s=300, start_ts=now, end_ts=now + 300,
                        up_token=up, down_token=down, resolution=CHAINLINK)


class FakeClob:
    def __init__(self, silent=False):
        self.silent = silent
        self.received = []
        self.conns = set()
        self.connections = 0

    async def handler(self, ws):
        self.connections += 1
        self.conns.add(ws)
        try:
            async for msg in ws:
                self.received.append(msg)
                if self.silent:
                    continue
                if msg == "PING":
                    await ws.send("PONG")
                    continue
                d = json.loads(msg)
                if d.get("type") == "market" or d.get("operation") == "subscribe":
                    await ws.send(json.dumps([
                        {"event_type": "book", "asset_id": t, "timestamp": str(int(time.time() * 1000)),
                         "bids": [{"price": "0.40", "size": "10"}], "asks": [{"price": "0.60", "size": "10"}]}
                        for t in d["assets_ids"]]))
        finally:
            self.conns.discard(ws)

    async def kick(self):
        for ws in list(self.conns):
            await ws.close()


async def wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_polymarket_feed_subscribes_pings_resubscribes_and_reconnects():
    async def scenario():
        fake = FakeClob()
        async with serve(fake.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            c = cfg(POLY_WS_URL=f"ws://127.0.0.1:{port}", POLY_APP_PING_S=1)
            hub = MarketDataHub(c, WallClock())
            w1, w2 = win("a", "A", "B"), win("b", "C", "D")
            hub.set_markets([w1])
            feeds = FeedSet(c, WallClock(), hub)
            task = asyncio.create_task(feeds.poly.ws.run())
            try:
                await feeds.poly.set_tokens(["A", "B"])
                await wait_until(lambda: hub.books["A"].synced and hub.books["B"].synced)
                assert json.loads(fake.received[0]) == {"assets_ids": ["A", "B"], "type": "market"}
                assert hub.books["A"].best_ask() == (0.60, 10.0)

                hub.set_markets([w1, w2])                      # a new window opens
                await feeds.poly.set_tokens(["A", "B", "C", "D"])
                await wait_until(lambda: hub.books["C"].synced and hub.books["D"].synced)
                assert {"assets_ids": ["C", "D"], "operation": "subscribe"} in [
                    json.loads(m) for m in fake.received if m != "PING"]
                assert fake.connections == 1                  # no reconnect needed

                await wait_until(lambda: "PING" in fake.received, timeout=3)
                await wait_until(lambda: hub.feed_age("polymarket", time.time()) < 1.5, timeout=3)

                await fake.kick()                             # server drops us
                await wait_until(lambda: feeds.poly.ws.connects >= 2)
                await wait_until(lambda: all(hub.books[t].synced for t in "ABCD"))
                resub = [json.loads(m) for m in fake.received if m != "PING" and "type" in json.loads(m)]
                assert resub[-1] == {"assets_ids": ["A", "B", "C", "D"], "type": "market"}
            finally:
                await feeds.poly.ws.stop()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_silence_watchdog_replaces_a_dead_connection():
    async def scenario():
        fake = FakeClob(silent=True)
        async with serve(fake.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            c = cfg(POLY_WS_URL=f"ws://127.0.0.1:{port}", WS_SILENCE_RECONNECT_S=2, POLY_APP_PING_S=10)
            hub = MarketDataHub(c, WallClock())
            hub.set_markets([win("a", "A", "B")])
            feeds = FeedSet(c, WallClock(), hub)
            feeds.poly.desired = {"A", "B"}
            task = asyncio.create_task(feeds.poly.ws.run())
            try:
                await wait_until(lambda: feeds.poly.ws.connects >= 2, timeout=6)
                assert not hub.books["A"].synced
            finally:
                await feeds.poly.ws.stop()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_subscription_payloads_for_other_venues():
    c = cfg(ASSETS=("btc", "eth"))
    feeds = FeedSet(c, WallClock(), MarketDataHub(c, WallClock()))
    assert feeds._binance_url() == ("wss://stream.binance.com:9443/stream?streams="
                                    "btcusdt@bookTicker/btcusdt@trade/ethusdt@bookTicker/ethusdt@trade")
    assert json.loads(next(feeds._coinbase_sub())) == {
        "type": "subscribe", "product_ids": ["BTC-USD", "ETH-USD"], "channels": ["ticker", "heartbeat"]}
    sub = json.loads(next(feeds._rtds_sub()))
    assert sub["action"] == "subscribe"
    assert [json.loads(s["filters"])["symbol"] for s in sub["subscriptions"]] == ["btc/usd", "eth/usd"]
    assert all(s["topic"] == "crypto_prices_chainlink" for s in sub["subscriptions"])

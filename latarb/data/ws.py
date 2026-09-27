"""Resilient WebSocket transport shared by all feeds.

  * reconnect forever with exponential backoff + full jitter
    (WS_BACKOFF_MIN_S .. WS_BACKOFF_MAX_S); the attempt counter resets only
    after a connection actually delivered data, so a server that accepts and
    immediately drops us is not hammered;
  * protocol-level ping (websockets keepalive) on every connection, plus an
    optional application-level text ping ("PING") for Polymarket endpoints;
  * silence watchdog: no frame at all for WS_SILENCE_RECONNECT_S -> the
    connection is considered dead and replaced (a half-open TCP connection
    otherwise looks "connected" forever);
  * handler exceptions are logged and counted, never allowed to kill the feed.

Staleness for TRADING is decided elsewhere (engine gates, per source and per
asset); this module only keeps connections alive and reports frames.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

from ..clock import Clock

log = logging.getLogger("latarb.ws")

MessageHandler = Callable[[str, float], None]
OpenHandler = Callable[[ClientConnection, float], Awaitable[None]]
EventHandler = Callable[[float], None]


class ResilientWS:
    def __init__(self, name: str, url: Callable[[], str], clock: Clock, *,
                 on_message: MessageHandler,
                 on_open: Optional[OpenHandler] = None,
                 on_close: Optional[EventHandler] = None,
                 should_connect: Callable[[], bool] = lambda: True,
                 app_ping_text: Optional[str] = None,
                 app_ping_interval_s: float = 10.0,
                 protocol_ping_s: float = 20.0,
                 silence_timeout_s: float = 30.0,
                 open_timeout_s: float = 10.0,
                 backoff_min_s: float = 0.5,
                 backoff_max_s: float = 30.0,
                 proxy: str = "",
                 max_size: int = 16 * 1024 * 1024) -> None:
        self.name = name
        self._url = url
        self.clock = clock
        self.on_message = on_message
        self.on_open = on_open
        self.on_close = on_close
        self.should_connect = should_connect
        self.app_ping_text = app_ping_text
        self.app_ping_interval_s = app_ping_interval_s
        self.protocol_ping_s = protocol_ping_s
        self.silence_timeout_s = silence_timeout_s
        self.open_timeout_s = open_timeout_s
        self.backoff_min_s = backoff_min_s
        self.backoff_max_s = backoff_max_s
        self.max_size = max_size
        # websockets semantics: True = from environment, None = direct, str = explicit proxy
        self.proxy = True if not proxy else (None if proxy.lower() == "none" else proxy)
        self._ws: Optional[ClientConnection] = None
        self._stop = False
        self._reconnect_now = False
        self._wake = asyncio.Event()
        self.connected = False
        self.connects = 0
        self.frames = 0
        self.handler_errors = 0
        self.last_error: str = ""

    # ------------------------------------------------------------------ control
    async def send_json(self, text: str) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(text)
            return True
        except ConnectionClosed:
            return False

    def reconnect(self, reason: str) -> None:
        log.info("%s: reconnect requested (%s)", self.name, reason)
        self._reconnect_now = True
        ws = self._ws
        if ws is not None:
            asyncio.ensure_future(ws.close())
        self._wake.set()

    def poke(self) -> None:
        """Wake a feed idling because should_connect() was False."""
        self._wake.set()

    async def stop(self) -> None:
        self._stop = True
        self._wake.set()
        if self._ws is not None:
            await self._ws.close()

    # ------------------------------------------------------------------ loop
    def _backoff(self, attempt: int) -> float:
        cap = min(self.backoff_max_s, self.backoff_min_s * (2 ** min(attempt, 16)))
        return random.uniform(self.backoff_min_s, max(cap, self.backoff_min_s))

    async def _app_ping(self, ws: ClientConnection) -> None:
        while True:
            await asyncio.sleep(self.app_ping_interval_s)
            try:
                await ws.send(self.app_ping_text)
            except ConnectionClosed:
                return

    async def _sleep_or_wake(self, seconds: float) -> None:
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def run(self) -> None:
        attempt = 0
        while not self._stop:
            if not self.should_connect():
                await self._sleep_or_wake(1.0)
                continue
            url = self._url()
            ping_task: Optional[asyncio.Task] = None
            got_data = False
            try:
                async with connect(url, open_timeout=self.open_timeout_s,
                                   ping_interval=self.protocol_ping_s, ping_timeout=self.protocol_ping_s,
                                   close_timeout=5, max_size=self.max_size, max_queue=1024,
                                   proxy=self.proxy) as ws:
                    self._ws = ws
                    self.connected = True
                    self.connects += 1
                    now = self.clock.now()
                    log.info("%s: connected (#%d)", self.name, self.connects)
                    if self.on_open is not None:
                        await self.on_open(ws, now)
                    if self.app_ping_text:
                        ping_task = asyncio.create_task(self._app_ping(ws))
                    while not self._stop:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=self.silence_timeout_s)
                        except asyncio.TimeoutError:
                            log.warning("%s: no frames for %.0fs - reconnecting", self.name, self.silence_timeout_s)
                            break
                        recv_ts = self.clock.now()
                        if isinstance(raw, (bytes, bytearray)):
                            raw = raw.decode("utf-8", "replace")
                        self.frames += 1
                        if not got_data:
                            got_data = True
                            attempt = 0
                        try:
                            self.on_message(raw, recv_ts)
                        except Exception as e:  # noqa: BLE001 - a bad frame must not kill the feed
                            self.handler_errors += 1
                            if self.handler_errors <= 20 or self.handler_errors % 1000 == 0:
                                log.exception("%s: handler error #%d: %s", self.name, self.handler_errors, e)
            except (OSError, asyncio.TimeoutError, ConnectionClosed, InvalidHandshake, InvalidURI) as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("%s: connection error: %s", self.name, self.last_error)
            finally:
                if ping_task is not None:
                    ping_task.cancel()
                was_connected = self.connected
                self.connected = False
                self._ws = None
                if was_connected and self.on_close is not None:
                    self.on_close(self.clock.now())
            if self._stop:
                break
            if self._reconnect_now:
                self._reconnect_now = False
                continue
            delay = self._backoff(attempt)
            attempt += 1
            log.info("%s: retry in %.1fs (attempt %d)", self.name, delay, attempt)
            await self._sleep_or_wake(delay)

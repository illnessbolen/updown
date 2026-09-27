"""Gamma REST: market discovery (deterministic probes + periodic scan) and outcomes.

All calls are blocking (requests + urllib3 Retry, the pattern of the previous
bot) and run in a worker thread via asyncio.to_thread, never on the event loop.

Discovery keeps a TTL cache so the tick loop does not re-query on every tick:
  * positive entries live until the window ends (+ a grace period);
  * slugs that are not listed yet are negatively cached for
    DISCOVERY_NEGATIVE_TTL_S;
  * the full scan (tag + "up or down" pattern, all durations) runs every
    DISCOVERY_SCAN_INTERVAL_S (60..300 s).
Every window is tracked on its own timeline; 5m, 15m, 1h, 4h and daily windows
open and close independently.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..clock import Clock
from ..config import LABEL_SECONDS, Settings
from .markets import MarketWindow, deterministic_slug, looks_like_updown, windows_from_event, winner_from_market

log = logging.getLogger("latarb.gamma")


def make_session(pool: int = 8) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
                  raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool * 2)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class GammaClient:
    def __init__(self, cfg: Settings, session: Optional[requests.Session] = None) -> None:
        self.base = cfg.GAMMA_URL.rstrip("/")
        self.timeout = cfg.HTTP_TIMEOUT_S
        self.session = session or make_session()

    def get_events(self, **params) -> List[dict]:
        r = self.session.get(self.base + "/events", params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    def event_by_slug(self, slug: str) -> Optional[dict]:
        events = self.get_events(slug=slug)
        return events[0] if events else None


class Discovery:
    KEEP_AFTER_END_S = 120.0

    def __init__(self, cfg: Settings, gamma: GammaClient, clock: Clock) -> None:
        self.cfg = cfg
        self.gamma = gamma
        self.clock = clock
        self.windows: Dict[str, MarketWindow] = {}
        self._negative: Dict[str, float] = {}
        self._last_scan = -math.inf
        self.rejected: Dict[str, str] = {}
        self.errors = 0
        self._stopped = False

    def stop(self) -> None:
        """Make an in-flight refresh (running in a worker thread) return at the next request."""
        self._stopped = True

    # ------------------------------------------------------------------
    def _add(self, ws: List[MarketWindow], rejected: List[Tuple[str, str]]) -> int:
        new = 0
        for w in ws:
            if w.slug not in self.windows:
                new += 1
            self.windows[w.slug] = w
        for slug, why in rejected:
            if slug not in self.rejected:
                log.info("discovery: skipping %s (%s)", slug, why)
            self.rejected[slug] = why
        return new

    def probe_deterministic(self, now: float) -> Optional[int]:
        """Number of new windows, or None if Gamma is unreachable (round aborted)."""
        new = 0
        for asset in self.cfg.ASSETS:
            for label in self.cfg.DETERMINISTIC_LABELS:
                dur = LABEL_SECONDS[label]
                cur = int(now // dur) * dur
                starts = [cur]
                if cur + dur - now <= self.cfg.DISCOVERY_PREFETCH_S:
                    starts.append(cur + dur)
                for start in starts:
                    slug = deterministic_slug(asset, label, start)
                    if slug in self.windows or self._negative.get(slug, 0.0) > now:
                        continue
                    if self._stopped:
                        return None
                    try:
                        ev = self.gamma.event_by_slug(slug)
                    except requests.ConnectionError as e:
                        # outage: every other probe would burn its own retries too - try next tick
                        self.errors += 1
                        log.warning("gamma unreachable (%s) - discovery round aborted", e)
                        return None
                    except (requests.RequestException, ValueError) as e:
                        self.errors += 1
                        log.warning("gamma probe %s failed: %s", slug, e)
                        self._negative[slug] = now + self.cfg.DISCOVERY_NEGATIVE_TTL_S
                        continue
                    if ev is None:
                        self._negative[slug] = now + self.cfg.DISCOVERY_NEGATIVE_TTL_S
                        continue
                    new += self._add(*windows_from_event(ev, self.cfg.ASSETS, now))
        return new

    def scan(self, now: float) -> int:
        """All active Up/Down events ending within the scan horizon, any duration."""
        new = 0
        size = self.cfg.GAMMA_SCAN_PAGE_SIZE
        for page in range(self.cfg.GAMMA_SCAN_MAX_PAGES):
            if self._stopped:
                break
            params = {
                "closed": "false", "limit": size, "offset": page * size,
                "end_date_min": _iso(now),
                "end_date_max": _iso(now + self.cfg.GAMMA_SCAN_HORIZON_H * 3600),
            }
            if self.cfg.GAMMA_SCAN_TAG:
                params["tag_slug"] = self.cfg.GAMMA_SCAN_TAG
            try:
                events = self.gamma.get_events(**params)
            except (requests.RequestException, ValueError) as e:
                self.errors += 1
                log.warning("gamma scan page %d failed: %s", page, e)
                break
            for ev in events:
                if looks_like_updown(str(ev.get("slug") or ""), str(ev.get("title") or "")):
                    new += self._add(*windows_from_event(ev, self.cfg.ASSETS, now))
            if len(events) < size:
                break
        return new

    def prune(self, now: float) -> None:
        for slug in [s for s, w in self.windows.items() if w.end_ts + self.KEEP_AFTER_END_S < now]:
            del self.windows[slug]
        for slug in [s for s, t in self._negative.items() if t < now]:
            del self._negative[slug]

    def refresh(self, force_scan: bool = False) -> Dict[str, MarketWindow]:
        now = self.clock.now()
        new = self.probe_deterministic(now)
        if new is None:
            new = 0                      # unreachable: skip the scan too, keep what is cached
        elif force_scan or now - self._last_scan >= self.cfg.DISCOVERY_SCAN_INTERVAL_S:
            self._last_scan = now
            new += self.scan(now)
        self.prune(now)
        if new:
            log.info("discovery: %d new window(s), %d tracked", new, len(self.windows))
        return dict(self.windows)

    def relevant(self, now: float) -> List[MarketWindow]:
        """Windows that are open, or open within the prefetch horizon."""
        return [w for w in self.windows.values()
                if w.start_ts - self.cfg.DISCOVERY_PREFETCH_S <= now < w.end_ts and not w.closed]


def fetch_winner(gamma: GammaClient, slug: str) -> Optional[str]:
    ev = gamma.event_by_slug(slug)
    if not ev:
        return None
    for m in ev.get("markets") or []:
        if (m.get("slug") or ev.get("slug")) == slug or len(ev.get("markets") or []) == 1:
            return winner_from_market(m)
    return None

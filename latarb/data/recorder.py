"""Tick recorder and reader for backtests.

Raw frames are stored exactly as received (plus local receive time and the
source tag), so a replay re-runs the real parsers and the real hub/engine code
path. Format: gzip'd TSV, one frame per line:

    <recv_ts>\t<src>\t<raw frame>

src is a feed name (binance, coinbase, rtds, polymarket) or a meta record:
    @open / @close   raw = feed name          (connection lifecycle -> book resets)
    @markets         raw = JSON list of MarketWindow dicts (discovery snapshot)
    @outcome         raw = JSON {"slug":..., "winner": "up"|"down"}

Files rotate every RECORD_ROTATE_S (aligned), named ticks-YYYYmmdd-HHMMSS.tsv.gz,
so lexical order is chronological.
"""
from __future__ import annotations

import glob
import gzip
import io
import logging
import math
import os
from datetime import datetime, timezone
from typing import Iterable, Iterator, List, Optional, Tuple

log = logging.getLogger("latarb.recorder")


class TickRecorder:
    def __init__(self, directory: str, rotate_s: float = 3600.0) -> None:
        self.directory = directory
        self.rotate_s = rotate_s
        os.makedirs(directory, exist_ok=True)
        self._fh: Optional[io.TextIOBase] = None
        self._rotate_at = -math.inf
        self.lines = 0
        self.path = ""

    def _rotate(self, ts: float) -> None:
        self.close()
        start = math.floor(ts / self.rotate_s) * self.rotate_s
        self._rotate_at = start + self.rotate_s
        name = datetime.fromtimestamp(start, timezone.utc).strftime("ticks-%Y%m%d-%H%M%S.tsv.gz")
        self.path = os.path.join(self.directory, name)
        self._fh = gzip.open(self.path, "at", compresslevel=5, encoding="utf-8")
        log.info("recording ticks -> %s", self.path)

    def write(self, recv_ts: float, src: str, raw: str) -> None:
        if recv_ts >= self._rotate_at or self._fh is None:
            self._rotate(recv_ts)
        if "\n" in raw:
            raw = raw.replace("\r", " ").replace("\n", " ")   # JSON whitespace only; strings keep escaped \n
        self._fh.write("%.6f\t%s\t%s\n" % (recv_ts, src, raw))
        self.lines += 1

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def expand_paths(paths: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(glob.glob(os.path.join(p, "ticks-*.tsv*")))
        else:
            out.extend(glob.glob(p) or [p])
    return sorted(set(out), key=os.path.basename)


def iter_recording(paths: Iterable[str]) -> Iterator[Tuple[float, str, str]]:
    for path in expand_paths(paths):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            try:
                for line in fh:
                    parts = line.rstrip("\n").split("\t", 2)
                    if len(parts) != 3:
                        continue
                    try:
                        ts = float(parts[0])
                    except ValueError:
                        continue
                    yield ts, parts[1], parts[2]
            except EOFError:
                # a file from a crashed session can end mid-block; keep what was readable
                log.warning("%s: truncated gzip stream, stopping at the last complete frame", path)

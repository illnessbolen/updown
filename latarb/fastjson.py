"""JSON helpers: orjson when installed (3-5x faster parsing on the hot path), stdlib otherwise."""
from __future__ import annotations

import json
from typing import Any

try:  # pragma: no cover - depends on the environment
    import orjson as _orjson

    def loads(raw: str | bytes) -> Any:
        return _orjson.loads(raw)

    def dumps(obj: Any) -> str:
        return _orjson.dumps(obj).decode()

except ImportError:  # pragma: no cover
    def loads(raw: str | bytes) -> Any:
        return json.loads(raw)

    def dumps(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"))

"""Machine-readable trace sinks for CARVE experiments."""

from __future__ import annotations

import json
import pathlib
import threading
from collections.abc import Mapping
from typing import Any

from .contracts import RuntimeTrace


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class JsonlTraceSink:
    """Thread-safe append-only writer with one durable row per policy call."""

    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, trace: RuntimeTrace) -> None:
        payload: Mapping[str, Any] = trace.to_dict()
        row = json.dumps(payload, default=_json_default, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(row)
                handle.write("\n")

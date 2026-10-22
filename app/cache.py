"""Small in-memory TTL cache.

Keys for private-league data include a hash of the caller's cookies, so one
user's private league is never served to anyone else. Cookie values themselves
are never part of a key or a value.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

from app.espn.client import EspnCredentials


def credentials_key(creds: EspnCredentials | None) -> str:
    if creds is None:
        return "public"
    digest = hashlib.sha256(f"{creds.swid}\x00{creds.espn_s2}".encode()).hexdigest()
    return f"cred:{digest[:32]}"


class TTLCache:
    def __init__(self, ttl_seconds: float, max_items: int = 256):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._items: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()

    def get(self, key: tuple) -> Any | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: tuple, value: Any) -> None:
        self._items[key] = (time.monotonic() + self.ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

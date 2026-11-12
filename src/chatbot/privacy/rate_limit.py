"""Límite de solicitudes por clave (sesión o IP) en memoria. Mitiga abuso y agotamiento de cuota."""
from __future__ import annotations

import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, max_calls: int = 20, window_s: float = 60.0):
        self.max_calls, self.window_s = max_calls, window_s
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.max_calls:
            return False
        q.append(now)
        return True

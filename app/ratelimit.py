"""Usage limits for a public deployment, so the shared Gemini key's quota can't be used up by one visitor."""
from __future__ import annotations

import threading
import time
from collections import deque

MINUTE = 60.0
DAY = 24 * 60 * 60.0


class RateLimited(Exception):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """Sliding-window limits per visitor (per minute, per day) and for everyone together (per day).
    A limit of 0 switches that check off. In-memory: limits reset when the server restarts."""

    def __init__(self, per_minute: int = 0, per_day: int = 0, global_per_day: int = 0, clock=time.monotonic):
        self.per_minute = per_minute
        self.per_day = per_day
        self.global_per_day = global_per_day
        self._clock = clock
        self._visitors: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.per_minute or self.per_day or self.global_per_day)

    def check(self, visitor: str) -> None:
        """Count one request for this visitor, or raise RateLimited without counting it."""
        if not self.enabled:
            return
        now = self._clock()
        with self._lock:
            _expire(self._all, now - DAY)
            times = self._visitors.setdefault(visitor, deque())
            _expire(times, now - DAY)
            if self.global_per_day and len(self._all) >= self.global_per_day:
                raise RateLimited("KnowSure has reached its shared daily limit. Please try again tomorrow.",
                                  _retry(self._all, now, DAY))
            if self.per_day and len(times) >= self.per_day:
                raise RateLimited(f"Daily limit reached ({self.per_day} questions per day). Please try again tomorrow.",
                                  _retry(times, now, DAY))
            recent = [t for t in times if t > now - MINUTE]
            if self.per_minute and len(recent) >= self.per_minute:
                wait = max(1, int(recent[0] + MINUTE - now) + 1)
                raise RateLimited(f"Too many questions in a minute ({self.per_minute}/min). "
                                  f"Please wait {wait} seconds.", wait)
            times.append(now)
            self._all.append(now)
            if len(self._visitors) > 10000:  # forget visitors with nothing left in the window
                for key in [k for k, v in self._visitors.items() if not v]:
                    del self._visitors[key]

    def status(self) -> dict[str, int]:
        return {"per_minute": self.per_minute, "per_day": self.per_day, "global_per_day": self.global_per_day}


def _expire(times: deque[float], cutoff: float) -> None:
    while times and times[0] <= cutoff:
        times.popleft()


def _retry(times: deque[float], now: float, window: float) -> int:
    return max(1, int(times[0] + window - now) + 1)

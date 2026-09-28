"""Precise timing helpers (Windows timer resolution + hybrid sleep/spin)."""
from __future__ import annotations

import sys
import time

_period_set = False


def enable_high_res_timer() -> None:
    """Ask Windows for 1 ms scheduler granularity (no-op elsewhere)."""
    global _period_set
    if sys.platform != "win32" or _period_set:
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        _period_set = True
    except Exception:
        pass


def precise_sleep_until(t_target: float, spin_margin: float = 0.0015) -> None:
    """Sleep until ``time.perf_counter() >= t_target`` with sub-millisecond accuracy."""
    while True:
        now = time.perf_counter()
        remaining = t_target - now
        if remaining <= 0:
            return
        if remaining > spin_margin:
            time.sleep(remaining - spin_margin)
        else:
            # spin for the last ~1.5 ms
            while time.perf_counter() < t_target:
                pass
            return


class RateTicker:
    """Fixed-rate ticker that tolerates jitter without drifting."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.next_t = time.perf_counter()

    def wait(self) -> float:
        precise_sleep_until(self.next_t)
        now = time.perf_counter()
        self.next_t += self.period
        if now - self.next_t > self.period * 4:  # fell far behind: resync
            self.next_t = now + self.period
        return now


class EMA:
    def __init__(self, alpha: float = 0.05, init: float = 0.0):
        self.alpha = alpha
        self.value = init
        self.n = 0

    def update(self, x: float) -> float:
        if self.n == 0:
            self.value = x
        else:
            self.value += self.alpha * (x - self.value)
        self.n += 1
        return self.value

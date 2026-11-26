"""App watchdog: is the game process still alive? (overnight Phase 0)

Liveness covers the stream (socket, server, ADB ping). A crashed or OOM-killed game
leaves the stream healthy and the screen on the launcher, which liveness would only
notice 30-60 s later as a stuck screen. ``pidof`` notices within one period.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class AppWatchdog:
    def __init__(self, adb, package: str, *, every_s: float = 10.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.adb, self.package, self.every_s, self.clock = adb, package, every_s, clock
        self._last = -1e18
        self.seen_alive = False  # pidof may be missing on old phones: never act before it has worked
        self.deaths = 0

    def due(self) -> bool:
        return self.clock() - self._last >= self.every_s

    def died(self) -> bool:
        """Check now. True only if the app was seen running before and isn't any more."""
        self._last = self.clock()
        alive = bool(self.adb.pidof(self.package))
        if alive:
            self.seen_alive = True
            return False
        if not self.seen_alive:
            return False
        self.deaths += 1
        self.seen_alive = False  # re-arm once the relaunched app shows up
        return True

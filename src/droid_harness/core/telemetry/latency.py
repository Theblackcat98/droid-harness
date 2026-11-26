"""Latency bookkeeping: per-stage timestamps and percentile summaries."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StageClock:
    """Marks named stages for one frame; durations are in ms from the previous mark."""

    t0_ns: int = field(default_factory=time.perf_counter_ns)
    marks: dict[str, int] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        self.marks[stage] = time.perf_counter_ns()

    def durations_ms(self) -> dict[str, float]:
        out, prev = {}, self.t0_ns
        for k, t in self.marks.items():
            out[k] = (t - prev) / 1e6
            prev = t
        out["total"] = (prev - self.t0_ns) / 1e6
        return out


class LatencyStats:
    def __init__(self, window: int = 5000) -> None:
        self._series: dict[str, deque] = {}
        self.window = window

    def add(self, name: str, ms: float) -> None:
        self._series.setdefault(name, deque(maxlen=self.window)).append(ms)

    def add_many(self, durations: dict[str, float], prefix: str = "") -> None:
        for k, v in durations.items():
            self.add(prefix + k, v)

    def summary(self, name: str) -> dict[str, float]:
        return percentiles(list(self._series.get(name, [])))

    def all(self) -> dict[str, dict[str, float]]:
        return {k: self.summary(k) for k in self._series}


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    a = np.asarray(values, float)
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "mean": float(a.mean())}

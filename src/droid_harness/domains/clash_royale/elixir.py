"""Elixir models: your elixir (HUD fused with a model) and the opponent's (an interval).

Official phases: x1 for the first two minutes, x2 in the third; in overtime x2 in the
fourth minute and x3 in the fifth. About 1 elixir per 2.8 s at x1, capped at 10.
(Some modes differ; pass a different phase table for them.)
"""

from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_ELIXIR_X1 = 2.8
MAX_ELIXIR = 10.0
PHASES: tuple[tuple[float, float, str, float], ...] = (
    (0.0, 120.0, "x1", 1.0),
    (120.0, 180.0, "x2", 2.0),
    (180.0, 240.0, "ot_x2", 2.0),
    (240.0, 300.0, "ot_x3", 3.0),
)


def phase_at(t_s: float, phases=PHASES) -> str:
    for a, b, name, _ in phases:
        if a <= t_s < b:
            return name
    return phases[-1][2] if t_s >= phases[-1][1] else phases[0][2]


def multiplier_at(t_s: float, phases=PHASES) -> float:
    for a, b, _, m in phases:
        if a <= t_s < b:
            return m
    return phases[-1][3] if t_s >= phases[-1][1] else phases[0][3]


def generated_between(t0_s: float, t1_s: float, phases=PHASES) -> float:
    """Uncapped elixir generated in [t0, t1]."""
    if t1_s <= t0_s:
        return 0.0
    total = 0.0
    for a, b, _, m in phases:
        lo, hi = max(a, t0_s), min(b, t1_s)
        if hi > lo:
            total += (hi - lo) * m / SECONDS_PER_ELIXIR_X1
    tail_start = max(phases[-1][1], t0_s)
    if t1_s > tail_start:
        total += (t1_s - tail_start) * phases[-1][3] / SECONDS_PER_ELIXIR_X1
    return total


def time_to_reach(current: float, target: float, t_s: float, phases=PHASES) -> float:
    """Seconds until elixir reaches ``target`` from ``current`` at match time ``t_s``."""
    if current >= target:
        return 0.0
    need, t, step = target - current, t_s, 0.05
    while need > 1e-9 and t - t_s < 60:
        g = generated_between(t, t + step, phases)
        if g >= need:
            return t + step * need / g - t_s
        need -= g
        t += step
    return t - t_s


class ElixirModel:
    """Your elixir: a model advanced by time, nudged toward HUD readings.

    Persistent HUD/model disagreement is reported via ``discrepancy_s`` so triage can tag
    it as a perception problem instead of a strategy problem.
    """

    def __init__(self, start: float = 5.0, *, blend: float = 0.3, snap_after_s: float = 1.0,
                 tolerance: float = 0.7) -> None:
        self.value = start
        self.t_s = 0.0
        self.leaked = 0.0
        self.blend = blend
        self.snap_after_s = snap_after_s
        self.tolerance = tolerance
        self._disagree_since: float | None = None
        self.discrepancy_s = 0.0
        self.last_discrepancy = 0.0

    def reset(self, value: float, t_s: float = 0.0) -> None:
        self.value, self.t_s, self.leaked, self._disagree_since = value, t_s, 0.0, None

    def advance(self, t_s: float) -> float:
        g = generated_between(self.t_s, t_s)
        room = MAX_ELIXIR - self.value
        if g > room:
            self.leaked += g - max(room, 0.0)
        self.value = min(MAX_ELIXIR, self.value + g)
        self.t_s = max(self.t_s, t_s)
        return self.value

    def spend(self, cost: float) -> None:
        self.value = max(0.0, self.value - cost)

    def observe(self, hud_value: float | None, t_s: float) -> float:
        self.advance(t_s)
        if hud_value is None:
            return self.value
        d = hud_value - self.value
        self.last_discrepancy = d
        if abs(d) <= self.tolerance:
            self._disagree_since = None
            self.discrepancy_s = 0.0
            self.value += self.blend * d
        else:
            if self._disagree_since is None:
                self._disagree_since = t_s
            self.discrepancy_s = t_s - self._disagree_since
            if self.discrepancy_s >= self.snap_after_s:
                self.value = hud_value  # trust the HUD, but triage sees the discrepancy
        self.value = min(MAX_ELIXIR, max(0.0, self.value))
        return self.value


@dataclass
class OpponentElixir:
    """Opponent elixir as an interval [low, high].

    Generation is deterministic and the 10 cap collapses the interval (a full bar is
    known). Uncertainty comes from plays you can't fully see: unknown card costs (a range),
    and possibly missed plays (``miss_widen_per_s`` lowers ``low`` over time).
    """

    low: float = 5.0
    high: float = 5.0
    t_s: float = 0.0
    miss_widen_per_s: float = 0.0

    def advance(self, t_s: float) -> None:
        g = generated_between(self.t_s, t_s)
        self.low = min(MAX_ELIXIR, self.low + g - self.miss_widen_per_s * max(0.0, t_s - self.t_s))
        self.high = min(MAX_ELIXIR, self.high + g)
        self.low = max(0.0, self.low)
        self.t_s = max(self.t_s, t_s)

    def on_play(self, cost: float | tuple[float, float], t_s: float) -> None:
        self.advance(t_s)
        cmin, cmax = (cost, cost) if isinstance(cost, int | float) else cost
        # They could afford it, so they had at least cmin before playing.
        self.low = max(self.low, cmin)
        self.high = max(self.high, self.low)
        self.low = max(0.0, self.low - cmax)
        self.high = max(0.0, self.high - cmin)

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2

    def interval(self) -> tuple[float, float]:
        return (round(self.low, 2), round(self.high, 2))

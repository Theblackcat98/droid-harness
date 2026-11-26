"""Small statistics helpers: Wilson intervals, Thompson screening, abort-rate comparison."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class Arm:
    name: str
    successes: float = 0.0  # e.g. wins, or a [0,1]-scaled score summed
    trials: float = 0.0


def thompson_pick(arms: list[Arm], rng: random.Random | None = None, prior: tuple[float, float] = (1, 1)) -> Arm:
    """Choose which candidate plays next: sample each arm's Beta posterior, take the max."""
    rng = rng or random.Random()
    a0, b0 = prior
    return max(arms, key=lambda a: rng.betavariate(a0 + a.successes, b0 + a.trials - a.successes))


def prob_best(arms: list[Arm], draws: int = 4000, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    counts = {a.name: 0 for a in arms}
    for _ in range(draws):
        counts[thompson_pick(arms, rng).name] += 1
    return {k: v / draws for k, v in counts.items()}


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """z statistic and two-sided p-value for p1 != p2 (e.g. infra-abort rates between arms)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (x1 / n1 - x2 / n2) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval

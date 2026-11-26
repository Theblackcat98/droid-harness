"""Sequential probability ratio tests for accepting/rejecting strategy changes.

Average games to decide (H0 = 50%, alpha = beta = 0.05): ~530 for H1 = 55%, ~205 for 58%,
~130 for 60%, ~57 for 65%. One serious test is about one night on two phones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SprtResult:
    decision: str  # accept_H1 | accept_H0 | continue
    llr: float
    lower: float
    upper: float
    wins: int
    losses: int


def bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def llr(wins: int, losses: int, p0: float, p1: float) -> float:
    return wins * math.log(p1 / p0) + losses * math.log((1 - p1) / (1 - p0))


def sprt(wins: int, losses: int, p0: float = 0.50, p1: float = 0.58, alpha: float = 0.05,
         beta: float = 0.05) -> SprtResult:
    """Bernoulli SPRT on decisive games (draws excluded; use a trinomial GSPRT to include them)."""
    lo, hi = bounds(alpha, beta)
    v = llr(wins, losses, p0, p1)
    d = "accept_H1" if v >= hi else "accept_H0" if v <= lo else "continue"
    return SprtResult(d, v, lo, hi, wins, losses)


def paired_sprt(candidate_only_wins: int, champion_only_wins: int, p1: float = 0.6, alpha: float = 0.05,
                beta: float = 0.05) -> SprtResult:
    """Paired gauntlet test: only discordant pairs (one arm won, the other lost) carry information.
    H0: P(candidate wins a discordant pair) = 0.5."""
    return sprt(candidate_only_wins, champion_only_wins, 0.5, p1, alpha, beta)


def expected_games(p0: float = 0.5, p1: float = 0.58, alpha: float = 0.05, beta: float = 0.05,
                   under: str = "H1") -> float:
    """Wald's average sample number when the true win rate is p1 (``under="H1"``) or p0 ("H0")."""
    lo, hi = bounds(alpha, beta)
    p = p1 if under == "H1" else p0
    step = p * math.log(p1 / p0) + (1 - p) * math.log((1 - p1) / (1 - p0))
    if under == "H1":
        return ((1 - beta) * hi + beta * lo) / step
    return (alpha * hi + (1 - alpha) * lo) / step


def elo_from_p(p: float) -> float:
    return 400 * math.log10(p / (1 - p))


def p_from_elo(elo: float) -> float:
    return 1 / (1 + 10 ** (-elo / 400))

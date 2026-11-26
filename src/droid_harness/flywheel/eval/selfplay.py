"""Two-phone self-play scheduling with strategy <-> (phone, account) swapping.

Phones differ in speed and thermals, and accounts differ in card levels. Alternating which
setup runs the candidate every game makes those differences cancel instead of silently
favoring one arm. Afterwards, check that phone/account don't predict the result.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Seat:
    phone: str
    account: str


@dataclass(frozen=True)
class SelfPlayGame:
    index: int
    candidate_seat: Seat
    champion_seat: Seat
    host: Seat  # who sends the friendly-battle invite


def schedule(n_games: int, seat_a: Seat, seat_b: Seat) -> list[SelfPlayGame]:
    games = []
    for i in range(n_games):
        cand, champ = (seat_a, seat_b) if i % 2 == 0 else (seat_b, seat_a)
        host = seat_a if (i // 2) % 2 == 0 else seat_b  # also alternate who hosts
        games.append(SelfPlayGame(i, cand, champ, host))
    return games


def balance_report(games: list[SelfPlayGame]) -> dict[str, dict[str, int]]:
    return {
        "candidate_phone": dict(Counter(g.candidate_seat.phone for g in games)),
        "candidate_account": dict(Counter(g.candidate_seat.account for g in games)),
        "host": dict(Counter(g.host.phone for g in games)),
    }


def seat_effect(results: list[tuple[SelfPlayGame, bool]]) -> dict[str, float]:
    """Candidate win rate by the phone it played on. A large gap means a device bias."""
    by: dict[str, list[int]] = {}
    for g, cand_won in results:
        by.setdefault(g.candidate_seat.phone, []).append(int(cand_won))
    return {k: sum(v) / len(v) for k, v in by.items() if v}

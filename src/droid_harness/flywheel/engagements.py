"""Lane engagements and their value, with credit shared across the decisions inside them.

Per-unit kill attribution isn't observable from pixels, and per-decision rolling windows
count the same swing several times. Instead, segment each lane into engagements and score
the whole exchange from observable quantities:

    net_me  = board_me_end  - board_me_start  - spent_me
    net_opp = board_opp_end - board_opp_start - spent_opp
    value   = (net_me - net_opp) + lambda * (tower_dmg_dealt - tower_dmg_taken)

Board value = sum over a side's units in the lane of (card cost / units per card) x hp.
Each of our plays gets ``value x its cost / spent_me``. Play value ranks rules and finds
problems; it is not used to accept changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domains.clash_royale.arena import Arena
from ..domains.clash_royale.cards import CardDB
from .timeline import MatchTimeline, Play, Point


@dataclass
class Engagement:
    eng_id: int
    lane: str
    t_start: int
    t_end: int
    plays: list[Play] = field(default_factory=list)
    spent_me: float = 0.0
    spent_opp: float = 0.0
    board_me_start: float = 0.0
    board_me_end: float = 0.0
    board_opp_start: float = 0.0
    board_opp_end: float = 0.0
    tower_dmg_dealt: int = 0
    tower_dmg_taken: int = 0
    value: float = 0.0
    credit: dict[int, float] = field(default_factory=dict)  # decision seq -> play value
    triage: str = "UNTAGGED"
    evidence: dict = field(default_factory=dict)


def _lane(arena: Arena, x: float) -> str:
    return arena.lane_of(x)


def board_value(p: Point, side: str, lane: str, arena: Arena, cards: CardDB) -> float:
    return sum(cards.unit_value(kind) * hp for _, kind, s, x, _y, _vx, _vy, hp in p.units
               if s == side and _lane(arena, x) == lane)


def contested(p: Point, lane: str, arena: Arena, advancing: float = 0.2) -> bool:
    """A lane is live while a unit is across the river, or still advancing toward it."""
    for _, _kind, s, x, y, _vx, vy, _hp in p.units:
        if _lane(arena, x) != lane:
            continue
        if s == "opp" and (y < arena.river_y[1] or vy < -advancing):
            return True
        if s == "me" and (y > arena.river_y[0] or vy > advancing):
            return True
    return False


def _towers(lane: str, side: str, towers: dict[str, int]) -> int:
    """HP that matters for a lane: the princess tower, plus the king once that princess is down."""
    princess = towers.get(f"{side}_{lane}", 0)
    king = towers.get(f"{side}_king", 0) if princess == 0 else 0
    return princess + king


def _at(points: list[Point], t: int) -> Point:
    best = points[0]
    for p in points:
        if p.t_ms <= t:
            best = p
        else:
            break
    return best


def segment(tl: MatchTimeline, arena: Arena, cards: CardDB, *, quiet_ms: int = 2000, cap_ms: int = 25_000,
            lam: float = 0.01) -> list[Engagement]:
    if not tl.points:
        return []
    out: list[Engagement] = []
    for lane in ("left", "right"):
        lane_plays = [pl for pl in tl.plays if pl.tile is not None and _lane(arena, pl.tile[0]) == lane]
        open_eng: Engagement | None = None
        last_active = 0
        pi = 0
        for p in tl.points:
            active = contested(p, lane, arena)
            while pi < len(lane_plays) and lane_plays[pi].t_ms <= p.t_ms:
                active = True
                if open_eng is None:
                    open_eng = Engagement(0, lane, lane_plays[pi].t_ms, lane_plays[pi].t_ms)
                    last_active = lane_plays[pi].t_ms
                pi += 1
            if active:
                if open_eng is None:
                    open_eng = Engagement(0, lane, p.t_ms, p.t_ms)
                last_active = p.t_ms
            if open_eng is not None:
                open_eng.t_end = p.t_ms
                if p.t_ms - last_active >= quiet_ms or p.t_ms - open_eng.t_start >= cap_ms:
                    out.append(open_eng)
                    open_eng = None
        if open_eng is not None:
            out.append(open_eng)
    out.sort(key=lambda e: (e.t_start, e.lane))
    for i, e in enumerate(out):
        e.eng_id = i
        _score(e, tl, arena, cards, lam)
    return out


def _score(e: Engagement, tl: MatchTimeline, arena: Arena, cards: CardDB, lam: float) -> None:
    start, end = _at(tl.points, e.t_start - 1), _at(tl.points, e.t_end)
    e.plays = [pl for pl in tl.plays if pl.tile is not None and _lane(arena, pl.tile[0]) == e.lane
               and e.t_start <= pl.t_ms <= e.t_end]
    e.spent_me = sum(cards.cost(pl.card) for pl in e.plays if pl.side == "me" and pl.card in cards)
    e.spent_opp = sum(cards.cost(pl.card) for pl in e.plays if pl.side == "opp" and pl.card in cards)
    e.board_me_start = board_value(start, "me", e.lane, arena, cards)
    e.board_me_end = board_value(end, "me", e.lane, arena, cards)
    e.board_opp_start = board_value(start, "opp", e.lane, arena, cards)
    e.board_opp_end = board_value(end, "opp", e.lane, arena, cards)
    e.tower_dmg_dealt = max(0, _towers(e.lane, "opp", start.towers) - _towers(e.lane, "opp", end.towers))
    e.tower_dmg_taken = max(0, _towers(e.lane, "me", start.towers) - _towers(e.lane, "me", end.towers))
    net_me = e.board_me_end - e.board_me_start - e.spent_me
    net_opp = e.board_opp_end - e.board_opp_start - e.spent_opp
    e.value = (net_me - net_opp) + lam * (e.tower_dmg_dealt - e.tower_dmg_taken)
    mine = [pl for pl in e.plays if pl.side == "me" and pl.decision_seq is not None and pl.card in cards]
    if e.spent_me > 0:
        for pl in mine:
            e.credit[pl.decision_seq] = e.credit.get(pl.decision_seq, 0.0) + e.value * cards.cost(pl.card) / e.spent_me

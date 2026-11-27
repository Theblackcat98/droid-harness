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


DEFAULT_PRINCESS_HP = 1400
DEFAULT_KING_HP = 2400


def _towers(lane: str, side: str, towers: dict[str, int], *, infer_missing: dict[str, int] | None = None) -> int:
    """HP that matters for a lane: the princess tower, plus the king once that princess is down."""
    pkey, kkey = f"{side}_{lane}", f"{side}_king"
    if pkey in towers:
        princess = towers[pkey]
    elif infer_missing and pkey in infer_missing:
        princess = infer_missing[pkey]
    else:
        princess = 0
    if princess == 0:
        if kkey in towers:
            king = towers[kkey]
        elif infer_missing and kkey in infer_missing:
            king = infer_missing[kkey]
        else:
            king = 0
    else:
        king = 0
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
    baselines = _infer_baselines(tl)
    for i, e in enumerate(out):
        e.eng_id = i
        _score(e, tl, arena, cards, lam, baselines=baselines)
    return out


def _opp_cost(card: str, cards: CardDB) -> float:
    if card in cards:
        return float(cards.cost(card))
    return 3.5 if card in ("other_ground", "other_air") else 0.0


def _infer_baselines(tl: MatchTimeline) -> dict[str, int]:
    """Infer baseline tower max HP across the match so legacy/unactivated towers get real destruction delta."""
    max_seen: dict[str, int] = {}
    ever_seen: set[str] = set()
    for pt in tl.points:
        for k, v in pt.towers.items():
            ever_seen.add(k)
            if v > max_seen.get(k, 0):
                max_seen[k] = v
    base: dict[str, int] = {}
    for side in ("me", "opp"):
        p_ref = max(max_seen.get(f"{side}_left", 0), max_seen.get(f"{side}_right", 0)) or DEFAULT_PRINCESS_HP
        for lane in ("left", "right"):
            pk = f"{side}_{lane}"
            if pk in ever_seen:
                base[pk] = max_seen.get(pk, 0) or p_ref
        kk = f"{side}_king"
        if kk in ever_seen or any(f"{side}_{ln}" in ever_seen for ln in ("left", "right")):
            base[kk] = max_seen.get(kk, 0) or (int(round(p_ref * 1.66)) if p_ref != DEFAULT_PRINCESS_HP else DEFAULT_KING_HP)
    return base


def _score(e: Engagement, tl: MatchTimeline, arena: Arena, cards: CardDB, lam: float,
           baselines: dict[str, int] | None = None) -> None:
    start, end = _at(tl.points, e.t_start - 1), _at(tl.points, e.t_end)
    e.plays = [pl for pl in tl.plays if pl.tile is not None and _lane(arena, pl.tile[0]) == e.lane
               and e.t_start <= pl.t_ms <= e.t_end]
    e.spent_me = sum(cards.cost(pl.card) for pl in e.plays if pl.side == "me" and pl.card in cards)
    e.spent_opp = sum(_opp_cost(pl.card, cards) for pl in e.plays if pl.side == "opp")
    e.board_me_start = board_value(start, "me", e.lane, arena, cards)
    e.board_me_end = board_value(end, "me", e.lane, arena, cards)
    e.board_opp_start = board_value(start, "opp", e.lane, arena, cards)
    e.board_opp_end = board_value(end, "opp", e.lane, arena, cards)
    base = baselines if baselines is not None else _infer_baselines(tl)
    e.tower_dmg_dealt = max(0, _towers(e.lane, "opp", start.towers, infer_missing=base)
                            - _towers(e.lane, "opp", end.towers, infer_missing=base))
    e.tower_dmg_taken = max(0, _towers(e.lane, "me", start.towers, infer_missing=base)
                            - _towers(e.lane, "me", end.towers, infer_missing=base))
    net_me = e.board_me_end - e.board_me_start - e.spent_me
    net_opp = e.board_opp_end - e.board_opp_start - e.spent_opp
    e.value = (net_me - net_opp) + lam * (e.tower_dmg_dealt - e.tower_dmg_taken)
    mine = [pl for pl in e.plays if pl.side == "me" and pl.decision_seq is not None and pl.card in cards]
    if e.spent_me > 0:
        for pl in mine:
            e.credit[pl.decision_seq] = e.credit.get(pl.decision_seq, 0.0) + e.value * cards.cost(pl.card) / e.spent_me

"""Root-cause triage: only strategy problems reach the coach.

PERCEPTION -> labeling/calibration queue · EXECUTION -> executor bug queue ·
LATENCY -> profiling queue · STRATEGY -> coach. Triage never removes matches from win-rate
tests (see ``eval/inclusion.py``); it only decides what the coach and diagnostics see.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domains.clash_royale.arena import Arena
from .engagements import Engagement
from .timeline import MatchTimeline

HARD_PERCEPTION_WARNINGS = {"elixir_discrepancy", "tower_hp_increase"}
SLOT_TRANSITION_WARNINGS = {"hand_unreadable", "affordability_mismatch"}
PERCEPTION_WARNINGS = HARD_PERCEPTION_WARNINGS | SLOT_TRANSITION_WARNINGS


@dataclass
class TriageParams:
    perception_warning_frames: int = 2
    slot_transition_pre_ms: int = 100
    slot_transition_post_ms: int = 600
    slot_warning_min_frames: int = 12
    slot_warning_min_share: float = 0.45
    exec_confirm_ms: int = 600
    exec_tile_error: float = 1.5
    latency_p99_ms: float = 100.0
    frame_gap_ms: float = 250.0
    strategy_value_threshold: float = -1.0


def triage(e: Engagement, tl: MatchTimeline, arena: Arena, p: TriageParams | None = None) -> Engagement:
    p = p or TriageParams()
    window = [pt for pt in tl.points if e.t_start <= pt.t_ms <= e.t_end]
    my_plays = [pl for pl in tl.plays if pl.side == "me"]

    hard_frames = sum(1 for pt in window if HARD_PERCEPTION_WARNINGS & set(pt.warnings))
    if hard_frames >= p.perception_warning_frames:
        kinds = sorted({w for pt in window for w in pt.warnings if w in HARD_PERCEPTION_WARNINGS})
        return _tag(e, "PERCEPTION", frames=hard_frames, warnings=kinds)

    def _in_play_transition(t_ms: int) -> bool:
        return any(-p.slot_transition_pre_ms <= (t_ms - pl.t_ms) <= p.slot_transition_post_ms for pl in my_plays)

    slot_frames = sum(
        1 for pt in window
        if (SLOT_TRANSITION_WARNINGS & set(pt.warnings)) and not _in_play_transition(pt.t_ms)
    )
    if window and slot_frames >= p.slot_warning_min_frames and (slot_frames / len(window)) >= p.slot_warning_min_share:
        kinds = sorted({w for pt in window for w in pt.warnings if w in SLOT_TRANSITION_WARNINGS})
        return _tag(e, "PERCEPTION", frames=slot_frames, warnings=kinds)
    for c in tl.chosen:
        exec_ms = c.execute_at_ms if c.execute_at_ms is not None else c.t_ms
        if not (e.t_start <= exec_ms <= e.t_end) or arena.lane_of(c.tile[0]) != e.lane:
            continue
        if c.rule.startswith("menu"):
            continue
        confirmed = [pl for pl in my_plays if pl.decision_seq == c.seq]
        if not confirmed and not any(pl.card == c.card and 0 <= pl.t_ms - exec_ms <= p.exec_confirm_ms
                                     for pl in my_plays):
            return _tag(e, "EXECUTION", decision_seq=c.seq, card=c.card, reason="card never left hand")

    loop = sorted(pt.loop_ms for pt in window if pt.loop_ms is not None)
    if loop:
        p99 = loop[min(len(loop) - 1, int(0.99 * len(loop)))]
        if p99 > p.latency_p99_ms:
            return _tag(e, "LATENCY", loop_p99_ms=p99)
    gaps = [b[0] - a[0] for a, b in zip(tl.frame_ms, tl.frame_ms[1:], strict=False)
            if a[0] >= e.t_start and b[0] <= e.t_end]  # battles always animate: a gap is an anomaly
    if gaps and max(gaps) > p.frame_gap_ms:
        return _tag(e, "LATENCY", frame_gap_ms=round(max(gaps), 1))

    if e.value < p.strategy_value_threshold:
        return _tag(e, "STRATEGY", value=round(e.value, 2))
    return _tag(e, "OK", value=round(e.value, 2))


def _tag(e: Engagement, tag: str, **evidence) -> Engagement:
    e.triage, e.evidence = tag, evidence
    return e

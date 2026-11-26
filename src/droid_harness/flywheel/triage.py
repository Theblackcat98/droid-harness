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

PERCEPTION_WARNINGS = {"elixir_discrepancy", "hand_unreadable", "affordability_mismatch", "tower_hp_increase"}


@dataclass
class TriageParams:
    perception_warning_frames: int = 2
    exec_confirm_ms: int = 600
    exec_tile_error: float = 1.5
    latency_p99_ms: float = 100.0
    frame_gap_ms: float = 250.0
    strategy_value_threshold: float = -1.0


def triage(e: Engagement, tl: MatchTimeline, arena: Arena, p: TriageParams | None = None) -> Engagement:
    p = p or TriageParams()
    window = [pt for pt in tl.points if e.t_start <= pt.t_ms <= e.t_end]

    warn_frames = sum(1 for pt in window if PERCEPTION_WARNINGS & set(pt.warnings))
    if warn_frames >= p.perception_warning_frames:
        kinds = sorted({w for pt in window for w in pt.warnings if w in PERCEPTION_WARNINGS})
        return _tag(e, "PERCEPTION", frames=warn_frames, warnings=kinds)

    my_plays = [pl for pl in tl.plays if pl.side == "me"]
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
